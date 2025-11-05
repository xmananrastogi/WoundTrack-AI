#!/usr/bin/env python3
"""
Robust cell_tracking module.

- Primary strategy: try trackpy (if available) for good detection/linking.
- Fallback: connected-components + Hungarian matching linking for dense fields.
- Outputs trajectories.csv, velocities.csv, trajectories_plot.png into output_dir.
- Returns a dictionary with keys expected by the rest of the pipeline.
- NEW: Calculates Directionality (mean cosine similarity to wound center).
"""
import os
import math
import csv
import numpy as np
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Optional

# Optional trackpy usage
try:
    import trackpy as tp

    TP_AVAILABLE = True
    tp.quiet()
except Exception:
    TP_AVAILABLE = False


# ---------- Utils: detect centroids from mask ----------
def detect_centroids_from_mask_array(mask: np.ndarray, min_area_px: int = 4):
    """
    Given a binary mask (0/255 or 0/1), return list of (cx, cy, area_px)
    """
    try:
        if mask.dtype != np.uint8:
            mask_u8 = (mask > 0).astype(np.uint8) * 255
        else:
            mask_u8 = mask.copy()
        # ensure binary
        _, bw = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        centers = []
        for i in range(1, nlabels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            centers.append((cx, cy, area))
        return centers
    except Exception:
        return []


def detect_centroids_from_mask_paths(mask_paths: List[str], min_area_px: int = 4):
    frames = []
    for p in mask_paths:
        try:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                frames.append([])
            else:
                frames.append(detect_centroids_from_mask_array(m, min_area_px=min_area_px))
        except Exception:
            frames.append([])
    return frames


# ---------- NEW: Helper to find wound centers ----------
def get_wound_centers(masks: List[Any]) -> List[Optional[tuple]]:
    """
    Given a list of wound masks (as arrays or paths), find the centroid of the wound area for each frame.
    This is the "target" for directionality.
    """
    centers = []
    for m in masks:
        try:
            mask_array = None
            if isinstance(m, str):
                mask_array = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            elif isinstance(m, np.ndarray):
                mask_array = m

            if mask_array is None or mask_array.max() == 0:
                centers.append(None)
                continue

            # Ensure binary
            if mask_array.max() > 1:
                mask_array = (mask_array > 127).astype(np.uint8)
            else:
                mask_array = mask_array.astype(np.uint8)

            M = cv2.moments(mask_array)
            if M["m00"] == 0:
                centers.append(None)
            else:
                cX = float(M["m10"] / M["m00"])
                cY = float(M["m01"] / M["m00"])
                centers.append((cX, cY))
        except Exception:
            centers.append(None)
    return centers


# ---------- Linking (Hungarian fallback) ----------
def link_centroids_hungarian(frames_centroids: List[List[tuple]], max_disp_px: float = 20.0, memory: int = 2):
    """
    Link centroids using a greedy/Hungarian matching with memory.
    frames_centroids[t] = [(x,y,area), ...]
    Returns tracks: dict track_id -> list of (frame_idx, x, y)
    """
    tracks = {}
    active = {}  # track_id -> {'pos':(x,y), 'missed':int, 'last_frame':int}
    next_id = 0

    for t, centers in enumerate(frames_centroids):
        pts = np.array([[c[0], c[1]] for c in centers], dtype=np.float32) if centers else np.zeros((0, 2),
                                                                                                   dtype=np.float32)

        if len(active) == 0 and pts.shape[0] > 0:
            # initialize tracks
            for p in pts:
                tracks[next_id] = [(t, float(p[0]), float(p[1]))]
                active[next_id] = {'pos': (float(p[0]), float(p[1])), 'missed': 0, 'last_frame': t}
                next_id += 1
            continue

        if pts.shape[0] == 0:
            for tid in list(active.keys()):
                active[tid]['missed'] += 1
                if active[tid]['missed'] > memory:
                    del active[tid]
            continue

        track_ids = list(active.keys())
        track_pos = np.array([active[tid]['pos'] for tid in track_ids], dtype=np.float32) if track_ids else np.zeros(
            (0, 2), dtype=np.float32)

        if track_pos.shape[0] == 0:
            for p in pts:
                tracks[next_id] = [(t, float(p[0]), float(p[1]))]
                active[next_id] = {'pos': (float(p[0]), float(p[1])), 'missed': 0, 'last_frame': t}
                next_id += 1
            continue

        D = np.linalg.norm(track_pos[:, None, :] - pts[None, :, :], axis=2)  # (n_tracks, n_pts)
        row_ind, col_ind = linear_sum_assignment(D)
        assigned_tracks = set()
        assigned_points = set()
        for r, c in zip(row_ind, col_ind):
            if D[r, c] <= max_disp_px:
                tid = track_ids[r]
                px, py = float(pts[c, 0]), float(pts[c, 1])
                tracks[tid].append((t, px, py))
                active[tid]['pos'] = (px, py)
                active[tid]['missed'] = 0
                active[tid]['last_frame'] = t
                assigned_tracks.add(tid)
                assigned_points.add(c)

        # new tracks for unassigned points
        for pi in range(pts.shape[0]):
            if pi not in assigned_points:
                p = pts[pi]
                tracks[next_id] = [(t, float(p[0]), float(p[1]))]
                active[next_id] = {'pos': (float(p[0]), float(p[1])), 'missed': 0, 'last_frame': t}
                next_id += 1

        # prune unassigned tracks
        for tid in list(active.keys()):
            if tid not in assigned_tracks:
                active[tid]['missed'] += 1
                if active[tid]['missed'] > memory:
                    del active[tid]

    # filter short tracks
    final_tracks = {tid: tr for tid, tr in tracks.items() if len(tr) >= 2}
    return final_tracks


# ---------- Trackpy wrapper ----------
def trackpy_linking_wrapper(frames_centroids, search_range=15, memory=3):
    """
    Use trackpy: expects DataFrame with x,y,frame entries. If trackpy fails, raise exception.
    """
    import pandas as pd
    df_rows = []
    for t, centers in enumerate(frames_centroids):
        for c in centers:
            df_rows.append({'x': float(c[0]), 'y': float(c[1]), 'frame': int(t)})

    if len(df_rows) == 0:
        return {}

    df = pd.DataFrame(df_rows)
    try:
        import trackpy as tp
        tp.quiet()
        linked = tp.link(df, search_range=search_range, memory=memory, t_column='frame')
        linked = tp.filter_stubs(linked, threshold=3)
        tracks = {}
        for pid, group in linked.groupby('particle'):
            lst = [(int(r['frame']), float(r['x']), float(r['y'])) for _, r in group.sort_values('frame').iterrows()]
            if len(lst) >= 2:
                tracks[int(pid)] = lst
        return tracks
    except Exception as e:
        raise


# ---------- Compute velocities & metrics ----------
def compute_tracking_metrics(tracks: Dict[int, List[tuple]],
                             wound_centers: List[Optional[tuple]],
                             time_interval_hours: float,
                             pixel_scale_um_per_px: float):
    """
    tracks: {tid: [(frame, x, y), ...]}
    wound_centers: list of (cx, cy) for each frame
    returns dict with metrics and rows for csv
    """
    traj_rows = []
    speeds = []
    efficiencies = []
    displacements = []
    path_lengths = []
    directionalities = []  # NEW

    for tid, pts in tracks.items():
        if len(pts) < 2:
            continue

        # --- Basic Metrics ---
        start_pt = pts[0]
        end_pt = pts[-1]
        start_frame = start_pt[0]

        # record traj rows
        for fr, x, y in pts:
            traj_rows.append([tid, fr, x, y])

        # compute stepwise distances (um) and velocities (um/min)
        prev_x, prev_y = start_pt[1], start_pt[2]
        total_path = 0.0
        for (fr, x, y) in pts[1:]:
            dx = (x - prev_x) * pixel_scale_um_per_px
            dy = (y - prev_y) * pixel_scale_um_per_px
            d = math.hypot(dx, dy)
            total_path += d
            prev_x, prev_y = x, y

        # net displacement
        dx_net = (end_pt[1] - start_pt[1]) * pixel_scale_um_per_px
        dy_net = (end_pt[2] - start_pt[2]) * pixel_scale_um_per_px
        net_disp = math.hypot(dx_net, dy_net)

        displacements.append(net_disp)
        path_lengths.append(total_path)
        eff = (net_disp / total_path) if total_path > 0 else 0.0
        efficiencies.append(eff)

        # compute mean velocity across steps
        step_dists = []
        prev_x, prev_y = pts[0][1], pts[0][2]
        for (fr, x, y) in pts[1:]:
            dx = (x - prev_x) * pixel_scale_um_per_px
            dy = (y - prev_y) * pixel_scale_um_per_px
            d = math.hypot(dx, dy)
            step_dists.append(d)
            prev_x, prev_y = x, y

        if time_interval_hours > 0 and len(step_dists) > 0:
            step_speeds = [sd / (time_interval_hours * 60.0) for sd in step_dists]
            speeds.extend(step_speeds)

        # --- NEW: Directionality ---
        # Get the "target" wound center from the cell's starting frame
        target_center = wound_centers[start_frame] if start_frame < len(wound_centers) else None

        if target_center is not None:
            # Vector from cell start to wound center (target vector)
            vec_target_x = target_center[0] - start_pt[1]
            vec_target_y = target_center[1] - start_pt[2]
            mag_target = math.hypot(vec_target_x, vec_target_y)

            # Vector of cell displacement (displacement vector)
            vec_disp_x = end_pt[1] - start_pt[1]
            vec_disp_y = end_pt[2] - start_pt[2]
            mag_disp = math.hypot(vec_disp_x, vec_disp_y)

            if mag_target > 0 and mag_disp > 0:
                # Cosine similarity
                dot_product = (vec_target_x * vec_disp_x) + (vec_target_y * vec_disp_y)
                directionality = dot_product / (mag_target * mag_disp)
                directionalities.append(directionality)
            else:
                directionalities.append(0.0)  # No movement or already at target

    num_tracks = int(len(path_lengths))
    mean_velocity_um_min = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    migration_efficiency_mean = float(np.mean(efficiencies)) if len(efficiencies) > 0 else 0.0
    mean_displacement = float(np.mean(displacements)) if len(displacements) > 0 else 0.0
    mean_path_length = float(np.mean(path_lengths)) if len(path_lengths) > 0 else 0.0
    mean_directionality = float(np.mean(directionalities)) if len(directionalities) > 0 else 0.0  # NEW

    return {
        'num_cells_tracked': num_tracks,
        'mean_velocity_um_min': mean_velocity_um_min,
        'migration_efficiency_mean': migration_efficiency_mean,
        'mean_directionality': mean_directionality,  # NEW
        'mean_displacement_um': mean_displacement,
        'mean_path_length_um': mean_path_length,
        'trajectories_rows': traj_rows
    }


# ---------- Visualization ----------
def save_trajectories_plot(tracks: Dict[int, List[tuple]], img_path: str, out_path: str, max_tracks: int = 100):
    # create canvas from first image if possible
    try:
        canvas = None
        if img_path and os.path.exists(img_path):
            canvas = cv2.imread(img_path)
            if canvas is not None:
                if len(canvas.shape) == 2 or canvas.shape[2] == 1:
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
                canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            else:
                canvas_rgb = None
        else:
            canvas_rgb = None

        fig, ax = plt.subplots(figsize=(8, 6))
        if canvas_rgb is not None:
            ax.imshow(canvas_rgb)
            ax.set_xlim(0, canvas_rgb.shape[1])
            ax.set_ylim(canvas_rgb.shape[0], 0)
        else:
            ax.set_facecolor('black')

        keys = list(tracks.keys())
        if len(keys) > max_tracks:
            keys = list(np.random.choice(keys, max_tracks, replace=False))

        cmap = plt.get_cmap('tab20')
        for i, tid in enumerate(keys):
            pts = tracks[tid]
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.plot(xs, ys, '-o', linewidth=1.2, markersize=3, color=cmap(i % 20), alpha=0.8)
        ax.set_title(f'Trajectories (n={len(keys)})')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception:
        return None


# ---------- Main entrypoint ----------
def track_cells_in_timeseries(image_files: List[str], masks: List[Any],
                              time_interval: float, pixel_scale: float,
                              output_dir: str):
    """
    image_files: list of image file paths (may be used for plotting)
    masks: list of numpy arrays (wound gap masks)
    time_interval: hours per frame
    pixel_scale: um per pixel
    output_dir: directory to write tracking outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- NEW PLAN ---
    # 1. Get wound centers from `masks` (wound gap masks). This is correct.
    wound_centers = get_wound_centers(masks)

    # 2. Detect cell centroids from *image_files*.
    #    We need a new detection function that uses the *image* not a mask.
    #    Let's use a simple blob detector on the *inverse* of the wound.

    frames_centroids = []
    for i, img_path in enumerate(image_files):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                frames_centroids.append([])
                continue

            wound_mask = masks[i] if i < len(masks) else None
            if wound_mask is None:
                # If no mask, try to detect on whole image (less ideal)
                wound_mask_u8 = np.zeros_like(img, dtype=np.uint8)
            else:
                # Ensure mask is correct format
                if wound_mask.dtype != np.uint8:
                    wound_mask_u8 = (wound_mask > 0).astype(np.uint8)
                else:
                    wound_mask_u8 = wound_mask

            # Create cell mask (inverse of wound mask)
            cell_area_mask = 1 - wound_mask_u8  # Invert

            # Apply cell mask to image
            cell_img = cv2.bitwise_and(img, img, mask=cell_area_mask)

            # Detect cells in this area
            # Using thresholding + connected components is a good substitute for blob detection
            # Apply Gaussian blur to reduce noise before thresholding
            blurred_img = cv2.GaussianBlur(cell_img, (5, 5), 0)

            _, thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply mask again in case threshold bleeds
            thresh = cv2.bitwise_and(thresh, thresh, mask=cell_area_mask)

            # Find components (cells)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

            centers = []
            for j in range(1, nlabels):  # Skip background
                area = int(stats[j, cv2.CC_STAT_AREA])
                if area >= 4 and area < 500:  # Filter noise and huge blobs
                    cx, cy = float(centroids[j][0]), float(centroids[j][1])
                    centers.append((cx, cy, area))
            frames_centroids.append(centers)

        except Exception as e:
            print(f"Error detecting cells in frame {i}: {e}")
            frames_centroids.append([])

    total_positions = sum(len(f) for f in frames_centroids)
    if total_positions == 0:
        return {'num_cells_tracked': 0, 'mean_velocity_um_min': 0.0, 'migration_efficiency_mean': 0.0,
                'mean_directionality': 0.0}

    # Attempt trackpy linking first (if available)
    tracks = {}
    if TP_AVAILABLE:
        try:
            tracks = trackpy_linking_wrapper(frames_centroids, search_range=15, memory=3)
        except Exception:
            tracks = {}

    # If trackpy didn't produce tracks, use Hungarian fallback
    if not tracks:
        tracks = link_centroids_hungarian(frames_centroids, max_disp_px=20.0, memory=2)

    # Compute metrics, NOW passing wound_centers
    metrics = compute_tracking_metrics(tracks, wound_centers,
                                       time_interval_hours=time_interval,
                                       pixel_scale_um_per_px=pixel_scale)

    # Save trajectories CSV
    traj_csv = os.path.join(output_dir, 'trajectories.csv')
    with open(traj_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['track_id', 'frame', 'x_px', 'y_px'])
        for row in metrics.get('trajectories_rows', []):
            w.writerow(row)

    # Compute velocities CSV (per-track summary)
    vel_csv = os.path.join(output_dir, 'velocities.csv')
    with open(vel_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['num_cells_tracked', 'mean_velocity_um_min', 'migration_efficiency_mean', 'mean_directionality',
                    'mean_displacement_um', 'mean_path_length_um'])
        w.writerow([
            metrics['num_cells_tracked'],
            metrics['mean_velocity_um_min'],
            metrics['migration_efficiency_mean'],
            metrics['mean_directionality'],  # NEW
            metrics['mean_displacement_um'],
            metrics['mean_path_length_um']
        ])

    # Save trajectory plot (overlay on first image if available)
    traj_png = os.path.join(output_dir, 'trajectories_plot.png')
    plot_res = save_trajectories_plot(tracks, image_files[0] if image_files else None, traj_png)
    if not plot_res:
        traj_png = None

    # Return all metrics
    return {
        'num_cells_tracked': int(metrics['num_cells_tracked']),
        'mean_velocity_um_min': float(metrics['mean_velocity_um_min']),
        'migration_efficiency_mean': float(metrics['migration_efficiency_mean']),
        'mean_directionality': float(metrics['mean_directionality']),  # NEW
        'mean_displacement_um': float(metrics['mean_displacement_um']),
        'mean_path_length_um': float(metrics['mean_path_length_um']),
        'trajectories_csv': traj_csv,
        'trajectories_plot': traj_png,
    }