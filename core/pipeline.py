# core/pipeline.py
import os
import tempfile
import time
import gc
import threading
import queue
import numpy as np
import cv2
import logging

from core.interfaces import AnalysisPipeline
from utils.errors import SegmentationError

# Import scientific modules
from modules.segmentation import segment_wound_from_array, compute_wound_quality_score, detect_wound_axis
from modules.quantification import estimate_flatfield, apply_flatfield_correction, analyze_time_series
from modules.cell_tracking import track_cells_in_timeseries

logger = logging.getLogger(__name__)


def _threaded_image_loader(image_files, img_queue, stop_event):
    """Worker thread to load images from disk to RAM."""
    for f in image_files:
        if stop_event.is_set():
            break
        try:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            # Use a timeout to occasionally check the stop_event even if queue is full
            while not stop_event.is_set():
                try:
                    img_queue.put(img, timeout=0.1)
                    break
                except queue.Full:
                    continue
        except Exception as e:
            logger.error(f"Threaded loader failed on {f}: {e}")
            while not stop_event.is_set():
                try:
                    img_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
    img_queue.put("STOP")


class WoundHealingPipeline(AnalysisPipeline):
    """
    Pure computational pipeline. Zero Flask dependencies.
    Implements Memory-Mapped (memmap) arrays to handle massive datasets safely.
    """

    def run(self, image_files: list, config: dict) -> dict:
        t_start = time.time()
        n_frames = len(image_files)
        if n_frames < 2:
            raise SegmentationError("Need at least 2 frames for analysis.")

        # 🔴 CRITICAL FIX 1: Sanitize Inputs!
        # Force disk_size to be an ODD integer to prevent OpenCV Gaussian Blur crashes
        try:
            raw_disk = float(config.get("disk_size", 10.0))
            disk_size = int(raw_disk)
            if disk_size % 2 == 0:
                disk_size += 1
        except (ValueError, TypeError):
            disk_size = 11

        # Force scales to be valid floats
        try:
            time_interval = float(config.get("time_interval", 0.25))
            pixel_scale = float(config.get("pixel_scale", 1.0))
        except (ValueError, TypeError):
            time_interval = 0.25
            pixel_scale = 1.0

        apply_flatfield = str(config.get("apply_flatfield", True)).lower() != "false"
        track_cells = str(config.get("track_cells", True)).lower() != "false"
        output_dir = config.get("output_dir", "")

        # 0. Autonomous Auto-Calibration (Scale Bar OCR)
        # We read the first frame outside the threaded loader for calibration
        first_img_raw = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        if first_img_raw is not None:
            from modules.segmentation import detect_scale_bar
            sb = detect_scale_bar(first_img_raw)
            if sb["detected"] and sb.get("um_per_px"):
                logger.info("✓ AUTO-CALIBRATION: Scale bar detected (%s). Using %.4f um/px", 
                            sb.get('scale_label', 'unlabeled'), sb['um_per_px'])
                pixel_scale = sb["um_per_px"]
            del first_img_raw

        # 1. Flatfield Estimation (Batched)
        flatfield = None
        if apply_flatfield:
            step = max(1, n_frames // 10)
            sampled_f = []
            for f in image_files[::step]:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    sampled_f.append(img.astype(np.float32) / 255.0)
            if sampled_f:
                flatfield = estimate_flatfield(sampled_f)
            del sampled_f

        # 2. Memmap Allocation for masks (Eliminates OOM crashes and disk stream bottleneck)
        first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        if first_img is None:
            raise SegmentationError(f"Could not read first image: {image_files[0]}")

        h, w = first_img.shape

        temp_file = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        masks_memmap = np.memmap(temp_path, dtype=np.uint8, mode='w+', shape=(n_frames, h, w))

        areas_px = []
        timepoints = []
        quality_scores = []
        low_quality_frames = []

        # 3. Threaded Loading & Processing
        img_queue = queue.Queue(maxsize=4)  # Small buffer to prevent RAM bloat
        stop_event = threading.Event()
        loader_thread = threading.Thread(
            target=_threaded_image_loader, 
            args=(image_files, img_queue, stop_event), 
            daemon=True
        )
        loader_thread.start()

        try:
            for idx in range(n_frames):
                img = img_queue.get()
                if isinstance(img, str) and img == "STOP":
                    break
                
                if img is None:
                    areas_px.append(0.0)
                    timepoints.append(idx * time_interval)
                    continue

                img_f = img.astype(np.float32) / 255.0
                if apply_flatfield and flatfield is not None:
                    img_f = apply_flatfield_correction(img_f, flatfield)

                try:
                    # Pass the odd integer disk_size
                    mask, area_px = segment_wound_from_array(img_f, disk_size=disk_size, apply_morph=True,
                                                             flatfield=None)
                except Exception as e:
                    logger.error(f"Segmentation failed on frame {idx}: {e}")
                    mask, area_px = np.zeros((h, w), dtype=np.uint8), 0.0

                # Early-stop: area anomaly check
                if len(areas_px) > 0 and areas_px[0] > 0:
                    if area_px > areas_px[0] * 2.2 or (area_px == 0 and idx > 0):
                        logger.warning("Frame %d: anomalous area %.0f — stopping early.", idx, area_px)
                        break

                masks_memmap[idx] = mask
                areas_px.append(float(area_px))
                timepoints.append(idx * time_interval)

                prev_area = areas_px[-2] if len(areas_px) > 1 else None
                qs = compute_wound_quality_score(mask, prev_area)
                quality_scores.append(qs["score"])
                if qs["score"] < 0.3:
                    low_quality_frames.append(idx)

                if idx % 20 == 0:
                    gc.collect()
        finally:
            stop_event.set()
            # Drain the queue to unblock the loader thread if it's stuck on a put()
            while not img_queue.empty():
                try:
                    img_queue.get_nowait()
                except queue.Empty:
                    break

        masks_memmap.flush()

        # Fallback to Cell Tracking Only if no wound is detected
        if not areas_px or areas_px[0] <= 0:
            tracking_res = {}
            if track_cells:
                try:
                    tracking_res = track_cells_in_timeseries(
                        image_files, [None] * n_frames, time_interval, pixel_scale, output_dir
                    )
                except Exception as e:
                    logger.warning("Tracking failed: %s", e)

            return {
                "metrics": {"num_timepoints": n_frames, "analysis_mode": "cell_tracking_only"},
                "tracking": tracking_res,
                "masks_memmap": None,
                "temp_path": temp_path
            }

        # 4. Temporal Smoothing — Reduces frame-to-frame segmentation jitter
        #    while preserving kinetic trend. Raw values kept for transparency.
        areas_raw = list(areas_px)  # preserve originals
        try:
            from scipy.signal import savgol_filter
            min_window = 7
            if len(areas_px) >= min_window:
                # Window must be odd and <= number of data points
                window = min(min_window, len(areas_px))
                if window % 2 == 0:
                    window -= 1
                if window >= 3:
                    areas_px = savgol_filter(areas_px, window_length=window, polyorder=2).tolist()
                    # Ensure smoothed areas are non-negative
                    areas_px = [max(0.0, a) for a in areas_px]
                    logger.info("✓ Temporal smoothing applied (Savgol window=%d)", window)
        except ImportError:
            logger.debug("scipy not available — skipping temporal smoothing")

        # 5. Quantification
        def get_mask(idx):
            return masks_memmap[idx].copy()

        metrics = analyze_time_series(
            time_points=timepoints,
            areas=areas_px,
            time_unit="hours",
            masks=get_mask,
            pixel_scale=pixel_scale
        )

        wound_axis = detect_wound_axis(get_mask(0))

        metrics.update({
            "timepoints": timepoints,
            "areas_px": areas_px,
            "areas_raw_px": areas_raw,
            "areas_um2": [a * pixel_scale ** 2 for a in areas_px],
            "quality_scores": quality_scores,
            "low_quality_frames": low_quality_frames,
            "pixel_scale_um_per_px": pixel_scale,
            "wound_angle_deg": wound_axis.get("angle_deg"),
            "processing_time_sec": time.time() - t_start,
            "_n_frames": n_frames
        })

        # 5. Tracking — only pass frames that were actually processed
        tracking_res = {}
        n_processed = len(areas_px)
        if track_cells and n_processed >= 4:
            try:
                masks_memmap.flush()
                # Only pass the image files that have valid masks
                processed_files = image_files[:n_processed]
                # Create a list-like mask accessor for the processed range
                processed_masks = [masks_memmap[i] for i in range(n_processed)]
                tracking_res = track_cells_in_timeseries(
                    processed_files, processed_masks, time_interval, pixel_scale, output_dir
                )
                logger.info("Cell tracking returned %d tracked cells",
                            tracking_res.get("num_cells_tracked", 0))
            except Exception as exc:
                import traceback as _tb
                logger.warning("Cell tracking skipped: %s\n%s", exc, _tb.format_exc())

        return {
            "metrics": metrics,
            "tracking": tracking_res,
            "masks_memmap": masks_memmap,
            "temp_path": temp_path
        }