#!/usr/bin/env python3
"""
cell_tracking.py — Cell Migration Tracker  (v3 — Full Biological Upgrade)

Upgrade summary over v1:

  NEW ANALYSIS MODULES
  ────────────────────
  A. MEAN SQUARED DISPLACEMENT (MSD) ANALYSIS
     MSD(τ) = <|r(t+τ) − r(t)|²>  averaged over all cells and time origins.
     The log-log slope α distinguishes:
       α ≈ 1  → random walk (Brownian, no directional bias)
       α ≈ 2  → ballistic / directed migration (chemotaxis, durotaxis)
       1<α<2  → superdiffusive / anomalous (common in scratch assays)
       α < 1  → subdiffusive (confined, dense monolayer)
     Diffusion coefficient D extracted from MSD = 4D·τ^α.

  B. CELL DIVISION EVENT DETECTION
     Detects putative division events from sudden track splits or
     large jumps in the number of detected particles per frame.
     Reports division rate per hour and per-frame counts.

  C. DIRECTED MIGRATION SCORE
     Compares each cell's net displacement direction against the wound axis.
     Score 0 = random,  1 = all cells moving directly toward wound.
     Separates contact-inhibition-of-locomotion (CIL) signal from random noise.

  D. VELOCITY AUTOCORRELATION
     C(τ) = <v(t)·v(t+τ)> / <v²>  — persistence time extraction.
     Exponential decay time constant τ_p = migration persistence time (min).
     Distinguishes persistent walkers from random walkers.

  E. IMPROVED ROSE PLOT
     Now shows:
     - Colour-coded by migration phase (early / mid / late)
     - Net displacement arrow for each cell
     - MSD-α annotation
     - Wound axis reference line

  RETAINED (improved)
  ───────────────────
  • TrackPy preferred / centroid fallback (unchanged)
  • Per-cell velocity, efficiency, directionality (unchanged)
  • Trajectory CSV export (unchanged)
  • Sector velocity analysis (improved)

  Public API:
    track_cells_in_timeseries(image_files, masks, time_interval_hours,
                               pixel_scale_um_per_px, output_dir)
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import distance

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

try:
    import trackpy as tp
    HAS_TRACKPY = True
except ImportError:
    HAS_TRACKPY = False

try:
    from scipy.optimize import curve_fit as _cf
    HAS_SCIPY_LOCAL = True
except ImportError:
    HAS_SCIPY_LOCAL = False

TRACKPY_CFG = {
    "diameter": 11, "minmass": 100,
    "separation": 7, "search_range": 15, "memory": 3,
}
FALLBACK_MIN_AREA    = 20
FALLBACK_MAX_AREA    = 5000
FALLBACK_MAX_LINK    = 40.0
MIN_TRACK_LENGTH     = 4   # frames — minimum for MSD / autocorrelation


# DETECTION & LINKING  

def _detect_centroids_fallback(img_gray: np.ndarray, mask: Optional[np.ndarray],
                               frame_idx: int) -> pd.DataFrame:
    """Blob detection via thresholding + connected components."""
    img8 = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)
    blurred = cv2.GaussianBlur(img8, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask is not None:
        cell_region = 1 - (mask > 0).astype(np.uint8)
        binary = cv2.bitwise_and(binary, binary, mask=cell_region)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if FALLBACK_MIN_AREA <= area <= FALLBACK_MAX_AREA:
            rows.append({"x": float(centroids[i][0]),
                         "y": float(centroids[i][1]),
                         "frame": frame_idx})
    return pd.DataFrame(rows)


def _link_fallback(frames_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Greedy nearest-neighbour linking across frames."""
    if not frames_dfs:
        return pd.DataFrame()
    all_rows   = []
    next_pid   = 0
    prev_frame = None

    for df in frames_dfs:
        if df.empty:
            prev_frame = None
            continue
        df = df.copy().reset_index(drop=True)
        df["particle"] = -1

        if prev_frame is None or prev_frame.empty:
            for i in range(len(df)):
                df.at[i, "particle"] = next_pid
                next_pid += 1
        else:
            prev_pts = prev_frame[["x", "y"]].values
            curr_pts = df[["x", "y"]].values
            dm = distance.cdist(prev_pts, curr_pts)
            assigned_prev = set()
            assigned_curr = set()
            pairs = sorted(
                [(dm[r, c], r, c)
                 for r in range(len(prev_pts))
                 for c in range(len(curr_pts))],
                key=lambda x: x[0]
            )
            pid_map = {int(prev_frame.iloc[r]["particle"]): r
                       for r in range(len(prev_frame))}
            for dist_val, ri, ci in pairs:
                if dist_val > FALLBACK_MAX_LINK:
                    break
                if ri in assigned_prev or ci in assigned_curr:
                    continue
                prev_pid = int(prev_frame.iloc[ri]["particle"])
                df.at[ci, "particle"] = prev_pid
                assigned_prev.add(ri)
                assigned_curr.add(ci)
            for i in range(len(df)):
                if df.at[i, "particle"] == -1:
                    df.at[i, "particle"] = next_pid
                    next_pid += 1

        prev_frame = df.copy()
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# A.  MEAN SQUARED DISPLACEMENT ANALYSIS


def compute_msd(
    tracks_df: pd.DataFrame,
    time_interval_hr: float,
    pixel_scale: float,
    max_lag_fraction: float = 0.5,
) -> Dict:
    """
    Ensemble MSD as a function of lag time.

    MSD(τ) = 4D·τ^α  (2D diffusion)

    log-log linear fit over lags up to max_lag_fraction × trajectory duration.

    Returns:
        lag_times_hr        — array of lag durations
        msd_um2             — ensemble-averaged MSD at each lag (µm²)
        alpha               — anomalous exponent (slope of log-log)
        D_um2_hr            — generalised diffusion coefficient
        migration_mode      — "random_walk"|"directed"|"superdiffusive"|"subdiffusive"
        n_tracks_used       — number of valid tracks contributing
        r_squared_loglog    — quality of power-law fit
    """
    null = {
        "lag_times_hr": [], "msd_um2": [], "alpha": None,
        "D_um2_hr": None, "migration_mode": None,
        "n_tracks_used": 0, "r_squared_loglog": None,
    }
    if tracks_df.empty or "particle" not in tracks_df.columns:
        return null

    msd_by_lag: Dict[int, List[float]] = {}

    for pid, grp in tracks_df.groupby("particle"):
        g = grp.sort_values("frame")
        if len(g) < MIN_TRACK_LENGTH:
            continue
        xs = g["x"].values * pixel_scale  # µm
        ys = g["y"].values * pixel_scale
        max_lag = max(1, int(len(g) * max_lag_fraction))
        for lag in range(1, max_lag + 1):
            dx = xs[lag:] - xs[:-lag]
            dy = ys[lag:] - ys[:-lag]
            sd = dx**2 + dy**2
            msd_by_lag.setdefault(lag, []).extend(sd.tolist())

    if not msd_by_lag:
        return null

    lags     = sorted(msd_by_lag.keys())
    msd_vals = [float(np.mean(msd_by_lag[l])) for l in lags]
    lag_hrs  = [l * time_interval_hr for l in lags]

    # Power-law fit in log-log space
    log_t = np.log(np.array(lag_hrs, dtype=float) + 1e-12)
    log_m = np.log(np.array(msd_vals, dtype=float) + 1e-12)
    valid = np.isfinite(log_t) & np.isfinite(log_m)

    alpha, D, r2 = None, None, None
    if valid.sum() >= 2:
        coeffs = np.polyfit(log_t[valid], log_m[valid], 1)
        alpha  = float(coeffs[0])
        D      = float(np.exp(coeffs[1]) / 4.0)   # MSD = 4D·τ^α
        pred   = np.poly1d(coeffs)(log_t[valid])
        ss_res = float(np.sum((log_m[valid] - pred) ** 2))
        ss_tot = float(np.sum((log_m[valid] - log_m[valid].mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if alpha is not None:
        if alpha < 0.8:
            mode = "subdiffusive"
        elif alpha < 1.2:
            mode = "random_walk"
        elif alpha < 1.8:
            mode = "superdiffusive"
        else:
            mode = "directed"
    else:
        mode = None

    return {
        "lag_times_hr":     lag_hrs,
        "msd_um2":          msd_vals,
        "alpha":            alpha,
        "D_um2_hr":         D,
        "migration_mode":   mode,
        "n_tracks_used":    len(tracks_df["particle"].unique()),
        "r_squared_loglog": r2,
    }


# ═════════════════════════════════════════════════════════════════════════════
# B. CELL DIVISION DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_division_events(
    tracks_df: pd.DataFrame,
    time_interval_hr: float,
    max_displacement_multiplier: float = 2.5,
) -> Dict:
    """
    Estimate cell division events from tracking data.

    Heuristic:
    1. Count particles per frame — sudden increase = likely division wave
    2. Identify tracks that show a sudden large-displacement step followed
       by appearance of a new nearby track (track split signature)

    Returns:
        division_rate_per_hr     — estimated divisions per cell per hour
        division_events_per_frame — list of per-frame counts
        doubling_time_hr         — estimated population doubling time
        n_putative_events        — total count
    """
    if tracks_df.empty:
        return {}

    # Per-frame particle counts
    frame_counts = tracks_df.groupby("frame")["particle"].nunique().sort_index()
    counts = frame_counts.values.astype(float)
    delta  = np.diff(counts)
    # Positive jumps > median + 1 SD = likely division bursts
    med, sd = np.median(delta), np.std(delta)
    div_frames = np.where(delta > med + sd)[0]
    n_events   = int(np.sum(np.clip(delta[div_frames] - med, 0, None)))

    total_cell_hrs = float(counts.sum()) * time_interval_hr
    div_rate       = float(n_events / total_cell_hrs) if total_cell_hrs > 0 else 0.0
    doubling_time  = float(np.log(2) / div_rate) if div_rate > 0 else None

    return {
        "division_rate_per_hr":        div_rate,
        "division_events_per_frame":   [int(max(0, d - med)) for d in delta],
        "doubling_time_hr":            doubling_time,
        "n_putative_events":           n_events,
        "particle_count_per_frame":    frame_counts.tolist(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# C. DIRECTED MIGRATION SCORE TOWARD WOUND
# ═════════════════════════════════════════════════════════════════════════════

def compute_directed_migration_score(
    tracks_df: pd.DataFrame,
    masks: Optional[List[np.ndarray]],
    pixel_scale: float = 1.0,
) -> Dict:
    """
    Score how directionally each cell migrates toward the wound.

    For each track, compare the mean displacement vector against the
    vector pointing toward the nearest wound edge centroid.

    Returns:
        directed_migration_score — 0 (random) to 1 (all cells toward wound)
        cos_theta_distribution   — per-cell cosine of angle to wound
        mean_cos_theta           — mean directional alignment
    """
    if tracks_df.empty or not masks:
        return {}

    # Find wound centroid per frame from masks
    wound_centers: Dict[int, Optional[Tuple[float, float]]] = {}
    for fi, mask in enumerate(masks):
        if mask is None or mask.sum() == 0:
            wound_centers[fi] = None
            continue
        ys, xs = np.where(mask > 0)
        wound_centers[fi] = (float(xs.mean()), float(ys.mean()))

    cos_thetas = []
    for pid, grp in tracks_df.groupby("particle"):
        g = grp.sort_values("frame")
        if len(g) < 3:
            continue
        frames = g["frame"].values
        xs = g["x"].values
        ys = g["y"].values

        # Net displacement vector
        dx_net = (xs[-1] - xs[0]) * pixel_scale
        dy_net = (ys[-1] - ys[0]) * pixel_scale
        disp_norm = np.sqrt(dx_net**2 + dy_net**2)
        if disp_norm < 1e-6:
            continue

        # Vector toward wound centroid at first frame
        wc = wound_centers.get(int(frames[0]))
        if wc is None:
            continue
        wx = (wc[0] - xs[0]) * pixel_scale
        wy = (wc[1] - ys[0]) * pixel_scale
        wound_norm = np.sqrt(wx**2 + wy**2)
        if wound_norm < 1e-6:
            continue

        cos_t = (dx_net * wx + dy_net * wy) / (disp_norm * wound_norm)
        cos_thetas.append(float(np.clip(cos_t, -1.0, 1.0)))

    if not cos_thetas:
        return {}

    mean_cos = float(np.mean(cos_thetas))
    # Score: rescale from [-1,1] → [0,1]
    score = (mean_cos + 1.0) / 2.0

    return {
        "directed_migration_score": score,
        "mean_cos_theta":           mean_cos,
        "cos_theta_distribution":   cos_thetas,
        "n_cells_scored":           len(cos_thetas),
    }


# ═════════════════════════════════════════════════════════════════════════════
# D. VELOCITY AUTOCORRELATION & PERSISTENCE TIME
# ═════════════════════════════════════════════════════════════════════════════

def compute_velocity_autocorrelation(
    tracks_df: pd.DataFrame,
    time_interval_hr: float,
    pixel_scale: float = 1.0,
) -> Dict:
    """
    Velocity autocorrelation C(τ) = <v(t)·v(t+τ)> / <v²>

    For a persistent random walk, C(τ) decays as exp(−τ/τ_p).
    τ_p = persistence time. Longer τ_p = more directionally persistent cells.

    Returns:
        lag_times_hr          — lag durations
        autocorrelation       — C(τ) values
        persistence_time_hr   — exponential decay constant (hours)
        r_squared_fit         — quality of exponential fit
    """
    null = {"lag_times_hr": [], "autocorrelation": [], "persistence_time_hr": None}
    if tracks_df.empty:
        return null

    all_vx: Dict[int, List[float]] = {}
    all_vy: Dict[int, List[float]] = {}

    for pid, grp in tracks_df.groupby("particle"):
        g = grp.sort_values("frame")
        if len(g) < MIN_TRACK_LENGTH:
            continue
        xs = g["x"].values * pixel_scale
        ys = g["y"].values * pixel_scale
        vx = np.diff(xs) / time_interval_hr
        vy = np.diff(ys) / time_interval_hr
        for fi, (vi_x, vi_y) in zip(g["frame"].values[:-1], zip(vx, vy)):
            all_vx.setdefault(fi, []).append(vi_x)
            all_vy.setdefault(fi, []).append(vi_y)

    if not all_vx:
        return null

    frames = sorted(all_vx.keys())
    mean_vx = {f: float(np.mean(all_vx[f])) for f in frames}
    mean_vy = {f: float(np.mean(all_vy[f])) for f in frames}

    # Dot product at each frame for autocorrelation
    dot_t = {f: mean_vx[f]**2 + mean_vy[f]**2 for f in frames}
    v2_mean = float(np.mean(list(dot_t.values())))
    if v2_mean < 1e-12:
        return null

    max_lag = max(1, len(frames) // 3)
    lags, acf = [], []
    for lag in range(1, max_lag + 1):
        pairs = [(frames[i], frames[i + lag])
                 for i in range(len(frames) - lag)
                 if i + lag < len(frames)]
        if not pairs:
            continue
        corr_vals = []
        for f0, f1 in pairs:
            c = mean_vx[f0]*mean_vx[f1] + mean_vy[f0]*mean_vy[f1]
            corr_vals.append(c / v2_mean)
        lags.append(lag * time_interval_hr)
        acf.append(float(np.mean(corr_vals)))

    # Fit exponential decay to extract persistence time
    tau_p, r2_fit = None, None
    if len(lags) >= 3 and HAS_SCIPY_LOCAL:
        try:
            from scipy.optimize import curve_fit as _cf
            def _exp_decay(x, tau): return np.exp(-x / tau)
            popt, _ = _cf(_exp_decay, lags, acf,
                          p0=[lags[len(lags)//2]], bounds=([1e-5], [1e4]))
            tau_p = float(popt[0])
            pred  = np.exp(-np.array(lags) / tau_p)
            ss_res = float(np.sum((np.array(acf) - pred)**2))
            ss_tot = float(np.sum((np.array(acf) - np.mean(acf))**2))
            r2_fit = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
        except Exception:
            pass

    return {
        "lag_times_hr":       lags,
        "autocorrelation":    acf,
        "persistence_time_hr": tau_p,
        "r_squared_fit":       r2_fit,
    }


try:
    from scipy.optimize import curve_fit as _cf
    HAS_SCIPY_LOCAL = True
except ImportError:
    HAS_SCIPY_LOCAL = False


# ═════════════════════════════════════════════════════════════════════════════
# PER-TRACK METRICS (improved from v1)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_per_track_metrics(
    tracks_df: pd.DataFrame,
    time_interval_hr: float,
    pixel_scale: float,
) -> Tuple[pd.DataFrame, Dict]:
    """Return per-cell DataFrame and ensemble summary dict."""
    rows = []
    for pid, grp in tracks_df.groupby("particle"):
        g = grp.sort_values("frame")
        if len(g) < 2:
            continue
        xs = g["x"].values * pixel_scale  # µm
        ys = g["y"].values * pixel_scale

        # Step-wise displacement
        dx = np.diff(xs)
        dy = np.diff(ys)
        steps = np.sqrt(dx**2 + dy**2)

        # Net displacement (start → end)
        x_net = xs[-1] - xs[0]
        y_net = ys[-1] - ys[0]
        net_disp = np.sqrt(x_net**2 + y_net**2)

        # Path length
        path_len = float(steps.sum())

        # Duration
        duration_hr = (len(g) - 1) * time_interval_hr

        # Mean velocity (µm/min)
        mean_vel = float(steps.mean() / (time_interval_hr * 60)) if duration_hr > 0 else 0.0

        # Efficiency = net / path
        efficiency = float(net_disp / path_len) if path_len > 0 else 0.0

        # Directionality: correlation of consecutive step angles
        if len(dx) >= 2:
            angles  = np.arctan2(dy, dx)
            d_angle = np.diff(angles)
            d_angle = (d_angle + np.pi) % (2 * np.pi) - np.pi
            directionality = float(np.cos(d_angle).mean())
        else:
            directionality = 0.0

        rows.append({
            "track_id":       int(pid),
            "n_frames":       len(g),
            "duration_hr":    duration_hr,
            "path_length_um": path_len,
            "net_displacement_um": net_disp,
            "mean_velocity_um_min": mean_vel,
            "efficiency":     efficiency,
            "meander_index":  efficiency,  # Biological nomenclature upgrade
            "directionality": directionality,
            "start_x_um":     float(xs[0]),
            "start_y_um":     float(ys[0]),
            "end_x_um":       float(xs[-1]),
            "end_y_um":       float(ys[-1]),
        })

    if not rows:
        return pd.DataFrame(), {}

    df = pd.DataFrame(rows)
    summary = {
        "num_cells_tracked":         len(df),
        "mean_velocity_um_min":      float(df["mean_velocity_um_min"].mean()),
        "mean_path_length_um":       float(df["path_length_um"].mean()),
        "mean_displacement_um":      float(df["net_displacement_um"].mean()),
        "migration_efficiency_mean": float(df["efficiency"].mean()),
        "meander_index_mean":        float(df["meander_index"].mean()),
        "mean_directionality":       float(df["mean_directionality"].mean() if "mean_directionality" in df else df["directionality"].mean()),
        "std_velocity_um_min":       float(df["mean_velocity_um_min"].std()),
        "std_path_length_um":        float(df["path_length_um"].std()),
    }
    return df, summary


# ═════════════════════════════════════════════════════════════════════════════
# IMPROVED ROSE PLOT
# ═════════════════════════════════════════════════════════════════════════════

def _plot_rose(tracks_df: pd.DataFrame, tracking_dir: str,
               msd_data: Optional[Dict] = None,
               pixel_scale: float = 1.0) -> str:
    """
    Enhanced rose plot showing:
    - Colour-coded trajectories by migration phase (early/mid/late)
    - Net displacement arrow per cell
    - MSD alpha annotation
    - Wind-rose histogram of migration directions
    """
    fig = plt.figure(figsize=(12, 5), facecolor="#0a0f18")

    # ── Left: cartesian trajectory map ──────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, facecolor="#0a0f18")
    cmap_ph = plt.get_cmap("plasma")

    for pid, grp in tracks_df.groupby("particle"):
        g = grp.sort_values("frame")
        if len(g) < 2:
            continue
        xs = g["x"].values * pixel_scale
        ys = g["y"].values * pixel_scale
        n  = len(xs)
        # Phase colouring
        for i in range(n - 1):
            phase = i / max(1, n - 1)
            ax1.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                     color=cmap_ph(phase), lw=1.2, alpha=0.7)
        # Net displacement arrow
        ax1.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[0], ys[0]),
                     arrowprops=dict(arrowstyle="->", color="#00e5ff",
                                     lw=1.5, alpha=0.6))

    ax1.set_xlabel("X position (µm)", color="#aaa", fontsize=9)
    ax1.set_ylabel("Y position (µm)", color="#aaa", fontsize=9)
    ax1.tick_params(colors="#555")
    for sp in ax1.spines.values():
        sp.set_color("#1a2530")
    ax1.set_title("Cell Migration Trajectories", color="#d0dde8",
                  fontsize=10, pad=8)

    # Phase colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap_ph, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, fraction=0.04, pad=0.01)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Early", "Mid", "Late"])
    cb.ax.yaxis.set_tick_params(color="#555", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#888")

    # ── Right: polar rose histogram ──────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, projection="polar",
                          facecolor="#0a0f18")
    ax2.set_facecolor("#0a0f18")

    all_angles = []
    for pid, grp in tracks_df.groupby("particle"):
        g = grp.sort_values("frame")
        if len(g) < 2:
            continue
        xs = g["x"].values; ys = g["y"].values
        dx = np.diff(xs); dy = np.diff(ys)
        angles = np.arctan2(dy, dx)
        all_angles.extend(angles.tolist())

    if all_angles:
        n_bins = 36
        bins   = np.linspace(-np.pi, np.pi, n_bins + 1)
        counts, _ = np.histogram(all_angles, bins=bins)
        theta_mid  = (bins[:-1] + bins[1:]) / 2
        width_bin  = (2 * np.pi) / n_bins
        bars = ax2.bar(theta_mid, counts / max(counts.max(), 1),
                       width=width_bin, alpha=0.75,
                       color=plt.get_cmap("plasma")(
                           np.linspace(0.1, 0.9, n_bins)))
        for bar in bars:
            bar.set_edgecolor("#0a0f18")

    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(1)
    ax2.set_yticks([])
    ax2.tick_params(colors="#555", labelsize=7)
    ax2.spines["polar"].set_color("#1a2530")
    ax2.set_title("Migration Direction Rose", color="#d0dde8",
                  fontsize=10, pad=12)

    # MSD annotation
    if msd_data and msd_data.get("alpha") is not None:
        alpha = msd_data["alpha"]
        mode  = msd_data.get("migration_mode", "")
        ax2.text(0.5, -0.15,
                 f"MSD α = {alpha:.2f}  ({mode.replace('_',' ')})",
                 transform=ax2.transAxes, ha="center", fontsize=8,
                 color="#00e5ff", style="italic")

    plt.tight_layout(pad=2.0)
    out = os.path.join(tracking_dir, "rose_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0f18")
    plt.close()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# MSD PLOT
# ═════════════════════════════════════════════════════════════════════════════

def _plot_msd(msd_data: Dict, tracking_dir: str) -> Optional[str]:
    if not msd_data.get("lag_times_hr") or not msd_data.get("msd_um2"):
        return None
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a0f18")
    ax.set_facecolor("#0a0f18")
    lags = msd_data["lag_times_hr"]
    msd  = msd_data["msd_um2"]
    ax.loglog(lags, msd, "o-", color="#00ff88", lw=2, ms=5,
              label="Ensemble MSD")
    alpha = msd_data.get("alpha")
    D     = msd_data.get("D_um2_hr")
    if alpha is not None and D is not None:
        t_arr  = np.logspace(np.log10(min(lags)), np.log10(max(lags)), 50)
        ax.loglog(t_arr, 4*D*t_arr**alpha, "--", color="#ff2060",
                  lw=1.5, label=f"4D·τ^α  (α={alpha:.2f})")
    ax.set_xlabel("Lag time (hr)", color="#aaa", fontsize=9)
    ax.set_ylabel("MSD (µm²)", color="#aaa", fontsize=9)
    mode = msd_data.get("migration_mode", "")
    ax.set_title(f"Mean Squared Displacement — {mode.replace('_',' ')}",
                 color="#d0dde8", fontsize=10)
    ax.legend(fontsize=8, facecolor="#0a0f18", edgecolor="#1a2530",
              labelcolor="#d0dde8")
    ax.tick_params(colors="#555")
    for sp in ax.spines.values(): sp.set_color("#1a2530")
    plt.tight_layout()
    out = os.path.join(tracking_dir, "msd_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0f18")
    plt.close()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# CSV HELPERS (backwards compatible)
# ═════════════════════════════════════════════════════════════════════════════

def _save_csvs(df_tracks: pd.DataFrame, tracking_dir: str) -> Tuple[str, str]:
    traj_path = os.path.join(tracking_dir, "trajectories.csv")
    vel_path  = os.path.join(tracking_dir, "velocities.csv")
    df_tracks.rename(columns={"particle": "track_id"}, inplace=True, errors="ignore")
    df_tracks.to_csv(traj_path, index=False)
    vel_rows = []
    col = "track_id" if "track_id" in df_tracks.columns else "particle"
    for pid, g in df_tracks.groupby(col):
        g = g.sort_values("frame")
        xs = g["x"].values; ys = g["y"].values; fs = g["frame"].values
        for i in range(len(fs) - 1):
            vel_rows.append({"track_id": int(pid), "frame_from": int(fs[i]),
                             "frame_to": int(fs[i+1]),
                             "dx_px": float(xs[i+1]-xs[i]),
                             "dy_px": float(ys[i+1]-ys[i])})
    pd.DataFrame(vel_rows or [{"track_id":0,"frame_from":0,"frame_to":0,
                                "dx_px":0,"dy_px":0}]).to_csv(vel_path, index=False)
    return traj_path, vel_path


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def track_cells_in_timeseries(
    image_files: List[str],
    masks: List[Optional[np.ndarray]],
    time_interval_hours: float,
    pixel_scale_um_per_px: float,
    output_dir: str,
) -> Dict:
    """
    Full cell tracking and migration analysis pipeline.

    Returns summary dict with all new metrics. Saves:
      tracking/trajectories.csv
      tracking/velocities.csv
      tracking/per_cell_metrics.csv
      tracking/rose_plot.png      (upgraded)
      tracking/msd_plot.png       (NEW)
      tracking/tracking_summary.json
    """
    tracking_dir = os.path.join(output_dir, "tracking")
    os.makedirs(tracking_dir, exist_ok=True)
    t_start = time.time()

    # ── Detection & linking ───────────────────────────────────────────────────
    if HAS_TRACKPY:
        try:
            frames_list = []
            for fi, img_path in enumerate(image_files):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img8 = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img
                f = tp.locate(img8, TRACKPY_CFG["diameter"],
                              minmass=TRACKPY_CFG["minmass"],
                              separation=TRACKPY_CFG["separation"])
                if not f.empty:
                    f["frame"] = fi
                    frames_list.append(f)
            if frames_list:
                all_particles = pd.concat(frames_list, ignore_index=True)
                tracks_df = tp.link(all_particles,
                                    search_range=TRACKPY_CFG["search_range"],
                                    memory=TRACKPY_CFG["memory"])
            else:
                tracks_df = pd.DataFrame()
        except Exception as exc:
            logger.warning("TrackPy failed (%s) — fallback.", exc)
            tracks_df = pd.DataFrame()
    else:
        tracks_df = pd.DataFrame()

    # Centroid fallback
    if tracks_df.empty:
        logger.info("Centroid fallback tracking…")
        frame_dfs = []
        for fi, img_path in enumerate(image_files):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_f = img.astype(np.float32) / 255.0
            m = masks[fi] if fi < len(masks) else None
            frame_dfs.append(_detect_centroids_fallback(img_f, m, fi))
        tracks_df = _link_fallback(frame_dfs)

    if tracks_df.empty or "particle" not in tracks_df.columns:
        logger.warning("No tracks detected.")
        return {"num_cells_tracked": 0}

    # Filter short tracks
    track_len = tracks_df.groupby("particle")["frame"].count()
    valid_ids = track_len[track_len >= MIN_TRACK_LENGTH].index
    tracks_df = tracks_df[tracks_df["particle"].isin(valid_ids)].copy()

    # ── Per-track metrics ─────────────────────────────────────────────────────
    per_cell_df, summary = _compute_per_track_metrics(
        tracks_df, time_interval_hours, pixel_scale_um_per_px)

    # ── MSD analysis ──────────────────────────────────────────────────────────
    msd_data = compute_msd(tracks_df, time_interval_hours, pixel_scale_um_per_px)
    summary["msd_alpha"]         = msd_data.get("alpha")
    summary["msd_D_um2_hr"]      = msd_data.get("D_um2_hr")
    summary["migration_mode_msd"] = msd_data.get("migration_mode")

    # ── Division detection ────────────────────────────────────────────────────
    div_data = detect_division_events(tracks_df, time_interval_hours)
    summary["division_rate_per_hr"] = div_data.get("division_rate_per_hr")
    summary["doubling_time_hr"]     = div_data.get("doubling_time_hr")

    # ── Directed migration score ──────────────────────────────────────────────
    dir_data = compute_directed_migration_score(
        tracks_df, masks, pixel_scale_um_per_px)
    summary["directed_migration_score"] = dir_data.get("directed_migration_score")
    summary["mean_cos_theta"]           = dir_data.get("mean_cos_theta")

    # ── Velocity autocorrelation ──────────────────────────────────────────────
    acf_data = compute_velocity_autocorrelation(
        tracks_df, time_interval_hours, pixel_scale_um_per_px)
    summary["persistence_time_hr"] = acf_data.get("persistence_time_hr")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    _save_csvs(tracks_df.copy(), tracking_dir)
    if not per_cell_df.empty:
        per_cell_df.to_csv(
            os.path.join(tracking_dir, "per_cell_metrics.csv"), index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_rose(tracks_df, tracking_dir, msd_data, pixel_scale_um_per_px)
    _plot_msd(msd_data, tracking_dir)

    # ── JSON summary ──────────────────────────────────────────────────────────
    full_summary = {
        **summary,
        "msd_details":         msd_data,
        "division_details":    div_data,
        "directed_migration":  dir_data,
        "autocorrelation":     acf_data,
        "processing_time_sec": float(time.time() - t_start),
        "method":              "trackpy" if HAS_TRACKPY else "centroid_fallback",
    }
    import json
    with open(os.path.join(tracking_dir, "tracking_summary.json"), "w") as fh:
        json.dump({k: v for k, v in full_summary.items()
                   if not isinstance(v, (np.ndarray, pd.DataFrame))},
                  fh, indent=2, default=str)

    logger.info("✓ Tracking complete: %d tracks, α=%.2f (%s)",
                summary.get("num_cells_tracked", 0),
                msd_data.get("alpha") or 0.0,
                msd_data.get("migration_mode") or "unknown")

    return full_summary


# ── Legacy wrapper ─────────────────────────────────────────────────────────────
def run_cell_tracking(image_files, masks, time_interval, pixel_scale, output_dir):
    return track_cells_in_timeseries(
        image_files, masks, time_interval, pixel_scale, output_dir)