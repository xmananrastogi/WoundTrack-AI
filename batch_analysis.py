#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, glob, argparse, cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import time, json, logging
from PIL import Image, ImageSequence
import imageio.v2 as imageio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from werkzeug.utils import secure_filename

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

try:
    from preprocessing import load_and_preprocess_image, normalize_intensity
    from segmentation import segment_wound_from_array, detect_wound_contours
    from quantification import calculate_wound_closure_percentage
    import cell_tracking
except ImportError as e:
    logger.error(f"Failed to import a required module: {e}")
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Batch analysis of scratch assay images')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input directory')
    parser.add_argument('--output', '-o', type=str, default='results', help='Output directory')
    parser.add_argument('--disk-size', '-d', type=int, default=0, help='Disk size (0 = auto-select)')
    parser.add_argument('--time-interval', '-t', type=float, default=0.25, help='Time interval (hours/frame)')
    parser.add_argument('--visualize', action='store_true', help='Generate plots')
    parser.add_argument('--track-cells', action='store_true', help='Run cell tracking')
    parser.add_argument('--save-masks', action='store_true', help='Save masks')
    parser.add_argument('--pixel-scale', '-p', type=float, default=1.0, help='Pixel scale (e.g., 0.65 um/pixel)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Specific name for the experiment output files (Sample ID)')
    return parser.parse_args()


def get_image_files(input_dir):
    image_files = []
    supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.gif']
    for ext in supported_formats:
        image_files.extend(sorted(glob.glob(os.path.join(input_dir, '**', ext), recursive=True)))

    image_files = sorted(list(set(f for f in image_files if '__MACOSX' not in f and '/._' not in f)))

    # If no static images, try video extraction
    if not image_files:
        video_formats = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        for ext in video_formats:
            video_files = sorted(glob.glob(os.path.join(input_dir, ext)))
            if video_files:
                logger.info(f"Video file detected: {video_files[0]}. Extracting frames...")
                temp_dir = os.path.join(input_dir, '_temp_frames')
                os.makedirs(temp_dir, exist_ok=True)
                return extract_frames_from_video(video_files[0], temp_dir)

    # If exactly one multi-frame image (tif/gif) present, extract frames
    if len(image_files) == 1 and image_files[0].lower().endswith(('.tif', '.tiff', '.gif')):
        try:
            logger.info(f"Multi-frame file detected: {image_files[0]}. Extracting frames...")
            temp_dir = os.path.join(input_dir, '_temp_frames')
            os.makedirs(temp_dir, exist_ok=True)
            extracted_files = []
            with Image.open(image_files[0]) as img:
                for i, frame in enumerate(ImageSequence.Iterator(img)):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    frame.convert('L').save(frame_path)
                    extracted_files.append(frame_path)
            extracted_files = sorted(extracted_files)
            logger.info(f"Extracted {len(extracted_files)} frames from multi-frame file.")
            return extracted_files
        except Exception as e:
            logger.error(f"Error processing multi-frame file: {e}", exc_info=True)
            return []

    return image_files


def extract_frames_from_video(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return []
    success, frame = vidcap.read()
    count = 0
    extracted_files = []
    while success:
        frame_path = os.path.join(output_dir, f"frame_{count:04d}.png")
        cv2.imwrite(frame_path, frame)
        extracted_files.append(frame_path)
        success, frame = vidcap.read()
        count += 1
    vidcap.release()
    extracted_files = sorted(extracted_files)
    logger.info(f"Extracted {count} frames from video.")
    return extracted_files


def auto_select_disk_size(first_image_path, last_image_path, sizes_to_try=[7, 10, 15, 20, 25]):
    logger.info("🤖 Running automatic disk size selection...")
    results = []
    try:
        img_first = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
        img_last = cv2.imread(last_image_path, cv2.IMREAD_GRAYSCALE)
        if img_first is None or img_last is None:
            logger.warning("Could not read images for auto-selection. Defaulting to 10.")
            return 10
        for size in sizes_to_try:
            try:
                _, area_first = segment_wound_from_array(img_first, disk_size=size)
                _, area_last = segment_wound_from_array(img_last, disk_size=size)
            except Exception as e:
                logger.warning(f"Segmentation error at size {size}: {e}")
                continue
            closure = (area_first - area_last) / area_first if area_first > 0 else -999
            logger.info(f"  - Testing Disk Size {size}: Closure={closure * 100:.1f}%")
            results.append({'size': size, 'closure': closure})
        if not results:
            logger.warning("Auto-selection failed. Defaulting to 10.")
            return 10
        best_result = max(results, key=lambda x: x['closure'])
        logger.info(f"✓ Auto-selected best disk size: {best_result['size']} (Closure: {best_result['closure'] * 100:.1f}%)")
        return best_result['size']
    except Exception as e:
        logger.error(f"Error during auto-selection: {e}. Defaulting to 10.", exc_info=True)
        return 10


def process_timeseries(image_files, disk_size, time_interval, save_masks, output_dir, pixel_scale):
    if not image_files:
        logger.warning("No images found to process.")
        return None

    timepoints, areas_px, masks = [], [], []
    logger.info(f"Processing {len(image_files)} images...")
    for idx, img_path in enumerate(tqdm(image_files, desc="Analyzing Frames")):
        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Could not read image {img_path}; skipping.")
                continue
            image_norm = normalize_intensity(image)
            wound_mask, wound_area_px = segment_wound_from_array(image_norm, disk_size=disk_size)
            timepoints.append(idx * time_interval)
            areas_px.append(float(wound_area_px))
            masks.append(wound_mask) # <-- This is the WOUND MASK (gap)
            if save_masks:
                mask_dir = os.path.join(output_dir, 'masks')
                os.makedirs(mask_dir, exist_ok=True)
                cv2.imwrite(os.path.join(mask_dir, f"mask_{idx:04d}.png"), (wound_mask * 255).astype(np.uint8))
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}", exc_info=True)
            continue

    if len(areas_px) < 2:
        logger.error("Processing failed: Not enough images were successfully processed.")
        return None

    logger.info("Calculating metrics...")
    areas_um2 = [a * (pixel_scale ** 2) for a in areas_px]
    initial_area_px, final_area_px = areas_px[0], areas_px[-1]
    initial_area_um2, final_area_um2 = areas_um2[0], areas_um2[-1]
    final_closure = calculate_wound_closure_percentage(final_area_px, initial_area_px)

    try:
        if len(timepoints) > 1:
            healing_rate_um2_per_hr = float(np.polyfit(timepoints, areas_um2, 1)[0])
        else:
            healing_rate_um2_per_hr = 0.0
    except Exception:
        healing_rate_um2_per_hr = 0.0

    if len(timepoints) > 2:
        coeffs_px = np.polyfit(timepoints, areas_px, 1)
        predictions_px = np.poly1d(coeffs_px)(timepoints)
        ss_res = np.sum((np.array(areas_px) - predictions_px) ** 2)
        ss_tot = np.sum((np.array(areas_px) - np.mean(areas_px)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    else:
        r_squared = 0.0

    closure_percentages = [calculate_wound_closure_percentage(a, initial_area_px) for a in areas_px]
    time_to_50 = None
    for i, closure in enumerate(closure_percentages):
        if closure >= 50.0:
            if i > 0:
                t1, t2 = timepoints[i - 1], timepoints[i]
                c1, c2 = closure_percentages[i - 1], closure_percentages[i]
                time_to_50 = t1 + (50.0 - c1) * (t2 - t1) / (c2 - c1) if (c2 - c1) != 0 else t1
            else:
                time_to_50 = timepoints[i]
            break

    frame_diffs = np.diff(areas_px) if len(areas_px) > 1 else np.array([0.0])
    healing_rate_mean_px_per_hr = float(np.mean(frame_diffs) / time_interval) if len(frame_diffs) > 0 else 0.0

    return {
        'timepoints': timepoints,
        'areas_px': areas_px,
        'areas_um2': areas_um2,
        'closure_percentages': closure_percentages,
        'masks': masks, # <-- This is the list of wound_mask arrays
        'pixel_scale_um_per_px': float(pixel_scale),
        'initial_area_px': float(initial_area_px),
        'final_area_px': float(final_area_px),
        'initial_area_um2': float(initial_area_um2),
        'final_area_um2': float(final_area_um2),
        'final_closure_pct': float(final_closure),
        'healing_rate_um2_per_hr': float(healing_rate_um2_per_hr),
        'r_squared': float(r_squared),
        'num_timepoints': int(len(timepoints)),
        'time_to_50_closure_hr': float(time_to_50) if time_to_50 is not None else None,
        'area_mean_px': float(np.mean(areas_px)),
        'area_std_px': float(np.std(areas_px)),
        'healing_rate_mean_px_per_hr': float(healing_rate_mean_px_per_hr),
        'processing_time_sec': 0.0,
        'tracking_results': {}
    }


def run_cell_tracking(image_files, masks, time_interval, pixel_scale, output_dir):
    try:
        logger.info(f"🔬 Starting Cell Tracking (output to {output_dir})...")
        os.makedirs(output_dir, exist_ok=True)
        # This call now correctly passes the image files and the WOUND masks
        tracking_results = cell_tracking.track_cells_in_timeseries(image_files, masks, time_interval, pixel_scale, output_dir)
        logger.info("✓ Cell Tracking Complete")
        return tracking_results
    except Exception as e:
        logger.error(f"Error during cell tracking: {e}", exc_info=True)
        return {}


def create_overlay_gallery(image_files, masks, output_dir, experiment_name):
    """
    Create overlay images with wound contours drawn and save them into output_dir.
    Returns a list of created overlay file paths.
    """
    gallery_dir = os.path.abspath(output_dir)
    os.makedirs(gallery_dir, exist_ok=True)
    logger.info(f"Creating overlay gallery at {gallery_dir}...")
    overlay_paths = []
    for idx, (img_path, mask) in enumerate(tqdm(list(zip(image_files, masks)), total=min(len(image_files), len(masks)), desc="Creating Gallery")):
        try:
            original_img = cv2.imread(img_path)
            if original_img is None:
                logger.warning(f"Could not read original image {img_path}; skipping overlay.")
                continue
            if len(original_img.shape) == 2 or (len(original_img.shape) == 3 and original_img.shape[2] == 1):
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
            contours, _ = detect_wound_contours(mask)
            if contours:
                cv2.drawContours(original_img, contours, -1, (0, 0, 255), 2)
            overlay_path = os.path.join(gallery_dir, f"{experiment_name}_frame_{idx:04d}.jpg")
            cv2.imwrite(overlay_path, original_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            overlay_paths.append(overlay_path)
        except Exception as e:
            logger.warning(f"Could not create overlay for frame {idx}: {e}", exc_info=True)
    return overlay_paths


def create_animation(overlay_paths, output_dir, experiment_name, time_interval):
    if not overlay_paths:
        logger.warning("No overlay images found to create animation.")
        return None
    video_path = os.path.join(output_dir, f'{experiment_name}_analysis_video.mp4')
    # time_interval is hours/frame -> fps = frames per second; choose reasonable mapping:
    fps = max(1, min(30, int(round(1.0 / max(time_interval, 1e-3)))))
    logger.info(f"Creating MP4 animation at {video_path} (FPS={fps})...")
    try:
        with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as writer:
            for img_path in tqdm(overlay_paths, desc="Animating MP4"):
                frame = imageio.imread(img_path)
                writer.append_data(frame)
        logger.info(f"✓ Animation saved: {video_path}")
        return video_path
    except Exception as e:
        logger.error(f"Failed to create MP4 animation: {e}. Is ffmpeg installed? (pip install imageio-ffmpeg)", exc_info=True)
        return None


def save_results(results, output_dir, experiment_name):
    """
    Save timeseries CSV and a summary JSON. Returns (csv_path, json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({
        'time(hours)': results['timepoints'],
        'wound_area(px)': results['areas_px'],
        'wound_area(um2)': results['areas_um2'],
        'closure_percentage': results['closure_percentages']
    })
    csv_path = os.path.join(output_dir, f'{experiment_name}_timeseries.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved timeseries data to: {csv_path}")

    # Compose summary
    summary = {k: v for k, v in results.items() if k not in ['masks', 'timepoints', 'areas_px', 'areas_um2', 'closure_percentages']}
    summary['experiment'] = experiment_name
    # merge tracking results if present
    summary.update(results.get('tracking_results', {}))

    json_path = os.path.join(output_dir, f'{experiment_name}_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved summary to: {json_path}")
    return csv_path, json_path


def create_visualization(results, output_dir, experiment_name):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Top-left: wound area (um2) over time
    ax = axes[0, 0]
    if 'areas_um2' in results and len(results['areas_um2']) > 0:
        ax.plot(results['timepoints'], results['areas_um2'], marker='o')
        ax.set_title('Wound Area (µm²) over Time')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Area (µm²)')
    # Top-right: closure %
    ax = axes[0, 1]
    if 'closure_percentages' in results and len(results['closure_percentages']) > 0:
        ax.plot(results['timepoints'], results['closure_percentages'], marker='o')
        ax.set_title('Closure (%) over Time')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Closure (%)')
    # Bottom-left: histogram of areas (px)
    ax = axes[1, 0]
    ax.hist(results.get('areas_px', []), bins=20)
    ax.set_title('Area Distribution (px)')
    # Bottom-right: leave space for tracking summary or last frame
    ax = axes[1, 1]
    ax.axis('off')
    plot_path = os.path.join(output_dir, f'{experiment_name}_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved static plot to: {plot_path}")
    return plot_path


def create_interactive_plot(csv_path, output_json_path):
    logger.info("Creating interactive Plotly JSON...")
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning("CSV empty; skipping interactive plot.")
            return
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        area_col = 'wound_area(um2)' if 'wound_area(um2)' in df.columns else 'wound_area(px)'
        area_unit = "µm²" if 'wound_area(um2)' in df.columns else "px"

        fig.add_trace(go.Scatter(x=df['time(hours)'], y=df[area_col], name=f'Wound Area ({area_unit})',
                                 mode='lines+markers'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['time(hours)'], y=df['closure_percentage'], name='Closure (%)',
                                 mode='lines+markers', line=dict(dash='dot')), secondary_y=True)

        fig.update_layout(title_text='Interactive Analysis', template='plotly_dark',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          hovermode="x unified")
        fig.update_yaxes(title_text=f"Wound Area ({area_unit})", secondary_y=False)
        fig.update_yaxes(title_text="Closure (%)", secondary_y=True, range=[0, 105])
        fig.update_xaxes(title_text="Time (hours)")

        fig.write_json(output_json_path)
        logger.info(f"Saved interactive plot to: {output_json_path}")
    except Exception as e:
        logger.error(f"Failed to create interactive plot: {e}", exc_info=True)


def main():
    args = parse_arguments()
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("🔬 WOUND HEALING BATCH ANALYSIS (REFACTORED & FIXED)")
    logger.info("=" * 70)
    logger.info(f"Input Directory: {args.input}")
    logger.info(f"Output Directory: {args.output}")
    logger.info(f"Disk Size: {'Auto-select' if args.disk_size == 0 else args.disk_size}")
    logger.info(f"Time Interval: {args.time_interval} hours/frame")
    logger.info(f"Pixel Scale: {args.pixel_scale} µm/px")
    logger.info(f"Cell Tracking Enabled: {args.track_cells}")
    logger.info("=" * 70)

    image_files = get_image_files(args.input)
    if len(image_files) < 2:
        logger.error(f"Not enough images found in {args.input} (found {len(image_files)}). Aborting.")
        sys.exit(1)
    logger.info(f"✓ Found {len(image_files)} images to analyze.")

    csv_dir = os.path.join(args.output, 'csv')
    plots_dir = os.path.join(args.output, 'plots')
    gallery_dir = os.path.join(args.output, 'gallery')
    video_dir = os.path.join(args.output, 'video')
    tracking_dir = os.path.join(args.output, 'tracking')

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(tracking_dir, exist_ok=True)

    selected_disk_size = auto_select_disk_size(image_files[0], image_files[-1]) if args.disk_size == 0 else args.disk_size
    logger.info(f"Using Disk Size: {selected_disk_size}")

    experiment_name = secure_filename(args.experiment_name) if args.experiment_name else os.path.basename(os.path.normpath(args.input))
    logger.info(f"Using Experiment Name: {experiment_name}")

    results = process_timeseries(image_files, selected_disk_size, args.time_interval, args.save_masks, args.output, args.pixel_scale)
    if results is None:
        logger.error("Time-series processing failed. Aborting.")
        sys.exit(1)

    if args.track_cells:
        # This now passes results['masks'] (the WOUND masks) to the tracking function
        results['tracking_results'] = run_cell_tracking(image_files, results['masks'], args.time_interval, args.pixel_scale, tracking_dir)

    overlay_paths = create_overlay_gallery(image_files, results['masks'], gallery_dir, experiment_name)
    create_animation(overlay_paths, video_dir, experiment_name, args.time_interval)

    processing_time = time.time() - start_time
    results['processing_time_sec'] = processing_time

    csv_path, json_path = save_results(results, csv_dir, experiment_name)
    plot_path = create_visualization(results, plots_dir, experiment_name)

    if args.visualize:
        interactive_plot_path = os.path.join(plots_dir, f'{experiment_name}_analysis_interactive.json')
        create_interactive_plot(csv_path, interactive_plot_path)

    logger.info("=" * 70)
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    logger.info(f"Results saved in: {args.output}")
    logger.info("-" * 20)
    logger.info(f"Final Closure: {results['final_closure_pct']:.1f}%")

    if args.pixel_scale != 1.0:
        logger.info(f"Healing Speed: {abs(results['healing_rate_um2_per_hr']):.2f} µm²/hr")
    else:
        logger.info(f"Healing Speed: {abs(results.get('healing_rate_mean_px_per_hr', 0)):.2f} px/hr")

    logger.info(f"Healing Consistency (R²): {results.get('r_squared', 0):.3f}")

    if results.get('tracking_results', {}).get('num_cells_tracked', 0) > 0:
        tr = results['tracking_results']
        logger.info(f"Cells Tracked: {tr['num_cells_tracked']}")
        logger.info(f"Mean Velocity: {tr.get('mean_velocity_um_min', 0):.2f} μm/min")
        logger.info(f"Migration Efficiency: {tr.get('migration_efficiency_mean', 0):.3f}")
        logger.info(f"Mean Directionality: {tr.get('mean_directionality', 0):.3f}") # NEW

    logger.info("=" * 70)


if __name__ == '__main__':
    main()