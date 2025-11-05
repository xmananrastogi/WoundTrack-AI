#!/usr/bin/env python3
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, abort
from flask_cors import CORS
import os, glob, json, subprocess, threading, base64, pandas as pd, cv2, numpy as np, posixpath, shutil
from scipy import stats
import io, zipfile, uuid, logging
from PIL import Image, ImageSequence
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from werkzeug.utils import secure_filename

# NEW: Import Plotly for backend plot generation
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from config import Config
import database  # --- Import database module ---
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Ensure results dir exists (upload dir is handled by Config)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

analysis_state = {'running': False, 'progress': 0, 'status': 'Idle', 'current': ''}

# METRIC NAMES WITH DESCRIPTIONS
METRIC_INFO = {
    'initial_area_px': {'name': 'Starting Wound Size (px)', 'unit': 'px²'},
    'final_area_px': {'name': 'Final Wound Size (px)', 'unit': 'px²'},
    'healing_rate_px_per_hr': {'name': 'Healing Speed (px/hr)', 'unit': 'px/hr'},
    'pixel_scale': {'name': 'Pixel Scale', 'unit': 'µm/px'},
    'initial_area_mm2': {'name': 'Starting Wound Size', 'unit': 'mm²'},
    'final_area_mm2': {'name': 'Final Wound Size', 'unit': 'mm²'},
    'healing_rate_um_per_hr': {'name': 'Healing Speed', 'unit': 'µm/hr'},
    'time_to_50_closure_hr': {'name': 'Time to 50% Closure', 'unit': 'hours'},
    'final_closure_pct': {'name': 'Wound Closure', 'unit': '%'},
    'r_squared': {'name': 'Healing Consistency (R²)', 'unit': ''},
    'num_timepoints': {'name': 'Frames Analyzed', 'unit': ''},
    'area_mean_mm2': {'name': 'Mean Wound Area', 'unit': 'mm²'},
    'area_std_mm2': {'name': 'Area Variability (SD)', 'unit': 'mm²'},
    'healing_rate_mean_um_per_hr': {'name': 'Mean Frame-to-Frame Speed', 'unit': 'µm/hr'},
    'healing_rate_std_um_per_hr': {'name': 'Speed Variability (SD)', 'unit': 'µm/hr'},
    'num_cells_tracked': {'name': 'Cells Tracked', 'unit': ''},
    'mean_velocity_um_min': {'name': 'Mean Cell Velocity', 'unit': 'µm/min'},
    'migration_efficiency_mean': {'name': 'Migration Efficiency', 'unit': ''},
    'mean_directionality': {'name': 'Mean Directionality', 'unit': ''},
    "initial_area_um2": {"name": "Starting Wound Size", "unit": "µm²"},
    "final_area_um2": {"name": "Final Wound Size", "unit": "µm²"},
    "healing_rate_um2_per_hr": {"name": "Healing Speed", "unit": "µm²/hr"},
    "pixel_scale_um_per_px": {"name": "Pixel Scale", "unit": "µm/px"}
}

CONDITION_NAMES = {
    'MDCK_Control': ('🧬 Epithelial Cells (Baseline)', 'Normal epithelial cells - baseline'),
    'MDCK_HGF': ('⚡ Epithelial + Growth Factor', 'Epithelial cells treated with HGF/SF'),
    'DA3_Control': ('🔬 Cancer Cells (Baseline)', 'Cancer cells - baseline'),
    'DA3_PHA': ('💊 Cancer + Immune Activation', 'Cancer cells with immune activation'),
    'DA3_HGF': ('🔥 Cancer + Growth Factor', 'Cancer cells with growth factor'),
    'Uploaded Data': ('📤 Uploaded Data', 'User-uploaded dataset')
}


# --- REMOVED: get_available_datasets() function is no longer needed ---


# ----------------- Robust results discovery -----------------
def get_all_results():
    """
    Find *_summary.json files under RESULTS_FOLDER and create a list of result dicts.
    """
    results = []
    base = os.path.abspath(app.config['RESULTS_FOLDER'])
    if not os.path.exists(base):
        return results

    pattern = os.path.join(base, '**', '*_summary.json')
    summary_files = glob.glob(pattern, recursive=True)

    for sfile in sorted(summary_files, key=os.path.getmtime, reverse=True):
        try:
            with open(sfile, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except Exception:
            raw = None

        rel = os.path.relpath(sfile, base).replace(os.sep, '/')
        parts = rel.split('/')
        if parts[0] == 'uploads' and len(parts) >= 3:
            result_id = posixpath.join('uploads', parts[1])  # e.g., uploads/uuid
            condition = 'Uploaded Data'
            experiment_name = raw.get('experiment', os.path.splitext(parts[-1])[0].replace('_summary', ''))
            base_result_dir = os.path.join(base, 'uploads', parts[1])
        else:
            if len(parts) >= 3:  # e.g., DA3_Control/CIL_43406/csv/DA3_Control_CIL_43406_summary.json
                condition = parts[0]
                experiment_name = parts[1]
                base_result_dir = os.path.join(base, parts[0], experiment_name)
                result_id = posixpath.join(parts[0], experiment_name)
            else:
                condition = 'Unknown'
                experiment_name = raw.get('experiment', os.path.splitext(parts[-1])[0].replace('_summary', ''))
                base_result_dir = os.path.dirname(sfile)
                result_id = experiment_name

        base_name = raw.get('experiment', os.path.splitext(os.path.basename(sfile))[0].replace('_summary', ''))

        plot_candidates = [
            os.path.join(base_result_dir, 'plots', f'{base_name}_analysis.png'),
            os.path.join(base_result_dir, 'plots', f'{base_name}.png'),
            os.path.join(base_result_dir, f'{base_name}.png'),
        ]
        csv_candidates = [
            os.path.join(base_result_dir, 'csv', f'{base_name}_timeseries.csv'),
            os.path.join(base_result_dir, f'{base_name}.csv'),
        ]
        video_candidates = [
            os.path.join(base_result_dir, 'video', f'{base_name}_analysis_video.mp4'),
            os.path.join(base_result_dir, 'video', f'{base_name}.mp4'),
        ]

        def pick_first_existing(cands):
            for c in cands:
                if c and os.path.exists(c):
                    return c
            return None

        plot_path = pick_first_existing(plot_candidates)
        csv_path = pick_first_existing(csv_candidates)
        video_path = pick_first_existing(video_candidates)

        gallery_dir = os.path.join(base_result_dir, 'gallery')
        gallery_files = []
        if os.path.isdir(gallery_dir):
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                gallery_files.extend(sorted(glob.glob(os.path.join(gallery_dir, ext))))
        gallery_urls = [path_to_url_for_result(p) for p in gallery_files if path_to_url_for_result(p)]

        interactive_json_path = os.path.join(base_result_dir, 'plots', f'{base_name}_analysis_interactive.json')
        if not os.path.exists(interactive_json_path):
            interactive_json_path = None

        cond_name = CONDITION_NAMES.get(condition, (condition, ''))[0]

        results.append({
            'id': result_id,
            'rel_summary_path': rel,
            'summary_path': sfile,
            'raw_summary': raw,
            'csv_path': csv_path,
            'plot_path': plot_path,
            'plot_url': path_to_url_for_result(plot_path),
            'interactive_plot_path': interactive_json_path,
            'interactive_plot_url': path_to_url_for_result(interactive_json_path),
            'gallery': gallery_files,
            'gallery_thumbs': gallery_urls,
            'video_path': video_path,
            'video_url': path_to_url_for_result(video_path),
            'condition': condition,
            'condition_name': cond_name,
            'experiment_name': experiment_name
        })

    results.sort(key=lambda r: os.path.getmtime(r['summary_path']) if os.path.exists(r['summary_path']) else 0,
                 reverse=True)
    return results


def path_to_url_for_result(fs_path: Optional[str]) -> Optional[str]:
    if not fs_path:
        return None
    base = os.path.abspath(app.config['RESULTS_FOLDER'])
    try:
        abs_path = os.path.abspath(fs_path)
        rel = os.path.relpath(abs_path, base)
    except Exception:
        return None
    if rel.startswith('..'):
        return None
    rel_url = rel.replace(os.sep, '/').lstrip('./')
    return f'/results_data/{rel_url}'


# ----------------- Statistics & helpers -----------------

# --- NEW: Plotting functions for Stats page ---
def create_correlation_heatmap_json(df: pd.DataFrame) -> str:
    """Generates a Plotly heatmap JSON from a metrics DataFrame."""
    if df.empty or len(df.columns) < 2:
        return "{}"
    try:
        # Compute correlation, ensuring only numeric columns are used
        corr = df.corr(numeric_only=True)
        if corr.empty:
            return "{}"

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Teal',  # Use our theme color
            zmin=-1, zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            hoverongaps=False))
        fig.update_layout(
            title="Metric Correlation Heatmap",
            template="plotly_dark",
            height=600,
            xaxis_showgrid=False, yaxis_showgrid=False,
            yaxis_autorange='reversed'
        )
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        return "{}"


def create_stats_box_plots_json(df: pd.DataFrame) -> str:
    """Generates Plotly box plots for key metrics, faceted by condition."""
    if df.empty:
        return "{}"
    try:
        # Melt dataframe to long format for easier plotting with plotly express
        metrics_to_plot = [
            'Closure (%)', 'Healing Speed (µm²/hr)', 'Consistency (R²)',
            'Cell Velocity (µm/min)', 'Efficiency', 'Directionality'
        ]
        # Ensure only available metrics are used
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        if not available_metrics:
            return "{}"

        df_melted = df.melt(id_vars=['Condition'], value_vars=available_metrics,
                            var_name='Metric', value_name='Value')

        fig = px.box(df_melted, x='Condition', y='Value',
                     color='Condition',
                     facet_row='Metric',
                     title="Metric Distributions by Condition")

        fig.update_layout(
            template="plotly_dark",
            height=min(max(1000, len(available_metrics) * 300), 2000),  # Dynamic height
            showlegend=False
        )
        # Make y-axes independent so scales are correct
        fig.update_yaxes(matches=None, showticklabels=True)
        # Clean up facet labels
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error creating box plots: {e}")
        return "{}"


# ----------------- Analysis runner -----------------
def run_analysis(input_dir, output_dir, disk_size, time_interval, pixel_scale, analysis_id, sample_id=None):
    global analysis_state
    try:
        analysis_state['running'] = True
        analysis_state['progress'] = 0
        analysis_state['status'] = f'Starting analysis for {analysis_id}'
        analysis_state['current'] = analysis_id
        os.makedirs(output_dir, exist_ok=True)

        safe_sample_id = secure_filename(sample_id) if sample_id else secure_filename(analysis_id)

        cmd = ['python', 'batch_analysis.py', '--input', input_dir, '--output', output_dir,
               '--disk-size', str(disk_size), '--time-interval', str(time_interval),
               '--pixel-scale', str(pixel_scale), '--visualize', '--track-cells',
               '--experiment-name', safe_sample_id]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                                   universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            logger.info(line.rstrip())
            if "Auto-selected best disk size" in line:
                analysis_state['status'] = f'Auto-selected parameters... {line.split(":")[-1].strip()}'
            elif "Analyzing" in line:
                analysis_state['status'] = f'Processing frames for {safe_sample_id}...'
            elif "Creating overlay gallery" in line:
                analysis_state['status'] = f'Creating gallery for {safe_sample_id}...'
                analysis_state['progress'] = 50
            elif "Creating animation" in line:
                analysis_state['status'] = f'Creating video for {safe_sample_id}...'
                analysis_state['progress'] = 75

        _, stderr = process.communicate()
        if process.returncode == 0:
            analysis_state['progress'] = 100
            analysis_state['status'] = f'✅ Complete: {safe_sample_id}'

            try:
                summary_path = os.path.join(output_dir, 'csv', f'{safe_sample_id}_summary.json')
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary_data = json.load(f)

                    base_results_dir = os.path.abspath(app.config['RESULTS_FOLDER'])
                    rel_output_dir = os.path.relpath(output_dir, base_results_dir)
                    result_id = rel_output_dir.replace(os.sep, '/')

                    summary_data['experiment_name'] = safe_sample_id
                    if 'uploads' in result_id:
                        summary_data['condition_name'] = 'Uploaded Data'
                    else:
                        condition_key = result_id.split('/')[0]
                        summary_data['condition_name'] = CONDITION_NAMES.get(condition_key, (condition_key,))[0]

                    database.upsert_experiment(summary_data, result_id)
                else:
                    logger.error(f"Could not find summary file to save to DB: {summary_path}")
            except Exception as e:
                logger.error(f"Failed to save result to database: {e}", exc_info=True)

        else:
            error_message = stderr[-200:] if stderr else "Unknown error"
            analysis_state['status'] = f'❌ Error running analysis for {safe_sample_id}: {error_message}'
            logger.error(f"Analysis Error: {stderr}")
    except Exception as e:
        analysis_state['status'] = f'❌ Error: {str(e)}'
        logger.exception(f"Exception in run_analysis: {e}")
    finally:
        analysis_state['running'] = False


# ----------------- Routes -----------------
@app.route('/')
def index():
    """Main route that renders the HTML page (enriches results to match template expectations)."""
    results = get_all_results()

    for r in results:
        r['data'] = r.get('raw_summary') or {}
        if 'condition_name' not in r or not r['condition_name']:
            cond = r.get('condition', 'Unknown')
            r['condition_name'] = CONDITION_NAMES.get(cond, (cond.replace('_', ' ').title(), ''))[0]

        r['plot_b64'] = None
        plot_path = r.get('plot_path')
        try:
            if plot_path and os.path.exists(plot_path):
                with open(plot_path, 'rb') as pf:
                    r['plot_b64'] = base64.b64encode(pf.read()).decode()
        except Exception as e:
            logger.warning(f"Warning reading plot for {r.get('id')}: {e}")
            r['plot_b64'] = None

        r['csv_b64'] = ""
        csv_path = r.get('csv_path')
        try:
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as cf:
                    r['csv_b64'] = base64.b64encode(cf.read().encode()).decode()
        except Exception as e:
            logger.warning(f"Warning reading csv for {r.get('id')}: {e}")
            r['csv_b64'] = ""

        gallery_list = r.get('gallery') or []
        thumbs = []
        if gallery_list:
            picks = [gallery_list[0]]
            if len(gallery_list) > 2:
                picks.append(gallery_list[len(gallery_list) // 2])
            if len(gallery_list) > 1:
                picks.append(gallery_list[-1])
            for p in sorted(list(dict.fromkeys(picks))):
                url = path_to_url_for_result(p)
                if url:
                    thumbs.append(url)
        r['gallery_thumbs'] = thumbs

        vpath = r.get('video_path')
        if vpath and os.path.exists(vpath):
            r['video_path'] = path_to_url_for_result(vpath)
        else:
            r['video_path'] = path_to_url_for_result(vpath) if vpath else None

    # Page-level aggregates
    # --- REMOVED: available_datasets ---
    total_exp = len(results)
    total_cond = len(set(r.get('condition', 'Unknown') for r in results))
    total_frames = sum(int((r.get('data') or {}).get('num_timepoints', 0)) for r in results)
    total_time = sum((r.get('data') or {}).get('processing_time_sec', 0) for r in results) / 60.0

    # --- NEW: Get stats from database ---
    cond_stats = database.get_stats_by_condition()
    pvalues = database.calculate_all_pvalues()

    # --- NEW: Generate plots for Stats page ---
    metrics_df = database.get_all_metrics_for_plots()
    correlation_json = create_correlation_heatmap_json(metrics_df)
    box_plots_json = create_stats_box_plots_json(metrics_df)

    all_conditions = sorted(
        set(r.get('condition', 'Unknown') for r in results if r.get('condition', 'Unknown') != 'Uploaded Data'))

    condition_names_safe = dict(CONDITION_NAMES)
    for cond in sorted(set(r.get('condition', 'Unknown') for r in results)):
        if cond not in condition_names_safe:
            condition_names_safe[cond] = (cond.replace('_', ' ').title(), '')

    return render_template('index.html',
                           results=results,
                           total_exp=total_exp,
                           total_cond=total_cond,
                           total_frames=total_frames,
                           total_time=total_time,
                           # --- REMOVED: available_datasets ---
                           cond_stats=cond_stats,
                           pvalues=pvalues,
                           all_conditions=all_conditions,
                           condition_names=condition_names_safe,
                           metric_info=METRIC_INFO,
                           correlation_json=correlation_json,  # NEW
                           box_plots_json=box_plots_json  # NEW
                           )


@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        frame_interval = int(request.form.get('frameInterval', 5))

        filename = secure_filename(file.filename)
        analysis_id = str(uuid.uuid4())
        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
        os.makedirs(input_dir, exist_ok=True)
        save_path = os.path.join(input_dir, filename)
        file.save(save_path)

        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(input_dir)
            os.remove(save_path)
        elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            vidcap = cv2.VideoCapture(save_path)
            success, image = vidcap.read()
            count, frame_num = 0, 0
            while success:
                if count % frame_interval == 0:
                    cv2.imwrite(os.path.join(input_dir, f"frame_{frame_num:04d}.png"), image)
                    frame_num += 1
                success, image = vidcap.read()
                count += 1
            os.remove(save_path)
        elif filename.lower().endswith(('.tif', '.tiff', '.gif')):
            img = Image.open(save_path)
            for i, page in enumerate(ImageSequence.Iterator(img)):
                page.convert('L').save(os.path.join(input_dir, f"frame_{i:04d}.png"))
            os.remove(save_path)

        extracted_items = os.listdir(input_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(input_dir, extracted_items[0])):
            if extracted_items[0] != "__MACOSX":
                sub_dir = os.path.join(input_dir, extracted_items[0])
                for item in os.listdir(sub_dir):
                    shutil.move(os.path.join(sub_dir, item), input_dir)
                os.rmdir(sub_dir)

        return jsonify({'status': 'uploaded', 'analysis_id': analysis_id})
    except Exception as e:
        logger.exception(f"Unhandled error in /api/upload: {e}")
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        if not analysis_id:
            return jsonify({'error': 'No analysis_id provided'}), 400

        disk_size = int(data.get('disk_size', 0))
        time_interval = float(data.get('time_interval', 0.25))
        pixel_scale = float(data.get('pixel_scale', 1.0))
        sample_id = data.get('sample_id') or None

        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
        output_dir = os.path.join(app.config['RESULTS_FOLDER'], 'uploads', analysis_id)

        thread_args = (input_dir, output_dir, disk_size, time_interval, pixel_scale, analysis_id, sample_id)
        thread = threading.Thread(target=run_analysis, args=thread_args)
        thread.daemon = True
        thread.start()
        return jsonify({'status': 'started', 'analysis_id': analysis_id})
    except Exception as e:
        logger.exception(f"Unhandled error in /api/analyze: {e}")
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500


# --- REMOVED: /api/analyze_existing route ---


@app.route('/api/status')
def api_status():
    return jsonify(analysis_state)


@app.route('/api/comparison_data')
def api_comparison_data():
    try:
        experiments = database.get_all_experiments_for_comparison()
        return jsonify(experiments)
    except Exception as e:
        logger.error(f"Error fetching comparison data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# --- NEW: Delete Experiment Endpoint ---
@app.route('/api/delete_experiment', methods=['POST'])
def api_delete_experiment():
    data = request.json
    result_id = data.get('result_id')
    if not result_id:
        return jsonify({'error': 'No result_id provided'}), 400

    try:
        # 1. Delete from database
        db_deleted = database.delete_experiment(result_id)

        # 2. Delete from file system
        # Sanitize the result_id to prevent path traversal
        safe_rel_path = os.path.normpath(result_id).replace('..', '')
        if safe_rel_path.startswith('/'):
            safe_rel_path = safe_rel_path[1:]

        fs_path = os.path.join(app.config['RESULTS_FOLDER'], safe_rel_path)

        if not os.path.exists(fs_path):
            logger.warning(f"File path not found for deletion, but DB record might be gone: {fs_path}")
            # If DB was deleted, we still count it as a success
            if db_deleted:
                return jsonify({'status': 'success', 'message': 'DB record deleted, file path not found.'})
            else:
                return jsonify({'error': 'Experiment not found in DB or file system.'}), 404

        # Check if path is safely within the RESULTS_FOLDER
        if not os.path.abspath(fs_path).startswith(os.path.abspath(app.config['RESULTS_FOLDER'])):
            logger.error(f"Potential security violation: attempt to delete path outside results: {fs_path}")
            return jsonify({'error': 'Invalid path'}), 400

        # Delete the whole directory
        shutil.rmtree(fs_path)
        logger.info(f"Successfully deleted experiment files: {fs_path}")

        return jsonify({'status': 'success', 'message': 'Experiment deleted from database and file system.'})

    except Exception as e:
        logger.error(f"Error deleting experiment '{result_id}': {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# --- End of New Endpoint ---


@app.route('/results_json/<path:exp_id>')
def get_result_json(exp_id):
    results = get_all_results()
    target_result = next((res for res in results if res['id'] == exp_id), None)
    if not target_result:
        return jsonify({'error': 'Result not found'}), 404

    data = dict(target_result.get('raw_summary') or {})

    data['plot_url'] = target_result.get('plot_url')
    data['plot_b64'] = None
    if target_result.get('plot_path') and os.path.exists(target_result['plot_path']):
        try:
            with open(target_result['plot_path'], 'rb') as pf:
                data['plot_b64'] = base64.b64encode(pf.read()).decode()
        except Exception:
            pass

    data['interactive_plot_url'] = target_result.get('interactive_plot_url')
    data['video_url'] = target_result.get('video_url')
    data['gallery_thumbs'] = target_result.get('gallery_thumbs', [])
    data['condition_name'] = target_result.get('condition_name')
    data['experiment_name'] = target_result.get('experiment_name')

    if target_result.get('csv_path') and os.path.exists(target_result['csv_path']):
        try:
            with open(target_result['csv_path'], 'r', encoding='utf-8') as cf:
                data['csv_b64'] = base64.b64encode(cf.read().encode()).decode()
            data['csv_url'] = path_to_url_for_result(target_result['csv_path'])
        except Exception:
            data['csv_b64'] = ""
            data['csv_url'] = None
    else:
        data['csv_b64'] = ""
        data['csv_url'] = None

    if target_result.get('interactive_plot_path') and os.path.exists(target_result['interactive_plot_path']):
        try:
            with open(target_result['interactive_plot_path'], 'r', encoding='utf-8') as f:
                data['plot_json'] = f.read()
        except Exception:
            data['plot_json'] = None
    else:
        data['plot_json'] = None

    return jsonify(data)


@app.route('/results_data/<path:filename>')
def send_result_file(filename):
    base_dir = os.path.abspath(app.config['RESULTS_FOLDER'])
    safe_rel = os.path.normpath(filename).replace('..', '')
    abs_path = os.path.abspath(os.path.join(base_dir, safe_rel))
    if not abs_path.startswith(base_dir):
        abort(404)
    if not os.path.exists(abs_path):
        abort(404)
    rel_for_send = os.path.relpath(abs_path, base_dir)
    return send_from_directory(base_dir, rel_for_send, as_attachment=False)


@app.route('/download-pdf/<path:exp_id>')
def download_pdf(exp_id):
    results = get_all_results()
    result = next((res for res in results if res['id'] == exp_id), None)
    if not result:
        return jsonify({'error': 'Not found'}), 404

    data = result['raw_summary'] or {}
    exp_name = result['experiment_name']
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
                                 textColor=colors.HexColor('#008B8B'), spaceAfter=30, alignment=1)
    h3_style = ParagraphStyle('CustomH3', parent=styles['Heading3'], fontSize=14, textColor=colors.HexColor('#333333'),
                              spaceAfter=10, spaceBefore=10)
    story.append(Paragraph(f'Wound Healing Analysis Report', title_style))
    story.append(Paragraph(f'<b>Experiment:</b> {exp_name}', styles['Normal']))
    story.append(Paragraph(f'<b>Condition:</b> {result.get("condition_name")}', styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph('Wound Area Analysis', h3_style))
    time_50_val = data.get('time_to_50_closure_hr')
    time_50_str = f"{time_50_val:.1f}" if time_50_val is not None else "N/A"
    pixel_scale = data.get('pixel_scale_um_per_px', 1.0)
    area_unit = "µm²" if pixel_scale != 1.0 else "pixels"
    speed_unit = "µm²/hr" if pixel_scale != 1.0 else "px/hr"
    initial_area_display = data.get('initial_area_um2', data.get('initial_area_px', 0))
    final_area_display = data.get('final_area_um2', data.get('final_area_px', 0))
    healing_rate_display = abs(data.get('healing_rate_um2_per_hr', data.get('healing_rate_mean_px_per_hr', 0)))
    mean_area_display = np.mean(data.get('areas_um2', data.get('areas_px', [0])))
    std_area_display = np.std(data.get('areas_um2', data.get('areas_px', [0])))
    table_data = [
        ['Metric', 'Value', 'Unit'],
        ['Starting Wound Size', f"{initial_area_display:.0f}", area_unit],
        ['Final Wound Size', f"{final_area_display:.0f}", area_unit],
        ['Wound Closure', f"{data.get('final_closure_pct', 0):.1f}", '%'],
        ['Healing Speed (Slope)', f"{healing_rate_display:.2f}", speed_unit],
        ['Healing Consistency (R²)', f"{data.get('r_squared', 0):.3f}", ''],
        ['Time to 50% Closure', time_50_str, 'hours'],
        ['Mean Wound Area', f"{mean_area_display:.0f}", area_unit],
        ['Area Variability (SD)', f"± {std_area_display:.0f}", area_unit],
        ['Total Frames', f"{data.get('num_timepoints', 0)}", ''],
    ]
    style_commands = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#008B8B')),
                      ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                      ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                      ('FONTSIZE', (0, 0), (-1, 0), 12), ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                      ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD')),
                      ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F4F4F4'), colors.white])]
    cell_count = data.get('num_cells_tracked', data.get('num_cells', 0))
    story.append(Table(table_data, colWidths=[2.5 * inch, 1.5 * inch, 1 * inch], style=style_commands))

    if cell_count > 0:
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph('Cell Migration Analysis', h3_style))
        mean_vel_val = data.get('mean_velocity_um_min', data.get('mean_velocity'))
        mean_vel_str = f"{mean_vel_val:.2f}" if mean_vel_val is not None else "N/A"
        mig_eff_val = data.get('migration_efficiency_mean', data.get('migration_efficiency'))
        mig_eff_str = f"{mig_eff_val:.3f}" if mig_eff_val is not None else "N/A"
        mean_disp_val = data.get('mean_displacement_um', data.get('mean_displacement'))
        mean_disp_str = f"{mean_disp_val:.2f}" if mean_disp_val is not None else "N/A"
        mean_path_val = data.get('mean_path_length_um', data.get('mean_path_length'))
        mean_path_str = f"{mean_path_val:.2f}" if mean_path_val is not None else "N/A"
        directionality_val = data.get('mean_directionality')
        directionality_str = f"{directionality_val:.3f}" if directionality_val is not None else "N/A"

        tracking_table_data = [
            ['Metric', 'Value', 'Unit'],
            ['Cells Tracked', f"{cell_count}", ''],
            ['Mean Velocity', mean_vel_str, 'μm/min'],
            ['Migration Efficiency', mig_eff_str, ''],
            ['Mean Directionality', directionality_str, ''],
            ['Mean Displacement', mean_disp_str, 'μm'],
            ['Mean Path Length', mean_path_str, 'μm']
        ]
        tracking_style_commands = style_commands.copy()
        tracking_style_commands[0] = ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0078D4'))
        story.append(
            Table(tracking_table_data, colWidths=[2.5 * inch, 1.5 * inch, 1 * inch], style=tracking_style_commands))

    story.append(PageBreak())
    story.append(Paragraph('Analysis Plot', h3_style))
    if result.get('plot_path') and os.path.exists(result['plot_path']):
        try:
            with open(result['plot_path'], 'rb') as pf:
                plot_data = pf.read()
            plot_img = RLImage(io.BytesIO(plot_data), width=6 * inch, height=4.5 * inch)
            plot_img.hAlign = 'CENTER'
            story.append(plot_img)
        except Exception as e:
            logger.warning(f"Error adding plot to PDF: {e}")
    tracking_plot_file = data.get('trajectory_plot', data.get('trajectories_plot'))
    if tracking_plot_file and os.path.exists(tracking_plot_file):
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph('Cell Trajectories', h3_style))
        try:
            with open(tracking_plot_file, 'rb') as f:
                traj_data = f.read()
            traj_img = RLImage(io.BytesIO(traj_data), width=6 * inch, height=4.5 * inch)
            traj_img.hAlign = 'CENTER'
            story.append(traj_img)
        except Exception as e:
            logger.warning(f"Error adding trajectory plot to PDF: {e}")

    doc.build(story)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, mimetype='application/pdf', download_name=f'{exp_name}_report.pdf', as_attachment=True)


if __name__ == '__main__':
    database.create_table()

    logger.info("\n" + "=" * 70)
    logger.info("🔬 WOUNDTRACK AI ANALYSIS SERVER (V4 - AUTO-SEGMENT)")
    logger.info("=" * 70)
    logger.info("✅ Refactored with templates, config, and robust error handling.")
    logger.info("✅ Running on: http://localhost:8080")
    logger.info("=" * 70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)