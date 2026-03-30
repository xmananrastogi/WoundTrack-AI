# api/routes.py
import os
import uuid
import zipfile
import shutil
import cv2
import base64
from io import BytesIO
from flask import Blueprint, jsonify, request, render_template, send_file, send_from_directory, abort, current_app
from werkzeug.utils import secure_filename
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

from config.constants import CONDITION_NAMES, METRIC_INFO, SCAFFOLD_DB
from services.analysis_service import AnalysisService
from services.result_service import ResultService
from services.visualization_service import VisualizationService
from services.biomaterial_service import BiomaterialService
from services.ai_service import AIService

# ✅ FIXED IMPORTS
from storage import database
from modules.preprocessing import extract_frames_from_video
from modules.quantification import compute_closure_wavefront, classify_healing_phase, compute_temporal_healing_entropy, compute_phenotypic_fingerprint, cosine_similarity_fingerprints

import logging
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)
pages_bp = Blueprint('pages', __name__)
analysis_service = AnalysisService()

# 🛡️ AUTHENTICATION DECORATOR
from functools import wraps
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_app.config.get("FLASK_ENV") == "development":
            return f(*args, **kwargs)
        
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != current_app.config.get("API_KEY"):
            return jsonify({"error": "Unauthorized — Valid API Key required"}), 401
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


@pages_bp.route("/")
def index():
    results = ResultService.get_all_results()
    for r in results:
        r["condition_name"] = \
        CONDITION_NAMES.get(r.get("condition"), (r.get("condition", "").replace("_", " ").title(), ""))[0]

    total_exp = len(results)
    total_cond = len(set(r.get("condition", "Unknown") for r in results))
    total_frames = sum(int((r.get("data") or {}).get("num_timepoints", 0)) for r in results)
    total_time = sum((r.get("data") or {}).get("processing_time_sec", 0) for r in results) / 60.0

    metrics_df = database.get_all_metrics_for_plots()
    stiffness_df = database.get_stiffness_healing_data()

    corr_json = VisualizationService.create_correlation_heatmap_json(metrics_df)
    box_json = VisualizationService.create_stats_box_plots_json(metrics_df)
    stiffness_json = VisualizationService.create_stiffness_scatter_json(stiffness_df)

    all_conditions = sorted(
        set(r.get("condition", "Unknown") for r in results if r.get("condition") != "Uploaded Data"))

    return render_template(
        "index.html",
        results=results, total_exp=total_exp, total_cond=total_cond,
        total_frames=total_frames, total_time=total_time,
        cond_stats=database.get_stats_by_condition(),
        comparisons={},
        all_conditions=all_conditions, condition_names=CONDITION_NAMES,
        metric_info=METRIC_INFO, correlation_json=corr_json,
        box_plots_json=box_json, stiffness_json=stiffness_json,
        scaffold_db=SCAFFOLD_DB,
        stats={"total_experiments": total_exp, "unique_conditions": total_cond, "total_frames": total_frames,
               "total_processing_time_sec": total_time * 60.0}
    )


@api_bp.route("/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format"}), 400
        
    analysis_id = str(uuid.uuid4())
    input_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], analysis_id)
    os.makedirs(input_dir, exist_ok=True)
    save_path = os.path.join(input_dir, secure_filename(file.filename))
    file.save(save_path)

    if save_path.endswith(".zip"):
        with zipfile.ZipFile(save_path, "r") as z:
            z.extractall(input_dir)
        os.remove(save_path)
    elif save_path.lower().endswith((".mp4", ".avi", ".mov")):
        extract_frames_from_video(save_path, input_dir, int(request.form.get("frameInterval", 5)))
        os.remove(save_path)
    elif save_path.lower().endswith((".tif", ".tiff")):
        img = Image.open(save_path)
        for i, page in enumerate(ImageSequence.Iterator(img)):
            page.convert("L").save(os.path.join(input_dir, f"frame_{i:04d}.png"))
        os.remove(save_path)

    items = os.listdir(input_dir)
    if len(items) == 1 and os.path.isdir(os.path.join(input_dir, items[0])):
        nested = os.path.join(input_dir, items[0])
        for item in os.listdir(nested): shutil.move(os.path.join(nested, item), input_dir)
        os.rmdir(nested)

    return jsonify({"status": "uploaded", "analysis_id": analysis_id})


@api_bp.route("/analyze", methods=["POST"])
def api_analyze():
    data = request.json or {}
    analysis_id = data.get("analysis_id")
    if not analysis_id: return jsonify({"error": "No id"}), 400

    input_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], analysis_id)
    output_dir = os.path.join(current_app.config["RESULTS_FOLDER"], "uploads", analysis_id)

    analysis_service.analyze_async(input_dir, output_dir, data)
    return jsonify({"status": "started", "analysis_id": analysis_id})


@api_bp.route("/status")
def api_status():
    return jsonify(analysis_service.get_status(request.args.get("analysis_id")))


@api_bp.route("/clean_all", methods=["POST"])
@require_api_key
def api_clean_all():
    for f in [current_app.config["UPLOAD_FOLDER"], current_app.config["RESULTS_FOLDER"]]:
        shutil.rmtree(f, ignore_errors=True)
        os.makedirs(f, exist_ok=True)
    ResultService.invalidate_cache()
    for db_name in ["woundtrack.db", "database.db", "woundtrack_results.db"]:
        if os.path.exists(db_name):
            try:
                os.remove(db_name)
            except OSError:
                pass
    database.create_table()
    return jsonify({"status": "success"})


@api_bp.route("/delete_experiment", methods=["POST"])
@require_api_key
def api_delete_experiment():
    body = request.json or {}
    result_id = body.get("result_id")
    if not result_id:
        return jsonify({"error": "No result_id provided"}), 400
    base = os.path.abspath(current_app.config["RESULTS_FOLDER"])
    target_dir = os.path.abspath(os.path.join(base, result_id))
    # Security check: must be inside RESULTS_FOLDER
    if not target_dir.startswith(base) or not os.path.exists(target_dir):
        return jsonify({"error": "Experiment not found"}), 404
    try:
        shutil.rmtree(target_dir, ignore_errors=True)
        # 🧪 CRITICAL FIX: Also remove from SQLite to prevent orphaned records in dashboard
        database.delete_experiment(result_id)
        ResultService.invalidate_cache()
        # Also clean up the upload directory if it exists
        upload_base = os.path.abspath(current_app.config["UPLOAD_FOLDER"])
        # For uploaded experiments, result_id looks like "uploads/<uuid>"
        parts = result_id.replace("/", os.sep).split(os.sep)
        if parts[0] == "uploads" and len(parts) >= 2:
            upload_dir = os.path.join(upload_base, parts[1])
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir, ignore_errors=True)
        return jsonify({"status": "success"})
    except Exception as exc:
        logger.error("Delete failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/system_memory")
def api_system_memory():
    import resource, sys
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin": usage *= 1024
    used_mb = usage / (1024 * 1024)
    return jsonify(
        {"used_mb": round(used_mb, 1), "total_mb": 8192, "percent": round(used_mb / 8192 * 100, 1), "status": "ok"})


@api_bp.route("/scaffold_design", methods=["POST"])
def api_scaffold_design():
    body = request.json or {}
    return jsonify(BiomaterialService.recommend_scaffold(
        body.get("material", "Collagen I"), float(body.get("stiffness_kpa", 1.0)),
        float(body.get("crosslink_density", 0.5)), body.get("cell_type", "epithelial")
    ))

from services.insilico_service import InSilicoService

@api_bp.route("/insilico/durotaxis_traction", methods=["POST"])
def api_insilico_durotaxis():
    body = request.json or {}
    return jsonify(InSilicoService.compute_durotaxis_surface(
        body.get("material", "Collagen I"),
        body.get("stiffness_kpa", 1.0),
        body.get("crosslink_density", 0.5),
        body.get("cell_type", "epithelial")
    ))

@api_bp.route("/insilico/immune_trajectory", methods=["POST"])
def api_insilico_immune_timeline():
    body = request.json or {}
    return jsonify(InSilicoService.compute_immune_trajectory(
        body.get("material", "Collagen I"),
        body.get("stiffness_kpa", 1.0),
        body.get("crosslink_density", 0.5),
        body.get("cell_type", "epithelial")
    ))


@api_bp.route("/ai_interpret", methods=["POST"])
def api_ai_interpret():
    body = request.json or {}
    exp_id = body.get("exp_id", "")
    result = next((r for r in ResultService.get_all_results() if r.get("id") == exp_id), None)
    if not result: return abort(404)
    out = AIService.generate_interpretation(result.get("experiment_name"), result.get("condition_name"),
                                            result.get("data"), body.get("mode", "interpret"))
    out["exp_id"] = exp_id
    return jsonify(out)


@api_bp.route("/ai_composite_score", methods=["POST"])
def api_ai_composite_score():
    body = request.json or {}
    result = next((r for r in ResultService.get_all_results() if r.get("id") == body.get("exp_id", "")), None)
    if not result: return abort(404)
    return jsonify(AIService.calculate_composite_score(result.get("data", {})))


# Advanced Science Routes
@api_bp.route("/novel/wavefront/<path:exp_id>")
def api_wavefront(exp_id):
    result = next((r for r in ResultService.get_all_results() if r["id"] == exp_id), None)
    if not result: return abort(404)
    d = result.get("data", {})
    widths = d.get("wound_width_px_series") or [float(a ** 0.5) for a in d.get("areas_px_series", [])]
    return jsonify(
        compute_closure_wavefront(widths, d.get("time_points", []), float(d.get("pixel_scale_um_per_px", 1.0))))


@api_bp.route("/novel/phase/<path:exp_id>")
def api_phase_classify(exp_id):
    result = next((r for r in ResultService.get_all_results() if r["id"] == exp_id), None)
    if not result: return abort(404)
    metrics = dict(result.get("data", {}))
    metrics.update({k: result.get(k) for k in ("condition_name", "substrate_material", "treatment") if result.get(k)})
    return jsonify(classify_healing_phase(metrics))


@api_bp.route("/novel/entropy/<path:exp_id>")
def api_entropy(exp_id):
    result = next((r for r in ResultService.get_all_results() if r["id"] == exp_id), None)
    if not result: return abort(404)
    d = result.get("data", {})
    return jsonify(compute_temporal_healing_entropy(d.get("closure_percentages", []), d.get("time_points", [])))


@api_bp.route("/novel/fingerprint/<path:exp_id>")
def api_fingerprint(exp_id):
    result = next((r for r in ResultService.get_all_results() if r["id"] == exp_id), None)
    if not result: return abort(404)
    d = result.get("data", {})
    metrics = dict(d)
    if d.get("closure_percentages") and d.get("time_points"):
        ent = compute_temporal_healing_entropy(d["closure_percentages"], d["time_points"])
        metrics["regularity_score"] = ent.get("regularity_score")
    return jsonify(compute_phenotypic_fingerprint(metrics))


@api_bp.route("/compare_stats", methods=["POST"])
def api_compare_stats():
    import scipy.stats as stats
    import numpy as np
    
    body = request.json or {}
    exp_ids = body.get("exp_ids", [])
    metric = body.get("metric", "")
    
    if len(exp_ids) < 2 or not metric:
        return jsonify({"valid": False, "error": "Require at least 2 datasets and a metric"})
    
    all_results = ResultService.get_all_results()
    distributions = []
    used_real_data = []  # Track whether each distribution is from real array data
    
    for eid in exp_ids:
        res = next((r for r in all_results if r.get("id") == eid), None)
        if not res or "data" not in res: continue
        d = res["data"]
        arr = []
        is_real = False
        
        # Strategy 1: Pull real distribution arrays from tracking data
        if metric in ["mean_velocity_um_min", "migration_efficiency_mean", "mean_directionality"]:
            arr = d.get("all_velocities_um_min", []) or d.get("all_track_displacements_um", [])
            is_real = bool(arr)
        
        # Strategy 2: Use time-series derivatives as distribution
        if not arr and metric in ["final_closure_pct", "time_to_50_closure_hr", 
                                   "healing_rate_um2_per_hr", "r_squared", "final_area_um2"]:
            clos = d.get("closure_percentages", [])
            if len(clos) > 2:
                arr = np.diff(clos).tolist()
                is_real = True
        
        # Strategy 3: Use raw area series as distribution for area-based metrics
        if not arr:
            areas = d.get("areas_px", []) or d.get("areas_raw_px", [])
            if len(areas) > 3:
                arr = np.diff(areas).tolist()
                is_real = True
        
        # NO FALLBACK: Do not fabricate pseudo-distributions from scalar values
        if arr:
            distributions.append(arr)
            used_real_data.append(is_real)
        else:
            # Report scalar value directly — cannot do distribution-based test
            distributions.append([])
            used_real_data.append(False)

    # Filter out empty distributions
    valid_dists = [d for d in distributions if len(d) >= 3]
    
    if len(valid_dists) < 2:
        # Rigorous fallback: report scalars only, do not attempt p-values on diffs
        scalars = []
        for eid in exp_ids:
            res = next((r for r in all_results if r.get("id") == eid), None)
            if res and "data" in res:
                val = res["data"].get(metric)
                if val is not None:
                    scalars.append(float(val))
        
        if len(scalars) >= 2:
            return jsonify({
                "valid": True,
                "p_value": None,
                "significance": "n/a",
                "test_type": "Descriptive Comparison (N < 3 per group)",
                "note": "A rigorous statistical test requires at least 3 samples per group. "
                        "Currently reporting raw values only. Please upload replicates for p-value calculation."
            })
        return jsonify({"valid": False, "error": "Insufficient data for biological statistical comparison."})
            
    try:
        if len(valid_dists) == 2:
            # Non-parametric: Mann-Whitney U (does not assume normality)
            stat, p_val = stats.mannwhitneyu(
                valid_dists[0], valid_dists[1], alternative='two-sided'
            )
            test_type = "Mann-Whitney U"
        else:
            # Non-parametric: Kruskal-Wallis (does not assume normality)
            stat, p_val = stats.kruskal(*valid_dists)
            test_type = "Kruskal-Wallis"
            
        if np.isnan(p_val): p_val = 1.0
        
        signif = "ns"
        if p_val < 0.001: signif = "***"
        elif p_val < 0.01: signif = "**"
        elif p_val < 0.05: signif = "*"
        
        return jsonify({
            "valid": True,
            "p_value": float(p_val),
            "significance": signif,
            "test_type": test_type,
            "n_per_group": [len(d) for d in valid_dists]
        })
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


# ── Results JSON (modal data endpoint) ──────────────────────────────────────
@pages_bp.route("/results_json/<path:exp_id>")
def results_json(exp_id):
    """Return all data needed by the experiment detail modal."""
    import json as _json

    target = next((r for r in ResultService.get_all_results() if r["id"] == exp_id), None)
    if not target:
        return jsonify({"error": "not found"}), 404

    data = dict(target.get("data") or {})
    data["experiment_name"] = target.get("experiment_name", "Experiment")
    data["condition"] = target.get("condition", "")
    data["gallery_thumbs"] = target.get("gallery_thumbs", [])

    # ── Normalise key names (pipeline keys → frontend keys) ─────────────────
    pixel_scale = float(data.get("pixel_scale_um_per_px", 1.0))
    _KEY_MAP = {
        "initial_area":             "initial_area_um2",
        "final_area":               "final_area_um2",
        "final_closure_percentage": "final_closure_pct",
        "healing_rate":             "healing_rate_um2_per_hr",
    }
    for old_key, new_key in _KEY_MAP.items():
        if old_key in data and new_key not in data:
            val = data[old_key]
            if val is not None:
                # Convert px² → µm² for area keys
                if "area" in old_key and pixel_scale != 1.0:
                    val = float(val) * pixel_scale ** 2
                data[new_key] = val

    # ── Merge tracking data from tracking_summary.json if available ─────────
    base = os.path.abspath(current_app.config["RESULTS_FOLDER"])
    exp_dir = os.path.join(base, exp_id.replace("/", os.sep))
    tracking_json = os.path.join(exp_dir, "tracking", "tracking_summary.json")
    if os.path.exists(tracking_json):
        try:
            with open(tracking_json, "r", encoding="utf-8") as fh:
                tracking_data = _json.load(fh)
            # Merge tracking keys into data (don't overwrite existing keys)
            _TRACKING_KEYS = [
                "num_cells_tracked", "mean_velocity_um_min", "migration_efficiency_mean",
                "mean_directionality", "mean_displacement_um", "mean_path_length_um",
                "msd_alpha", "msd_D_um2_hr", "migration_mode_msd",
                "directed_migration_score", "mean_cos_theta", "persistence_time_hr",
                "division_rate_per_hr", "doubling_time_hr",
                "std_velocity_um_min", "std_path_length_um",
            ]
            for k in _TRACKING_KEYS:
                if k in tracking_data and k not in data:
                    data[k] = tracking_data[k]
        except Exception as exc:
            logger.warning("Could not read tracking_summary.json: %s", exc)

    # ── Interactive Plotly JSON ──────────────────────────────────────────────
    iplot_path = target.get("interactive_plot_path")
    if iplot_path and os.path.exists(iplot_path):
        try:
            with open(iplot_path, "r", encoding="utf-8") as fh:
                data["plot_json"] = fh.read()
        except Exception:
            data["plot_json"] = None
    else:
        data["plot_json"] = None

    # ── Static plot as base64 ───────────────────────────────────────────────
    plot_path = target.get("plot_path")
    if plot_path and os.path.exists(plot_path):
        try:
            with open(plot_path, "rb") as fh:
                data["plot_b64"] = base64.b64encode(fh.read()).decode("ascii")
        except Exception:
            data["plot_b64"] = None
    else:
        data["plot_b64"] = None

    # ── Rose plot as base64 (try result_service path, then tracking dir) ────
    rose_path = target.get("rose_plot_path")
    if not rose_path or not os.path.exists(rose_path or ""):
        rose_path = os.path.join(exp_dir, "tracking", "rose_plot.png")
    if rose_path and os.path.exists(rose_path):
        try:
            with open(rose_path, "rb") as fh:
                data["rose_plot_b64"] = base64.b64encode(fh.read()).decode("ascii")
        except Exception:
            data["rose_plot_b64"] = None
    else:
        data["rose_plot_b64"] = None

    # ── MSD plot as base64 (NEW) ────────────────────────────────────────────
    msd_path = os.path.join(exp_dir, "tracking", "msd_plot.png")
    if os.path.exists(msd_path):
        try:
            with open(msd_path, "rb") as fh:
                data["msd_plot_b64"] = base64.b64encode(fh.read()).decode("ascii")
        except Exception:
            data["msd_plot_b64"] = None

    # ── CSV file as base64 ──────────────────────────────────────────────────
    csv_path = target.get("csv_path")
    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as fh:
                data["csv_b64"] = base64.b64encode(fh.read()).decode("ascii")
        except Exception:
            data["csv_b64"] = None
    else:
        data["csv_b64"] = None

    return jsonify(data)


# ── Base Data Rendering Routes ──────────────────────────────────────────────
@api_bp.route("/stiffness_scatter")
def api_stiffness_scatter():
    return jsonify(
        {"plot_json": VisualizationService.create_stiffness_scatter_json(database.get_stiffness_healing_data())})


@api_bp.route("/gallery_frames/<path:exp_id>")
def api_gallery_frames(exp_id):
    target = next((r for r in ResultService.get_all_results() if r["id"] == exp_id), None)
    if not target: return jsonify({"frames": []})
    frames = [u for u in (ResultService._path_to_url(p) for p in target.get("gallery", [])) if u]
    return jsonify({"frames": frames, "count": len(frames)})


@pages_bp.route("/results_data/<path:filename>")
def send_result_file(filename):
    base_dir = os.path.abspath(current_app.config["RESULTS_FOLDER"])
    abs_path = os.path.abspath(os.path.join(base_dir, filename))
    if not abs_path.startswith(base_dir) or not os.path.exists(abs_path): abort(404)
    return send_from_directory(base_dir, os.path.relpath(abs_path, base_dir))


@pages_bp.route("/download-fiji/<path:exp_id>")
def download_fiji(exp_id):
    target = next((r for r in ResultService.get_all_results() if r["id"] == exp_id), None)
    if not target: abort(404)
    exp_dir = os.path.join(os.path.abspath(current_app.config["RESULTS_FOLDER"]), exp_id)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(exp_dir):
            for file in files: zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), exp_dir))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", download_name=f"{target.get('experiment_name')}_fiji_export.zip",
                     as_attachment=True)

@api_bp.route("/comparison_data")
def api_comparison_data():
    """Returns all experiments in a format suitable for the comparison UI."""
    from storage import database
    try:
        rows = database.get_all_experiments_for_comparison()
        return jsonify(rows)
    except Exception as exc:
        logger.error("comparison_data error: %s", exc)
        return jsonify([])