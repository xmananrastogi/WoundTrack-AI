# services/analysis_service.py
import os
import json
import logging
import threading
import traceback
from werkzeug.utils import secure_filename
from flask import current_app

from core.pipeline import WoundHealingPipeline
from services.visualization_service import VisualizationService
from services.result_service import ResultService
from config.constants import CONDITION_NAMES

from storage import database
from modules.preprocessing import get_image_files

logger = logging.getLogger(__name__)

analysis_states = {}
states_lock = threading.Lock()


class AnalysisService:
    """Orchestrates the entire execution: Threads -> Pipeline -> Visuals -> DB."""

    def __init__(self):
        self.pipeline = WoundHealingPipeline()
        self.vis_service = VisualizationService()

    def get_status(self, analysis_id):
        with states_lock:
            if analysis_id and analysis_id in analysis_states:
                return analysis_states[analysis_id]
            running = [s for s in analysis_states.values() if s["running"]]
            if running:
                return running[-1]
        return {"running": False, "progress": 0, "status": "Idle", "current": ""}

    def analyze_async(self, input_dir, output_dir, config):
        analysis_id = config["analysis_id"]

        # 🔴 CRITICAL FIX 2: Capture the actual Flask app object so the thread can use it
        app_context = current_app._get_current_object()

        thread = threading.Thread(
            target=self._run_task,
            args=(input_dir, output_dir, config, app_context),
            daemon=True
        )
        thread.start()
        return analysis_id

    def _run_task(self, input_dir, output_dir, config, app):
        # 🔴 CRITICAL FIX 3: Push the app context so database.py can access current_app.config safely
        with app.app_context():
            analysis_id = config["analysis_id"]
            with states_lock:
                analysis_states[analysis_id] = {"running": True, "progress": 0, "status": "Initialising…",
                                                "current": analysis_id}
            state = analysis_states[analysis_id]

            try:
                os.makedirs(output_dir, exist_ok=True)
                safe_id = secure_filename(config.get("sample_id") or analysis_id)
                config["output_dir"] = output_dir

                state["status"] = "🔬 Loading image stack…"
                image_files = get_image_files(input_dir)
                if not image_files:
                    raise ValueError(f"No valid image files in {input_dir}")
                state["progress"] = 10

                # Auto disk size (Safely forcing to float)
                raw_ds = config.get("disk_size", 0)
                try:
                    ds_val = float(raw_ds)
                except Exception:
                    ds_val = 0.0

                if ds_val == 0.0:
                    state["status"] = "⚙️ Auto-calibrating parameters…"
                    config["disk_size"] = self._auto_select_disk_size(image_files)
                state["progress"] = 20

                state["status"] = f"📡 Scanning {len(image_files)} frames via Memmap Engine…"
                result = self.pipeline.run(image_files, config)
                state["progress"] = 80

                state["status"] = "🎨 Generating visualisations…"

                # Pass memmap directly — supports [i] indexing without materialising all frames
                # VisualizationService uses only ~20 sampled frames for gallery
                resolved_masks = result.get("masks_memmap")  # np.memmap — NOT a full list

                self.vis_service.generate_all(
                    image_files, resolved_masks,
                    result["metrics"], result["tracking"],
                    output_dir, safe_id, config
                )
                state["progress"] = 90

                state["status"] = "💾 Saving to database…"
                summary_path = os.path.join(output_dir, "csv", f"{safe_id}_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path) as fh:
                        summary_data = json.load(fh)

                    db_folder = app.config["RESULTS_FOLDER"]
                    result_id = os.path.relpath(output_dir, os.path.abspath(db_folder)).replace(os.sep, "/")
                    summary_data["experiment_name"] = safe_id
                    summary_data["condition_name"] = "Uploaded Data" if "uploads" in result_id else \
                    CONDITION_NAMES.get(result_id.split("/")[0], (result_id.split("/")[0],))[0]
                    summary_data["substrate_material"] = config.get("substrate_material")
                    summary_data["substrate_stiffness_kpa"] = config.get("substrate_stiffness_kpa")
                    summary_data["treatment"] = config.get("treatment")

                    database.upsert_experiment(summary_data, result_id)

                state["progress"] = 100
                state["status"] = f"✅ Complete: {safe_id}"
                ResultService.invalidate_cache()

            except Exception as exc:
                state["status"] = f"❌ Error: {exc}"
                logger.error("[%s] Task failed:\n%s", analysis_id, traceback.format_exc())
            finally:
                state["running"] = False
                if 'result' in locals() and result is not None:
                    mm = result.get("masks_memmap")
                    if mm is not None:
                        try:
                            del mm
                            del result["masks_memmap"]
                        except Exception:
                            pass
                    tp = result.get("temp_path")
                    if tp and os.path.exists(str(tp)):
                        try:
                            os.remove(tp)
                        except OSError:
                            pass
                import gc; gc.collect()

    def _auto_select_disk_size(self, image_files):
        from modules.segmentation import segment_wound_from_array
        from modules.preprocessing import normalize_intensity
        import cv2

        if len(image_files) < 2: return 11.0
        results = []
        try:
            img0 = normalize_intensity(cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE))
            img1 = normalize_intensity(cv2.imread(image_files[-1], cv2.IMREAD_GRAYSCALE))
            # Force odd numbers during auto-select to prevent crashes
            for size in [7.0, 11.0, 15.0, 21.0, 25.0]:
                _, a0 = segment_wound_from_array(img0, disk_size=int(size))
                _, a1 = segment_wound_from_array(img1, disk_size=int(size))
                if a0 > 0:
                    results.append({"size": size, "closure": (a0 - a1) / a0})
        except Exception:
            pass
        if not results: return 11.0
        positive = [r for r in results if r["closure"] > 0]
        best = max(positive or results, key=lambda r: r["closure"])
        return float(best["size"])