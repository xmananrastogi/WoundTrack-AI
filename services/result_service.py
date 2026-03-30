# services/result_service.py
import os
import glob
import json
import time
import posixpath
from flask import current_app

_results_cache = {"data": None, "ts": 0.0}
_RESULTS_TTL = 5.0


class ResultService:
    """Handles fetching and structuring historical analyses from the filesystem."""

    @staticmethod
    def invalidate_cache():
        _results_cache["ts"] = 0.0

    @staticmethod
    def get_all_results():
        if (_results_cache["data"] is not None and time.time() - _results_cache["ts"] < _RESULTS_TTL):
            return _results_cache["data"]
        data = ResultService._scan_results_from_disk()
        _results_cache["data"] = data
        _results_cache["ts"] = time.time()
        return data

    @staticmethod
    def _scan_results_from_disk():
        results = []
        base = os.path.abspath(current_app.config["RESULTS_FOLDER"])
        if not os.path.exists(base): return results

        for sfile in sorted(glob.glob(os.path.join(base, "**", "*_summary.json"), recursive=True), key=os.path.getmtime,
                            reverse=True):
            if os.path.basename(sfile) == "tracking_summary.json":
                continue  # Prevents tracking duplicates breaking UI

            try:
                with open(sfile, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except Exception:
                raw = None

            rel = os.path.relpath(sfile, base).replace(os.sep, "/")
            parts = rel.split("/")

            if parts[0] == "uploads" and len(parts) >= 3:
                result_id, condition = posixpath.join("uploads", parts[1]), "Uploaded Data"
                experiment_name = (raw or {}).get("experiment", os.path.splitext(parts[-1])[0].replace("_summary", ""))
                base_result_dir = os.path.join(base, "uploads", parts[1])
            elif len(parts) >= 3:
                condition, experiment_name = parts[0], parts[1]
                base_result_dir = os.path.join(base, parts[0], experiment_name)
                result_id = posixpath.join(parts[0], experiment_name)
            else:
                condition = "Unknown"
                experiment_name = (raw or {}).get("experiment", "Unknown")
                base_result_dir = os.path.dirname(sfile)
                result_id = experiment_name

            bn = (raw or {}).get("experiment", os.path.splitext(os.path.basename(sfile))[0].replace("_summary", ""))

            def _pick(*cs):
                for c in cs:
                    if c and os.path.exists(c): return c
                return None

            gallery_dir = os.path.join(base_result_dir, "gallery")
            gallery_files = []
            if os.path.isdir(gallery_dir):
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    gallery_files.extend(sorted(glob.glob(os.path.join(gallery_dir, ext))))

            iplot = os.path.join(base_result_dir, "plots", f"{bn}_interactive.json")

            results.append({
                "id": result_id,
                "summary_path": sfile,
                "raw_summary": raw or {},
                "data": raw or {},
                "csv_path": _pick(os.path.join(base_result_dir, "csv", f"{bn}_timeseries.csv")),
                "plot_path": _pick(os.path.join(base_result_dir, "plots", f"{bn}_analysis.png")),
                "rose_plot_path": _pick(os.path.join(base_result_dir, "tracking", "rose_plot.png")),
                "plot_url": ResultService._path_to_url(
                    _pick(os.path.join(base_result_dir, "plots", f"{bn}_analysis.png"))),
                "interactive_plot_path": iplot if os.path.exists(iplot) else None,
                "interactive_plot_url": ResultService._path_to_url(iplot) if os.path.exists(iplot) else None,
                "gallery": gallery_files,
                "gallery_thumbs": [u for u in (ResultService._path_to_url(p) for p in gallery_files) if u],
                "video_path": _pick(os.path.join(base_result_dir, "video", f"{bn}_healing_timelapse.mp4")),
                "video_url": ResultService._path_to_url(
                    _pick(os.path.join(base_result_dir, "video", f"{bn}_healing_timelapse.mp4"))),
                "condition": condition,
                "experiment_name": experiment_name,
            })

        results.sort(key=lambda r: os.path.getmtime(r["summary_path"]) if os.path.exists(r["summary_path"]) else 0,
                     reverse=True)
        return results

    @staticmethod
    def _path_to_url(fs_path):
        if not fs_path: return None
        base = os.path.abspath(current_app.config["RESULTS_FOLDER"])
        try:
            rel = os.path.relpath(os.path.abspath(fs_path), base)
        except Exception:
            return None
        if rel.startswith(".."): return None
        return "/results_data/" + rel.replace(os.sep, "/").lstrip("./")