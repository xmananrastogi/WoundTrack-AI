# services/visualization_service.py
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
import logging

# ✅ FIXED IMPORT
from modules.segmentation import detect_wound_contours
try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
except ImportError:
    go = px = pio = None

logger = logging.getLogger(__name__)


class VisualizationService:
    """Generates plots, timelapses, overlays, and CSV records independently of pipeline core."""

    def generate_all(self, image_files, masks, metrics, tracking, output_dir, exp_name, config):
        for d in ["plots", "gallery", "video", "csv"]:
            os.makedirs(os.path.join(output_dir, d), exist_ok=True)

        pixel_scale = config.get("pixel_scale", 1.0)
        time_interval = config.get("time_interval", 0.25)

        if masks is not None:
            overlay_paths = self.create_overlay_gallery(image_files, masks, os.path.join(output_dir, "gallery"),
                                                        exp_name)
            self.create_animation(overlay_paths, os.path.join(output_dir, "video"), exp_name, time_interval)
            self.create_visualization(metrics, os.path.join(output_dir, "plots"), exp_name, pixel_scale)
            csv_path, _ = self.save_results(metrics, tracking, os.path.join(output_dir, "csv"), exp_name)
            self.create_interactive_plot(csv_path, os.path.join(output_dir, "plots", f"{exp_name}_interactive.json"),
                                         pixel_scale)
        else:
            # Fallback for cell-tracking only
            self._create_basic_gallery(image_files, os.path.join(output_dir, "gallery"), exp_name)
            self.save_results(metrics, tracking, os.path.join(output_dir, "csv"), exp_name)

            import shutil
            rose_path = os.path.join(output_dir, "tracking", "rose_plot.png")
            if os.path.exists(rose_path):
                shutil.copy(rose_path, os.path.join(output_dir, "plots", f"{exp_name}_analysis.png"))

    def create_overlay_gallery(self, image_files, masks, output_dir, experiment_name):
        step = max(1, len(masks) // 20)
        paths = []
        for idx in range(0, len(masks), step):
            if idx >= len(image_files): break
            try:
                img = cv2.imread(image_files[idx])
                if img is None: continue
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                mask = masks[idx]
                contours, _ = detect_wound_contours(mask)
                if contours:
                    cv2.drawContours(img, contours, -1, (0, 80, 255), 2)
                out = os.path.join(output_dir, f"{experiment_name}_frame_{idx:04d}.jpg")
                cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                paths.append(out)
            except Exception as exc:
                logger.warning("Overlay %d failed: %s", idx, exc)
        return paths

    def _create_basic_gallery(self, image_files, output_dir, exp_name):
        step = max(1, len(image_files) // 20)
        for idx in range(0, len(image_files), step):
            img = cv2.imread(image_files[idx])
            if img is not None:
                out = os.path.join(output_dir, f"{exp_name}_frame_{idx:04d}.jpg")
                cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def create_animation(self, overlay_paths, output_dir, experiment_name, time_interval):
        if not overlay_paths: return None
        video_path = os.path.join(output_dir, f"{experiment_name}_healing_timelapse.mp4")
        fps = max(5, min(30, len(overlay_paths) // 7))
        try:
            with imageio.get_writer(video_path, fps=fps, codec="libx264") as w:
                for p in overlay_paths:
                    w.append_data(imageio.imread(p))
            return video_path
        except Exception as exc:
            logger.error("MP4 failed: %s", exc)
            return None

    def save_results(self, metrics, tracking, output_dir, experiment_name):
        n = metrics.get("num_timepoints", 0)
        edge_w = metrics.get("wound_width_px_series", [])
        sig_fit = metrics.get("sigmoid_fitted_values", [])
        qs = metrics.get("quality_scores", [])

        df_dict = {
            "time_hours": metrics.get("timepoints", []),
            "wound_area_px": metrics.get("areas_px", []),
            "wound_area_um2": metrics.get("areas_um2", []),
            "closure_pct": metrics.get("closure_percentages", []),
        }
        if edge_w and len(edge_w) == n: df_dict["wound_width_px"] = edge_w
        if sig_fit and len(sig_fit) == n: df_dict["sigmoid_fitted_closure_pct"] = sig_fit
        if qs and len(qs) == n: df_dict["segmentation_quality_score"] = qs

        df = pd.DataFrame(df_dict)
        csv_path = os.path.join(output_dir, f"{experiment_name}_timeseries.csv")
        df.to_csv(csv_path, index=False)

        _SKIP = {"timepoints", "areas_px", "areas_um2", "closure_percentages", "sigmoid_fitted_values",
                 "wound_width_px_series", "quality_scores", "low_quality_frames", "edge_velocity_details",
                 "tortuosity_details",
                 # Large tracking sub-dicts (saved separately in tracking_summary.json)
                 "msd_details", "division_details", "directed_migration", "autocorrelation",
                 "cos_theta_distribution", "division_events_per_frame", "particle_count_per_frame"}
        summary = {k: v for k, v in metrics.items() if k not in _SKIP and not k.startswith("_")}
        summary["experiment"] = experiment_name
        # Merge tracking scalar metrics (skip nested dicts/lists)
        if tracking:
            for k, v in tracking.items():
                if k in _SKIP or k.startswith("_"):
                    continue
                if isinstance(v, (dict, list)):
                    continue  # Skip nested structures
                summary[k] = v

        json_path = os.path.join(output_dir, f"{experiment_name}_summary.json")
        with open(json_path, "w") as fh:
            json.dump(summary, fh, indent=4, default=str)
        return csv_path, json_path

    def create_visualization(self, results, output_dir, experiment_name, pixel_scale):
        t = results.get("timepoints", [])
        closure = results.get("closure_percentages", [])
        if not t or not closure: return None

        sig_fit = results.get("sigmoid_fitted_values", [])
        sig_mod = results.get("sigmoid_model", "insufficient_data")

        areas, area_label = (results.get("areas_um2", []), "Wound Area (µm²)") if pixel_scale != 1.0 else (
        results.get("areas_px", []), "Wound Area (px²)")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Wound Healing Analysis: {experiment_name}", fontsize=15, fontweight="bold", y=1.01)
        sns.set_theme(style="whitegrid")

        # Panel 1: Area vs time
        ax = axes[0, 0]
        ax.plot(t, areas, "o-", lw=2.5, ms=5, label="Cell-Free Area", color="#0066cc")
        if len(t) > 1:
            poly = np.polyfit(t, areas, 1)
            ax.plot(t, np.poly1d(poly)(t), "--", lw=1.5, label=f"Linear fit  R²={results.get('r_squared', 0):.3f}",
                    color="#cc3300")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(area_label)
        ax.set_title("Cell-Free Area Depletion", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)

        # Panel 2: Closure % + sigmoid fit
        ax = axes[0, 1]
        ax.plot(t, closure, "s-", lw=2.5, ms=5, color="#009966", label="Measured")
        ax.fill_between(t, 0, closure, alpha=0.12, color="#009966")

        if sig_fit and len(sig_fit) == len(t) and sig_mod != "insufficient_data":
            ax.plot(t, sig_fit, "--", lw=2, color="#ff6600", alpha=0.9,
                    label=f"Sigmoid fit ({sig_mod}, R²={results.get('sigmoid_r_squared', 0):.3f})")
            if results.get("sigmoid_lag_phase_hr"):
                ax.axvline(results["sigmoid_lag_phase_hr"], ls=":", color="#884400", lw=1.2,
                           label=f"Lag phase {results['sigmoid_lag_phase_hr']:.1f}h")

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Area Fraction Encroached (%)")
        ax.set_title("Wound Closure Kinetics (% Area Fraction)", fontweight="bold")
        ax.set_ylim(0, 108)
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)

        # Panel 3: Dual edge velocity
        ax = axes[1, 0]
        ev_l = results.get("edge_velocity_details", {}).get("edge_velocities_left", [])
        ev_r = results.get("edge_velocity_details", {}).get("edge_velocities_right", [])
        ev_t = results.get("edge_velocity_details", {}).get("edge_time_points", [])

        if ev_l and ev_r and ev_t and len(ev_l) == len(ev_r):
            t_mid = [(ev_t[i] + ev_t[i + 1]) / 2 for i in range(len(ev_t) - 1)] if len(ev_t) > 1 else ev_t
            if len(t_mid) == len(ev_l):
                ax.plot(t_mid, ev_l, "o-", lw=2, ms=4, color="#3388ff", label="Left margin")
                ax.plot(t_mid, ev_r, "s-", lw=2, ms=4, color="#ff6633", label="Right margin")
                ax.axhline(0, color="grey", lw=1, ls="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "Front tracking requires masks", ha="center", va="center", transform=ax.transAxes,
                    color="gray")

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Advancement Velocity (µm/hr)")
        ax.set_title("Migration Front Velocities", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)

        # Panel 4: Wound quality score
        ax = axes[1, 1]
        qs = results.get("quality_scores", [])
        if qs and len(qs) == len(t):
            ax.bar(t, qs, width=float(t[1] - t[0]) * 0.7 if len(t) > 1 else 0.1, color="#4488ff", alpha=0.35,
                   label="Confidence score")
            ax.axhline(0.3, ls="--", color="red", lw=1, alpha=0.7, label="Confidence threshold")
            ax.set_ylabel("Mask Confidence [0–1]")
            ax.set_ylim(0, 1.25)

        ax.set_xlabel("Time (hours)")
        ax.set_title("Mask Confidence Score", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plot_path = os.path.join(output_dir, f"{experiment_name}_analysis.png")
        plt.savefig(plot_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return plot_path

    def create_interactive_plot(self, csv_path, output_json_path, pixel_scale):
        if not go: return
        try:
            df = pd.read_csv(csv_path)
            if df.empty: return
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            area_col, area_lbl = ("wound_area_um2", "µm²") if ("wound_area_um2" in df.columns) else (
            "wound_area_px", "px²")

            fig.add_trace(
                go.Scatter(x=df["time_hours"], y=df[area_col], name=f"Cell-Free Area ({area_lbl})", mode="lines+markers"),
                secondary_y=False)
            fig.add_trace(go.Scatter(x=df["time_hours"], y=df["closure_pct"], name="Area Fraction Encroached (%)", mode="lines+markers",
                                     line=dict(dash="dot")), secondary_y=True)

            if "sigmoid_fitted_closure_pct" in df.columns:
                fig.add_trace(
                    go.Scatter(x=df["time_hours"], y=df["sigmoid_fitted_closure_pct"], name="Sigmoid Curve Fit", mode="lines",
                               line=dict(dash="dash", color="orange")), secondary_y=True)

            fig.update_layout(title_text="Gap Closure Kinetics", template="plotly_dark", hovermode="x unified",
                              height=600)
            fig.update_yaxes(title_text=f"Wound Area ({area_lbl})", secondary_y=False)
            fig.update_yaxes(title_text="Closure / Sigmoid (%)", secondary_y=True,
                             range=[0, max(105, df["closure_pct"].max() * 1.1)])
            fig.write_json(output_json_path)
        except Exception as exc:
            logger.error("Interactive plot failed: %s", exc)

    @staticmethod
    def create_correlation_heatmap_json(df: pd.DataFrame) -> str:
        if not go or df.empty or len(df.columns) < 2: return "{}"
        try:
            corr = df.corr(numeric_only=True)
            if corr.empty: return "{}"
            fig = go.Figure(
                data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Teal", zmin=-1, zmax=1,
                                text=corr.values, texttemplate="%{text:.2f}", hoverongaps=False))
            fig.update_layout(title="Metric Correlation Heatmap", template="plotly_dark", height=600,
                              xaxis_showgrid=False, yaxis_showgrid=False, yaxis_autorange="reversed")
            return pio.to_json(fig)
        except Exception:
            return "{}"

    @staticmethod
    def create_stats_box_plots_json(df: pd.DataFrame) -> str:
        if not px or df.empty: return "{}"
        try:
            wanted = ["Closure (%)", "Healing Speed (µm²/hr)", "Consistency (R²)", "Cell Velocity (µm/min)",
                      "Efficiency", "Directionality"]
            available = [m for m in wanted if m in df.columns]
            if not available: return "{}"
            melted = df.melt(id_vars=["Condition"], value_vars=available, var_name="Metric", value_name="Value")
            fig = px.box(melted, x="Condition", y="Value", color="Condition", facet_row="Metric",
                         title="Metric Distributions by Condition")
            fig.update_layout(template="plotly_dark", height=min(max(1000, len(available) * 300), 2000),
                              showlegend=False)
            fig.update_yaxes(matches=None, showticklabels=True)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            return pio.to_json(fig)
        except Exception:
            return "{}"

    @staticmethod
    def create_stiffness_scatter_json(df: pd.DataFrame) -> str:
        if not px or df.empty or "substrate_stiffness_kpa" not in df.columns: return "{}"
        try:
            fig = px.scatter(df, x="substrate_stiffness_kpa", y="healing_rate_um2_per_hr", color="substrate_material",
                             size="final_closure_pct", hover_name="experiment_name", log_x=True, template="plotly_dark")
            fig.update_layout(height=500, legend_title_text="Substrate")
            return pio.to_json(fig)
        except Exception:
            return "{}"