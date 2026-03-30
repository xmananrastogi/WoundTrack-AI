import os
# services/ai_service.py
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


class AIService:
    """Handles external API calls to Claude and internal composite score algorithms."""

    @staticmethod
    def generate_interpretation(exp_name: str, condition: str, metrics_clean: dict, mode: str) -> dict:
        mode_prompts = {
            "interpret": f"""You are an expert cell biologist specialising in wound healing assays and cell migration.
Analyse this scratch assay experiment and provide a rigorous scientific interpretation.
Experiment: {exp_name} | Condition: {condition}
Metrics: {json.dumps(metrics_clean, indent=2)}
Provide these sections:
1. BIOLOGICAL INTERPRETATION: What do these metrics tell us about cell behaviour?
2. KINETIC PROFILE: Comment on lag phase, peak rate, sigmoid vs linear kinetics.
3. MIGRATION vs PROLIFERATION: Based on migration_fraction, what drives closure?
4. MSD ANALYSIS: Interpret diffusion exponent α — directed, random, or superdiffusive?
5. EDGE ANALYSIS: Interpret tortuosity and asymmetry.
6. SUBSTRATE EFFECTS: Comment on mechanosensing/durotaxis if biomaterial data present.
7. ANOMALY FLAGS: List any metrics that are unusual or potentially erroneous.
8. PUBLICATION STATEMENT: One concise sentence for a results section.
Be precise, use quantitative values, cite specific numbers.""",

            "anomaly": f"""You are a bioinformatics QC specialist reviewing scratch assay data.
Experiment: {exp_name} | Condition: {condition}
Metrics: {json.dumps(metrics_clean, indent=2)}
Identify ALL anomalies and quality issues:
1. METRIC INCONSISTENCIES: Biologically impossible combinations?
2. STATISTICAL RED FLAGS: Values outside typical biological ranges?
3. DATA COMPLETENESS: Which important metrics are missing?
4. REPRODUCIBILITY: Would you trust these for publication?
Rate each issue: [CRITICAL] [WARNING] [INFO]
End with overall QC score: PASS / CONDITIONAL PASS / FAIL""",

            "predict": f"""You are a predictive biology AI analysing wound healing trajectories.
Experiment: {exp_name} | Condition: {condition}
Current metrics: {json.dumps(metrics_clean, indent=2)}
Predict:
1. FINAL CLOSURE: Estimated % at 48hr and 72hr with uncertainty ranges.
2. RATE TRAJECTORY: Accelerating, decelerating, or steady-state?
3. BIOLOGICAL ENDPOINT: Full closure or arrested migration?
4. THERAPEUTIC RESPONSE: If treatment noted, characterise response magnitude.
5. CONFIDENCE: How confident? What additional data would improve accuracy?""",
        }

        prompt = mode_prompts.get(mode, mode_prompts["interpret"])
        api_payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"interpretation": "AI interpretation unavailable — set ANTHROPIC_API_KEY environment variable.", "mode": mode, "metrics_used": 0}

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=api_payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            api_data = json.loads(resp.read().decode("utf-8"))

        interpretation = "".join(b["text"] for b in api_data.get("content", []) if b.get("type") == "text")
        return {"interpretation": interpretation, "mode": mode, "metrics_used": len(metrics_clean)}

    @staticmethod
    def calculate_composite_score(d: dict) -> dict:
        sig_r2 = float(d.get("sigmoid_r_squared") or d.get("r_squared") or 0)
        closure = float(d.get("final_closure_pct") or 0)
        lag = float(d.get("sigmoid_lag_phase_hr") or 0)
        kinetics_score = max(0, min(25, sig_r2 * 15 + closure / 100 * 8 - min(lag / 6, 1) * 2))

        alpha = float(d.get("msd_alpha") or 1.0)
        dir_score = float(d.get("directed_migration_score") or 0.5)
        mig_frac = float(d.get("migration_fraction") or 50) / 100
        alpha_score = max(0, 1 - abs(alpha - 1.5) / 1.5) * 15
        migration_score = max(0, min(25, alpha_score + dir_score * 6 + mig_frac * 4))

        tort_i = float(d.get("initial_tortuosity") or 1.0)
        tort_f = float(d.get("final_tortuosity") or 1.0)
        asym = float(d.get("edge_asymmetry_index") or 0.0)
        tort_improvement = max(0, (tort_i - tort_f) / max(tort_i, 0.001))
        morpho_score = max(0, min(25, tort_improvement * 15 + (1 - min(asym, 1)) * 10))

        n_frames = int(d.get("num_timepoints") or 0)
        n_cells = int(d.get("num_cells_tracked") or 0)
        lin_r2 = float(d.get("r_squared") or 0)
        confidence_score = max(0, min(25, min(n_frames / 20, 1) * 10 + min(n_cells / 30, 1) * 8 + lin_r2 * 7))

        total_wts = kinetics_score + migration_score + morpho_score + confidence_score
        grade = "A — Excellent" if total_wts >= 80 else "B — Good" if total_wts >= 65 else "C — Acceptable" if total_wts >= 50 else "D — Poor" if total_wts >= 35 else "F — Unreliable"

        return {
            "wts_total": round(total_wts, 1),
            "grade": grade,
            "kinetics_score": round(kinetics_score, 1),
            "migration_score": round(migration_score, 1),
            "morpho_score": round(morpho_score, 1),
            "confidence_score": round(confidence_score, 1),
            "interpretation": f"WoundTrack Score {total_wts:.1f}/100 ({grade}) — Kinetics {kinetics_score:.1f} · Migration {migration_score:.1f} · Morphology {morpho_score:.1f} · Confidence {confidence_score:.1f}"
        }