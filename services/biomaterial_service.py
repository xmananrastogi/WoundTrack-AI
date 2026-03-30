# services/biomaterial_service.py
import math
from config.constants import SCAFFOLD_DB


class BiomaterialService:
    """Virtual Scaffold Designer logic extracted from Flask routes."""

    @staticmethod
    def recommend_scaffold(material: str, stiffness_kpa: float, crosslink_density: float, cell_type: str) -> dict:
        info = SCAFFOLD_DB.get(material, SCAFFOLD_DB["Collagen I"])
        lo, hi = info["stiffness_range"]
        stiffness = max(lo, min(hi, stiffness_kpa))

        cell_params = {
            "epithelial": {"peak_kpa": 5.0, "base_rate": 1200, "base_vel": 0.8},
            "fibroblast": {"peak_kpa": 20.0, "base_rate": 2500, "base_vel": 1.4},
            "cancer": {"peak_kpa": 50.0, "base_rate": 3500, "base_vel": 2.1},
        }
        p = cell_params.get(cell_type, cell_params["epithelial"])

        stiff_factor = math.exp(-0.5 * ((math.log10(max(stiffness, 0.01)) - math.log10(p["peak_kpa"])) / 1.0) ** 2)
        crosslink_penalty = 1.0 - 0.4 * crosslink_density
        adhesion_bonus = {"Very High": 1.25, "High": 1.10, "Medium": 0.95, "Low": 0.75,
                          "Low (requires functionalization)": 0.65, "Medium (coating-dependent)": 0.90}.get(
            info["cell_adhesion"], 1.0)

        healing_rate = p["base_rate"] * stiff_factor * crosslink_penalty * adhesion_bonus
        velocity = p["base_vel"] * stiff_factor * crosslink_penalty * adhesion_bonus
        closure_24h = min(99.0, healing_rate / 50.0)

        if stiffness < 1.0:
            regime = "Amoeboid (soft substrate — reduced traction)"
        elif stiffness < 20.0:
            regime = "Mesenchymal (optimal matrix remodelling zone)"
        elif stiffness < 200.0:
            regime = "Mechanosensing plateau (high durotaxis drive)"
        else:
            regime = "Frustrated migration (substrate too rigid)"

        if abs(math.log10(max(stiffness, 0.01)) - math.log10(p["peak_kpa"])) < 0.3:
            rec = f"✅ Optimal stiffness for {cell_type} migration on {material}."
        elif stiffness < p["peak_kpa"]:
            rec = f"⬆️ Increase stiffness toward {p['peak_kpa']:.0f} kPa for faster closure."
        else:
            rec = f"⬇️ Reduce stiffness toward {p['peak_kpa']:.0f} kPa — current value suppresses migration."

        porosity = info["porosity"] * (1 - 0.3 * crosslink_density)

        return {
            "healing_rate_um2_hr": round(healing_rate, 1),
            "velocity_um_min": round(velocity, 3),
            "closure_pct_24h": round(closure_24h, 1),
            "migration_mode": regime,
            "recommendation": rec,
            "effective_porosity": round(porosity * 100, 1),
            "degradation_days": info["degradation_days"],
            "cell_adhesion": info["cell_adhesion"],
            "common_use": info["common_use"],
            "stiffness_used_kpa": round(stiffness, 2),
        }