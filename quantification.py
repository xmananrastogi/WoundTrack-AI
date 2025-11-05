"""Quantification Module"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def calculate_wound_closure_percentage(area_t: float, area_0: float) -> float:
    if area_0 == 0:
        return 100.0
    closure = (1 - (area_t / area_0)) * 100
    return max(0, min(100, closure))

def calculate_healing_rate(time_points: List[float], areas: List[float]) -> Tuple[float, float, float]:
    if len(time_points) != len(areas) or len(time_points) < 2:
        return 0.0, 0.0, 0.0
    t = np.array(time_points)
    a = np.array(areas)
    coeffs = np.polyfit(t, a, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    predicted = slope * t + intercept
    ss_res = np.sum((a - predicted) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        r_squared = 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    return slope, r_squared, intercept

def analyze_time_series(time_points: List[float], areas: List[float], time_unit: str = 'hours') -> Dict:
    if len(time_points) == 0 or len(areas) == 0:
        return {}
    area_0 = areas[0]
    area_final = areas[-1]
    closure_percentages = [calculate_wound_closure_percentage(a, area_0) for a in areas]
    healing_rate, r_squared, intercept = calculate_healing_rate(time_points, areas)
    time_to_50_closure = None
    for i, closure in enumerate(closure_percentages):
        if closure >= 50.0:
            if i > 0:
                t1, t2 = time_points[i-1], time_points[i]
                c1, c2 = closure_percentages[i-1], closure_percentages[i]
                time_to_50_closure = t1 + (50.0 - c1) * (t2 - t1) / (c2 - c1)
            else:
                time_to_50_closure = time_points[i]
            break
    results = {
        'initial_area': area_0,
        'final_area': area_final,
        'final_closure_percentage': closure_percentages[-1],
        'healing_rate': healing_rate,
        'healing_rate_unit': f'pixels/{time_unit}',
        'r_squared': r_squared,
        'time_to_50_closure': time_to_50_closure,
        'time_unit': time_unit,
        'num_timepoints': len(time_points),
        'closure_percentages': closure_percentages
    }
    return results

def create_results_dataframe(experiments: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for exp_name, results in experiments.items():
        row = {
            'experiment': exp_name,
            'initial_area_px': results.get('initial_area', np.nan),
            'final_area_px': results.get('final_area', np.nan),
            'final_closure_pct': results.get('final_closure_percentage', np.nan),
            'healing_rate': results.get('healing_rate', np.nan),
            'r_squared': results.get('r_squared', np.nan),
            'time_to_50_closure': results.get('time_to_50_closure', np.nan),
            'num_timepoints': results.get('num_timepoints', np.nan)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    print("Quantification module loaded!")
