# storage/database.py
import sqlite3
import os
import pandas as pd
from flask import current_app


def get_db_path():
    return os.path.join(current_app.config["RESULTS_FOLDER"], "database.db")


def create_table():
    os.makedirs(current_app.config["RESULTS_FOLDER"], exist_ok=True)
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    # Fully Normalized Architecture
    c.execute('''CREATE TABLE IF NOT EXISTS experiments
                 (id TEXT PRIMARY KEY, name TEXT, condition TEXT, substrate TEXT,
                  stiffness REAL, treatment TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS metrics
                 (exp_id TEXT PRIMARY KEY, closure_pct REAL, healing_rate REAL,
                  time_to_50 REAL, sig_r2 REAL, sig_max_rate REAL, sig_lag REAL,
                  FOREIGN KEY(exp_id) REFERENCES experiments(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS tracking
                 (exp_id TEXT PRIMARY KEY, num_cells INTEGER, velocity REAL,
                  directionality REAL, msd_alpha REAL,
                  FOREIGN KEY(exp_id) REFERENCES experiments(id))''')
    conn.commit()
    conn.close()


def upsert_experiment(summary_data, result_id):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute(
        "REPLACE INTO experiments (id, name, condition, substrate, stiffness, treatment) VALUES (?, ?, ?, ?, ?, ?)",
        (result_id, summary_data.get("experiment_name", "Unknown"), summary_data.get("condition_name", "Unknown"),
         summary_data.get("substrate_material"), summary_data.get("substrate_stiffness_kpa"),
         summary_data.get("treatment")))

    c.execute(
        "REPLACE INTO metrics (exp_id, closure_pct, healing_rate, time_to_50, sig_r2, sig_max_rate, sig_lag) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (result_id, summary_data.get("final_closure_pct"), summary_data.get("healing_rate_um2_per_hr"),
         summary_data.get("time_to_50_closure_hr"), summary_data.get("sigmoid_r_squared"),
         summary_data.get("sigmoid_max_rate_pct_hr"), summary_data.get("sigmoid_lag_phase_hr")))

    c.execute("REPLACE INTO tracking (exp_id, num_cells, velocity, directionality, msd_alpha) VALUES (?, ?, ?, ?, ?)",
              (result_id, summary_data.get("num_cells_tracked"), summary_data.get("mean_velocity_um_min"),
               summary_data.get("mean_directionality"), summary_data.get("msd_alpha")))
    conn.commit()
    conn.close()


def delete_experiment(result_id):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    for table in ["experiments", "metrics", "tracking"]:
        c.execute(f"DELETE FROM {table} WHERE {'id' if table == 'experiments' else 'exp_id'}=?", (result_id,))
    conn.commit()
    conn.close()


def get_all_metrics_for_plots():
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query('''SELECT e.condition as Condition, m.closure_pct as "Closure (%)",
                              m.healing_rate as "Healing Speed (µm²/hr)", m.sig_r2 as "Consistency (R²)",
                              t.velocity as "Cell Velocity (µm/min)", t.directionality as "Directionality"
                              FROM experiments e LEFT JOIN metrics m ON e.id = m.exp_id LEFT JOIN tracking t ON e.id = t.exp_id''',
                           conn)
    conn.close()
    return df


def get_stiffness_healing_data():
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query('''SELECT e.name as experiment_name, e.substrate as substrate_material,
                              e.stiffness as substrate_stiffness_kpa, m.healing_rate as healing_rate_um2_per_hr,
                              m.closure_pct as final_closure_pct
                              FROM experiments e JOIN metrics m ON e.id = m.exp_id WHERE e.stiffness IS NOT NULL''',
                           conn)
    conn.close()
    return df


def get_stats_by_condition():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('''SELECT e.condition, COUNT(e.id), AVG(m.closure_pct), AVG(m.healing_rate), AVG(t.velocity)
                 FROM experiments e LEFT JOIN metrics m ON e.id = m.exp_id LEFT JOIN tracking t ON e.id = t.exp_id GROUP BY e.condition''')
    stats = {r[0]: {"n": r[1], "mean_closure": r[2] or 0, "mean_rate": r[3] or 0, "mean_velocity": r[4] or 0} for r in
             c.fetchall()}
    conn.close()
    return stats


def get_all_experiments_for_comparison():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('''SELECT e.id, e.name, e.condition, m.closure_pct, m.healing_rate, t.velocity
                 FROM experiments e LEFT JOIN metrics m ON e.id = m.exp_id LEFT JOIN tracking t ON e.id = t.exp_id''')
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "experiment_name": r[1], "condition": r[2], "final_closure_pct": r[3],
             "healing_rate_um2_per_hr": r[4], "mean_velocity_um_min": r[5]} for r in rows]


def calculate_all_pvalues():
    return {}  # Connect scipy.stats if p-values are required in future