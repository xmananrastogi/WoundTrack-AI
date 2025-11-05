"""
Database module for handling SQLite operations.
"""
import sqlite3
import logging
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from config import Config

logger = logging.getLogger(__name__)
DATABASE_URL = Config.DATABASE_URL

def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
    return conn

def create_table():
    """Create the experiments table if it doesn't exist."""
    sql_create_table = """
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        experiment_name TEXT NOT NULL,
        condition_name TEXT,
        final_closure_pct REAL,
        healing_rate_um2_per_hr REAL,
        r_squared REAL,
        time_to_50_closure_hr REAL,
        num_cells_tracked INTEGER,
        mean_velocity_um_min REAL,
        migration_efficiency_mean REAL,
        mean_directionality REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = create_connection()
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute(sql_create_table)
            conn.commit()
            logger.info("Database table 'experiments' is ready.")
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")
        finally:
            conn.close()

def upsert_experiment(summary_data: dict, result_id: str):
    """
    Insert or update an experiment's results in the database.
    'result_id' is the unique identifier (e.g., 'uploads/uuid' or 'DA3_Control/CIL_43406')
    """
    conn = create_connection()
    if conn is None:
        return

    sql = """
    INSERT OR REPLACE INTO experiments (
        id, experiment_name, condition_name, final_closure_pct,
        healing_rate_um2_per_hr, r_squared, time_to_50_closure_hr,
        num_cells_tracked, mean_velocity_um_min, migration_efficiency_mean,
        mean_directionality
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    try:
        # Extract data, using .get() to provide defaults (None) if key is missing
        data_tuple = (
            result_id,
            summary_data.get('experiment_name'),
            summary_data.get('condition_name'),
            summary_data.get('final_closure_pct'),
            summary_data.get('healing_rate_um2_per_hr'),
            summary_data.get('r_squared'),
            summary_data.get('time_to_50_closure_hr'),
            summary_data.get('num_cells_tracked'),
            summary_data.get('mean_velocity_um_min'),
            summary_data.get('migration_efficiency_mean'),
            summary_data.get('mean_directionality')
        )

        c = conn.cursor()
        c.execute(sql, data_tuple)
        conn.commit()
        logger.info(f"Successfully upserted experiment '{result_id}' to database.")
    except sqlite3.Error as e:
        logger.error(f"Error upserting experiment '{result_id}': {e}")
    finally:
        conn.close()

def get_all_experiments_for_comparison():
    """
    Query the database for all experiments to populate the comparison UI.
    """
    conn = create_connection()
    if conn is None:
        return []

    try:
        c = conn.cursor()
        # Fetch all data, including condition_name
        c.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
        rows = c.fetchall()
        # Convert sqlite.Row objects to plain dictionaries
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error fetching comparison data: {e}")
        return []
    finally:
        if conn:
            conn.close()

def delete_experiment(result_id: str):
    """
    Delete an experiment from the database by its ID.
    """
    conn = create_connection()
    if conn is None:
        logger.error("Could not connect to database to delete.")
        return False

    sql = "DELETE FROM experiments WHERE id = ?"

    try:
        c = conn.cursor()
        c.execute(sql, (result_id,))
        conn.commit()
        # Check if any row was actually deleted
        was_deleted = c.rowcount > 0
        if was_deleted:
            logger.info(f"Successfully deleted experiment '{result_id}' from database.")
        else:
            logger.warning(f"No experiment with id '{result_id}' found in database to delete.")
        return was_deleted
    except sqlite3.Error as e:
        logger.error(f"Error deleting experiment '{result_id}' from database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_stats_by_condition():
    """
    Calculates aggregate statistics (mean, std, n, ci) for key metrics,
    grouped by condition. This replaces the old file-based logic.
    """
    conn = create_connection()
    if conn is None:
        return {}

    # --- FIXED ---
    # 1. Select the raw data, not the aggregates.
    query = """
    SELECT
        condition_name,
        final_closure_pct,
        healing_rate_um2_per_hr,
        r_squared
    FROM experiments
    WHERE condition_name IS NOT NULL
    """

    summary = {}
    try:
        # 2. Read all data into pandas
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return {}

        # 3. Use pandas to calculate all stats, which handles STDEV correctly.
        grouped = df.groupby('condition_name')

        # Note: pandas .std() uses ddof=1 by default (sample stdev), which is correct.
        stats_df = grouped.agg(
            n=('final_closure_pct', 'count'),
            closure_mean=('final_closure_pct', 'mean'),
            closure_std=('final_closure_pct', 'std'),
            healing_mean=('healing_rate_um2_per_hr', 'mean'),
            healing_std=('healing_rate_um2_per_hr', 'std'),
            r2_mean=('r_squared', 'mean')
        )

        # 4. Filter for n >= 2 and fill NaNs (which happen if std dev is 0)
        stats_df = stats_df[stats_df['n'] >= 2].fillna(0)

        # 5. Build the summary dictionary from the pandas DataFrame
        for condition_name, row in stats_df.iterrows():
            n = row['n']
            closure_std = row['closure_std']
            healing_std = row['healing_std']

            # Check if healing_rate_um2_per_hr was used
            uses_scientific = row['healing_mean'] != 0

            summary[condition_name] = {
                'n': int(n),
                'closure_mean': float(row['closure_mean']),
                'closure_std': float(closure_std),
                'closure_ci': float(1.96 * closure_std / np.sqrt(n)) if n > 0 else 0,
                'healing_mean': float(row['healing_mean']),
                'healing_std': float(healing_std),
                'healing_ci': float(1.96 * healing_std / np.sqrt(n)) if n > 0 else 0,
                'r2_mean': float(row['r2_mean']),
                'uses_scientific': bool(uses_scientific),
                'healing_unit': 'µm²/hr' if bool(uses_scientific) else 'px/hr'
            }

    except Exception as e:
        logger.error(f"Error calculating stats by condition: {e}")
    finally:
        if conn:
            conn.close()
    return summary
    # --- END FIX ---

def get_significance_stars(p_value):
    if p_value is None or np.isnan(p_value):
        return 'n/a'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'

def calculate_all_pvalues():
    """
    Calculates t-test p-values for all combinations of conditions.
    This replaces the old file-based logic.
    """
    conn = create_connection()
    if conn is None:
        return {}

    query = """
    SELECT condition_name, final_closure_pct, healing_rate_um2_per_hr
    FROM experiments
    WHERE condition_name IS NOT NULL
    """
    pvalues = {}

    try:
        df = pd.read_sql_query(query, conn)
        df = df.dropna(subset=['final_closure_pct', 'healing_rate_um2_per_hr'])

        conditions = df['condition_name'].unique()

        for cond1, cond2 in combinations(conditions, 2):
            pair = f"{cond1} vs {cond2}"

            data1_closure = df[df['condition_name'] == cond1]['final_closure_pct']
            data2_closure = df[df['condition_name'] == cond2]['final_closure_pct']

            data1_healing = df[df['condition_name'] == cond1]['healing_rate_um2_per_hr']
            data2_healing = df[df['condition_name'] == cond2]['healing_rate_um2_per_hr']

            if len(data1_closure) < 2 or len(data2_closure) < 2:
                continue

            _, p_closure = stats.ttest_ind(data1_closure, data2_closure, equal_var=False, nan_policy='omit')
            _, p_healing = stats.ttest_ind(data1_healing, data2_healing, equal_var=False, nan_policy='omit')

            pvalues[pair] = {
                'closure_p': float(p_closure),
                'healing_p': float(p_healing),
                'closure_sig': get_significance_stars(p_closure),
                'healing_sig': get_significance_stars(p_healing)
            }

    except Exception as e:
        logger.error(f"Error calculating p-values: {e}")
    finally:
        if conn:
            conn.close()
    return pvalues

def get_all_metrics_for_plots():
    """
    Fetches all key metrics from the database for creating
    correlation and box plots. Returns a pandas DataFrame.
    """
    conn = create_connection()
    if conn is None:
        return pd.DataFrame()

    query = """
    SELECT
        condition_name,
        final_closure_pct,
        healing_rate_um2_per_hr,
        r_squared,
        mean_velocity_um_min,
        migration_efficiency_mean,
        mean_directionality
    FROM experiments
    WHERE condition_name IS NOT NULL
    """
    try:
        df = pd.read_sql_query(query, conn)

        # Rename columns for prettier plot labels
        df = df.rename(columns={
            'condition_name': 'Condition',
            'final_closure_pct': 'Closure (%)',
            'healing_rate_um2_per_hr': 'Healing Speed (µm²/hr)',
            'r_squared': 'Consistency (R²)',
            'mean_velocity_um_min': 'Cell Velocity (µm/min)',
            'migration_efficiency_mean': 'Efficiency',
            'mean_directionality': 'Directionality'
        })
        return df.dropna()

    except Exception as e:
        logger.error(f"Error fetching metrics for plots: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()