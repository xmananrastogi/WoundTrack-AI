# config/constants.py

METRIC_INFO = {
    # Area & linear metrics
    "initial_area_px":             {"name": "Initial Cell-Free Area", "unit": "px²"},
    "final_area_px":               {"name": "Final Cell-Free Area",   "unit": "px²"},
    "initial_area_um2":            {"name": "Initial Cell-Free Area", "unit": "µm²"},
    "final_area_um2":              {"name": "Final Cell-Free Area",   "unit": "µm²"},
    "final_closure_pct":           {"name": "Wound Area Fraction (Closure)", "unit": "%"},
    "healing_rate_um2_per_hr":     {"name": "Gap Closure Rate",      "unit": "µm²/hr"},
    "healing_rate_px_per_hr":      {"name": "Gap Closure Rate",      "unit": "px/hr"},
    "r_squared":                   {"name": "Linear Fit R²",         "unit": ""},
    "num_timepoints":              {"name": "Frames Analysed",       "unit": ""},
    "pixel_scale_um_per_px":       {"name": "Pixel Scale",           "unit": "µm/px"},

    # Multi-threshold & Sigmoidal kinetics
    "time_to_25_closure_hr":       {"name": "t₂₅ Gap Closure",       "unit": "hr"},
    "time_to_50_closure_hr":       {"name": "t₅₀ Gap Closure",       "unit": "hr"},
    "time_to_75_closure_hr":       {"name": "t₇₅ Gap Closure",       "unit": "hr"},
    "time_to_90_closure_hr":       {"name": "t₉₀ Gap Closure",       "unit": "hr"},
    "sigmoid_model":               {"name": "Sigmoid Curve Fit",     "unit": ""},
    "sigmoid_asymptote_pct":       {"name": "Predicted Max Closure", "unit": "%"},
    "sigmoid_max_rate_pct_hr":     {"name": "Maximum Kinetic Rate",  "unit": "%/hr"},
    "sigmoid_lag_phase_hr":        {"name": "Lag Phase (Initiation)", "unit": "hr"},
    "sigmoid_inflection_hr":       {"name": "Inflection Point",      "unit": "hr"},
    "sigmoid_r_squared":           {"name": "Sigmoid R²",            "unit": ""},
    "sigmoid_aic":                 {"name": "Sigmoid AIC",           "unit": ""},

    # Dual edge velocity & Tortuosity
    "left_edge_velocity_um_hr":    {"name": "Left Margin Advancement", "unit": "µm/hr"},
    "right_edge_velocity_um_hr":   {"name": "Right Margin Advancement","unit": "µm/hr"},
    "edge_asymmetry_index":        {"name": "Advancement Asymmetry", "unit": "0–1"},
    "initial_tortuosity":          {"name": "Initial Margin Tortuosity", "unit": "arc-chord"},
    "final_tortuosity":            {"name": "Final Margin Tortuosity",   "unit": "arc-chord"},
    "edge_smoothing_rate":         {"name": "Margin Reorganisation Rate", "unit": "/hr"},

    # Proliferation correction & Segmentation quality
    "migration_fraction":          {"name": "Directed Motility Fraction","unit": "%"},
    "proliferation_fraction":      {"name": "Proliferation Fraction",    "unit": "%"},
    "migration_rate_px2_hr":       {"name": "Directed Motility Rate",    "unit": "px²/hr"},
    "proliferation_rate_px2_hr":   {"name": "Proliferation Rate",        "unit": "px²/hr"},
    "flatfield_applied":           {"name": "Flatfield Corrected",       "unit": ""},
    "segmentation_method":         {"name": "Segmentation Engine",       "unit": ""},
    "wound_angle_deg":             {"name": "Wound Axis Angle",          "unit": "°"},

    # Cell tracking
    "num_cells_tracked":           {"name": "Cells Path-Tracked",        "unit": ""},
    "mean_velocity_um_min":        {"name": "Cellular Motility Rate",    "unit": "µm/min"},
    "migration_efficiency_mean":   {"name": "Chemotactic Efficiency",    "unit": ""},
    "mean_directionality":         {"name": "Directional Persistance",   "unit": ""},
    "mean_displacement_um":        {"name": "Net Vector Displacement",   "unit": "µm"},
    "mean_path_length_um":         {"name": "Total Path Length",         "unit": "µm"},
    "msd_alpha":                   {"name": "MSD Exponent α",            "unit": ""},
    "msd_D_um2_hr":                {"name": "Diffusion Coeff. D",        "unit": "µm²/hr"},
    "migration_mode_msd":          {"name": "Motility Mode (MSD)",       "unit": ""},
    "directed_migration_score":    {"name": "Directed Migration Score",  "unit": "0–1"},
    "persistence_time_hr":         {"name": "Persistance Time",          "unit": "hr"},
    "division_rate_per_hr":        {"name": "Mitotic Event Rate",        "unit": "/hr"},
    "doubling_time_hr":            {"name": "Population Doubling Time",  "unit": "hr"},

    # Biomaterial
    "substrate_material":          {"name": "Substrate",             "unit": ""},
    "substrate_stiffness_kpa":     {"name": "Stiffness",             "unit": "kPa"},
    "treatment":                   {"name": "Treatment",             "unit": ""},
}

CONDITION_NAMES = {
    "MDCK_Control": ("🧬 Epithelial Cells (Baseline)", "Normal epithelial cells"),
    "MDCK_HGF":     ("⚡ Epithelial + Growth Factor",  "With HGF/SF treatment"),
    "DA3_Control":  ("🔬 Cancer Cells (Baseline)",      "Cancer cells baseline"),
    "DA3_PHA":      ("💊 Cancer + Immune Activation",   "With immune activation"),
    "DA3_HGF":      ("🔥 Cancer + Growth Factor",       "With growth factor"),
    "Uploaded Data":("📤 Uploaded Data",                "User-uploaded dataset"),
}

SCAFFOLD_DB = {
    "Collagen I":      {"stiffness_range": (0.1, 4.0), "default_stiffness": 1.0, "porosity": 0.85, "degradation_days": 14, "cell_adhesion": "High", "common_use": "Skin, connective tissue models", "color": "#e8d5b7"},
    "Fibrin":          {"stiffness_range": (0.05, 2.0), "default_stiffness": 0.5, "porosity": 0.90, "degradation_days": 7, "cell_adhesion": "High", "common_use": "Wound healing, vascularization", "color": "#f4c2a1"},
    "Matrigel":        {"stiffness_range": (0.1, 0.6), "default_stiffness": 0.3, "porosity": 0.92, "degradation_days": 5, "cell_adhesion": "Very High", "common_use": "Tumor invasion, angiogenesis assays", "color": "#c8e6c9"},
    "PEGDA":           {"stiffness_range": (1.0, 50.0), "default_stiffness": 10.0, "porosity": 0.70, "degradation_days": 90, "cell_adhesion": "Low (requires functionalization)", "common_use": "Cartilage, tunable stiffness studies", "color": "#bbdefb"},
    "Alginate":        {"stiffness_range": (1.0, 100.0), "default_stiffness": 20.0, "porosity": 0.75, "degradation_days": 30, "cell_adhesion": "Low", "common_use": "Encapsulation, drug delivery", "color": "#e1bee7"},
    "GelMA":           {"stiffness_range": (0.5, 30.0), "default_stiffness": 5.0, "porosity": 0.80, "degradation_days": 21, "cell_adhesion": "High", "common_use": "Bioprinting, wound healing", "color": "#fff9c4"},
    "Hyaluronic Acid": {"stiffness_range": (0.05, 10.0), "default_stiffness": 1.5, "porosity": 0.88, "degradation_days": 10, "cell_adhesion": "Medium", "common_use": "Cartilage, skin repair", "color": "#f8bbd9"},
    "Glass (control)": {"stiffness_range": (10000, 10000), "default_stiffness": 10000, "porosity": 0.0, "degradation_days": 0, "cell_adhesion": "Medium (coating-dependent)", "common_use": "Rigid substrate control", "color": "#e0e0e0"},
}