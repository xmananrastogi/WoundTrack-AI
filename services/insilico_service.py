import numpy as np

class InSilicoService:
    @staticmethod
    def compute_durotaxis_surface(material, stiffness_kpa, crosslink_density, cell_type):
        """
        Computes a 3D surface meshgrid array of traction forces acting across the wound margin.
        Uses exponential decay mathematics governed by substrate stiffness (Durotaxis effect).
        """
        # Create a 2D mesh grid representing spatial coordinates around the wound edge
        x = np.linspace(-50, 50, 50)
        y = np.linspace(-50, 50, 50)
        X, Y = np.meshgrid(x, y)
        
        stiffness = float(stiffness_kpa)
        density = float(crosslink_density)
        
        # Calculate a peak traction force magnitude
        # Force scales logarithmically with stiffness. Modulated negatively if crosslink_density is too sparse or rigid
        peak_force = np.log10(stiffness + 1.0) * 15.0 * (1.0 - abs(density - 0.5))
        
        # Z = F * exp(-((X^2 + Y^2)/(2*variance)))
        # Stiffer substrates transmit physical mechanical signals over larger distances
        variance = 300.0 + stiffness * 2.0 
        Z = peak_force * np.exp(-((X**2 + Y**2) / (2 * variance)))
        
        # Inject minor stochastic focal-adhesion biologic noise 
        noise = np.random.normal(0, max(0.1, peak_force * 0.05), Z.shape)
        Z = np.abs(Z + noise)
        
        # Format the Plotly structure
        return {
            "data": [{
                "x": x.tolist(),
                "y": y.tolist(),
                "z": Z.tolist(),
                "type": "surface",
                "colorscale": [[0, "#050a07"], [0.4, "#9b59b6"], [0.8, "#1abc9c"], [1.0, "#2ecc71"]],
                "colorbar": {"title": "Traction (nN)", "tickfont": {"color": "#7a9c80"}, "titlefont": {"color":"#1abc9c"}}
            }],
            "layout": {
                "title": f"Durotaxis Edge Traction: {material} ({stiffness:.1f} kPa)",
                "scene": {
                    "xaxis": {"title": "X-Front (µm)", "gridcolor": "rgba(46,204,113,0.1)", "zerolinecolor": "rgba(46,204,113,0.2)", "color": "#7a9c80"},
                    "yaxis": {"title": "Y-Front (µm)", "gridcolor": "rgba(46,204,113,0.1)", "zerolinecolor": "rgba(46,204,113,0.2)", "color": "#7a9c80"},
                    "zaxis": {"title": "Force (nN)", "gridcolor": "rgba(46,204,113,0.1)", "zerolinecolor": "rgba(46,204,113,0.2)", "color": "#7a9c80"},
                    "bgcolor": "transparent"
                },
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
                "font": {"family": "DM Mono", "color": "#7a9c80"},
                "margin": {"l": 10, "r": 10, "t": 50, "b": 10}
            }
        }

    @staticmethod
    def compute_immune_trajectory(material, stiffness_kpa, crosslink_density, cell_type):
        """
        Computes the differential equations simulating Foreign Body Response vs Natural Regeneration.
        Plots major cellular timeline cascades.
        """
        days = np.linspace(0, 21, 50)
        
        # Evaluate standard categorical descriptors to shape biological behavior
        is_synthetic = any(x in material for x in ["Steel", "Ceramic", "PEGDA", "PHEMA"])
        
        # Neutrophils (Initial spike, acute response)
        neutro = 100 * np.exp(-1.5 * days)
        
        # M1 Macrophages (Pro-inflammatory, eats debris)
        # Fast onset, delayed decay if synthetic surface triggers foreign body encapsulation
        m1_peak_day = 3.5 if is_synthetic else 1.5
        m1_decay = 0.15 if is_synthetic else 0.45
        m1 = 80 * np.exp(-m1_decay * (days - m1_peak_day)**2)
        
        # M2 Macrophages (Pro-repair, secretes healing cytokines)
        m2_peak_day = 12 if is_synthetic else 6
        m2_decay = 0.08
        m2 = 60 * np.exp(-m2_decay * (days - m2_peak_day)**2)
        
        # Fibroblasts (Matrix deposition, wound closure)
        # Extreme stiffness accelerates proliferation (fibrosis)
        fibro_max = np.log10(float(stiffness_kpa) + 2.0) * 40
        fibro = fibro_max / (1 + np.exp(-0.8 * (days - (m1_peak_day + 3))))

        # Smooth lines
        return {
            "data": [
                {"x": days.tolist(), "y": m1.tolist(), "type": "scatter", "mode": "lines", "line": {"shape": "spline", "smoothing": 1.3, "color": "#e74c3c", "width": 2}, "name": "M1 Macrophage"},
                {"x": days.tolist(), "y": m2.tolist(), "type": "scatter", "mode": "lines", "line": {"shape": "spline", "smoothing": 1.3, "color": "#3498db", "width": 2}, "name": "M2 Macrophage"},
                {"x": days.tolist(), "y": fibro.tolist(), "type": "scatter", "mode": "lines", "line": {"shape": "spline", "smoothing": 1.3, "color": "#2ecc71", "width": 2}, "name": "Fibroblasts"},
                {"x": days.tolist(), "y": neutro.tolist(), "type": "scatter", "mode": "lines", "line": {"shape": "spline", "smoothing": 1.3, "color": "#9b59b6", "dash": "dot", "width": 1.5}, "name": "Neutrophils"}
            ],
            "layout": {
                "title": f"Immune Inflammatory Temporal Pipeline: {material}",
                "xaxis": {"title": "Time (Days)", "gridcolor": "rgba(255,255,255,0.05)", "color": "#7a9c80", "zeroline": False},
                "yaxis": {"title": "Relative Cell Density", "gridcolor": "rgba(255,255,255,0.05)", "color": "#7a9c80", "zeroline": False},
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
                "font": {"family": "DM Mono", "color": "#7a9c80"},
                "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
                "legend": {"orientation": "h", "y": -0.25}
            }
        }
