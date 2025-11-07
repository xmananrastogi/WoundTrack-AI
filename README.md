# WoundTrack AI 🔬

**A full-stack web platform for quantitative, multi-scale analysis of *in-vitro* wound healing (scratch) assays.**

This project moves beyond simple area-based measurements by integrating a robust image processing backend with a multi-object cell tracker, allowing for the simultaneous analysis of both macroscopic healing kinetics and microscopic single-cell migratory behavior.

The app is a complete, end-to-end solution: upload your time-lapse images, and get back a full suite of interactive plots, videos, and statistical analyses.

---

## 📸 Gallery

<p align="center">
  <strong>Main Dashboard & Analysis Page</strong><br>
  <img src=123.jpeg alt="WoundTrack AI Dashboard">
</p>

<p align="center">
  <strong>Results Dashboard with Analysis Cards</strong><br>
  <img src=456.jpeg alt="WoundTrack AI Results">
</p>

<p align="center">
  <strong>"View Details" Modal with Interactive Plot & Segmentation Video</strong><br>
  <img src=789.jpeg alt="WoundTrack AI Modal">
</p>

<p align="center">
  <strong>"Statistics" Tab with Aggregate Box Plots & Correlation Heatmap</strong><br>
  <img src=101.jpeg alt="WoundTrack AI Statistics">
</p>

---

## ✨ Core Features

* **Automated Data Ingestion:** Handles `ZIP`, `TIF` (multi-page), `MP4`, `AVI`, and folders of images.
* **Persistent Database:** Uses **SQLite** to save, browse, and manage all analysis results.
* **Delete Functionality:** Clean up bad runs from both the database and the file system.

### Macroscopic Analysis (Wound Level)

* **Automated Segmentation:** Uses a **Shannon Entropy filter** (`scikit-image`) and **Otsu's Thresholding** (`opencv`) to provide an objective, bias-free segmentation of the wound area.
* **Kinetic Calculations:** Automatically calculates:
    * Wound Area vs. Time ($\mu m^2$)
    * Percent Closure vs. Time
    * Healing Rate (via Linear Regression)
    * Healing Consistency ($R^2$)

### Microscopic Analysis (Cell Level)

* **Cell Tracking:** Implements **Trackpy** to track individual cells at the wound front.
* **Advanced Metrics:** Quantifies the *mechanism* of healing by calculating:
    * Mean Cell Velocity
    * Migration Efficiency (Chemotaxis Index)
    * **Mean Directionality** (cosine similarity to the wound center)

### Data Visualization Platform

* **Interactive Plots:** All results are plotted with **Plotly.js** for interactive viewing.
* **Video Generation:** Creates and displays an `.mp4` video overlay of the segmentation on the original footage.
* **Compare Tab:** Select any two (or more) experiments for a side-by-side bar chart comparison.
* **Statistics Tab:** Automatically generates **Box Plots** (to compare distributions between conditions) and a **Correlation Heatmap** (to find relationships between metrics).
* **Report Generation:** Download any result as a full **PDF report** or a simple **CSV** timeseries.

---

## 🛠️ Tech Stack

| Category | Technology |
| :--- | :--- |
| **Backend** | Python, Flask, Gunicorn |
| **Frontend** | HTML5, CSS3, JavaScript, Plotly.js |
| **Data Analysis** | Pandas, NumPy, SciPy |
| **Image Processing** | OpenCV, Scikit-image, Pillow |
| **Cell Tracking** | Trackpy, Pims |
| **Database** | SQLite |
| **Deployment** | Docker, Docker Compose |
| **Reporting** | ReportLab |

---

https://xmananrastogi-woundtrack-ai.hf.space/


