# Wound Healing Scratch Assay Analysis Pipeline

## Overview
This project provides an automated pipeline for analyzing wound healing scratch assays from microscopy images. Using Python and OpenCV, it segments wound areas and extracts healing metrics. Results are visualized interactively through a Flask web dashboard with filtering and export features.

## Features
- Image segmentation via adaptive thresholding and morphological filtering
- Extraction of metrics: wound area, closure percentage, healing rate, and fit quality
- Batch processing of multiple experimental datasets
- Interactive Flask dashboard with experiment filtering, tooltips, and CSV downloads
- Complete research paper with methodology and biological implications included

### Prerequisites
- Python 3.8 or higher
- Virtual environment recommended

## Repository Structure

/data/ # Raw and processed microscopy image data
/results/ # Wound healing metrics and visualizations
/app.py # Main analysis and Flask web server
/compareallconditions.py # Experimental comparison scripts
/requirements.txt # Python package requirements
/README.md # This documentation

text

## Contributing

Contributions are welcome! Please fork the repo and submit pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

Thanks to the entire team for collaboration and the open-source community for supporting tools.

---
