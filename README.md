# Wound Healing Scratch Assay Analysis Pipeline

## Overview
This project provides an automated pipeline for analyzing wound healing scratch assays from microscopy images. Using Python and OpenCV, it segments wound areas and extracts healing metrics. Results are visualized interactively through a Flask web dashboard with filtering and export features.

## Features
- Image segmentation via adaptive thresholding and morphological filtering
- Extraction of metrics: wound area, closure percentage, healing rate, and fit quality
- Batch processing of multiple experimental datasets
- Interactive Flask dashboard with experiment filtering, tooltips, and CSV downloads
- Complete research paper with methodology and biological implications included

  ...

## Dataset

The microscopy image datasets used in this project are publicly available from the [Cell Image Library](https://www.cellimagelibrary.org/).

Please download the relevant wound healing scratch assay datasets from the Cell Image Library prior to running the analysis pipeline. Organize the downloaded image sequences into the `/data/raw` folder structure as expected by the pipeline.

If you are unfamiliar with the Cell Image Library:

- Visit [Cell Image Library](https://www.cellimagelibrary.org/).
- Search for "wound healing scratch assay" or relevant keywords.
- Download the time-lapse image series in TIFF or supported formats.
- Follow folder naming conventions consistent with this project.

This approach helps keep the project repository size manageable by excluding large original datasets, while ensuring reproducible research.

...


### Prerequisites
- Python 3.8 or higher
- Virtual environment recommended

### Setup
Clone the repo:
git clone https://github.com/xmananrastogi/wound-healing-analysis.git
cd wound-healing-analysis

Install dependencies:
pip install -r requirements.txt

### Running the analysis
Run the main pipeline:
python app.py

Access the dashboard via:
http://localhost:5000

## Repository Structure

/data/ # Raw and processed microscopy image data
/results/ # Wound healing metrics and visualizations
/app.py # Main analysis and Flask web server
/compareallconditions.py # Experimental comparison scripts
/requirements.txt # Python package requirements
/README.md # This documentation


## Contributing

Contributions are welcome! Please fork the repo and submit pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

Thanks to the entire team for collaboration and the open-source community for supporting tools.

---


