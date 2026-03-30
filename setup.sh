#!/usr/bin/env bash
# WoundTrack AI Setup Script

echo "Setting up WoundTrack AI dependencies..."

# Check if venv exists, if not create it
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required dependencies including new biological analysis modules
echo "Installing packages..."
pip install Flask Werkzeug numpy opencv-contrib-python-headless pandas scipy matplotlib seaborn plotly Pillow numba scikit-image trackpy pytesseract python-dotenv gunicorn

echo "Setup complete! Run 'source .venv/bin/activate' and then 'python3 app.py' to start."