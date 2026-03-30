# WoundTrack AI — Production Docker Container
# High-performance clinical analysis environment

# Use scientific python base
FROM python:3.11-slim

# Set system dependencies for OpenCV & Tesseract
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements/setup (Installing directly for speed)
COPY setup.sh .
COPY . .

# Run installation
RUN pip install --no-cache-dir \
    Flask Werkzeug numpy opencv-contrib-python-headless \
    pandas scipy matplotlib seaborn plotly Pillow numba \
    scikit-image trackpy pytesseract python-dotenv gunicorn

# Initialize directories and set ownership for Hugging Face (UID 1000)
RUN mkdir -p uploads results_data && chmod -R 777 uploads results_data

# Production Environment
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=7860

# Port 7860 is required by Hugging Face Spaces
EXPOSE 7860

# Run with Gunicorn (4 workers, 1 thread each for safety with OpenCV)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "4", "--threads", "1", "app:app"]
