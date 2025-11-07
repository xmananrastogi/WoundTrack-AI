# 1. Start with a Python 3.12 base image
FROM python:3.12-slim

# 2. Set up the environment
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# 3. Install the complex system libraries that OpenCV needs
# This is the "magic" that Docker handles for us.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    --no-install-recommends
RUN rm -rf /var/lib/apt/lists/*

# 4. Copy and install our Python requirements
COPY requirements-py312.txt .
RUN pip install --no-cache-dir -r requirements-py312.txt

# 5. Copy all our application code into the container
COPY . .

# 6. Expose the port our app will run on
EXPOSE 8080

# 7. The command to run the app using gunicorn
# It will run 'app:app' (the 'app' object inside 'app.py')
# It binds to port 8080, has 2 workers, and a 5-minute (300s) timeout
# for your long-running analysis.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "300", "app:app"]
