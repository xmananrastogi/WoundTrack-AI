#!/usr/bin/env bash
# 🔬 WoundTrack AI — Unified Run Script

# 1. Automate environment activation
if [ -d ".venv" ]; then
    echo "Using existing .venv..."
    source .venv/bin/activate
else
    echo "Virtual environment not found. Running setup.sh..."
    chmod +x setup.sh
    ./setup.sh
    source .venv/bin/activate
fi

# 2. Start the Flask application
echo "Starting WoundTrack AI v3 Architecture..."
python3 app.py
