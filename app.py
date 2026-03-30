"""
app.py — WoundTrack AI Flask Server (Refactored Clean Architecture)
This file is strictly for booting the Flask application. All logic
has been delegated to the `api/`, `services/`, and `core/` modules.
"""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

from config.config import Config
from storage import database

# Import routing blueprints from the API module
from api.routes import api_bp, pages_bp

# Set up clean top-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize application
# Explicitly set the template folder using an absolute path to prevent TemplateNotFound errors
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config.from_object(Config)

# Enable CORS and initialize required system directories
CORS(app)
Config.init_dirs()

# Initialize the main SQLite tables inside the application context
# This prevents "Working outside of application context" errors during boot
with app.app_context():
    database.create_table()

# Register separated blueprints
app.register_blueprint(api_bp, url_prefix="/api")
app.register_blueprint(pages_bp)

# 🛡️ SECURITY HEADERS & DEPLOYMENT PROTECTION
@app.after_request
def add_security_headers(response):
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    # Prevent MIME-type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    # High-level script/style protection
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' cdn.plot.ly; "
        "style-src 'self' 'unsafe-inline' fonts.googleapis.com; "
        "font-src 'self' fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self';"
    )
    response.headers['Content-Security-Policy'] = csp
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    # Hide traceback in production
    logger.error("Unhandled Exception: %s", e)
    if app.config.get("FLASK_ENV") == "production":
        return jsonify({"error": "An internal server error occurred"}), 500
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("🔬 WoundTrack AI v4 Hardened Architecture — http://localhost:%s", os.environ.get("PORT", 8080))

    # use_reloader=False prevents macOS semaphore thread leaking and double-init issues
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=(app.config.get("FLASK_ENV") == "development"),
        use_reloader=False,
        threaded=True,
    )