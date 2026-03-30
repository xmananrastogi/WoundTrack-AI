import os
import sys
from dotenv import load_dotenv

# Load from .env if present
load_dotenv()

class Config:
    # 🛡️ MANDATORY SECURITY CONFIGURATION
    SECRET_KEY = os.environ.get("SECRET_KEY")
    API_KEY = os.environ.get("WOUNDTRACK_API_KEY")

    if not SECRET_KEY or not API_KEY:
        print("\n[CRITICAL ERROR] Missing mandatory security environment variables!")
        print("Please ensure SECRET_KEY and WOUNDTRACK_API_KEY are defined in your .env file.")
        print("Run 'setup.sh' or manually generate them to proceed safely.\n")
        sys.exit(1)

    FLASK_ENV = os.environ.get("FLASK_ENV", "production")

    # BASE_DIR is project root (parent of config folder)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    RESULTS_FOLDER = os.path.join(BASE_DIR, "results_data")
    
    # Secure File Handling
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH_MB", 500)) * 1024 * 1024
    ALLOWED_EXTENSIONS = {'zip', 'mp4', 'avi', 'mov', 'tif', 'tiff', 'png', 'jpg', 'jpeg'}

    @classmethod
    def init_dirs(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.RESULTS_FOLDER, exist_ok=True)