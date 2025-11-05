import os


class Config:
    """Flask configuration variables."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-very-secret-key-that-you-should-change'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024

    # Define application folders
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    DB_FOLDER = 'db'  # --- NEW ---

    # Define database file path
    DATABASE_URL = os.path.join(DB_FOLDER, 'analysis.db')  # --- NEW ---

    # Ensure all directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(DB_FOLDER, exist_ok=True)  # --- NEW ---