from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "Dataset"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

CASCADE_PATH = ASSETS_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = MODELS_DIR / "trainer.yml"
USERS_PATH = DATA_DIR / "users.json"

SAMPLE_SIZE = 200


def ensure_dirs():
    for d in (DATASET_DIR, MODELS_DIR, DATA_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
