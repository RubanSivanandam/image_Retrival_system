"""
Global settings for the CrewAI garment retrieval system
"""
from pathlib import Path
import os

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
IMAGES_DIR   = DATA_DIR / "images"
EMB_DIR      = DATA_DIR / "embeddings"
META_DIR     = DATA_DIR / "metadata"

for p in (DATA_DIR, IMAGES_DIR, EMB_DIR, META_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Image & vector settings
SUPPORTED_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
BATCH_SIZE           = 32
IMG_MAX_SIZE         = (512, 512)
EMBED_DIM            = 512
TOP_K                = 8
SIM_THRESHOLD        = 0.25

# Model names
CLIP_MODEL           = "openai/clip-vit-base-patch32"
SENTENCE_MODEL       = "all-MiniLM-L6-v2"

# Saved artefacts
INDEX_FILE           = EMB_DIR / "garment_index.faiss"
META_FILE            = META_DIR / "garment_metadata.json"

# Disable CrewAI telemetry
os.environ["CREWAI_TELEMETRY"] = "false"
