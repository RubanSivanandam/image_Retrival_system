"""
Global settings for the CrewAI garment retrieval system
"""
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Root paths - FIXED to use project root correctly
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Go up from src/config/settings.py to project root
DATA_DIR = PROJECT_ROOT / "data"  # This should be D:\Garment-Image-Retrieval-System\data
IMAGES_DIR = DATA_DIR / "images"
EMB_DIR = DATA_DIR / "embeddings"
META_DIR = DATA_DIR / "metadata"

# Create directories if they don't exist
for directory in (DATA_DIR, IMAGES_DIR, EMB_DIR, META_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified directory: {directory}")

# Image & vector settings
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
BATCH_SIZE = 32
IMG_MAX_SIZE = (512, 512)
EMBED_DIM = 512
TOP_K = 8
SIM_THRESHOLD = 0.25

# Model names
CLIP_MODEL = "openai/clip-vit-base-patch32"
SENTENCE_MODEL = "all-MiniLM-L6-v2"

# Saved artefacts
INDEX_FILE = EMB_DIR / "garment_index.faiss"
META_FILE = META_DIR / "garment_metadata.json"

# Disable CrewAI telemetry
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Yolo model fallback
YOLO_MODEL_URL = "yolov8n.pt"  # Use standard YOLOv8 model instead of custom garment model

# Debug: Print paths to verify they're correct
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"IMAGES_DIR: {IMAGES_DIR}")
