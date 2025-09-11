# Create enhanced src/config/settings.py:

"""
PROFESSIONAL FASHION INDUSTRY CONFIGURATION
Optimized for fashion design teams and industry professionals
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
META_DIR = DATA_DIR / "metadata"

# File paths
INDEX_FILE = EMBEDDINGS_DIR / "fashion_industry_index.faiss"
META_FILE = META_DIR / "fashion_metadata.json"

# Model configuration
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Fashion industry image processing
IMG_MAX_SIZE = (512, 512)
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# FASHION INDUSTRY SEARCH CONFIGURATION
FASHION_PRECISION_DEFAULT = "high"  # exact|high|moderate|exploratory
FASHION_CONFIDENCE_DEFAULT = "high"  # high|medium|low
TOP_K = 20  # Comprehensive candidate pool for fashion filtering
BATCH_SIZE = 16  # Optimized for fashion image processing

# Professional fashion search parameters
FASHION_INDUSTRY_CONFIG = {
    "precision_levels": {
        "exact": {"max_results": 1, "quality_min": "excellent", "threshold": 1.2},
        "high": {"max_results": 2, "quality_min": "very_good", "threshold": 1.0},
        "moderate": {"max_results": 3, "quality_min": "good", "threshold": 0.8},
        "exploratory": {"max_results": 5, "quality_min": "acceptable", "threshold": 0.6}
    },
    "fashion_bonuses": {
        "color_harmony_weight": 0.12,
        "garment_match_weight": 0.18,
        "seasonal_alignment_weight": 0.14,
        "style_consistency_weight": 0.13,
        "fabric_compatibility_weight": 0.11,
        "occasion_suitability_weight": 0.10,
        "sophistication_weight": 0.08,
        "trend_relevance_weight": 0.07,
        "designer_precision_weight": 0.09
    },
    "quality_thresholds": {
        "exceptional": 0.80,
        "excellent": 0.70,
        "very_good": 0.60,
        "good": 0.50,
        "acceptable": 0.40
    },
    "designer_confidence_multipliers": {
        "high": 1.15,
        "medium": 1.0,
        "low": 0.85
    }
}

# Fashion terminology and patterns (abbreviated for settings file)
FASHION_KEYWORDS = {
    "luxury_indicators": ["couture", "designer", "luxury", "premium", "bespoke", "tailored"],
    "seasonal_keywords": ["spring", "summer", "fall", "winter", "resort", "cruise", "pre-fall"],
    "style_keywords": ["classic", "bohemian", "minimalist", "romantic", "edgy", "preppy"],
    "occasion_keywords": ["work", "formal", "casual", "party", "wedding", "business"]
}

# Create directories
for directory in [DATA_DIR, IMAGES_DIR, EMBEDDINGS_DIR, META_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"🎨 Fashion Industry AI System Initialized")
print(f"📊 Precision Level: {FASHION_PRECISION_DEFAULT}")
print(f"🎯 Designer Confidence: {FASHION_CONFIDENCE_DEFAULT}")
print(f"🖼️ Images Directory: {IMAGES_DIR}")