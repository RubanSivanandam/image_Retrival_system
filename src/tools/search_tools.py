"""
Turn free-text query into CLIP text embedding + heuristic tags.
"""
import json
import re
import logging
from typing import Optional
import torch
from transformers import CLIPProcessor, CLIPModel
from src.config.settings import *

# Configure logging
log = logging.getLogger(__name__)

# Global model variables (lazy loading)
clip_model: Optional[CLIPModel] = None
clip_processor: Optional[CLIPProcessor] = None

# Pattern matching for extracting features from queries
COLORS = r"(black|white|red|blue|green|yellow|pink|purple|orange|brown|gray|grey|beige|navy|maroon|teal|olive|cyan|magenta|silver|gold)"
SEASONS = r"(winter|summer|spring|fall|autumn)"
MATERIALS = r"(cotton|silk|wool|leather|denim|polyester|linen|cashmere|velvet|satin)"
STYLES = r"(casual|formal|vintage|modern|bohemian|classic|trendy|elegant|sporty|business)"

def _initialize_models():
    global clip_model, clip_processor
    if clip_model is None or clip_processor is None:
        try:
            log.info(f"Loading CLIP model for text processing: {CLIP_MODEL}")
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            log.info("CLIP text models loaded successfully")
        except Exception as e:
            log.error(f"Failed to load CLIP models: {e}")
            raise RuntimeError(f"Could not initialize CLIP models: {e}")

def parse_query(text: str) -> str:
    """
    Convert a free-text query into a CLIP text embedding and extract semantic tags.
    Args:
        text (str): The input text query to process
    Returns:
        str: JSON string containing the query, embedding, and extracted features
    """
    try:
        # Initialize models if needed
        _initialize_models()

        # Clean and normalize text
        cleaned_text = text.strip().lower()
        if not cleaned_text:
            return json.dumps({"error": "Empty query text. Cannot generate embedding."})

        log.info(f"Processing query: '{text}'")

        # Generate CLIP text embedding
        inputs = clip_processor(text=[text], return_tensors="pt")

        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        # Extract semantic features using regex patterns
        extracted_features = {
            "colors": re.findall(COLORS, cleaned_text, re.IGNORECASE),
            "seasons": re.findall(SEASONS, cleaned_text, re.IGNORECASE),  
            "materials": re.findall(MATERIALS, cleaned_text, re.IGNORECASE),
            "styles": re.findall(STYLES, cleaned_text, re.IGNORECASE)
        }

        # Remove duplicates while preserving order
        for key in extracted_features:
            extracted_features[key] = list(dict.fromkeys(extracted_features[key]))

        result = {
            "query": text,
            "query_normalized": cleaned_text,
            "embedding": embedding.squeeze().numpy().tolist(),  # <-- This is ESSENTIAL
            "features": extracted_features,
            "embedding_dim": embedding.shape[-1]
        }

        log.info(f"Successfully processed query. Found features: {extracted_features}")
        return json.dumps(result)

    except Exception as e:
        log.error(f"Error processing query '{text}': {e}")
        return json.dumps({"error": f"Failed to process query: {str(e)}"})
