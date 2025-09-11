"""
Turn free-text query into CLIP text embedding + heuristic tags.
Bulletproof implementation with fallbacks.
"""
import json
import re
import logging
from typing import Optional
import numpy as np

# Configure logging
log = logging.getLogger(__name__)

# Global model variables (lazy loading)
clip_model: Optional = None
clip_processor: Optional = None

# Pattern matching for extracting features from queries
COLORS = r"(black|white|red|blue|green|yellow|pink|purple|orange|brown|gray|grey|beige|navy|maroon|teal|olive|cyan|magenta|silver|gold)"
SEASONS = r"(winter|summer|spring|fall|autumn)"
MATERIALS = r"(cotton|silk|wool|leather|denim|polyester|linen|cashmere|velvet|satin)"  
STYLES = r"(casual|formal|vintage|modern|bohemian|classic|trendy|elegant|sporty|business)"

def _initialize_models():
    """Initialize CLIP models with robust error handling."""
    global clip_model, clip_processor
    
    if clip_model is None or clip_processor is None:
        try:
            print("🔄 Loading CLIP model for text processing...")
            
            # Try importing required libraries
            import torch
            from transformers import CLIPProcessor, CLIPModel
            from src.config.settings import CLIP_MODEL
            
            # Load models
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            
            print("✅ CLIP models loaded successfully")
            return True
            
        except ImportError as e:
            print(f"❌ Missing dependencies: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to load CLIP models: {e}")
            return False
    return True

def _generate_dummy_embedding(text: str, dim: int = 512) -> list:
    """Generate a deterministic dummy embedding for fallback."""
    # Create a deterministic embedding based on text hash
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to numbers
    embedding = []
    for i in range(0, min(len(text_hash), dim//16)):
        chunk = text_hash[i:i+2]
        try:
            val = int(chunk, 16) / 255.0  # Normalize to 0-1
            embedding.extend([val] * 16)  # Repeat to fill dimension
        except:
            embedding.extend([0.5] * 16)
    
    # Fill remaining dimensions
    while len(embedding) < dim:
        embedding.append(0.5)
    
    # Normalize
    norm = sum(x**2 for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x/norm for x in embedding]
    
    return embedding[:dim]

def parse_query(text: str) -> str:
    """
    Convert a free-text query into a CLIP text embedding and extract semantic tags.
    
    Args:
        text (str): The input text query to process
        
    Returns:
        str: JSON string containing the query, embedding, and extracted features
    """
    try:
        # Clean and normalize text
        cleaned_text = text.strip().lower()
        if not cleaned_text:
            return json.dumps({"error": "Empty query text"})

        print(f"🔍 Processing query: '{text}'")

        # Extract semantic features using regex patterns
        extracted_features = {
            "colors": list(set(re.findall(COLORS, cleaned_text, re.IGNORECASE))),
            "seasons": list(set(re.findall(SEASONS, cleaned_text, re.IGNORECASE))),
            "materials": list(set(re.findall(MATERIALS, cleaned_text, re.IGNORECASE))),
            "styles": list(set(re.findall(STYLES, cleaned_text, re.IGNORECASE)))
        }

        # Try to generate CLIP embedding
        embedding = None
        if _initialize_models():
            try:
                import torch
                inputs = clip_processor(text=[text], return_tensors="pt")
                
                with torch.no_grad():
                    embedding_tensor = clip_model.get_text_features(**inputs)
                    embedding_tensor = embedding_tensor / embedding_tensor.norm(dim=-1, keepdim=True)
                    embedding = embedding_tensor.squeeze().numpy().tolist()
                    
                print("✅ Generated CLIP embedding")
                
            except Exception as e:
                print(f"⚠️  CLIP embedding failed: {e}")
                embedding = None
        
        # Fallback to dummy embedding if CLIP fails
        if embedding is None:
            embedding = _generate_dummy_embedding(text)
            print("⚠️  Using deterministic fallback embedding")

        result = {
            "query": text,
            "query_normalized": cleaned_text,
            "embedding": embedding,
            "features": extracted_features,
            "embedding_dim": len(embedding),
            "embedding_type": "clip" if clip_model is not None else "fallback"
        }

        print(f"✅ Query processed successfully. Features: {extracted_features}")
        return json.dumps(result)

    except Exception as e:
        print(f"❌ Error processing query '{text}': {e}")
        # Even in error case, return a valid embedding for system continuity
        fallback_embedding = _generate_dummy_embedding(text if text else "error")
        return json.dumps({
            "query": text,
            "embedding": fallback_embedding,
            "features": {},
            "embedding_dim": len(fallback_embedding),
            "embedding_type": "error_fallback",
            "error_note": f"Processing failed: {str(e)}"
        })