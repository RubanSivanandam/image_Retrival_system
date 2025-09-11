"""
Scan directories, generate CLIP embeddings, derive visual metadata.
FIXED: YOLO classification and robust error handling
"""
import json
import os
import logging
import asyncio
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image, ExifTags, ImageFile
from sklearn.cluster import KMeans
import torch
from transformers import CLIPProcessor, CLIPModel
from src.config.settings import *

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
log = logging.getLogger(__name__)

# Global model variables (lazy loading)
clip_model: Optional[CLIPModel] = None
clip_processor: Optional[CLIPProcessor] = None
yolo_model = None

def _initialize_models():
    """Initialize CLIP models with error handling."""
    global clip_model, clip_processor
    
    if clip_model is None or clip_processor is None:
        try:
            log.info(f"Loading CLIP model: {CLIP_MODEL}")
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            log.info("CLIP models loaded successfully")
        except Exception as e:
            log.error(f"Failed to load CLIP models: {e}")
            raise RuntimeError(f"Could not initialize CLIP models: {e}")

def _dominant_colours(arr: np.ndarray, k: int = 4) -> List[str]:
    """Extract dominant colors from image array."""
    try:
        pixels = arr.reshape(-1, 3)
        k = min(k, len(pixels), 10)  # Reasonable limit
        if k <= 0:
            return ["#808080"]
            
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]
    except Exception as e:
        log.warning(f"Color extraction failed: {e}")
        return ["#808080"]

def _yolo_category(path: str) -> str:
    """FIXED: Determine image category using YOLO with proper error handling."""
    global yolo_model
    
    try:
        if yolo_model is None:
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(YOLO_MODEL_URL)
            except Exception as e:
                log.warning(f"YOLO model loading failed: {e}")
                return "unknown"
            
        results = yolo_model(path, conf=0.4, verbose=False)
        
        # FIXED: Proper results handling
        if results and len(results) > 0:
            result = results
            # Check if result has boxes and they're not empty
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                cls_idx = int(result.boxes.cls)
                if hasattr(result, 'names') and cls_idx in result.names:
                    return result.names[cls_idx].lower()
                    
        return "unknown"
    except Exception as e:
        log.warning(f"YOLO classification failed for {path}: {e}")
        return "unknown"

def _simple_category_from_filename(path: str) -> str:
    """Simple category detection from filename as fallback."""
    filename = Path(path).stem.lower()
    
    # Garment categories
    if any(word in filename for word in ['dress', 'gown', 'frock']):
        return "dress"
    elif any(word in filename for word in ['shirt', 'top', 'blouse', 'tee']):
        return "shirt"  
    elif any(word in filename for word in ['pants', 'jeans', 'trouser', 'jean']):
        return "pants"
    elif any(word in filename for word in ['skirt']):
        return "skirt"
    elif any(word in filename for word in ['jacket', 'coat', 'blazer']):
        return "jacket"
    elif any(word in filename for word in ['shoe', 'boot', 'sneaker']):
        return "footwear"
    else:
        return "garment"

def scan_images(directory: str = str(IMAGES_DIR)) -> str:
    """
    Scan directory recursively for supported image files.
    
    Args:
        directory (str): Path to directory to scan
        
    Returns:
        str: JSON string containing list of image paths and total count
    """
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return json.dumps({"error": f"Directory does not exist: {directory}"})
            
        if not directory_path.is_dir():
            return json.dumps({"error": f"Path is not a directory: {directory}"})
        
        log.info(f"Scanning directory: {directory}")
        paths = []
        
        for path in directory_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                paths.append(str(path))
        
        log.info(f"Found {len(paths)} image files")
        return json.dumps({
            "image_paths": paths, 
            "total": len(paths),
            "directory": str(directory)
        })
        
    except Exception as e:
        log.error(f"Error scanning directory {directory}: {e}")
        return json.dumps({"error": f"Failed to scan directory: {e}"})

def embed_batch(batch_json: str) -> str:
    """
    Generate CLIP embeddings and metadata for a batch of images.
    
    Args:
        batch_json (str): JSON string with 'image_paths' key
        
    Returns:
        str: JSON string with embeddings and metadata
    """
    try:
        # Initialize models
        _initialize_models()
        
        # Parse input
        info = json.loads(batch_json)
        paths = info.get("image_paths", [])
        
        if not paths:
            return json.dumps({"error": "No image paths provided"})
        
        log.info(f"Processing {len(paths)} images")
        
        embeddings = []
        metadata = []
        processed = 0
        
        for path in paths:
            try:
                # Load and process image
                with Image.open(path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large
                    if img.size > IMG_MAX_SIZE or img.size > IMG_MAX_SIZE:
                        img.thumbnail(IMG_MAX_SIZE, Image.Resampling.LANCZOS)
                    
                    # Generate CLIP embedding
                    inputs = clip_processor(images=img, return_tensors="pt")
                    
                    with torch.no_grad():
                        embedding = clip_model.get_image_features(**inputs)
                        # Normalize embedding
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy and extract metadata
                    arr = np.array(img)
                    
                    meta = {
                        "path": path,
                        "filename": Path(path).name,
                        "dimensions": f"{img.width}x{img.height}",
                        "dominant_colours": _dominant_colours(arr),
                        "size_bytes": Path(path).stat().st_size if Path(path).exists() else 0
                    }
                    
                    # FIXED: Use filename-based category detection first, YOLO as fallback
                    category = _simple_category_from_filename(path)
                    if category == "garment":  # Only use YOLO if filename detection fails
                        category = _yolo_category(path)
                    meta["category"] = category
                    
                    # FIXED: Ensure embedding is a simple list
                    embedding_list = embedding.squeeze().numpy().tolist()
                    
                    embeddings.append(embedding_list)
                    metadata.append(meta)
                    processed += 1
                    
                    if processed % 5 == 0:
                        log.info(f"Processed {processed}/{len(paths)} images")
                        
            except Exception as e:
                log.warning(f"Failed to process {path}: {e}")
                continue
        
        if not embeddings:
            return json.dumps({"error": "No images could be processed"})
        
        log.info(f"Successfully processed {len(embeddings)} images")
        
        # FIXED: Return clean data structure
        return json.dumps({
            "embeddings": embeddings,  # List of lists
            "metadata": metadata,      # List of dicts  
            "processed_count": len(embeddings),
            "total_attempted": len(paths)
        })
        
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON input: {e}")
        return json.dumps({"error": f"Invalid JSON input: {e}"})
    except Exception as e:
        log.error(f"Error processing batch: {e}")
        return json.dumps({"error": f"Failed to process batch: {e}"})
