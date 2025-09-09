"""
Scan directories, generate CLIP embeddings, derive visual metadata.
Handles random filenames by inspecting EXIF first, then YOLOv8 if needed.
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
from crewai.tools import tool
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
        # Handle edge case where image has fewer pixels than clusters
        k = min(k, len(pixels))
        if k <= 0:
            return ["#000000"]
            
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]
    except Exception as e:
        log.warning(f"Color extraction failed: {e}")
        return ["#000000"]

def _yolo_category(path: str) -> str:
    """Determine image category using YOLO (fallback implementation)."""
    global yolo_model
    
    try:
        if yolo_model is None:
            from ultralytics import YOLO
            # Use standard YOLOv8 model instead of custom garment model
            yolo_model = YOLO(YOLO_MODEL_URL)
            
        results = yolo_model(path, conf=0.4, verbose=False)
        
        if results and len(results) > 0:
            result = results
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the class with highest confidence
                cls_idx = int(result.boxes.cls)
                if hasattr(result, 'names') and cls_idx in result.names:
                    return result.names[cls_idx].lower()
                    
        return "unknown"
    except Exception as e:
        log.warning(f"YOLO classification failed for {path}: {e}")
        return "unknown"

# --------- tools --------------------------------------------------------

@tool("scan_images")
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

@tool("embed_batch")
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
                    
                    # Try to get category from EXIF first
                    try:
                        exif = img.getexif()
                        if exif:
                            desc = ""
                            for tag_id, value in exif.items():
                                tag = ExifTags.TAGS.get(tag_id, tag_id)
                                if tag == "ImageDescription":
                                    desc = str(value).lower()
                                    break
                            meta["category"] = desc if desc else _yolo_category(path)
                        else:
                            meta["category"] = _yolo_category(path)
                    except Exception:
                        meta["category"] = _yolo_category(path)
                    
                    embeddings.append(embedding.squeeze().numpy())
                    metadata.append(meta)
                    processed += 1
                    
                    if processed % 10 == 0:
                        log.info(f"Processed {processed}/{len(paths)} images")
                        
            except Exception as e:
                log.warning(f"Failed to process {path}: {e}")
                continue
        
        if not embeddings:
            return json.dumps({"error": "No images could be processed"})
        
        # Stack embeddings
        embeddings_array = np.stack(embeddings)
        
        log.info(f"Successfully processed {len(embeddings)} images")
        
        return json.dumps({
            "embeddings": embeddings_array.tolist(),
            "metadata": metadata,
            "processed_count": len(embeddings),
            "total_attempted": len(paths)
        })
        
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON input: {e}")
        return json.dumps({"error": f"Invalid JSON input: {e}"})
    except Exception as e:
        log.error(f"Error processing batch: {e}")
        return json.dumps({"error": f"Failed to process batch: {e}"})
