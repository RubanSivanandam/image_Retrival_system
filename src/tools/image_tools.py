"""
Scan directories, generate CLIP embeddings, derive visual metadata.
Handles random filenames by inspecting EXIF first, then YOLOv8 if needed.
"""
import json, os, logging, asyncio
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ExifTags
from sklearn.cluster import KMeans
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO          # tiny garment weights < 10 MB
from crewai.tools import tool

from src.config.settings import *

log = logging.getLogger(__name__)
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
yolo_model     = None                 # lazy-load

# ---------- helpers ---------------------------------------------------------
def _dominant_colours(arr: np.ndarray, k: int = 4) -> List[str]:
    pixels = arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(pixels)
    cent   = kmeans.cluster_centers_.astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cent]

def _yolo_category(path: str) -> str:
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("https://huggingface.co/vision-weights/garments-yolov8n.pt")
    res = yolo_model(path, conf=0.4, verbose=False)[0]
    if res.boxes and res.names:
        cls = int(res.boxes.cls[0])
        return res.names[cls]
    return "unknown"

# ---------- tools -----------------------------------------------------------
@tool("scan_images")
def scan_images(directory: str = str(IMAGES_DIR)) -> str:
    """Return JSON list of all image files under directory."""
    paths = [str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    return json.dumps({"image_paths": paths, "total": len(paths)})

@tool("embed_batch")
def embed_batch(batch_json: str) -> str:
    """
    Accepts JSON with key 'image_paths'.
    Returns embeddings list and rich metadata for each image.
    """
    info  = json.loads(batch_json)
    paths = info["image_paths"]
    embs, metas = [], []

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            inp = clip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                emb = clip_model.get_image_features(**inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            arr = np.array(img)
            meta = {
                "path": p,
                "filename": Path(p).name,
                "dimensions": f"{img.width}x{img.height}",
                "dominant_colours": _dominant_colours(arr),
            }
            # category inference (EXIF or YOLO)
            try:
                exif = img.getexif()
                desc = exif.get(ExifTags.TAGS.get("ImageDescription", 0), "")
            except Exception:
                desc = ""
            meta["category"] = desc.lower() if desc else _yolo_category(p)
            embs.append(emb.squeeze().numpy())
            metas.append(meta)
        except Exception as e:
            log.warning(f"Skip {p}: {e}")

    return json.dumps({"embeddings": np.stack(embs).tolist(), "metadata": metas})
