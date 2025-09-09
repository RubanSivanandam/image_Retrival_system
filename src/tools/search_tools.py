"""
Turn free-text query into CLIP text embedding + heuristic tags.
"""
import json, re
from crewai.tools import tool
import torch
from transformers import CLIPProcessor, CLIPModel
from src.config.settings import *

clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL)

COLORS  = r"(black|white|red|blue|green|yellow|pink|purple|orange|brown|gray|grey|beige|navy)"
SEASONS = r"(winter|summer|spring|fall|autumn)"

@tool("parse_query")
def parse_query(text: str) -> str:
    """
    Converts a free-text query into a CLIP text embedding and extracts heuristic tags.

    Args:
        text (str): The input text query to process.

    Returns:
        str: JSON string containing the query, its CLIP embedding, and extracted color/season tags.
    """
    inp = clip_proc(text=[text], return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_text_features(**inp)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return json.dumps({
        "query": text,
        "embedding": emb.squeeze().numpy().tolist(),
        "color": re.findall(COLORS, text, re.I),
        "season": re.findall(SEASONS, text, re.I)
    })