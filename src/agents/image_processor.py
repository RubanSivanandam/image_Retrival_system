"""Image processor agent using Ollama AI framework."""

from ..agent_framework import Agent, Tool
from ..tools.image_tools import embed_batch  # CORRECT IMPORT

def processor():
    """Create an AI-powered image processor agent."""
    return Agent(
        role="Image Embedding Processor", 
        goal="Generate CLIP embeddings and extract comprehensive metadata from images",
        backstory="You are a specialist in computer vision and deep learning with extensive experience in image processing and feature extraction. You excel at generating high-quality embeddings and extracting meaningful metadata from visual content.",
        tools=[Tool("embed_batch", embed_batch, "Generate CLIP embeddings and metadata for a batch of images")],
        verbose=True,
        allow_delegation=False
    )
