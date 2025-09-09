"""Image processor agent for generating embeddings."""

from crewai import Agent
from ..tools.image_tools import embed_batch

def processor():
    """Create an image processor agent."""
    return Agent(
        role="Image Embedding Processor",
        goal="Generate CLIP embeddings and extract metadata from images",
        backstory="You are a specialist in computer vision and deep learning with extensive experience in image processing and feature extraction. You excel at generating high-quality embeddings and extracting meaningful metadata from visual content.",
        tools=[embed_batch],
        verbose=True,
        allow_delegation=False
    )
