"""Image processor agent using free framework."""

from ..agent_framework import Agent, Tool
from ..tools.image_tools import embed_batch

def processor():
    """Create an image processor agent."""
    return Agent(
        role="Image Embedding Processor", 
        goal="Generate CLIP embeddings and extract metadata from images",
        backstory="You are a specialist in computer vision and deep learning with extensive experience in image processing and feature extraction.",
        tools=[Tool("embed_batch", embed_batch, "Generate embeddings for image batch")],
        verbose=True,
        allow_delegation=False
    )
