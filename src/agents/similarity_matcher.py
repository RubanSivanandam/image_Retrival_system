"""Similarity matcher agent using Ollama AI framework."""

from ..agent_framework import Agent, Tool
from ..tools.vector_tools import similarity_search  # CORRECT IMPORT

def matcher():
    """Create an AI-powered similarity matcher agent."""
    return Agent(
        role="Image Similarity Matcher",
        goal="Find and retrieve the most similar images based on query embeddings with high precision",
        backstory="You are a specialist in vector similarity search and image retrieval with expertise in matching visual content based on semantic similarity. You understand how to find the most relevant images that match user queries.",
        tools=[Tool("similarity_search", similarity_search, "Search FAISS index for similar images and return ranked results")],
        verbose=True,
        allow_delegation=False
    )
