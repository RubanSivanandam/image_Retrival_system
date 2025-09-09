"""Similarity matcher agent using free framework."""

from ..agent_framework import Agent, Tool
from ..tools.vector_tools import similarity_search

def matcher():
    """Create a similarity matcher agent."""
    return Agent(
        role="Image Similarity Matcher",
        goal="Find and retrieve the most similar images based on query embeddings",
        backstory="You are a specialist in vector similarity search and image retrieval with expertise in matching visual content based on semantic similarity.",
        tools=[Tool("similarity_search", similarity_search, "Search for similar images")],
        verbose=True,
        allow_delegation=False
    )
