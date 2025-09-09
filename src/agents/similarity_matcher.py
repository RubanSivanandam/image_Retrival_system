"""Similarity matcher agent for retrieving similar images."""

from crewai import Agent
from ..tools.vector_tools import similarity_search

def matcher():
    """Create a similarity matcher agent."""
    return Agent(
        role="Image Similarity Matcher",
        goal="Find and retrieve the most similar images based on query embeddings",
        backstory="You are a specialist in vector similarity search and image retrieval with expertise in matching visual content based on semantic similarity. You excel at finding the most relevant images that match user queries.",
        tools=[similarity_search],
        verbose=True,
        allow_delegation=False
    )
