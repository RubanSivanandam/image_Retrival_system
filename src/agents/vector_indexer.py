"""Vector indexer agent for building FAISS indices."""

from crewai import Agent
from ..tools.vector_tools import build_index

def indexer():
    """Create a vector indexer agent."""
    return Agent(
        role="Vector Index Builder",
        goal="Build and manage FAISS vector indices for similarity search",
        backstory="You are an expert in vector databases and similarity search optimization with deep knowledge of FAISS indexing techniques. You specialize in creating efficient, scalable vector indices for fast similarity retrieval.",
        tools=[build_index],
        verbose=True,
        allow_delegation=False
    )
