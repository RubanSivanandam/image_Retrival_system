"""Vector indexer agent using free framework."""

from ..agent_framework import Agent, Tool
from ..tools.vector_tools import build_index

def indexer():
    """Create a vector indexer agent."""
    return Agent(
        role="Vector Index Builder",
        goal="Build and manage FAISS vector indices for similarity search", 
        backstory="You are an expert in vector databases and similarity search optimization with deep knowledge of FAISS indexing techniques.",
        tools=[Tool("build_index", build_index, "Build FAISS index from embeddings")],
        verbose=True,
        allow_delegation=False
    )
