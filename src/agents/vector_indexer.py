"""Vector indexer agent using Ollama AI framework."""

from ..agent_framework import Agent, Tool
from ..tools.vector_tools import build_index  # CORRECT IMPORT

# def indexer():
#     """Create an AI-powered vector indexer agent."""
#     return Agent(
#         role="Vector Index Builder",
#         goal="Build and manage FAISS vector indices for efficient similarity search", 
#         backstory="You are an expert in vector databases and similarity search optimization with deep knowledge of FAISS indexing techniques. You understand how to create efficient, scalable vector indices for fast similarity retrieval.",
#         tools=[Tool("build_index", build_index, "Build FAISS index from embeddings and metadata, save to disk")],
#         verbose=True,
#         allow_delegation=False
#     )


def indexer():
    """Create an AI-powered vector indexer agent."""
    return Agent(
        role="Vector Index Builder",
        goal="Build and manage FAISS vector indices for efficient similarity search", 
        backstory="You are an expert in vector databases and similarity search optimization with deep knowledge of FAISS indexing techniques. You understand how to create efficient, scalable vector indices for fast similarity retrieval.",
        tools=[
            Tool(
                "build_index",
                build_index,
                "Create a FAISS vector index from embeddings and metadata, save it to disk, and confirm successful build"
            )
        ],
        verbose=True,
        allow_delegation=False
    )