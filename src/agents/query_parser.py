"""Query parser agent using Ollama AI framework."""

from ..agent_framework import Agent, Tool
from ..tools.search_tools import parse_query  # CORRECT IMPORT

def parser():
    """Create an AI-powered query parser agent."""
    return Agent(
        role="Natural Language Query Parser",
        goal="Parse and understand natural language search queries with semantic analysis",
        backstory="You are an expert in natural language processing and semantic understanding with years of experience in interpreting user queries and extracting meaningful search intent from text. You excel at understanding context and user intent.",
        tools=[Tool("parse_query", parse_query, "Parse natural language query into CLIP embedding and semantic features")],
        verbose=True,
        allow_delegation=False
    )
