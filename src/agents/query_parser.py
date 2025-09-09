"""Query parser agent using free framework."""

from ..agent_framework import Agent, Tool
from ..tools.search_tools import parse_query

def parser():
    """Create a query parser agent."""
    return Agent(
        role="Natural Language Query Parser",
        goal="Parse and understand natural language search queries",
        backstory="You are an expert in natural language processing and semantic understanding with years of experience in interpreting user queries.",
        tools=[Tool("parse_query", parse_query, "Parse natural language queries")],
        verbose=True,
        allow_delegation=False
    )
