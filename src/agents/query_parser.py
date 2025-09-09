"""Query parser agent for processing natural language queries."""

from crewai import Agent
from ..tools.search_tools import parse_query

def parser():
    """Create a query parser agent."""
    return Agent(
        role="Natural Language Query Parser",
        goal="Parse and understand natural language search queries",
        backstory="You are an expert in natural language processing and semantic understanding with years of experience in interpreting user queries and extracting meaningful search intent from text.",
        tools=[parse_query],
        verbose=True,
        allow_delegation=False
    )
