from crewai import Agent
from ..tools import parse_query


def parser():
    return Agent(
        role="NLP",
        goal="Understand query",
        backstory="Expert in natural language processing and semantic understanding.",
        tools=[parse_query],
        verbose=True
    )