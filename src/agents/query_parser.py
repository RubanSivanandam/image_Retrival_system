from crewai import Agent
from ..tools.search_tools import parse_query
def parser(): return Agent(role="NLP", goal="Understand query", tools=[parse_query], verbose=True)