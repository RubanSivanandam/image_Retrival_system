from crewai import Agent
from ..tools.vector_tools import similarity_search
def matcher(): return Agent(role="Matcher", goal="Retrieve similar images", tools=[similarity_search], verbose=True)