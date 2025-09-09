from crewai import Agent
from ..tools.vector_tools import build_index
def indexer(): return Agent(role="Indexer", goal="Index vectors", tools=[build_index], verbose=True)