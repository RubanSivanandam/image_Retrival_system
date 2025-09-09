from crewai import Agent
from ..tools.image_tools import embed_batch
def processor(): return Agent(role="Embedder", goal="Embed images", tools=[embed_batch], verbose=True)