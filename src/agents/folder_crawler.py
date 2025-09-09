from crewai import Agent
from ..tools.image_tools import scan_images
def crawler(): return Agent(role="Crawler", goal="List images", tools=[scan_images], verbose=True)