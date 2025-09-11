"""Folder crawler agent using Ollama AI framework."""

from ..agent_framework import Agent, Tool
from ..tools.image_tools import scan_images  # CORRECT IMPORT

def crawler():
    """Create an AI-powered folder crawler agent."""
    return Agent(
        role="Image Directory Crawler",
        goal="Scan directories and locate all supported image files efficiently",
        backstory="You are an expert file system navigator with years of experience in efficiently scanning directories and identifying image files. You understand file systems and can handle various image formats.",
        tools=[Tool("scan_images", scan_images, "Scan directory recursively for supported image files and return JSON with paths")],
        verbose=True,
        allow_delegation=False
    )
