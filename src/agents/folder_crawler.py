"""Folder crawler agent using free framework."""

from ..agent_framework import Agent, Tool
from ..tools.image_tools import scan_images

def crawler():
    """Create a folder crawler agent."""
    return Agent(
        role="Image Directory Crawler",
        goal="Scan directories and locate all supported image files",
        backstory="You are an expert file system navigator with years of experience in efficiently scanning directories and identifying image files.",
        tools=[Tool("scan_images", scan_images, "Scan directory for image files")],
        verbose=True,
        allow_delegation=False
    )
