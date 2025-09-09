"""Folder crawler agent for scanning image directories."""

from crewai import Agent
from ..tools.image_tools import scan_images

def crawler():
    """Create a folder crawler agent."""
    return Agent(
        role="Image Directory Crawler",
        goal="Scan directories and locate all supported image files",
        backstory="You are an expert file system navigator with years of experience in efficiently scanning directories and identifying image files. You have a keen eye for finding all image formats and organizing file paths systematically.",
        tools=[scan_images],
        verbose=True,
        allow_delegation=False
    )
