from crewai import Crew, Task, Process  # Import from crewai, not src.crew
from .agents.folder_crawler import crawler
from .agents.image_processor import processor
from .agents.vector_indexer import indexer
from .agents.query_parser import parser
from .agents.similarity_matcher import matcher
from src.config.settings import *

class RetrievalCrew:
    def __init__(self, img_dir: str = str(IMAGES_DIR)):
        self.img_dir = img_dir

    # ---- one-time indexing crew --------------------------------------------
    def build_index(self):
        crawl = Task("Scan directory", crawler(),
                     expected_output="JSON list of image paths",
                     input={"directory": self.img_dir})
        embed = Task("Embed all images", processor(),
                     context=[crawl],
                     expected_output="JSON embeddings + metadata")
        make  = Task("Create FAISS index", indexer(),
                     context=[embed],
                     expected_output="Index built confirmation")
        Crew([crawler(), processor(), indexer()],
             [crawl, embed, make],
             process=Process.sequential,
             verbose=True).kickoff()

    # ---- query crew --------------------------------------------------------
    def search(self, query: str):
        qtask  = Task("Parse query", parser(),
                      input={"text": query},
                      expected_output="JSON with embedding")
        stask  = Task("Similar images", matcher(),
                      context=[qtask],
                      expected_output="JSON results")
        result = Crew([parser(), matcher()],
                      [qtask, stask],
                      process=Process.sequential,
                      verbose=False).kickoff()
        return result
