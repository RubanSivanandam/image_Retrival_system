"""
Main CrewAI orchestrator for the image retrieval system.
"""
import logging
from crewai import Crew, Task, Process
from .agents.folder_crawler import crawler
from .agents.image_processor import processor
from .agents.vector_indexer import indexer
from .agents.query_parser import parser
from .agents.similarity_matcher import matcher
from .config.settings import *

# Configure logging
log = logging.getLogger(__name__)

class RetrievalCrew:
    """
    Main orchestrator class for the image retrieval system.
    Manages both index building and search operations.
    """
    
    def __init__(self, img_dir: str = str(IMAGES_DIR)):
        """
        Initialize the retrieval crew.
        
        Args:
            img_dir (str): Path to the images directory
        """
        self.img_dir = img_dir
        log.info(f"Initialized RetrievalCrew with image directory: {img_dir}")
    
    def build_index(self):
        """
        Build the image index using a sequential crew workflow.
        
        Returns:
            str: Result of the index building process
        """
        try:
            log.info("Starting index building process...")
            
            # Create agents
            crawler_agent = crawler()
            processor_agent = processor()
            indexer_agent = indexer()
            
            # Define tasks for index building - FIXED initialization
            crawl_task = Task(
                description=f"Scan the directory {self.img_dir} and find all supported image files. Return a JSON list of image paths with total count.",
                expected_output="JSON list of image paths with total count",
                agent=crawler_agent
            )
            
            embed_task = Task(
                description="Generate CLIP embeddings and extract metadata for all discovered images. Process images in batches for efficiency.",
                expected_output="JSON with embeddings and metadata for all images",
                agent=processor_agent,
                context=[crawl_task]
            )
            
            index_task = Task(
                description="Create a FAISS vector index from the generated embeddings and save it to disk along with metadata.",
                expected_output="Confirmation that the index has been built and saved successfully",
                agent=indexer_agent,
                context=[embed_task]
            )
            
            # Create and execute the indexing crew
            indexing_crew = Crew(
                agents=[crawler_agent, processor_agent, indexer_agent],
                tasks=[crawl_task, embed_task, index_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = indexing_crew.kickoff()
            log.info("Index building completed successfully")
            return result
            
        except Exception as e:
            log.error(f"Error during index building: {e}")
            raise RuntimeError(f"Failed to build index: {e}")
    
    def search(self, query: str):
        """
        Search for images similar to the given query.
        
        Args:
            query (str): Natural language search query
            
        Returns:
            str: JSON string with search results
        """
        try:
            log.info(f"Starting search for query: '{query}'")
            
            if not query.strip():
                raise ValueError("Empty search query")
            
            # Create agents
            parser_agent = parser()
            matcher_agent = matcher()
            
            # Define tasks for search - FIXED initialization
            parse_task = Task(
                description=f"Parse the natural language query '{query}' and convert it to a CLIP text embedding with extracted semantic features.",
                expected_output="JSON with query embedding and extracted features",
                agent=parser_agent
            )
            
            search_task = Task(
                description="Use the query embedding to find the most similar images in the FAISS index. Return ranked results with similarity scores.",
                expected_output="JSON with ranked list of similar images and their metadata",
                agent=matcher_agent,
                context=[parse_task]
            )
            
            # Create and execute the search crew
            search_crew = Crew(
                agents=[parser_agent, matcher_agent],
                tasks=[parse_task, search_task],
                process=Process.sequential,
                verbose=False
            )
            
            result = search_crew.kickoff()
            log.info("Search completed successfully")
            return result
            
        except Exception as e:
            log.error(f"Error during search: {e}")
            raise RuntimeError(f"Failed to perform search: {e}")
