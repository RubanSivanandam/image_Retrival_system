#!/usr/bin/env python3
"""
Main entry point for the Free Open-Source Garment Image Retrieval System
"""
import argparse
import json
import pprint
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from src.crew import RetrievalCrew
    from src.config.settings import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def main():
    """Main function to handle command line arguments and execute crew operations."""
    parser = argparse.ArgumentParser(
        description="Free Open-Source Garment Image Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --index                    # Build the image index
  python main.py "red dress"               # Search for red dress
  python main.py "blue jeans casual"      # Search for blue casual jeans
        """
    )
    
    parser.add_argument("--index", action="store_true", help="Build the image index from the images directory")
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--images-dir", type=str, default=str(IMAGES_DIR), 
                       help=f"Path to images directory (default: {IMAGES_DIR})")
    
    args = parser.parse_args()
    
    try:
        # Initialize the retrieval crew
        crew = RetrievalCrew(img_dir=args.images_dir)
        
        if args.index:
            print("Building image index using free AI agents...")
            result = crew.build_index()
            print("Index building completed!")
            return
            
        if not args.query:
            print("Error: Please provide a search query or use --index to build the index")
            print("Use 'python main.py --help' for more information")
            return
            
        # Perform search
        query_text = " ".join(args.query)
        print(f"Searching for: '{query_text}' using free AI agents...")
        
        result = crew.search(query_text)
        
        # Parse and display results
        try:
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result
                
            if "results" in data:
                print("\n" + "="*50)
                print("SEARCH RESULTS:")
                print("="*50)
                for i, item in enumerate(data["results"], 1):
                    print(f"\n{i}. {item.get('filename', 'Unknown')}")
                    print(f"   Path: {item.get('path', 'Unknown')}")
                    print(f"   Similarity: {item.get('similarity_score', 0):.3f}")
                    if item.get('dimensions'):
                        print(f"   Dimensions: {item['dimensions']}")
                    if item.get('category'):
                        print(f"   Category: {item['category']}")
            else:
                print("Results:")
                pprint.pp(data)
                
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Result parsing error: {e}")
            print("Raw result:")
            print(result)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
