#!/usr/bin/env python3
"""
AI-Powered Garment Image Retrieval System using Ollama
100% Free and Local with Real AI Reasoning
"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def check_ollama():
    """Check if Ollama is available."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🤖 AI-Powered Garment Image Retrieval System")
    print("Powered by Ollama - Running in Local")
    print("=" * 50)
    
    # Check Ollama availability
    if check_ollama():
        print("✅ Ollama is running - Using AI reasoning")
    else:
        print("⚠️  Ollama not detected - Using rule-based fallback")
        print("💡 To use AI features, run: ollama serve")
    
    parser = argparse.ArgumentParser(
        description="AI-Powered Garment Image Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --index                    # Build image index with AI
  python main.py "red dress"               # AI-powered search
  python main.py "blue jeans casual"      # Smart semantic search
        """
    )
    
    parser.add_argument("--index", action="store_true", help="Build the image index")
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--images-dir", type=str, help="Path to images directory")
    
    args = parser.parse_args()
    
    try:
        from src.crew import RetrievalCrew
        from src.config.settings import IMAGES_DIR
        
        images_dir = args.images_dir if args.images_dir else str(IMAGES_DIR)
        crew = RetrievalCrew(img_dir=images_dir)
        
        if args.index:
            print("\n📚 Building image index with AI agents...")
            print("-" * 50)
            result = crew.build_index()
            
            try:
                if isinstance(result, str):
                    data = json.loads(result)
                else:
                    data = result
                    
                if "error" in data:
                    print(f"❌ Index building failed: {data['error']}")
                    return 1
                else:
                    print("\n✅ AI-powered index building completed!")
                    if "indexed" in data:
                        print(f"🧠 Indexed {data['indexed']} images using AI")
                    return 0
            except:
                print("✅ Index building completed!")
                return 0
        
        if not args.query:
            print("\n💡 Usage:")
            print("  Build index: python main.py --index")
            print("  AI Search:   python main.py \"your search query\"")
            return 0
        
        query_text = " ".join(args.query)
        print(f"\n🔍 AI-powered search for: '{query_text}'")
        print("-" * 50)
        
        result = crew.search(query_text)
        
        try:
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result
                
            if "error" in data:
                print(f"\n❌ Search failed: {data['error']}")
                if "index" in data["error"].lower():
                    print("💡 Try: python main.py --index")
                return 1
                
            if "results" in data and data["results"]:
                print(f"\n🎯 AI SEARCH RESULTS:")
                print("=" * 50)
                
                for i, item in enumerate(data["results"], 1):
                    print(f"\n{i}. 📷 {item.get('filename', 'Unknown')}")
                    print(f"   📊 Similarity: {item.get('similarity_score', 0):.3f}")
                    print(f"   📍 {item.get('path', 'Unknown')}")
                    if item.get('category'):
                        print(f"   🏷️  Category: {item['category']}")
                    if item.get('dominant_colours'):
                        colors = item['dominant_colours'][:3]
                        print(f"   🎨 Colors: {', '.join(colors)}")
                
                total = data.get("total_found", len(data["results"]))
                print(f"\n🧠 AI found {total} matches")
            else:
                print("\n🤷 No results found.")
                print("💡 Try different search terms or rebuild index")
                
        except Exception as e:
            print(f"\n❌ Error parsing results: {e}")
            return 1
            
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Install: pip install -r requirements.txt")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
