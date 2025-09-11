"""
Build & query FAISS index for vector similarity search.
FIXED: Proper dimension handling and FAISS index creation
"""
import json
import logging
from pathlib import Path
import numpy as np

# Configure logging
log = logging.getLogger(__name__)

def build_index(data_json: str) -> str:
    """
    Build a FAISS index from embeddings and save metadata.
    
    Returns:
        str: JSON string with indexing results
    """
    try:
        print("🔄 Building FAISS index...")
        log.info("Starting FAISS index building process")
        
        # Import FAISS
        try:
            import faiss
        except ImportError:
            error_msg = "FAISS not installed. Run: pip install faiss-cpu"
            log.error(error_msg)
            return json.dumps({"error": error_msg})
        
        # Parse input data
        try:
            data = json.loads(data_json)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON input: {e}"
            log.error(error_msg)
            return json.dumps({"error": error_msg})
        
        embeddings = data.get("embeddings", [])
        metadata = data.get("metadata", [])
        
        if not embeddings:
            return json.dumps({"error": "No embeddings provided"})
            
        if not metadata:
            return json.dumps({"error": "No metadata provided"})
            
        if len(embeddings) != len(metadata):
            return json.dumps({"error": "Embeddings and metadata length mismatch"})
        
        # FIXED: Proper numpy array conversion and dimension handling
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Ensure 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim != 2:
            return json.dumps({"error": f"Invalid embeddings shape: {vectors.shape}"})
        
        if vectors.size == 0:
            return json.dumps({"error": "Empty vectors array"})
        
        # FIXED: Extract dimension properly as integer
        num_vectors, dimension = vectors.shape
        dimension = int(dimension)  # Ensure it's an integer
        
        log.info(f"Building FAISS index with {num_vectors} vectors of dimension {dimension}")
        print(f"📊 Building index: {num_vectors} vectors × {dimension} dimensions")
        
        # FIXED: Create FAISS index with proper integer dimension
        try:
            index = faiss.IndexFlatIP(dimension)  # dimension is now guaranteed to be int
        except Exception as e:
            error_msg = f"Failed to create FAISS index: {e}"
            log.error(error_msg)
            return json.dumps({"error": error_msg})
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Add vectors to index
        try:
            index.add(vectors)
        except Exception as e:
            error_msg = f"Failed to add vectors to index: {e}"
            log.error(error_msg)
            return json.dumps({"error": error_msg})
        
        # Get paths from settings
        try:
            from src.config.settings import INDEX_FILE, META_FILE
        except ImportError as e:
            error_msg = f"Cannot import settings: {e}"
            log.error(error_msg)
            return json.dumps({"error": error_msg})
        
        # Ensure output directories exist
        try:
            INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
            META_FILE.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            error_msg = f"Cannot create directories: {e}"
            log.error(error_msg)
            return json.dumps({"error": error_msg})
        
        # Save index and metadata
        try:
            faiss.write_index(index, str(INDEX_FILE))
            META_FILE.write_text(json.dumps(metadata, indent=2))
        except Exception as e:
            error_msg = f"Cannot save index files: {e}"
            log.error(error_msg)
            return json.dumps({"error": error_msg})
        
        result = {
            "indexed": int(index.ntotal),
            "dimension": dimension,
            "num_vectors": num_vectors,
            "index_file": str(INDEX_FILE),
            "metadata_file": str(META_FILE)
        }
        
        log.info(f"Successfully built index with {index.ntotal} vectors")
        print(f"✅ Successfully built index with {index.ntotal} vectors")
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Failed to build index: {e}"
        log.error(error_msg)
        print(f"❌ Index building error: {error_msg}")
        return json.dumps({"error": error_msg})

def similarity_search(query_json: str) -> str:
    """
    Perform similarity search using FAISS index.
    
    Returns:
        str: JSON string with search results
    """
    try:
        print("🔍 Performing similarity search...")
        log.info("Starting similarity search")
        
        # Import FAISS
        try:
            import faiss
        except ImportError:
            return json.dumps({"error": "FAISS not installed. Run: pip install faiss-cpu"})
        
        # Parse query
        try:
            query_data = json.loads(query_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON input: {e}"})
        
        query_embedding = query_data.get("embedding")
        k = query_data.get("k", 5)
        
        if not query_embedding:
            return json.dumps({"error": "No query embedding provided"})
        
        # Get paths from settings
        try:
            from src.config.settings import INDEX_FILE, META_FILE, SIM_THRESHOLD
        except ImportError as e:
            return json.dumps({"error": f"Cannot import settings: {e}"})
        
        # Check if index exists
        if not INDEX_FILE.exists():
            return json.dumps({"error": "Index file not found. Please build index first with --index"})
            
        if not META_FILE.exists():
            return json.dumps({"error": "Metadata file not found. Please build index first with --index"})
        
        print(f"📋 Searching for top {k} similar images...")
        
        # Load index and metadata
        try:
            index = faiss.read_index(str(INDEX_FILE))
            metadata = json.loads(META_FILE.read_text())
        except Exception as e:
            return json.dumps({"error": f"Cannot load index/metadata: {e}"})
        
        # Prepare query vector
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search for similar vectors
        search_k = min(k * 2, index.ntotal)
        try:
            distances, indices = index.search(query_vector, search_k)
        except Exception as e:
            return json.dumps({"error": f"Search failed: {e}"})
        
        # Process results
        results = []
        for score, idx in zip(distances, indices):
            if idx == -1 or idx >= len(metadata):
                continue
                
            if score < SIM_THRESHOLD:
                continue
            
            result_item = metadata[idx].copy()
            result_item["similarity_score"] = float(score)
            results.append(result_item)
            
            if len(results) >= k:
                break
        
        search_result = {
            "results": results[:k],
            "total_found": len(results),
            "query_info": {
                "embedding_dim": len(query_embedding),
                "k_requested": k,
                "similarity_threshold": SIM_THRESHOLD,
                "query_text": query_data.get("query", "unknown")
            }
        }
        
        log.info(f"Found {len(results)} similar images")
        print(f"✅ Found {len(results)} similar images")
        return json.dumps(search_result)
        
    except Exception as e:
        error_msg = f"Failed to perform similarity search: {e}"
        log.error(error_msg)
        return json.dumps({"error": error_msg})
