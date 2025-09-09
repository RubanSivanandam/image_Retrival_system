"""
Build & query FAISS index for vector similarity search.
"""
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import faiss
from crewai.tools import tool
from src.config.settings import *

# Configure logging
log = logging.getLogger(__name__)

@tool("build_index")
def build_index(data_json: str) -> str:
    """
    Build a FAISS index from embeddings and save metadata.
    
    Args:
        data_json (str): JSON string with 'embeddings' and 'metadata' keys
        
    Returns:
        str: JSON string with indexing results
    """
    try:
        # Parse input data
        data = json.loads(data_json)
        
        embeddings = data.get("embeddings", [])
        metadata = data.get("metadata", [])
        
        if not embeddings:
            return json.dumps({"error": "No embeddings provided"})
            
        if not metadata:
            return json.dumps({"error": "No metadata provided"})
            
        if len(embeddings) != len(metadata):
            return json.dumps({"error": "Embeddings and metadata length mismatch"})
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        if vectors.size == 0:
            return json.dumps({"error": "Empty vectors array"})
        
        log.info(f"Building FAISS index with {len(vectors)} vectors of dimension {vectors.shape}")
        
        # Create FAISS index (Inner Product for normalized vectors)
        dimension = vectors.shape
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize vectors (important for cosine similarity with Inner Product)
        faiss.normalize_L2(vectors)
        
        # Add vectors to index
        index.add(vectors)
        
        # Ensure output directories exist
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        META_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index and metadata
        faiss.write_index(index, str(INDEX_FILE))
        META_FILE.write_text(json.dumps(metadata, indent=2))
        
        result = {
            "indexed": int(index.ntotal),
            "dimension": dimension,
            "index_file": str(INDEX_FILE),
            "metadata_file": str(META_FILE),
            "vectors_shape": list(vectors.shape)
        }
        
        log.info(f"Successfully built index with {index.ntotal} vectors")
        return json.dumps(result)
        
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON input: {e}")
        return json.dumps({"error": f"Invalid JSON input: {e}"})
    except Exception as e:
        log.error(f"Error building index: {e}")
        return json.dumps({"error": f"Failed to build index: {e}"})

@tool("similarity_search")
def similarity_search(query_json: str) -> str:
    """
    Perform similarity search using FAISS index.
    
    Args:
        query_json (str): JSON string with 'embedding' and optional 'k' keys
        
    Returns:
        str: JSON string with search results
    """
    try:
        # Parse query
        query_data = json.loads(query_json)
        
        query_embedding = query_data.get("embedding")
        k = query_data.get("k", TOP_K)
        
        if not query_embedding:
            return json.dumps({"error": "No query embedding provided"})
        
        # Check if index exists
        if not INDEX_FILE.exists():
            return json.dumps({"error": "Index file not found. Please build index first."})
            
        if not META_FILE.exists():
            return json.dumps({"error": "Metadata file not found. Please build index first."})
        
        log.info(f"Performing similarity search with k={k}")
        
        # Load index and metadata
        index = faiss.read_index(str(INDEX_FILE))
        metadata = json.loads(META_FILE.read_text())
        
        # Prepare query vector
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search for similar vectors (retrieve more than k to filter by threshold)
        search_k = min(k * 3, index.ntotal)  # Search more to allow for filtering
        distances, indices = index.search(query_vector, search_k)
        
        # Process results
        results = []
        for score, idx in zip(distances, indices):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            if score < SIM_THRESHOLD:  # Filter by similarity threshold
                continue
            
            if idx >= len(metadata):  # Safety check
                continue
                
            # Combine metadata with similarity score
            result_item = metadata[idx].copy()
            result_item["similarity_score"] = float(score)
            results.append(result_item)
            
            if len(results) >= k:  # Stop when we have enough results
                break
        
        search_result = {
            "results": results[:k],
            "total_found": len(results),
            "query_info": {
                "embedding_dim": len(query_embedding),
                "k_requested": k,
                "similarity_threshold": SIM_THRESHOLD
            }
        }
        
        log.info(f"Found {len(results)} similar images")
        return json.dumps(search_result)
        
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON input: {e}")
        return json.dumps({"error": f"Invalid JSON input: {e}"})
    except Exception as e:
        log.error(f"Error during similarity search: {e}")
        return json.dumps({"error": f"Failed to perform similarity search: {e}"})
