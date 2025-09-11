"""
Build & query FAISS index for vector similarity search.
FINAL BULLETPROOF FIX: Universal array handling for all FAISS versions
"""
import json
import logging
from pathlib import Path
import numpy as np

# Configure logging
log = logging.getLogger(__name__)

def build_index(data_json: str) -> str:
    """Build a FAISS index from embeddings and save metadata."""
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
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Ensure 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim != 2:
            return json.dumps({"error": f"Invalid embeddings shape: {vectors.shape}"})
        
        if vectors.size == 0:
            return json.dumps({"error": "Empty vectors array"})
        
        # Extract dimensions
        num_vectors, dimension = vectors.shape
        dimension = int(dimension)
        
        log.info(f"Building FAISS index with {num_vectors} vectors of dimension {dimension}")
        print(f"📊 Building index: {num_vectors} vectors × {dimension} dimensions")
        
        # Create FAISS index
        try:
            index = faiss.IndexFlatIP(dimension)
        except Exception as e:
            error_msg = f"Failed to create FAISS index: {e}"
            log.error(error_msg)
            return json.dumps({"error": error_msg})
        
        # Normalize and add vectors
        faiss.normalize_L2(vectors)
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
    FINAL BULLETPROOF FIX: Universal similarity search that handles all array formats.
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
        
        # Get settings
        try:
            from src.config.settings import INDEX_FILE, META_FILE, SIM_THRESHOLD
        except ImportError as e:
            return json.dumps({"error": f"Cannot import settings: {e}"})
        
        # Check if files exist
        if not INDEX_FILE.exists():
            return json.dumps({"error": "Index file not found. Please build index first with --index"})
        if not META_FILE.exists():
            return json.dumps({"error": "Metadata file not found. Please build index first with --index"})
        
        print(f"📋 Searching for top {k} similar images...")
        
        # Load index and metadata
        try:
            index = faiss.read_index(str(INDEX_FILE))
            metadata = json.loads(META_FILE.read_text())
            print(f"📊 Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")
        except Exception as e:
            return json.dumps({"error": f"Cannot load index/metadata: {e}"})
        
        # Prepare query vector
        try:
            query_vector = np.array(query_embedding, dtype=np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            elif query_vector.ndim != 2:
                return json.dumps({"error": f"Invalid query embedding shape: {query_vector.shape}"})
            
            faiss.normalize_L2(query_vector)
            print(f"🎯 Query vector shape: {query_vector.shape}")
        except Exception as e:
            return json.dumps({"error": f"Query vector preparation failed: {e}"})
        
        # Perform search
        search_k = min(k * 2, index.ntotal)
        try:
            distances, indices = index.search(query_vector, search_k)
            print(f"🔍 FAISS search returned distances shape: {distances.shape}, indices shape: {indices.shape}")
        except Exception as e:
            return json.dumps({"error": f"FAISS search failed: {e}"})
        
        # BULLETPROOF ARRAY PROCESSING
        results = []
        try:
            # Convert FAISS results to simple Python lists
            # FAISS always returns 2D arrays: (num_queries, k) - we have 1 query
            distances_flat = distances.flatten().tolist()  # Convert to flat Python list
            indices_flat = indices.flatten().tolist()      # Convert to flat Python list
            
            print(f"🐛 Debug - Converted to lists: {len(distances_flat)} distances, {len(indices_flat)} indices")
            
            # Process each result with guaranteed Python scalars
            for i, (score, idx) in enumerate(zip(distances_flat, indices_flat)):
                print(f"🐛 Debug - Result {i}: score={score} (type: {type(score)}), idx={idx} (type: {type(idx)})")
                
                # These are now guaranteed Python float/int
                try:
                    score = float(score)
                    idx = int(idx)
                except (ValueError, TypeError) as e:
                    log.warning(f"Conversion failed for result {i}: {e}")
                    continue
                
                # Skip invalid indices
                if idx == -1:
                    print(f"🐛 Debug - Skipping invalid index: {idx}")
                    continue
                
                # Skip low similarity scores  
                similarity_threshold = float(SIM_THRESHOLD)
                print(f"🐛 Debug - Comparing score {score} with threshold {similarity_threshold}")
                if score < similarity_threshold:
                    print(f"🐛 Debug - Skipping low score: {score} < {similarity_threshold}")
                    continue
                
                # Safety check for metadata bounds
                if idx >= len(metadata):
                    print(f"🐛 Debug - Index out of bounds: {idx} >= {len(metadata)}")
                    continue
                
                # Add to results
                result_item = metadata[idx].copy()
                result_item["similarity_score"] = score
                results.append(result_item)
                
                print(f"✅ Added result {len(results)}: {result_item.get('filename', 'unknown')} (score: {score})")
                
                # Stop when we have enough
                if len(results) >= k:
                    break
            
        except Exception as e:
            log.error(f"Error in result processing: {e}")
            return json.dumps({"error": f"Results processing failed: {e}"})
        
        # If no results due to threshold, try with lower threshold
        if len(results) == 0:
            print("⚠️  No results found with current threshold, trying with lower threshold...")
            try:
                # Retry with much lower threshold
                for i, (score, idx) in enumerate(zip(distances_flat, indices_flat)):
                    score = float(score)
                    idx = int(idx)
                    
                    if idx == -1 or idx >= len(metadata):
                        continue
                    
                    # Use very low threshold (0.1) to get some results
                    if score >= 0.1:
                        result_item = metadata[idx].copy()
                        result_item["similarity_score"] = score
                        results.append(result_item)
                        
                        if len(results) >= k:
                            break
                            
                print(f"🔍 Found {len(results)} results with lower threshold")
            except Exception as e:
                log.warning(f"Retry with lower threshold failed: {e}")
        
        # Prepare final result
        search_result = {
            "results": results[:k],
            "total_found": len(results),
            "query_info": {
                "embedding_dim": len(query_embedding),
                "k_requested": k,
                "similarity_threshold": float(SIM_THRESHOLD),
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
