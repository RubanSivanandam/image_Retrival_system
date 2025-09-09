"""
Build & query FAISS index.
"""
import json, logging
import numpy as np, faiss
from crewai.tools import tool
from src.config.settings import *

log = logging.getLogger(__name__)

@tool("build_index")
def build_index(data_json: str) -> str:
    """
    Builds a FAISS index from a JSON string containing embeddings and metadata.

    Args:
        data_json (str): JSON string with 'embeddings' (list of vectors) and 'metadata' (list of metadata).

    Returns:
        str: JSON string with the number of indexed vectors or an error message.
    """
    d = json.loads(data_json)
    vecs = np.array(d["embeddings"], dtype=np.float32)
    meta = d["metadata"]
    if vecs.size == 0:
        return json.dumps({"error": "no vectors"})
    idx = faiss.IndexFlatIP(vecs.shape[1])
    faiss.normalize_L2(vecs)
    idx.add(vecs)
    faiss.write_index(idx, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(meta, indent=2))
    return json.dumps({"indexed": int(idx.ntotal)})

@tool("similarity_search")
def similarity_search(query_json: str) -> str:
    """
    Performs a similarity search on a FAISS index using a query embedding.

    Args:
        query_json (str): JSON string with 'embedding' (query vector) and optional 'k' (number of results).

    Returns:
        str: JSON string with search results (metadata and similarity scores) or an error message.
    """
    q = json.loads(query_json)
    query_vec = np.array(q["embedding"], dtype=np.float32).reshape(1, -1)
    k = q.get("k", TOP_K)
    if not INDEX_FILE.exists():
        return json.dumps({"error": "index missing"})
    idx = faiss.read_index(str(INDEX_FILE))
    faiss.normalize_L2(query_vec)
    D, I = idx.search(query_vec, k*2)
    meta = json.loads(META_FILE.read_text())
    results = []
    for score, idx in zip(D[0], I[0]):
        if score < SIM_THRESHOLD: continue
        m = meta[idx] | {"similarity": float(score)}
        results.append(m)
        if len(results) == k: break
    return json.dumps({"results": results[:k]})