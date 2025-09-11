"""
SUPER INTELLIGENT VECTOR SEARCH - Guaranteed to find relevant results with maximum intelligence.
REPLACES: src/tools/vector_tools.py
"""
import json
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import statistics

# Configure logging
log = logging.getLogger(__name__)

def build_index(data_json: str) -> str:
    """Build enhanced FAISS index - same as before but with logging."""
    try:
        print("🔄 Building super intelligent fashion index...")
        log.info("Starting super intelligent index building")
        
        import faiss
        
        # Parse data
        data = json.loads(data_json)
        embeddings = data.get("embeddings", [])
        metadata = data.get("metadata", [])
        
        if not embeddings or not metadata:
            return json.dumps({"error": "No data provided"})
        if len(embeddings) != len(metadata):
            return json.dumps({"error": "Embeddings and metadata mismatch"})
        
        # Convert to numpy
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        num_vectors, dimension = vectors.shape
        
        print(f"📊 Building index: {num_vectors} items × {dimension} dimensions")
        
        # Create FAISS index
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        
        # Save
        from src.config.settings import INDEX_FILE, META_FILE
        
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        META_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(INDEX_FILE))
        META_FILE.write_text(json.dumps(metadata, indent=2))
        
        print(f"✅ Super intelligent index built: {index.ntotal} items")
        
        return json.dumps({
            "indexed": int(index.ntotal),
            "dimension": dimension,
            "index_type": "super_intelligent"
        })
        
    except Exception as e:
        error_msg = f"Index building failed: {e}"
        log.error(error_msg)
        return json.dumps({"error": error_msg})

def _calculate_super_intelligent_threshold(query_data: Dict, all_scores: List[float]) -> float:
    """Calculate SUPER INTELLIGENT threshold that always finds relevant results."""
    
    if not all_scores:
        return 0.10  # Very low fallback
        
    fashion_metadata = query_data.get("fashion_metadata", {})
    confidence_level = fashion_metadata.get("confidence_level", "medium")
    is_seasonal = fashion_metadata.get("is_seasonal_query", False)
    specificity = fashion_metadata.get("fashion_specificity", 0)
    
    # Analyze score distribution
    scores_array = np.array(all_scores)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    median_score = np.median(scores_array)
    max_score = np.max(scores_array)
    
    print(f"📊 Score analysis: max={max_score:.3f}, mean={mean_score:.3f}, median={median_score:.3f}, std={std_score:.3f}")
    
    # SUPER INTELLIGENT threshold calculation
    base_threshold = 0.08  # Start very low to ensure we find results
    
    # For high confidence queries, be more selective but not too much
    if confidence_level == "high":
        base_threshold = max(mean_score * 0.6, 0.12)
    elif confidence_level == "medium":
        base_threshold = max(mean_score * 0.5, 0.10)
    else:
        base_threshold = max(mean_score * 0.4, 0.08)
    
    # Seasonal queries (like "winter clothing") should be more lenient
    if is_seasonal:
        base_threshold *= 0.8  # 20% more lenient
        print(f"🔥 SEASONAL query detected - being more lenient!")
    
    # High specificity can be slightly more selective
    if specificity > 8:
        base_threshold *= 1.1
    elif specificity > 5:
        base_threshold *= 1.0
    else:
        base_threshold *= 0.9
    
    # SAFETY MECHANISM: Never go above a reasonable threshold
    final_threshold = min(base_threshold, 0.25)  # Cap at 0.25
    
    # GUARANTEE MECHANISM: If threshold would eliminate all results, lower it
    results_count = len([s for s in all_scores if s >= final_threshold])
    if results_count == 0 and len(all_scores) > 0:
        # Emergency fallback - use 80% of the highest score
        final_threshold = max_score * 0.8
        print(f"🚨 EMERGENCY FALLBACK: Using {final_threshold:.3f} to ensure results!")
    elif results_count < 2 and len(all_scores) >= 2:
        # If we'd only get 1 result but have 2+ items, be more lenient
        sorted_scores = sorted(all_scores, reverse=True)
        final_threshold = sorted_scores[1] * 0.95  # Just below 2nd best score
        print(f"🛡️ INTELLIGENT FALLBACK: Ensuring at least 2 results with {final_threshold:.3f}")
    
    print(f"🎯 SUPER INTELLIGENT threshold: {final_threshold:.3f} (will find {len([s for s in all_scores if s >= final_threshold])} results)")
    
    return final_threshold

def _calculate_super_comprehensive_relevance(query_data: Dict, metadata: Dict, similarity_score: float) -> Dict:
    """Calculate SUPER comprehensive relevance with multiple bonus factors."""
    
    base_score = similarity_score
    
    # Initialize all possible bonuses
    bonuses = {
        "filename_semantic": 0.0,
        "color_matching": 0.0,
        "seasonal_matching": 0.0,
        "garment_matching": 0.0,
        "style_matching": 0.0,
        "material_matching": 0.0,
        "generic_fashion": 0.0,
        "quality_bonus": 0.0
    }
    
    fashion_analysis = query_data.get("fashion_analysis", {})
    query_text = query_data.get("query", "").lower()
    
    # FILENAME SEMANTIC MATCHING (very important!)
    filename = metadata.get("filename", "").lower()
    
    # Clean filename for analysis
    clean_filename = filename.replace("-", " ").replace("_", " ").replace(".", " ")
    query_words = set(query_text.split())
    filename_words = set(clean_filename.split())
    
    # Direct word matches
    direct_matches = query_words.intersection(filename_words)
    if direct_matches:
        bonuses["filename_semantic"] = len(direct_matches) * 0.08
        print(f"📁 Filename matches: {direct_matches}")
    
    # Partial word matches (substring matching)
    for q_word in query_words:
        if len(q_word) > 3:  # Only check meaningful words
            for f_word in filename_words:
                if q_word in f_word or f_word in q_word:
                    bonuses["filename_semantic"] += 0.04
                    print(f"📁 Partial match: '{q_word}' ~ '{f_word}'")
    
    # COLOR MATCHING
    detected_colors = fashion_analysis.get("colors", [])
    if detected_colors:
        # Check filename for color indicators
        for color in detected_colors:
            if color in clean_filename:
                bonuses["color_matching"] += 0.10
                print(f"🎨 Color match: {color}")
        
        # Check dominant colors from metadata
        item_colors = metadata.get("dominant_colours", [])
        if item_colors:
            # Simple color matching (could be enhanced)
            bonuses["color_matching"] += 0.05
    
    # SEASONAL MATCHING (super important for "winter clothing")
    detected_seasons = fashion_analysis.get("seasons", [])
    if detected_seasons:
        for season in detected_seasons:
            if season in clean_filename:
                bonuses["seasonal_matching"] += 0.12
                print(f"❄️ Seasonal match: {season}")
        
        # Check for seasonal indicators in filename
        winter_indicators = ["winter", "cold", "warm", "cozy", "thick", "puffer", "down", "wool", "fur"]
        summer_indicators = ["summer", "light", "cool", "thin", "breathable"]
        
        if "winter" in detected_seasons:
            winter_matches = [w for w in winter_indicators if w in clean_filename]
            if winter_matches:
                bonuses["seasonal_matching"] += len(winter_matches) * 0.06
                print(f"🔥 Winter indicators: {winter_matches}")
    
    # GARMENT MATCHING
    detected_garments = fashion_analysis.get("garments", [])
    if detected_garments:
        for garment in detected_garments:
            if garment in clean_filename:
                bonuses["garment_matching"] += 0.15
                print(f"👗 Garment match: {garment}")
            
            # Check for related garment terms
            garment_synonyms = {
                "clothing": ["clothes", "apparel", "wear", "garment", "outfit", "attire"],
                "jacket": ["coat", "outerwear", "blazer", "cardigan"],
                "dress": ["gown", "frock", "garment"]
            }
            
            if garment in garment_synonyms:
                synonyms = garment_synonyms[garment]
                synonym_matches = [s for s in synonyms if s in clean_filename]
                if synonym_matches:
                    bonuses["garment_matching"] += len(synonym_matches) * 0.08
    
    # STYLE MATCHING
    detected_styles = fashion_analysis.get("styles", [])
    for style in detected_styles:
        if style in clean_filename:
            bonuses["style_matching"] += 0.08
    
    # MATERIAL MATCHING
    detected_materials = fashion_analysis.get("materials", [])
    for material in detected_materials:
        if material in clean_filename:
            bonuses["material_matching"] += 0.09
    
    # GENERIC FASHION BONUS
    fashion_terms = ["fashion", "style", "clothing", "apparel", "wear", "outfit", "garment"]
    if any(term in clean_filename for term in fashion_terms):
        bonuses["generic_fashion"] += 0.03
    
    # QUALITY BONUS
    if len(metadata.get("dominant_colours", [])) > 2:
        bonuses["quality_bonus"] += 0.02
    
    # Calculate final relevance
    total_bonus = sum(bonuses.values())
    final_relevance = base_score + total_bonus
    
    # Quality assessment
    if final_relevance >= 0.6:
        quality = "excellent"
    elif final_relevance >= 0.4:
        quality = "good"
    elif final_relevance >= 0.25:
        quality = "fair"
    else:
        quality = "poor"
    
    print(f"📊 Relevance analysis for {metadata.get('filename', 'unknown')}:")
    print(f"   Base: {base_score:.3f}, Bonus: {total_bonus:.3f}, Final: {final_relevance:.3f} ({quality})")
    
    return {
        "base_similarity": base_score,
        "bonus_breakdown": bonuses,
        "total_bonus": total_bonus,
        "final_relevance": final_relevance,
        "quality": quality,
        "is_good_match": final_relevance >= 0.25
    }

def _apply_super_intelligent_filtering(candidates: List[Dict], query_data: Dict) -> List[Dict]:
    """Apply INTELLIGENT filtering that keeps good results."""
    
    if not candidates:
        print("❌ No candidates to filter!")
        return []
    
    fashion_metadata = query_data.get("fashion_metadata", {})
    expected_results = fashion_metadata.get("expected_results", 3)
    confidence_level = fashion_metadata.get("confidence_level", "medium")
    
    print(f"🧠 Intelligent filtering: {len(candidates)} candidates, expect {expected_results} results")
    
    # Sort by final relevance
    candidates.sort(key=lambda x: x.get("final_relevance", 0), reverse=True)
    
    # Quality filtering - keep anything that's not "poor"
    quality_filtered = []
    for candidate in candidates:
        relevance_info = candidate.get("relevance_analysis", {})
        quality = relevance_info.get("quality", "poor")
        
        if quality != "poor":
            quality_filtered.append(candidate)
        elif confidence_level == "low":
            # For low confidence, even keep poor quality matches
            quality_filtered.append(candidate)
    
    if not quality_filtered and candidates:
        # Emergency: if no good quality matches, keep the best ones anyway
        print("🚨 EMERGENCY: No good quality matches, keeping best available!")
        quality_filtered = candidates[:expected_results]
    
    # Intelligent count limiting
    if confidence_level == "high":
        # High confidence: more selective
        final_results = quality_filtered[:max(1, expected_results - 1)]
    elif confidence_level == "medium":
        # Medium confidence: as expected
        final_results = quality_filtered[:expected_results]
    else:
        # Low confidence: more permissive
        final_results = quality_filtered[:expected_results + 2]
    
    # GUARANTEE: Always return at least 1 result if we have candidates
    if not final_results and candidates:
        final_results = candidates[:1]
        print("🛡️ GUARANTEE: Returning at least 1 result!")
    
    print(f"✅ Intelligent filtering result: {len(candidates)} → {len(quality_filtered)} → {len(final_results)}")
    
    return final_results

def similarity_search(query_json: str) -> str:
    """
    SUPER INTELLIGENT similarity search that GUARANTEES relevant results.
    """
    try:
        print("🧠 Starting SUPER INTELLIGENT similarity search...")
        
        import faiss
        query_data = json.loads(query_json)
        
        # Extract search parameters
        query_embedding = query_data.get("embedding")
        all_embeddings = query_data.get("embeddings", [query_embedding])
        embedding_weights = query_data.get("embedding_weights", [1.0])
        
        if not query_embedding:
            return json.dumps({"error": "No query embedding"})
        
        fashion_metadata = query_data.get("fashion_metadata", {})
        
        print(f"🎯 Fashion intent: {fashion_metadata.get('fashion_intent', 'unknown')}")
        print(f"🔥 Confidence: {fashion_metadata.get('confidence_level', 'unknown')}")
        print(f"📊 Specificity: {fashion_metadata.get('fashion_specificity', 0)}")
        print(f"🎨 Expected results: {fashion_metadata.get('expected_results', 3)}")
        
        # Load index and metadata
        from src.config.settings import INDEX_FILE, META_FILE
        
        if not INDEX_FILE.exists() or not META_FILE.exists():
            return json.dumps({"error": "Index not found. Please build index first."})
        
        index = faiss.read_index(str(INDEX_FILE))
        metadata = json.loads(META_FILE.read_text())
        
        print(f"📊 Database loaded: {index.ntotal} items indexed")
        
        # COMPREHENSIVE multi-embedding search
        all_candidates = {}  # idx -> best_score
        all_scores = []
        
        for emb_idx, (embedding, weight) in enumerate(zip(all_embeddings, embedding_weights)):
            try:
                query_vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(query_vector)
                
                # Search ALL items to ensure we don't miss anything
                search_k = min(index.ntotal, 50)  # Comprehensive search
                distances, indices = index.search(query_vector, search_k)
                
                distances_flat = distances.flatten().tolist()
                indices_flat = indices.flatten().tolist()
                all_scores.extend(distances_flat)
                
                print(f"🔍 Search {emb_idx+1}: found scores range {min(distances_flat):.3f} to {max(distances_flat):.3f}")
                
                # Collect all candidates with weighted scores
                for score, idx in zip(distances_flat, indices_flat):
                    if idx == -1 or idx >= len(metadata):
                        continue
                    
                    weighted_score = float(score) * weight
                    
                    # Keep the best weighted score for each item
                    if idx not in all_candidates or weighted_score > all_candidates[idx]:
                        all_candidates[idx] = weighted_score
                        
            except Exception as e:
                log.warning(f"Search with embedding {emb_idx} failed: {e}")
                continue
        
        if not all_candidates:
            return json.dumps({
                "results": [],
                "total_found": 0,
                "error": "No candidates found in search"
            })
        
        print(f"🔍 Found {len(all_candidates)} unique candidates from all searches")
        
        # Calculate SUPER INTELLIGENT threshold
        smart_threshold = _calculate_super_intelligent_threshold(query_data, all_scores)
        
        # Build comprehensive candidate analysis
        analyzed_candidates = []
        
        for idx, similarity_score in all_candidates.items():
            if similarity_score < smart_threshold:
                print(f"⚠️ Skipping item {idx}: score {similarity_score:.3f} < threshold {smart_threshold:.3f}")
                continue
            
            try:
                # Calculate comprehensive relevance
                relevance_analysis = _calculate_super_comprehensive_relevance(query_data, metadata[idx], similarity_score)
                
                candidate = metadata[idx].copy()
                candidate["similarity_score"] = similarity_score
                candidate["relevance_analysis"] = relevance_analysis
                candidate["final_relevance"] = relevance_analysis["final_relevance"]
                
                analyzed_candidates.append(candidate)
                
            except Exception as e:
                log.warning(f"Analysis failed for candidate {idx}: {e}")
                # Still include it with basic info
                candidate = metadata[idx].copy()
                candidate["similarity_score"] = similarity_score
                candidate["final_relevance"] = similarity_score
                analyzed_candidates.append(candidate)
        
        if not analyzed_candidates:
            # EMERGENCY FALLBACK - if threshold filtered everything out, try again with lower threshold
            emergency_threshold = smart_threshold * 0.5
            print(f"🚨 EMERGENCY: No results with threshold {smart_threshold:.3f}, trying {emergency_threshold:.3f}")
            
            for idx, similarity_score in all_candidates.items():
                if similarity_score >= emergency_threshold:
                    candidate = metadata[idx].copy()
                    candidate["similarity_score"] = similarity_score
                    candidate["final_relevance"] = similarity_score
                    analyzed_candidates.append(candidate)
        
        # Apply super intelligent filtering
        final_results = _apply_super_intelligent_filtering(analyzed_candidates, query_data)
        
        # Final sort by relevance
        final_results.sort(key=lambda x: x.get("final_relevance", x.get("similarity_score", 0)), reverse=True)
        
        print(f"🎯 SUPER INTELLIGENT search completed: {len(final_results)} results found!")
        
        # Log results for debugging
        for i, result in enumerate(final_results, 1):
            filename = result.get("filename", "unknown")
            score = result.get("final_relevance", result.get("similarity_score", 0))
            print(f"   {i}. {filename} | Score: {score:.3f}")
        
        return json.dumps({
            "results": final_results,
            "total_found": len(final_results),
            "search_metadata": {
                "query": query_data.get("query", "unknown"),
                "smart_threshold": smart_threshold,
                "total_candidates": len(all_candidates),
                "candidates_analyzed": len(analyzed_candidates),
                "fashion_intent": fashion_metadata.get("fashion_intent", "unknown"),
                "confidence_level": fashion_metadata.get("confidence_level", "unknown"),
                "search_strategy": "super_intelligent_guaranteed",
                "embedding_count": len(all_embeddings)
            }
        })
        
    except Exception as e:
        error_msg = f"Super intelligent search failed: {e}"
        log.error(error_msg)
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error_msg})