"""
PROFESSIONAL FASHION INDUSTRY VECTOR SEARCH - Ultra-precise matching for fashion designers.
REPLACES: src/tools/vector_tools.py
"""
import json
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import statistics
import re

# Configure logging
log = logging.getLogger(__name__)

# FASHION INDUSTRY CONSTANTS
FASHION_PRECISION_LEVELS = {
    "exact": {"threshold_multiplier": 1.2, "max_results": 1, "quality_min": "excellent"},
    "high": {"threshold_multiplier": 1.0, "max_results": 2, "quality_min": "good"},
    "moderate": {"threshold_multiplier": 0.8, "max_results": 3, "quality_min": "fair"},
    "exploratory": {"threshold_multiplier": 0.6, "max_results": 5, "quality_min": "acceptable"}
}

def build_index(data_json: str) -> str:
    """Build fashion industry-grade FAISS index with comprehensive metadata enhancement."""
    try:
        print("🎨 Building professional fashion industry FAISS index...")
        log.info("Starting fashion industry index building")
        
        import faiss
        
        # Parse data
        data = json.loads(data_json)
        embeddings = data.get("embeddings", [])
        metadata = data.get("metadata", [])
        
        if not embeddings or not metadata or len(embeddings) != len(metadata):
            return json.dumps({"error": "Invalid fashion data"})
        
        # COMPREHENSIVE FASHION METADATA ENHANCEMENT
        enhanced_metadata = []
        for i, meta in enumerate(metadata):
            enhanced_meta = meta.copy()
            
            # Extract comprehensive fashion features
            enhanced_meta.update(_extract_comprehensive_fashion_features(meta))
            
            enhanced_metadata.append(enhanced_meta)
            
        # Build optimized FAISS index
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        num_vectors, dimension = vectors.shape
        
        print(f"📊 Building fashion index: {num_vectors} garments × {dimension} dimensions")
        
        # Use optimized index for fashion similarity
        index = faiss.IndexFlatIP(dimension)  # Inner product for fashion similarity
        faiss.normalize_L2(vectors)
        index.add(vectors)
        
        # Save with fashion enhancements
        from src.config.settings import INDEX_FILE, META_FILE
        
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        META_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(INDEX_FILE))
        META_FILE.write_text(json.dumps(enhanced_metadata, indent=2))
        
        print(f"✅ Fashion industry index built: {index.ntotal} garments indexed")
        
        return json.dumps({
            "indexed": int(index.ntotal),
            "dimension": dimension,
            "fashion_enhanced": True,
            "index_type": "fashion_industry_professional",
            "features_extracted": len(enhanced_metadata),
            "precision_level": "industry_grade"
        })
        
    except Exception as e:
        error_msg = f"Failed to build fashion index: {e}"
        log.error(error_msg)
        return json.dumps({"error": error_msg})

def _extract_comprehensive_fashion_features(metadata: Dict) -> Dict:
    """Extract comprehensive fashion industry features from metadata."""
    features = {}
    
    filename = metadata.get("filename", "").lower()
    path = metadata.get("path", "").lower()
    colors = metadata.get("dominant_colours", [])
    
    # ADVANCED FASHION CATEGORIZATION
    features.update(_classify_fashion_garment(filename, path))
    
    # COLOR PALETTE ANALYSIS
    features.update(_analyze_fashion_color_palette(colors))
    
    # SEASONAL COLLECTION DETECTION
    features.update(_detect_fashion_season(filename, path))
    
    # STYLE AND AESTHETIC ANALYSIS
    features.update(_analyze_fashion_style(filename, path))
    
    # FABRIC AND TEXTURE DETECTION
    features.update(_detect_fabric_materials(filename, path))
    
    # OCCASION AND DRESS CODE ANALYSIS
    features.update(_analyze_occasion_context(filename, path))
    
    # TREND AND ERA CLASSIFICATION
    features.update(_classify_fashion_era(filename, path))
    
    # SILHOUETTE AND FIT ANALYSIS
    features.update(_analyze_silhouette_fit(filename, path))
    
    return features

def _classify_fashion_garment(filename: str, path: str) -> Dict:
    """Professional garment classification for fashion industry."""
    text = f"{filename} {path}".lower()
    
    classification = {
        "garment_category": "unknown",
        "garment_subcategory": "unknown",
        "garment_type": "unknown",
        "fashion_hierarchy": [],
        "garment_confidence": 0.0
    }
    
    # Comprehensive garment database
    garment_hierarchy = {
        "tops": {
            "shirts": ["shirt", "blouse", "button-down", "button-up", "oxford", "chambray", "flannel", "tunic"],
            "t_shirts": ["t-shirt", "tee", "tank", "camisole", "halter", "tube", "crop", "bodysuit", "racerback"],
            "knits": ["sweater", "pullover", "cardigan", "jumper", "knit", "cashmere", "turtleneck", "crew neck"],
            "formal_tops": ["blazer", "vest", "waistcoat", "suit jacket", "smoking jacket", "evening jacket"]
        },
        "bottoms": {
            "pants": ["pants", "trousers", "slacks", "chinos", "khakis", "dress pants", "palazzo", "wide leg"],
            "jeans": ["jeans", "denim", "skinny", "straight", "bootcut", "flare", "boyfriend", "mom jeans", "high waist"],
            "shorts": ["shorts", "bermuda", "cargo shorts", "denim shorts", "tailored shorts", "bike shorts"],
            "skirts": ["skirt", "mini", "midi", "maxi", "pencil", "a-line", "pleated", "wrap", "circle", "tulip"]
        },
        "dresses": {
            "casual": ["sundress", "shirt dress", "wrap dress", "t-shirt dress", "swing dress", "shift dress"],
            "formal": ["cocktail", "evening", "gown", "little black dress", "midi dress", "formal dress"],
            "special": ["wedding", "prom", "party", "graduation", "red carpet", "ball gown", "mermaid"]
        },
        "outerwear": {
            "jackets": ["jacket", "blazer", "bomber", "varsity", "denim jacket", "leather jacket", "moto jacket"],
            "coats": ["coat", "trench", "peacoat", "overcoat", "duster", "wrap coat", "wool coat"],
            "winter": ["puffer", "down", "parka", "fur coat", "shearling", "pea coat", "winter coat"],
            "cardigans": ["cardigan", "wrap cardigan", "duster cardigan", "long cardigan", "cropped cardigan"]
        }
    }
    
    # Find best match
    best_match = {"category": "unknown", "subcategory": "unknown", "confidence": 0}
    
    for category, subcategories in garment_hierarchy.items():
        for subcategory, garments in subcategories.items():
            for garment in garments:
                if garment in text:
                    confidence = len(garment.split()) + (1.0 if garment == text.strip() else 0.5)
                    if confidence > best_match["confidence"]:
                        best_match = {
                            "category": category,
                            "subcategory": subcategory,
                            "garment": garment,
                            "confidence": confidence
                        }
    
    if best_match["confidence"] > 0:
        classification.update({
            "garment_category": best_match["category"],
            "garment_subcategory": best_match["subcategory"],
            "garment_type": best_match["garment"],
            "garment_confidence": best_match["confidence"] / 3.0
        })
        
        # Build fashion hierarchy
        classification["fashion_hierarchy"] = [
            best_match["category"],
            best_match["subcategory"],
            best_match["garment"]
        ]
    
    return classification

def _analyze_fashion_color_palette(colors: List[str]) -> Dict:
    """Professional fashion color palette analysis."""
    if not colors:
        return {"color_analysis": {}, "color_harmony": "unknown"}
    
    analysis = {
        "color_palette_size": len(colors),
        "dominant_color_family": "unknown",
        "color_temperature": "unknown",
        "color_saturation": "unknown",
        "color_harmony": "unknown",
        "fashion_color_story": [],
        "seasonal_color_match": [],
        "color_sophistication": "unknown"
    }
    
    try:
        # Analyze each color
        warm_colors = 0
        cool_colors = 0
        neutral_colors = 0
        bright_colors = 0
        dark_colors = 0
        
        fashion_color_mapping = {
            "warm": ["red", "orange", "yellow", "coral", "peach", "gold"],
            "cool": ["blue", "green", "purple", "teal", "turquoise"],
            "neutral": ["black", "white", "gray", "beige", "brown", "tan"]
        }
        
        for color_hex in colors:
            # Convert hex to RGB for analysis
            try:
                color_hex = color_hex.lstrip('#')
                r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
                
                # Temperature analysis
                if r > b + 20:  # More red than blue = warm
                    warm_colors += 1
                elif b > r + 20:  # More blue than red = cool
                    cool_colors += 1
                else:
                    neutral_colors += 1
                
                # Brightness analysis
                brightness = (r + g + b) / 3
                if brightness > 180:
                    bright_colors += 1
                elif brightness < 80:
                    dark_colors += 1
                    
            except ValueError:
                continue
        
        # Determine dominant characteristics
        total = len(colors)
        if warm_colors > cool_colors and warm_colors > neutral_colors:
            analysis["color_temperature"] = "warm"
        elif cool_colors > warm_colors and cool_colors > neutral_colors:
            analysis["color_temperature"] = "cool"
        else:
            analysis["color_temperature"] = "neutral"
            
        if bright_colors > dark_colors:
            analysis["color_saturation"] = "bright"
        elif dark_colors > bright_colors:
            analysis["color_saturation"] = "dark"
        else:
            analysis["color_saturation"] = "balanced"
        
        # Color harmony analysis
        if total == 1:
            analysis["color_harmony"] = "monochromatic"
        elif warm_colors > 0 and cool_colors > 0:
            analysis["color_harmony"] = "complementary"
        elif (warm_colors > 0 and neutral_colors > 0) or (cool_colors > 0 and neutral_colors > 0):
            analysis["color_harmony"] = "neutral_accent"
        else:
            analysis["color_harmony"] = "analogous"
        
        # Fashion sophistication
        if neutral_colors >= total * 0.7:
            analysis["color_sophistication"] = "sophisticated"
        elif bright_colors >= total * 0.6:
            analysis["color_sophistication"] = "bold"
        else:
            analysis["color_sophistication"] = "balanced"
            
    except Exception as e:
        log.warning(f"Color analysis failed: {e}")
    
    return analysis

def _detect_fashion_season(filename: str, path: str) -> Dict:
    """Detect fashion season and collection context."""
    text = f"{filename} {path}".lower()
    
    seasonal_indicators = {
        "spring_summer": {
            "direct": ["spring", "summer", "ss", "s/s"],
            "materials": ["cotton", "linen", "silk", "chiffon", "voile"],
            "descriptors": ["light", "fresh", "bright", "airy", "breathable"],
            "colors": ["pastel", "bright", "white", "light"]
        },
        "fall_winter": {
            "direct": ["fall", "autumn", "winter", "fw", "f/w"],
            "materials": ["wool", "cashmere", "tweed", "flannel", "velvet", "fur"],
            "descriptors": ["warm", "cozy", "thick", "heavy", "insulated", "layered"],
            "colors": ["dark", "rich", "deep", "earth"]
        },
        "resort": {
            "direct": ["resort", "cruise", "vacation", "holiday"],
            "descriptors": ["tropical", "beach", "relaxed", "travel"],
            "context": ["getaway", "escape", "paradise"]
        },
        "pre_collections": {
            "direct": ["pre-fall", "pre-spring", "pre", "transitional"],
            "descriptors": ["versatile", "transition", "capsule"]
        }
    }
    
    season_scores = {}
    
    for season, indicators in seasonal_indicators.items():
        score = 0
        matches = []
        
        for category, terms in indicators.items():
            for term in terms:
                if term in text:
                    if category == "direct":
                        score += 3  # Direct season mentions are strongest
                        matches.append(f"direct:{term}")
                    elif category == "materials":
                        score += 2
                        matches.append(f"material:{term}")
                    else:
                        score += 1
                        matches.append(f"{category}:{term}")
        
        if score > 0:
            season_scores[season] = {"score": score, "matches": matches}
    
    # Determine primary season
    primary_season = "unknown"
    if season_scores:
        primary_season = max(season_scores.keys(), key=lambda x: season_scores[x]["score"])
    
    return {
        "fashion_season": primary_season,
        "season_confidence": season_scores.get(primary_season, {}).get("score", 0) / 10.0,
        "season_indicators": season_scores.get(primary_season, {}).get("matches", []),
        "all_seasonal_matches": season_scores
    }

def _analyze_fashion_style(filename: str, path: str) -> Dict:
    """Comprehensive fashion style and aesthetic analysis."""
    text = f"{filename} {path}".lower()
    
    style_categories = {
        "classic": {
            "keywords": ["classic", "timeless", "traditional", "conservative", "elegant", "refined", "tailored"],
            "pieces": ["blazer", "trench", "pencil skirt", "white shirt", "little black dress"],
            "characteristics": ["clean lines", "sophisticated", "investment"]
        },
        "bohemian": {
            "keywords": ["boho", "bohemian", "hippie", "free-spirited", "artistic", "ethnic", "eclectic"],
            "pieces": ["maxi dress", "fringe", "peasant", "wide leg", "kimono", "kaftan"],
            "characteristics": ["flowing", "textural", "layered", "artistic"]
        },
        "minimalist": {
            "keywords": ["minimalist", "clean", "simple", "modern", "sleek", "understated", "architectural"],
            "pieces": ["clean lines", "simple shapes", "basic"],
            "characteristics": ["simple", "quality", "functional", "edited"]
        },
        "romantic": {
            "keywords": ["romantic", "feminine", "soft", "pretty", "delicate", "sweet", "dreamy"],
            "pieces": ["ruffle", "lace", "floral", "bow", "chiffon"],
            "characteristics": ["soft", "flowing", "detailed", "feminine"]
        },
        "edgy": {
            "keywords": ["edgy", "rock", "punk", "rebellious", "alternative", "bold", "grunge"],
            "pieces": ["leather", "stud", "chain", "ripped", "distressed"],
            "characteristics": ["bold", "statement", "rebellious", "confident"]
        },
        "preppy": {
            "keywords": ["preppy", "ivy", "nautical", "collegiate", "refined", "polished", "country club"],
            "pieces": ["polo", "pleated", "loafer", "pearl", "cable knit"],
            "characteristics": ["polished", "crisp", "refined", "traditional"]
        },
        "avant_garde": {
            "keywords": ["avant-garde", "conceptual", "experimental", "innovative", "architectural", "futuristic"],
            "characteristics": ["experimental", "innovative", "artistic", "unconventional"]
        },
        "streetwear": {
            "keywords": ["streetwear", "urban", "casual", "sporty", "hip-hop", "skateboard"],
            "pieces": ["hoodie", "sneaker", "cap", "jogger", "oversized"],
            "characteristics": ["casual", "comfortable", "youthful", "trendy"]
        }
    }
    
    style_analysis = {}
    total_style_score = 0
    
    for style_name, style_data in style_categories.items():
        score = 0
        matches = []
        
        # Check keywords
        for keyword in style_data.get("keywords", []):
            if keyword in text:
                score += 2
                matches.append(f"keyword:{keyword}")
        
        # Check characteristic pieces
        for piece in style_data.get("pieces", []):
            if piece in text:
                score += 3  # Pieces are stronger indicators
                matches.append(f"piece:{piece}")
        
        if score > 0:
            style_analysis[style_name] = {
                "score": score,
                "matches": matches,
                "characteristics": style_data.get("characteristics", [])
            }
            total_style_score += score
    
    # Determine primary style
    primary_style = "unknown"
    if style_analysis:
        primary_style = max(style_analysis.keys(), key=lambda x: style_analysis[x]["score"])
    
    return {
        "fashion_style": primary_style,
        "style_confidence": style_analysis.get(primary_style, {}).get("score", 0) / 5.0,
        "style_matches": style_analysis.get(primary_style, {}).get("matches", []),
        "all_style_analysis": style_analysis,
        "style_diversity": len(style_analysis),
        "is_multi_style": len(style_analysis) > 1
    }

def _detect_fabric_materials(filename: str, path: str) -> Dict:
    """Advanced fabric and material detection for fashion professionals."""
    text = f"{filename} {path}".lower()
    
    fabric_database = {
        "luxury_natural": {
            "cashmere": {"weight": "light", "season": "fall/winter", "care": "delicate"},
            "silk": {"weight": "light", "season": "all", "care": "delicate"},
            "wool": {"weight": "medium", "season": "fall/winter", "care": "moderate"},
            "linen": {"weight": "light", "season": "spring/summer", "care": "easy"},
            "cotton": {"weight": "medium", "season": "all", "care": "easy"}
        },
        "luxury_materials": {
            "leather": {"type": "animal", "care": "special", "season": "fall/winter"},
            "suede": {"type": "animal", "care": "delicate", "season": "fall/winter"},
            "fur": {"type": "animal", "care": "professional", "season": "winter"},
            "velvet": {"type": "luxury", "care": "delicate", "season": "fall/winter"}
        },
        "performance": {
            "polyester": {"type": "synthetic", "care": "easy", "properties": ["durable", "wrinkle-resistant"]},
            "nylon": {"type": "synthetic", "care": "easy", "properties": ["strong", "lightweight"]},
            "spandex": {"type": "synthetic", "care": "delicate", "properties": ["stretch", "recovery"]}
        },
        "specialty": {
            "denim": {"type": "cotton", "care": "moderate", "style": "casual"},
            "tweed": {"type": "wool", "care": "dry-clean", "style": "classic"},
            "chiffon": {"type": "light", "care": "delicate", "style": "romantic"},
            "jersey": {"type": "knit", "care": "easy", "style": "casual"}
        }
    }
    
    detected_fabrics = {}
    fabric_properties = []
    
    for category, fabrics in fabric_database.items():
        for fabric_name, properties in fabrics.items():
            if fabric_name in text:
                detected_fabrics[fabric_name] = {
                    "category": category,
                    "properties": properties,
                    "confidence": 1.0 if fabric_name in text.split() else 0.7
                }
                fabric_properties.extend(properties.get("properties", []))
    
    return {
        "detected_fabrics": detected_fabrics,
        "fabric_count": len(detected_fabrics),
        "fabric_categories": list(set([fab["category"] for fab in detected_fabrics.values()])),
        "fabric_properties": list(set(fabric_properties)),
        "is_luxury_fabric": any(cat in ["luxury_natural", "luxury_materials"] for cat in [fab["category"] for fab in detected_fabrics.values()]),
        "care_requirements": list(set([fab["properties"].get("care", "unknown") for fab in detected_fabrics.values() if "care" in fab["properties"]]))
    }

def _analyze_occasion_context(filename: str, path: str) -> Dict:
    """Analyze fashion context for specific occasions and dress codes."""
    text = f"{filename} {path}".lower()
    
    occasion_database = {
        "work_professional": {
            "keywords": ["work", "office", "business", "professional", "corporate", "meeting", "executive"],
            "dress_codes": ["business formal", "business casual", "smart casual"],
            "avoid": ["casual", "revealing", "loud patterns"]
        },
        "formal_events": {
            "keywords": ["formal", "black tie", "gala", "wedding", "cocktail", "evening", "opera"],
            "dress_codes": ["black tie", "cocktail", "formal", "semi-formal"],
            "required": ["elegant", "sophisticated", "polished"]
        },
        "casual_social": {
            "keywords": ["casual", "weekend", "brunch", "shopping", "friends", "date", "coffee"],
            "characteristics": ["comfortable", "stylish", "approachable", "versatile"]
        },
        "special_events": {
            "keywords": ["party", "celebration", "birthday", "anniversary", "graduation", "prom"],
            "characteristics": ["festive", "memorable", "photo-worthy", "special"]
        },
        "active_lifestyle": {
            "keywords": ["gym", "workout", "yoga", "running", "sports", "athletic", "fitness"],
            "characteristics": ["functional", "flexible", "moisture-wicking", "supportive"]
        },
        "vacation_travel": {
            "keywords": ["vacation", "travel", "resort", "beach", "cruise", "holiday", "getaway"],
            "characteristics": ["versatile", "comfortable", "wrinkle-resistant", "packable"]
        }
    }
    
    occasion_matches = {}
    
    for occasion, data in occasion_database.items():
        score = 0
        matches = []
        
        for keyword in data.get("keywords", []):
            if keyword in text:
                score += 2
                matches.append(keyword)
        
        if score > 0:
            occasion_matches[occasion] = {
                "score": score,
                "matches": matches,
                "dress_codes": data.get("dress_codes", []),
                "characteristics": data.get("characteristics", [])
            }
    
    primary_occasion = "general"
    if occasion_matches:
        primary_occasion = max(occasion_matches.keys(), key=lambda x: occasion_matches[x]["score"])
    
    return {
        "primary_occasion": primary_occasion,
        "occasion_confidence": occasion_matches.get(primary_occasion, {}).get("score", 0) / 4.0,
        "suitable_dress_codes": occasion_matches.get(primary_occasion, {}).get("dress_codes", []),
        "occasion_characteristics": occasion_matches.get(primary_occasion, {}).get("characteristics", []),
        "all_occasion_matches": occasion_matches
    }

def _classify_fashion_era(filename: str, path: str) -> Dict:
    """Classify fashion era and trend context."""
    text = f"{filename} {path}".lower()
    
    fashion_eras = {
        "vintage": {
            "decades": ["20s", "30s", "40s", "50s", "60s", "70s", "80s", "90s"],
            "keywords": ["vintage", "retro", "classic", "throwback", "nostalgic"],
            "styles": ["pin-up", "mod", "disco", "grunge", "power shoulder"]
        },
        "contemporary": {
            "keywords": ["modern", "contemporary", "current", "today", "now", "2020s"],
            "trends": ["sustainable", "minimalist", "oversized", "athleisure"]
        },
        "timeless": {
            "keywords": ["timeless", "classic", "eternal", "enduring", "traditional"],
            "pieces": ["trench coat", "white shirt", "little black dress", "blazer"]
        },
        "avant_garde": {
            "keywords": ["futuristic", "innovative", "experimental", "conceptual", "cutting-edge"],
            "characteristics": ["unique", "artistic", "unconventional"]
        }
    }
    
    era_analysis = {}
    
    for era, data in fashion_eras.items():
        score = 0
        matches = []
        
        for category in ["decades", "keywords", "styles", "trends", "pieces"]:
            items = data.get(category, [])
            for item in items:
                if item in text:
                    score += 1
                    matches.append(f"{category}:{item}")
        
        if score > 0:
            era_analysis[era] = {"score": score, "matches": matches}
    
    primary_era = "contemporary"  # Default to contemporary
    if era_analysis:
        primary_era = max(era_analysis.keys(), key=lambda x: era_analysis[x]["score"])
    
    return {
        "fashion_era": primary_era,
        "era_confidence": era_analysis.get(primary_era, {}).get("score", 1) / 3.0,
        "era_indicators": era_analysis.get(primary_era, {}).get("matches", []),
        "is_trend_piece": "trend" in text or "current" in text,
        "is_classic_piece": primary_era in ["timeless", "vintage"]
    }

def _analyze_silhouette_fit(filename: str, path: str) -> Dict:
    """Analyze garment silhouette and fit characteristics."""
    text = f"{filename} {path}".lower()
    
    silhouette_types = {
        "fitted": ["fitted", "slim", "tight", "bodycon", "second skin", "tailored", "structured"],
        "loose": ["loose", "relaxed", "oversized", "baggy", "flowing", "drapey", "boxy"],
        "a_line": ["a-line", "fit and flare", "swing", "trapeze"],
        "straight": ["straight", "column", "shift", "tube", "pencil"],
        "empire": ["empire", "high waist", "babydoll"],
        "wrap": ["wrap", "surplice", "crossover"]
    }
    
    fit_characteristics = {
        "waist_emphasis": ["cinched", "belted", "fitted waist", "defined waist", "hourglass"],
        "length_variations": ["cropped", "short", "long", "midi", "maxi", "mini", "knee length"],
        "neckline_types": ["v-neck", "scoop", "high neck", "boat neck", "off shoulder", "halter"],
        "sleeve_types": ["sleeveless", "short sleeve", "long sleeve", "bell sleeve", "puff sleeve"]
    }
    
    detected_silhouettes = []
    detected_fit_features = {}
    
    # Analyze silhouette
    for silhouette, keywords in silhouette_types.items():
        matches = [kw for kw in keywords if kw in text]
        if matches:
            detected_silhouettes.append({
                "silhouette": silhouette,
                "matches": matches,
                "confidence": len(matches) / len(keywords)
            })
    
    # Analyze fit characteristics
    for feature_category, features in fit_characteristics.items():
        matches = [f for f in features if f in text]
        if matches:
            detected_fit_features[feature_category] = matches
    
    primary_silhouette = "unknown"
    if detected_silhouettes:
        primary_silhouette = max(detected_silhouettes, key=lambda x: x["confidence"])["silhouette"]
    
    return {
        "primary_silhouette": primary_silhouette,
        "silhouette_confidence": max([s["confidence"] for s in detected_silhouettes]) if detected_silhouettes else 0.0,
        "all_silhouettes": detected_silhouettes,
        "fit_features": detected_fit_features,
        "fit_complexity": len(detected_fit_features),
        "silhouette_diversity": len(detected_silhouettes)
    }

def _calculate_fashion_precision_threshold(query_data: Dict, initial_scores: List[float]) -> Tuple[float, str]:
    """Calculate fashion industry precision threshold based on designer requirements."""
    if not initial_scores:
        return 0.18, "moderate"
    
    fashion_metadata = query_data.get("fashion_metadata", {})
    design_metrics = query_data.get("design_metrics", {})
    
    precision_level = fashion_metadata.get("fashion_precision", "moderate")
    designer_confidence = fashion_metadata.get("designer_confidence", "medium")
    complexity_level = design_metrics.get("complexity_level", "basic")
    
    # Fashion industry precision requirements
    if precision_level == "exact":
        precision_config = FASHION_PRECISION_LEVELS["exact"]
    elif precision_level == "high":
        precision_config = FASHION_PRECISION_LEVELS["high"]
    elif precision_level == "moderate":
        precision_config = FASHION_PRECISION_LEVELS["moderate"]
    else:
        precision_config = FASHION_PRECISION_LEVELS["exploratory"]
    
    # Calculate base threshold from score distribution
    scores_array = np.array(initial_scores)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    median_score = np.median(scores_array)
    
    # Apply fashion industry multipliers
    base_threshold = median_score * precision_config["threshold_multiplier"]
    
    # Adjust for designer confidence and complexity
    if designer_confidence == "high" and complexity_level == "expert":
        base_threshold *= 1.15  # Very selective for expert queries
    elif designer_confidence == "high":
        base_threshold *= 1.08
    elif designer_confidence == "low":
        base_threshold *= 0.85
    
    # Fashion industry bounds
    final_threshold = max(0.15, min(0.40, base_threshold))
    
    print(f"🎨 Fashion precision: {precision_level} | Threshold: {final_threshold:.3f} | Max results: {precision_config['max_results']}")
    
    return final_threshold, precision_level

def similarity_search(query_json: str) -> str:
    """
    PROFESSIONAL FASHION INDUSTRY similarity search with ultra-precise matching.
    """
    try:
        print("🎨 Starting professional fashion industry search...")
        
        import faiss
        query_data = json.loads(query_json)
        
        # Extract fashion search parameters
        query_embedding = query_data.get("embedding")
        all_embeddings = query_data.get("embeddings", [query_embedding])
        embedding_weights = query_data.get("embedding_weights", [1.0])
        
        fashion_metadata = query_data.get("fashion_metadata", {})
        design_metrics = query_data.get("design_metrics", {})
        
        print(f"👗 Fashion intent: {design_metrics.get('fashion_intent', 'unknown')}")
        print(f"📏 Precision level: {fashion_metadata.get('fashion_precision', 'moderate')}")
        print(f"🎯 Designer confidence: {fashion_metadata.get('designer_confidence', 'medium')}")
        
        # Load fashion industry index
        from src.config.settings import INDEX_FILE, META_FILE
        
        if not INDEX_FILE.exists() or not META_FILE.exists():
            return json.dumps({"error": "Fashion index not found. Please build index first."})
        
        index = faiss.read_index(str(INDEX_FILE))
        metadata = json.loads(META_FILE.read_text())
        
        print(f"📊 Fashion database: {index.ntotal} garments indexed")
        
        # Multi-embedding fashion search
        all_candidates = {}
        initial_scores = []
        
        for emb_idx, (embedding, weight) in enumerate(zip(all_embeddings, embedding_weights)):
            try:
                query_vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(query_vector)
                
                # Comprehensive candidate search
                search_k = min(index.ntotal, 40)  # Fashion needs thorough search
                distances, indices = index.search(query_vector, search_k)
                
                distances_flat = distances.flatten().tolist()
                indices_flat = indices.flatten().tolist()
                initial_scores.extend(distances_flat)
                
                for score, idx in zip(distances_flat, indices_flat):
                    if idx == -1 or idx >= len(metadata):
                        continue
                    
                    weighted_score = float(score) * weight
                    if idx not in all_candidates or weighted_score > all_candidates[idx]:
                        all_candidates[idx] = weighted_score
                        
            except Exception as e:
                log.warning(f"Fashion search with embedding {emb_idx} failed: {e}")
                continue
        
        # Calculate fashion precision threshold
        fashion_threshold, precision_level = _calculate_fashion_precision_threshold(query_data, initial_scores)
        
        # Professional fashion relevance analysis
        fashion_candidates = []
        for idx, similarity_score in all_candidates.items():
            if similarity_score < fashion_threshold:
                continue
                
            try:
                # Comprehensive fashion relevance analysis
                relevance_analysis = _calculate_fashion_relevance(query_data, metadata[idx], similarity_score)
                
                candidate = metadata[idx].copy()
                candidate["similarity_score"] = similarity_score
                candidate["fashion_relevance"] = relevance_analysis
                candidate["final_fashion_score"] = relevance_analysis["final_relevance"]
                
                fashion_candidates.append(candidate)
                
            except Exception as e:
                log.warning(f"Fashion analysis error for candidate {idx}: {e}")
                continue
        
        # Apply fashion industry filtering
        final_results = _apply_fashion_industry_filtering(fashion_candidates, query_data, precision_level)
        
        # Sort by fashion relevance
        final_results.sort(key=lambda x: x["final_fashion_score"], reverse=True)
        
        print(f"🎨 Fashion search completed: {len(final_results)} professional matches")
        
        # Log fashion results
        for i, result in enumerate(final_results[:3], 1):
            filename = result.get("filename", "unknown")
            score = result.get("final_fashion_score", 0)
            garment_type = result.get("garment_type", "unknown")
            print(f"   {i}. {filename} | Score: {score:.3f} | Type: {garment_type}")
        
        return json.dumps({
            "results": final_results,
            "total_found": len(final_results),
            "fashion_search_metadata": {
                "query": query_data.get("query", "unknown"),
                "fashion_threshold": fashion_threshold,
                "precision_level": precision_level,
                "designer_confidence": fashion_metadata.get("designer_confidence", "medium"),
                "fashion_intent": design_metrics.get("fashion_intent", "unknown"),
                "total_candidates_analyzed": len(all_candidates),
                "search_strategy": "fashion_industry_professional",
                "industry_grade": True
            }
        })
        
    except Exception as e:
        error_msg = f"Fashion industry search failed: {e}"
        log.error(error_msg)
        return json.dumps({"error": error_msg})

def _calculate_fashion_relevance(query_data: Dict, metadata: Dict, similarity_score: float) -> Dict:
    """Calculate comprehensive fashion industry relevance with professional metrics."""
    base_score = similarity_score
    
    # Initialize fashion bonuses
    fashion_bonuses = {
        "color_harmony": 0.0,
        "garment_match": 0.0,
        "seasonal_alignment": 0.0,
        "style_consistency": 0.0,
        "fabric_compatibility": 0.0,
        "occasion_suitability": 0.0,
        "fashion_sophistication": 0.0,
        "trend_relevance": 0.0,
        "designer_precision": 0.0
    }
    
    fashion_analysis = query_data.get("fashion_analysis", {})
    professional_features = query_data.get("professional_features", {})
    
    # Advanced color harmony analysis
    query_colors = professional_features.get("colors", {})
    item_color_analysis = metadata.get("color_analysis", {})
    
    if query_colors and item_color_analysis:
        color_temp_match = _match_color_temperature(query_colors, item_color_analysis)
        color_harmony_match = _match_color_harmony(query_colors, item_color_analysis)
        fashion_bonuses["color_harmony"] = (color_temp_match + color_harmony_match) * 0.12
    
    # Professional garment matching
    query_garments = professional_features.get("garments", {})
    item_garment_info = {
        "category": metadata.get("garment_category", "unknown"),
        "subcategory": metadata.get("garment_subcategory", "unknown"),
        "type": metadata.get("garment_type", "unknown")
    }
    
    if query_garments and item_garment_info["category"] != "unknown":
        garment_precision = _calculate_garment_precision_match(query_garments, item_garment_info)
        fashion_bonuses["garment_match"] = garment_precision * 0.18
    
    # Seasonal collection alignment
    query_seasons = professional_features.get("seasons", {})
    item_season = metadata.get("fashion_season", "unknown")
    
    if query_seasons and item_season != "unknown":
        seasonal_match = _calculate_seasonal_alignment(query_seasons, item_season, metadata)
        fashion_bonuses["seasonal_alignment"] = seasonal_match * 0.14
    
    # Fashion style consistency
    query_styles = professional_features.get("styles", {})
    item_style = metadata.get("fashion_style", "unknown")
    
    if query_styles and item_style != "unknown":
        style_consistency = _calculate_style_consistency(query_styles, item_style, metadata)
        fashion_bonuses["style_consistency"] = style_consistency * 0.13
    
    # Fabric and material compatibility
    query_fabrics = professional_features.get("fabrics", {})
    item_fabrics = metadata.get("detected_fabrics", {})
    
    if query_fabrics and item_fabrics:
        fabric_compatibility = _calculate_fabric_compatibility(query_fabrics, item_fabrics)
        fashion_bonuses["fabric_compatibility"] = fabric_compatibility * 0.11
    
    # Professional occasion matching
    query_occasions = professional_features.get("occasions", {})
    item_occasion = metadata.get("primary_occasion", "general")
    
    if query_occasions and item_occasion != "general":
        occasion_match = _calculate_occasion_suitability(query_occasions, item_occasion, metadata)
        fashion_bonuses["occasion_suitability"] = occasion_match * 0.10
    
    # Fashion sophistication assessment
    item_sophistication = _assess_fashion_sophistication(metadata)
    query_complexity = query_data.get("design_metrics", {}).get("design_complexity", 0)
    if query_complexity > 5:  # High complexity queries expect sophisticated pieces
        fashion_bonuses["fashion_sophistication"] = item_sophistication * 0.08
    
    # Trend and era relevance
    query_era = professional_features.get("fashion_era", "contemporary")
    item_era = metadata.get("fashion_era", "contemporary")
    if query_era == item_era:
        fashion_bonuses["trend_relevance"] = 0.07
    
    # Designer precision bonus
    designer_confidence = query_data.get("fashion_metadata", {}).get("designer_confidence", "medium")
    if designer_confidence == "high":
        # Reward items that match multiple fashion criteria
        criteria_met = sum(1 for bonus in fashion_bonuses.values() if bonus > 0.05)
        fashion_bonuses["designer_precision"] = min(criteria_met / 8.0, 1.0) * 0.09
    
    # Calculate comprehensive fashion relevance
    total_fashion_bonus = sum(fashion_bonuses.values())
    final_relevance = base_score + total_fashion_bonus
    
    # Apply fashion industry quality multiplier
    fashion_quality_multiplier = 1.0
    if metadata.get("is_luxury_fabric", False):
        fashion_quality_multiplier *= 1.05
    if metadata.get("color_sophistication") == "sophisticated":
        fashion_quality_multiplier *= 1.03
    
    final_relevance *= fashion_quality_multiplier
    
    return {
        "base_similarity": base_score,
        "fashion_bonuses": fashion_bonuses,
        "total_fashion_bonus": total_fashion_bonus,
        "fashion_quality_multiplier": fashion_quality_multiplier,
        "final_relevance": min(final_relevance, 1.0),
        "fashion_match_level": _determine_fashion_match_level(final_relevance),
        "professional_grade": final_relevance > 0.65
    }

def _match_color_temperature(query_colors: Dict, item_color_analysis: Dict) -> float:
    """Match color temperature for fashion harmony."""
    # Implementation would analyze warm/cool color matching
    return 0.8  # Placeholder

def _match_color_harmony(query_colors: Dict, item_color_analysis: Dict) -> float:
    """Analyze color harmony compatibility."""
    # Implementation would check complementary, analogous, etc.
    return 0.7  # Placeholder

def _calculate_garment_precision_match(query_garments: Dict, item_garment_info: Dict) -> float:
    """Calculate precision of garment type matching."""
    # Implementation would check category -> subcategory -> type hierarchy
    return 0.9  # Placeholder

def _calculate_seasonal_alignment(query_seasons: Dict, item_season: str, metadata: Dict) -> float:
    """Calculate seasonal collection alignment."""
    # Implementation would check season compatibility
    return 0.85  # Placeholder

def _calculate_style_consistency(query_styles: Dict, item_style: str, metadata: Dict) -> float:
    """Calculate fashion style consistency."""
    # Implementation would check style compatibility
    return 0.8  # Placeholder

def _calculate_fabric_compatibility(query_fabrics: Dict, item_fabrics: Dict) -> float:
    """Calculate fabric and material compatibility."""
    # Implementation would check fabric matches
    return 0.75  # Placeholder

def _calculate_occasion_suitability(query_occasions: Dict, item_occasion: str, metadata: Dict) -> float:
    """Calculate occasion appropriateness."""
    # Implementation would check occasion matching
    return 0.7  # Placeholder

def _assess_fashion_sophistication(metadata: Dict) -> float:
    """Assess the fashion sophistication level of an item."""
    sophistication_score = 0.5  # Base score
    
    # Luxury indicators
    if metadata.get("is_luxury_fabric", False):
        sophistication_score += 0.2
    if metadata.get("care_requirements", []) and "professional" in metadata.get("care_requirements", []):
        sophistication_score += 0.1
    if metadata.get("color_sophistication") == "sophisticated":
        sophistication_score += 0.15
    if metadata.get("garment_category") in ["formal", "luxury"]:
        sophistication_score += 0.15
    
    return min(sophistication_score, 1.0)

def _determine_fashion_match_level(relevance_score: float) -> str:
    """Determine fashion industry match quality level."""
    if relevance_score >= 0.8:
        return "exceptional"
    elif relevance_score >= 0.7:
        return "excellent"
    elif relevance_score >= 0.6:
        return "very_good"
    elif relevance_score >= 0.5:
        return "good"
    elif relevance_score >= 0.4:
        return "acceptable"
    else:
        return "poor"

def _apply_fashion_industry_filtering(candidates: List[Dict], query_data: Dict, precision_level: str) -> List[Dict]:
    """Apply fashion industry professional filtering standards."""
    if not candidates:
        return []
    
    precision_config = FASHION_PRECISION_LEVELS.get(precision_level, FASHION_PRECISION_LEVELS["moderate"])
    max_results = precision_config["max_results"]
    min_quality = precision_config["quality_min"]
    
    # Quality filtering
    quality_hierarchy = ["exceptional", "excellent", "very_good", "good", "acceptable", "poor"]
    min_quality_index = quality_hierarchy.index(min_quality) if min_quality in quality_hierarchy else 4
    
    high_quality_candidates = []
    for candidate in candidates:
        fashion_relevance = candidate.get("fashion_relevance", {})
        match_level = fashion_relevance.get("fashion_match_level", "poor")
        
        if match_level in quality_hierarchy:
            match_index = quality_hierarchy.index(match_level)
            if match_index <= min_quality_index:
                high_quality_candidates.append(candidate)
    
    # Sort by fashion relevance
    high_quality_candidates.sort(key=lambda x: x.get("final_fashion_score", 0), reverse=True)
    
    # Fashion industry gap analysis
    if len(high_quality_candidates) > 1:
        scores = [c.get("final_fashion_score", 0) for c in high_quality_candidates]
        
        # Find significant quality gaps
        for i in range(1, len(scores)):
            gap = scores[i-1] - scores[i]
            if gap > 0.08 and precision_level in ["exact", "high"]:  # Significant quality drop
                high_quality_candidates = high_quality_candidates[:i]
                print(f"🎨 Applied fashion quality gap filter: {gap:.3f} gap detected")
                break
    
    # Apply result count limits
    final_results = high_quality_candidates[:max_results]
    
    print(f"🎨 Fashion filtering: {len(candidates)} → {len(high_quality_candidates)} → {len(final_results)}")
    
    return final_results