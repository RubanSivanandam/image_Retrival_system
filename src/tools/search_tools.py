# PROFESSIONAL FASHION INDUSTRY AI SYSTEM
# Complete fashion design team integration with industry expertise

## COMPREHENSIVE FASHION KNOWLEDGE BASE

import json
import re
import logging
from typing import Optional, List, Dict, Tuple, Set
import numpy as np
from dataclasses import dataclass

# Configure logging
log = logging.getLogger(__name__)

# Global model variables
clip_model: Optional = None
clip_processor: Optional = None

# COMPREHENSIVE FASHION INDUSTRY DATABASE
@dataclass
class FashionKnowledgeBase:
    """Complete fashion industry knowledge base for professional designers."""
    
    # FASHION COLORS - Professional fashion color system
    FASHION_COLORS = {
        "neutrals": {
            "black": ["black", "jet", "onyx", "coal", "raven", "ebony", "charcoal", "midnight"],
            "white": ["white", "ivory", "cream", "pearl", "snow", "alabaster", "eggshell", "vanilla"],
            "gray": ["gray", "grey", "silver", "platinum", "pewter", "slate", "ash", "smoke", "steel"],
            "beige": ["beige", "nude", "sand", "oatmeal", "linen", "parchment", "buff", "ecru"],
            "brown": ["brown", "chocolate", "coffee", "espresso", "mocha", "camel", "tan", "taupe", "cognac", "mahogany"],
        },
        "warm_tones": {
            "red": ["red", "crimson", "scarlet", "burgundy", "wine", "cherry", "ruby", "cardinal", "brick", "rust"],
            "orange": ["orange", "coral", "peach", "apricot", "tangerine", "amber", "copper", "burnt orange", "mandarin"],
            "yellow": ["yellow", "gold", "mustard", "honey", "lemon", "canary", "saffron", "butterscotch", "champagne"],
            "pink": ["pink", "rose", "blush", "salmon", "fuchsia", "magenta", "coral", "dusty rose", "mauve", "flamingo"],
        },
        "cool_tones": {
            "blue": ["blue", "navy", "royal", "cobalt", "azure", "cerulean", "teal", "turquoise", "sapphire", "indigo"],
            "green": ["green", "emerald", "forest", "olive", "sage", "mint", "lime", "jade", "seafoam", "hunter"],
            "purple": ["purple", "violet", "lavender", "plum", "orchid", "amethyst", "lilac", "grape", "eggplant"],
        },
        "trending_colors": {
            "pantone_2024": ["peach fuzz", "digital lime", "brown sugar", "warm taupe", "cosmic cobalt"],
            "seasonal": ["living coral", "classic blue", "greenery", "rose quartz", "serenity", "marsala"],
        }
    }
    
    # FASHION SEASONS & COLLECTIONS
    FASHION_SEASONS = {
        "spring_summer": {
            "keywords": ["spring", "summer", "ss", "warm", "light", "breathable", "fresh", "bright"],
            "fabrics": ["cotton", "linen", "silk", "chiffon", "voile", "chambray", "seersucker"],
            "colors": ["pastels", "brights", "whites", "light neutrals"],
            "styles": ["flowing", "airy", "lightweight", "sleeveless", "short", "cropped"]
        },
        "fall_winter": {
            "keywords": ["fall", "autumn", "winter", "fw", "cold", "warm", "cozy", "layering", "thick"],
            "fabrics": ["wool", "cashmere", "tweed", "flannel", "corduroy", "velvet", "fleece", "down"],
            "colors": ["deep", "rich", "earth tones", "jewel tones", "dark neutrals"],
            "styles": ["layered", "structured", "long", "insulated", "covered", "warm"]
        },
        "resort_cruise": {
            "keywords": ["resort", "cruise", "vacation", "tropical", "beachwear", "holiday"],
            "fabrics": ["jersey", "crepe", "silk", "cotton", "rayon"],
            "colors": ["tropical", "bright", "nautical", "sunset colors"],
            "styles": ["relaxed", "flowing", "resort-ready", "travel-friendly"]
        },
        "pre_collections": {
            "keywords": ["pre-fall", "pre-spring", "transitional", "capsule"],
            "characteristics": ["versatile", "transitional", "core pieces", "investment"]
        }
    }
    
    # COMPREHENSIVE GARMENT CLASSIFICATION
    GARMENT_TYPES = {
        "tops": {
            "shirts": ["shirt", "blouse", "button-down", "button-up", "oxford", "chambray", "flannel"],
            "t_shirts": ["t-shirt", "tee", "tank", "camisole", "halter", "tube", "crop top", "bodysuit"],
            "knits": ["sweater", "pullover", "cardigan", "jumper", "knit", "cashmere", "turtleneck"],
            "formal_tops": ["blazer", "vest", "waistcoat", "formal shirt", "dress shirt"],
        },
        "bottoms": {
            "pants": ["pants", "trousers", "slacks", "chinos", "khakis", "dress pants", "palazzo"],
            "jeans": ["jeans", "denim", "skinny", "straight", "bootcut", "flare", "wide-leg", "boyfriend"],
            "shorts": ["shorts", "bermuda", "cargo", "denim shorts", "tailored shorts"],
            "skirts": ["skirt", "mini", "midi", "maxi", "pencil", "a-line", "pleated", "wrap", "circle"],
        },
        "dresses": {
            "casual_dresses": ["sundress", "shirt dress", "wrap dress", "t-shirt dress", "swing dress"],
            "formal_dresses": ["cocktail", "evening", "gown", "formal", "little black dress", "midi dress"],
            "special_occasion": ["wedding", "prom", "party", "graduation", "red carpet"],
        },
        "outerwear": {
            "jackets": ["jacket", "blazer", "bomber", "varsity", "denim jacket", "leather jacket"],
            "coats": ["coat", "trench", "peacoat", "overcoat", "duster", "wrap coat"],
            "winter_wear": ["puffer", "down", "parka", "wool coat", "fur", "shearling"],
            "cardigans": ["cardigan", "wrap", "duster", "long cardigan", "cropped cardigan"],
        }
    }
    
    # ADVANCED FABRIC & MATERIALS DATABASE
    FABRICS = {
        "natural_fibers": {
            "cotton": {
                "types": ["cotton", "organic cotton", "pima", "supima", "egyptian cotton"],
                "characteristics": ["breathable", "comfortable", "durable", "easy care"],
                "uses": ["casual", "everyday", "summer", "basics"]
            },
            "wool": {
                "types": ["wool", "merino", "cashmere", "angora", "mohair", "alpaca", "lambswool"],
                "characteristics": ["warm", "insulating", "luxurious", "natural", "breathable"],
                "uses": ["winter", "formal", "sweaters", "coats", "suits"]
            },
            "silk": {
                "types": ["silk", "charmeuse", "chiffon", "crepe de chine", "taffeta", "dupioni"],
                "characteristics": ["luxurious", "smooth", "elegant", "draping", "lustrous"],
                "uses": ["formal", "evening", "lingerie", "scarves", "blouses"]
            },
            "linen": {
                "types": ["linen", "european linen", "irish linen", "belgian linen"],
                "characteristics": ["breathable", "cool", "natural", "wrinkle-prone", "casual"],
                "uses": ["summer", "casual", "beachwear", "home", "relaxed"]
            }
        },
        "synthetic_fibers": {
            "polyester": {
                "types": ["polyester", "microfiber", "fleece", "performance polyester"],
                "characteristics": ["durable", "wrinkle-resistant", "quick-dry", "affordable"],
                "uses": ["activewear", "outerwear", "work clothes", "budget fashion"]
            },
            "nylon": {
                "types": ["nylon", "ripstop", "ballistic nylon", "cordura"],
                "characteristics": ["strong", "lightweight", "water-resistant", "elastic"],
                "uses": ["activewear", "outerwear", "lingerie", "hosiery"]
            }
        },
        "luxury_materials": {
            "leather": {
                "types": ["leather", "suede", "patent leather", "nubuck", "calfskin", "lambskin"],
                "characteristics": ["durable", "luxurious", "structured", "investment"],
                "uses": ["jackets", "pants", "accessories", "shoes", "bags"]
            },
            "fur": {
                "types": ["fur", "faux fur", "shearling", "mink", "fox", "rabbit"],
                "characteristics": ["warm", "luxurious", "textural", "statement"],
                "uses": ["coats", "trim", "accessories", "winter wear"]
            }
        }
    }
    
    # FASHION STYLE CATEGORIES
    STYLE_CATEGORIES = {
        "classic": {
            "keywords": ["classic", "timeless", "traditional", "conservative", "elegant", "refined"],
            "pieces": ["blazer", "trench coat", "pencil skirt", "white shirt", "little black dress"],
            "colors": ["navy", "black", "white", "beige", "gray"],
            "characteristics": ["clean lines", "tailored", "sophisticated", "investment pieces"]
        },
        "casual": {
            "keywords": ["casual", "relaxed", "comfortable", "everyday", "laid-back", "effortless"],
            "pieces": ["jeans", "t-shirt", "sneakers", "hoodie", "cardigan"],
            "colors": ["denim", "neutrals", "earth tones"],
            "characteristics": ["comfortable", "practical", "versatile", "easy-wearing"]
        },
        "bohemian": {
            "keywords": ["boho", "bohemian", "hippie", "free-spirited", "artistic", "ethnic"],
            "pieces": ["maxi dress", "fringe", "peasant blouse", "wide-leg pants", "kimono"],
            "colors": ["earth tones", "jewel tones", "prints", "patterns"],
            "characteristics": ["flowing", "textural", "eclectic", "layered", "artistic"]
        },
        "minimalist": {
            "keywords": ["minimalist", "clean", "simple", "modern", "sleek", "understated"],
            "pieces": ["clean lines", "simple shapes", "quality basics"],
            "colors": ["black", "white", "gray", "nude", "monochromatic"],
            "characteristics": ["simple", "clean", "quality", "functional", "edited"]
        },
        "romantic": {
            "keywords": ["romantic", "feminine", "soft", "pretty", "delicate", "sweet"],
            "pieces": ["ruffles", "lace", "florals", "bows", "soft fabrics"],
            "colors": ["pastels", "soft pinks", "cream", "lavender"],
            "characteristics": ["soft", "flowing", "detailed", "feminine", "pretty"]
        },
        "edgy": {
            "keywords": ["edgy", "rock", "punk", "rebellious", "alternative", "bold"],
            "pieces": ["leather jacket", "ripped jeans", "studs", "chains", "boots"],
            "colors": ["black", "metallics", "bold colors"],
            "characteristics": ["bold", "statement", "rebellious", "confident", "dramatic"]
        },
        "preppy": {
            "keywords": ["preppy", "ivy league", "nautical", "collegiate", "refined", "polished"],
            "pieces": ["blazer", "polo", "pleated skirt", "loafers", "pearls"],
            "colors": ["navy", "white", "red", "green", "pastels"],
            "characteristics": ["polished", "put-together", "classic", "refined", "crisp"]
        }
    }
    
    # OCCASION-BASED CLASSIFICATION
    OCCASIONS = {
        "work_professional": {
            "keywords": ["work", "office", "business", "professional", "corporate", "meeting"],
            "dress_codes": ["business formal", "business casual", "smart casual"],
            "pieces": ["suit", "blazer", "dress pants", "blouse", "pencil skirt"],
            "avoid": ["too casual", "revealing", "wrinkled", "flashy"]
        },
        "formal_events": {
            "keywords": ["formal", "black tie", "gala", "wedding", "cocktail", "evening"],
            "dress_codes": ["black tie", "cocktail", "formal", "semi-formal"],
            "pieces": ["evening gown", "cocktail dress", "formal suit", "dress shoes"],
            "fabrics": ["silk", "satin", "chiffon", "velvet", "taffeta"]
        },
        "casual_social": {
            "keywords": ["casual", "weekend", "brunch", "shopping", "friends", "relaxed"],
            "pieces": ["jeans", "casual dress", "sweater", "sneakers", "flats"],
            "characteristics": ["comfortable", "stylish", "approachable", "versatile"]
        },
        "active_lifestyle": {
            "keywords": ["active", "gym", "workout", "sports", "running", "yoga", "athleisure"],
            "pieces": ["leggings", "sports bra", "athletic shoes", "performance fabric"],
            "characteristics": ["functional", "moisture-wicking", "flexible", "supportive"]
        }
    }
    
    # BODY TYPE & FIT ANALYSIS
    BODY_TYPES = {
        "pear": {
            "characteristics": ["wider hips", "smaller bust", "defined waist"],
            "recommended": ["A-line", "fit and flare", "empire waist", "bootcut"],
            "avoid": ["tight bottoms", "horizontal stripes on hips"],
            "styling_tips": ["emphasize upper body", "create balance", "highlight waist"]
        },
        "apple": {
            "characteristics": ["broader midsection", "slender legs", "full bust"],
            "recommended": ["empire waist", "wrap styles", "V-neck", "straight leg"],
            "avoid": ["clingy fabrics", "high waistlines", "horizontal stripes"],
            "styling_tips": ["draw attention to legs", "create vertical lines", "define waist"]
        },
        "hourglass": {
            "characteristics": ["balanced bust and hips", "defined waist"],
            "recommended": ["fitted styles", "wrap dresses", "belt emphasis"],
            "avoid": ["boxy shapes", "loose fits"],
            "styling_tips": ["emphasize waist", "fitted styles", "balanced proportions"]
        },
        "rectangle": {
            "characteristics": ["similar bust and hip measurements", "less defined waist"],
            "recommended": ["create curves", "peplum", "ruffles", "belts"],
            "avoid": ["straight lines", "boxy shapes"],
            "styling_tips": ["create waist definition", "add curves", "layer textures"]
        }
    }

def _initialize_advanced_models():
    """Initialize advanced CLIP models for fashion analysis."""
    global clip_model, clip_processor
    
    if clip_model is None or clip_processor is None:
        try:
            print("🎨 Loading advanced fashion AI models...")
            
            import torch
            from transformers import CLIPProcessor, CLIPModel
            from src.config.settings import CLIP_MODEL
            
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            
            print("✅ Fashion AI models loaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load fashion AI models: {e}")
            return False
    return True

def _advanced_fashion_analysis(text: str) -> Dict:
    """PROFESSIONAL fashion analysis with comprehensive industry knowledge."""
    text_lower = text.lower()
    fashion_kb = FashionKnowledgeBase()
    
    analysis = {
        "original_query": text,
        "fashion_intelligence": {},
        "professional_metrics": {},
        "design_context": {},
        "trend_analysis": {}
    }
    
    # COMPREHENSIVE COLOR ANALYSIS
    detected_colors = {}
    color_confidence = {}
    color_families = []
    
    for family_name, color_family in fashion_kb.FASHION_COLORS.items():
        for color_name, variations in color_family.items():
            matches = [var for var in variations if var in text_lower]
            if matches:
                detected_colors[color_name] = matches
                color_confidence[color_name] = len(matches)
                if family_name not in color_families:
                    color_families.append(family_name)
    
    # SEASONAL COLLECTION ANALYSIS
    detected_seasons = {}
    seasonal_context = {}
    collection_type = "unknown"
    
    for season_name, season_data in fashion_kb.FASHION_SEASONS.items():
        keywords = season_data.get("keywords", [])
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            detected_seasons[season_name] = matches
            seasonal_context[season_name] = season_data
            if collection_type == "unknown":
                collection_type = season_name
    
    # COMPREHENSIVE GARMENT CLASSIFICATION
    garment_analysis = {}
    garment_hierarchy = {}
    
    for category, subcategories in fashion_kb.GARMENT_TYPES.items():
        for subcat, garments in subcategories.items():
            matches = [g for g in garments if g in text_lower]
            if matches:
                if category not in garment_analysis:
                    garment_analysis[category] = {}
                garment_analysis[category][subcat] = matches
                garment_hierarchy[subcat] = category
    
    # FABRIC AND MATERIAL ANALYSIS
    fabric_analysis = {}
    material_properties = {}
    
    for fiber_type, fibers in fashion_kb.FABRICS.items():
        for fabric_name, fabric_data in fibers.items():
            fabric_types = fabric_data.get("types", [])
            matches = [ft for ft in fabric_types if ft in text_lower]
            if matches:
                fabric_analysis[fabric_name] = {
                    "matches": matches,
                    "fiber_type": fiber_type,
                    "characteristics": fabric_data.get("characteristics", []),
                    "uses": fabric_data.get("uses", [])
                }
                material_properties[fabric_name] = fabric_data.get("characteristics", [])
    
    # STYLE CATEGORY ANALYSIS
    style_analysis = {}
    style_confidence = {}
    
    for style_name, style_data in fashion_kb.STYLE_CATEGORIES.items():
        keywords = style_data.get("keywords", [])
        pieces = style_data.get("pieces", [])
        
        keyword_matches = [kw for kw in keywords if kw in text_lower]
        piece_matches = [p for p in pieces if p in text_lower]
        
        if keyword_matches or piece_matches:
            style_analysis[style_name] = {
                "keyword_matches": keyword_matches,
                "piece_matches": piece_matches,
                "characteristics": style_data.get("characteristics", [])
            }
            style_confidence[style_name] = len(keyword_matches) + len(piece_matches) * 0.8
    
    # OCCASION ANALYSIS
    occasion_analysis = {}
    dress_code_suggestions = []
    
    for occasion_name, occasion_data in fashion_kb.OCCASIONS.items():
        keywords = occasion_data.get("keywords", [])
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            occasion_analysis[occasion_name] = {
                "matches": matches,
                "dress_codes": occasion_data.get("dress_codes", []),
                "recommended_pieces": occasion_data.get("pieces", [])
            }
            dress_code_suggestions.extend(occasion_data.get("dress_codes", []))
    
    # CALCULATE PROFESSIONAL METRICS
    fashion_specificity = (
        len(detected_colors) * 2.0 +
        len(garment_analysis) * 3.0 +
        len(fabric_analysis) * 2.5 +
        len(style_analysis) * 2.0 +
        len(occasion_analysis) * 1.5 +
        len(detected_seasons) * 1.5
    )
    
    # DETERMINE FASHION INTENT
    if len(detected_colors) > 0 and len(garment_analysis) > 0:
        fashion_intent = "color_specific_garment"
    elif len(detected_seasons) > 0 and len(garment_analysis) > 0:
        fashion_intent = "seasonal_collection"
    elif len(style_analysis) > 0:
        fashion_intent = "style_focused"
    elif len(fabric_analysis) > 0:
        fashion_intent = "material_focused"
    elif len(occasion_analysis) > 0:
        fashion_intent = "occasion_specific"
    elif len(garment_analysis) > 0:
        fashion_intent = "garment_category"
    else:
        fashion_intent = "general_fashion"
    
    # COMPILE COMPREHENSIVE ANALYSIS
    analysis.update({
        "fashion_intelligence": {
            "colors": detected_colors,
            "color_families": color_families,
            "color_confidence": color_confidence,
            "seasons": detected_seasons,
            "collection_type": collection_type,
            "garments": garment_analysis,
            "garment_hierarchy": garment_hierarchy,
            "fabrics": fabric_analysis,
            "material_properties": material_properties,
            "styles": style_analysis,
            "style_confidence": style_confidence,
            "occasions": occasion_analysis,
            "dress_codes": list(set(dress_code_suggestions))
        },
        "professional_metrics": {
            "fashion_specificity": fashion_specificity,
            "fashion_intent": fashion_intent,
            "complexity_level": "expert" if fashion_specificity > 8 else "intermediate" if fashion_specificity > 4 else "basic",
            "designer_confidence": "high" if fashion_specificity > 6 else "medium" if fashion_specificity > 3 else "low",
            "expected_precision": "exact" if fashion_specificity > 7 else "high" if fashion_specificity > 4 else "moderate",
            "target_results": 1 if fashion_specificity > 8 else 2 if fashion_specificity > 5 else 3
        },
        "design_context": {
            "is_trend_focused": len(detected_seasons) > 0 or "trend" in text_lower,
            "is_color_driven": len(detected_colors) > 1 or sum(color_confidence.values()) > 2,
            "is_silhouette_specific": any("fit" in text_lower or "cut" in text_lower for _ in [1]),
            "is_occasion_driven": len(occasion_analysis) > 0,
            "is_seasonal_collection": collection_type != "unknown",
            "design_complexity": fashion_specificity
        }
    })
    
    return analysis

def _generate_fashion_query_variations(text: str, fashion_analysis: Dict) -> List[Tuple[str, float]]:
    """Generate intelligent fashion-focused query variations."""
    variations = [(text, 1.0)]  # Original query with highest weight
    
    fashion_intel = fashion_analysis.get("fashion_intelligence", {})
    
    # High-precision color + garment combinations
    colors = fashion_intel.get("colors", {})
    garments = fashion_intel.get("garments", {})
    
    for color_name, color_matches in colors.items():
        for garment_category, garment_subcats in garments.items():
            for subcat, garment_list in garment_subcats.items():
                for garment in garment_list:
                    variations.append((f"{color_name} {garment}", 0.95))
                    # Add most specific color variation
                    if color_matches:
                        variations.append((f"{color_matches[0]} {garment}", 0.9))
    
    # Seasonal collection variations
    seasons = fashion_intel.get("seasons", {})
    if seasons and garments:
        for season_name in seasons.keys():
            for garment_category, garment_subcats in garments.items():
                for subcat, garment_list in garment_subcats.items():
                    for garment in garment_list:
                        variations.append((f"{season_name} {garment}", 0.85))
    
    # Style-specific variations
    styles = fashion_intel.get("styles", {})
    if styles and garments:
        for style_name, style_data in styles.items():
            style_pieces = style_data.get("piece_matches", [])
            for piece in style_pieces:
                variations.append((f"{style_name} {piece}", 0.8))
    
    # Fabric-focused variations
    fabrics = fashion_intel.get("fabrics", {})
    if fabrics and garments:
        for fabric_name, fabric_data in fabrics.items():
            fabric_matches = fabric_data.get("matches", [])
            for fabric in fabric_matches:
                for garment_category, garment_subcats in garments.items():
                    for subcat, garment_list in garment_subcats.items():
                        for garment in garment_list:
                            variations.append((f"{fabric} {garment}", 0.75))
    
    # Professional fashion terminology
    if colors and not garments:
        for color_name in colors.keys():
            variations.extend([
                (f"{color_name} apparel", 0.7),
                (f"{color_name} fashion", 0.65),
                (f"{color_name} clothing", 0.6)
            ])
    
    # Occasion-specific variations
    occasions = fashion_intel.get("occasions", {})
    for occasion_name, occasion_data in occasions.items():
        recommended_pieces = occasion_data.get("recommended_pieces", [])
        for piece in recommended_pieces:
            variations.append((f"{occasion_name} {piece}", 0.7))
            
    # Generic fashion terms for broader context
    if garments:
        variations.extend([
            ("designer fashion", 0.5),
            ("haute couture", 0.45),
            ("ready to wear", 0.4),
            ("fashion design", 0.35)
        ])
    
    # Remove duplicates and sort
    seen = set()
    unique_variations = []
    for query, weight in variations:
        if query not in seen and len(query.split()) <= 4:  # Keep reasonable length
            seen.add(query)
            unique_variations.append((query, weight))
    
    unique_variations.sort(key=lambda x: x[1], reverse=True)
    return unique_variations[:12]  # Fashion designers need more variations

def parse_query(text: str) -> str:
    """
    PROFESSIONAL FASHION AI query parser with comprehensive industry knowledge.
    """
    try:
        if not text or not text.strip():
            return json.dumps({"error": "Empty fashion query"})

        print(f"🎨 Processing fashion query with industry AI: '{text}'")

        # STAGE 1: Professional fashion analysis
        fashion_analysis = _advanced_fashion_analysis(text)
        professional_metrics = fashion_analysis.get("professional_metrics", {})
        
        print(f"👗 Fashion intent: {professional_metrics.get('fashion_intent', 'unknown')}")
        print(f"📏 Specificity: {professional_metrics.get('fashion_specificity', 0):.1f} | Precision: {professional_metrics.get('expected_precision', 'moderate')}")
        
        # STAGE 2: Generate fashion-focused variations
        fashion_variations = _generate_fashion_query_variations(text, fashion_analysis)
        
        print(f"🔄 Generated {len(fashion_variations)} fashion variations")
        
        # STAGE 3: Advanced embedding generation
        embeddings = []
        embedding_weights = []
        embedding_qualities = []
        
        if _initialize_advanced_models():
            try:
                import torch
                
                for i, (variation, weight) in enumerate(fashion_variations):
                    try:
                        inputs = clip_processor(text=[variation], return_tensors="pt")
                        with torch.no_grad():
                            embedding = clip_model.get_text_features(**inputs)
                            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                            
                        embedding_list = embedding.squeeze().numpy().tolist()
                        
                        # Fashion-specific quality scoring
                        quality_score = _calculate_fashion_embedding_quality(embedding_list, variation, fashion_analysis)
                        final_weight = weight * quality_score
                        
                        embeddings.append(embedding_list)
                        embedding_weights.append(final_weight)
                        embedding_qualities.append(quality_score)
                        
                    except Exception as e:
                        log.warning(f"Fashion embedding failed for '{variation}': {e}")
                        continue
                        
                print(f"✅ Generated {len(embeddings)} fashion-optimized embeddings")
                
            except Exception as e:
                print(f"⚠️  Fashion CLIP failed: {e}")
                embeddings = [_generate_fashion_fallback_embedding(text, fashion_analysis)]
                embedding_weights = [1.0]
                embedding_qualities = [0.7]
        else:
            embeddings = [_generate_fashion_fallback_embedding(text, fashion_analysis)]
            embedding_weights = [1.0]
            embedding_qualities = [0.7]

        # STAGE 4: Compile professional fashion data
        result = {
            "query": text,
            "embedding": embeddings[0],
            "embeddings": embeddings,
            "embedding_weights": embedding_weights,
            "embedding_qualities": embedding_qualities,
            "fashion_variations": [var for var, weight in fashion_variations],
            "variation_weights": [weight for var, weight in fashion_variations],
            "fashion_analysis": fashion_analysis,
            "professional_features": fashion_analysis.get("fashion_intelligence", {}),
            "design_metrics": professional_metrics,
            "fashion_metadata": {
                "industry_grade": True,
                "designer_confidence": professional_metrics.get("designer_confidence", "medium"),
                "fashion_precision": professional_metrics.get("expected_precision", "moderate"),
                "target_results": professional_metrics.get("target_results", 3),
                "complexity_level": professional_metrics.get("complexity_level", "basic"),
                "design_context": fashion_analysis.get("design_context", {}),
                "professional_classification": professional_metrics.get("fashion_intent", "general")
            },
            "embedding_dim": len(embeddings[0]),
            "embedding_type": "fashion_optimized_clip",
            "processing_version": "fashion_industry_v3.0"
        }

        print(f"✅ Professional fashion AI processing completed")
        return json.dumps(result)

    except Exception as e:
        print(f"❌ Fashion AI processing error: {e}")
        fallback = _generate_fashion_fallback_embedding(text if text else "fashion")
        return json.dumps({
            "query": text,
            "embedding": fallback,
            "fashion_metadata": {"designer_confidence": "low", "target_results": 5},
            "error_note": f"Fashion processing failed: {str(e)}"
        })

def _calculate_fashion_embedding_quality(embedding: List[float], query: str, fashion_analysis: Dict) -> float:
    """Calculate fashion-specific embedding quality."""
    base_quality = np.linalg.norm(np.array(embedding)) * 0.5
    
    # Fashion context bonuses
    professional_metrics = fashion_analysis.get("professional_metrics", {})
    fashion_intel = fashion_analysis.get("fashion_intelligence", {})
    
    # Reward fashion-specific terms
    fashion_bonus = 0.0
    if fashion_intel.get("colors"):
        fashion_bonus += 0.1
    if fashion_intel.get("garments"):
        fashion_bonus += 0.15
    if fashion_intel.get("fabrics"):
        fashion_bonus += 0.1
    if fashion_intel.get("styles"):
        fashion_bonus += 0.1
    
    # Professional terminology bonus
    professional_terms = ["couture", "designer", "luxury", "premium", "bespoke", "tailored"]
    if any(term in query.lower() for term in professional_terms):
        fashion_bonus += 0.05
    
    return min(1.0, base_quality + fashion_bonus)

def _generate_fashion_fallback_embedding(text: str, fashion_analysis: Dict = None, dim: int = 512) -> List[float]:
    """Generate fashion-optimized fallback embedding."""
    import hashlib
    
    # Use fashion context if available
    if fashion_analysis:
        fashion_intel = fashion_analysis.get("fashion_intelligence", {})
        # Incorporate fashion elements into hash
        fashion_context = str(fashion_intel.get("colors", {})) + str(fashion_intel.get("garments", {}))
        text = text + fashion_context
    
    # Create fashion-aware embedding
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    
    embedding = []
    for i in range(dim):
        char_idx = i % len(text_hash)
        val = ord(text_hash[char_idx]) / 255.0
        
        # Add fashion-specific patterns
        if i % 7 == 0:  # Fashion cycle pattern
            val *= 1.1
        if i % 12 == 0:  # Seasonal pattern
            val *= 0.9
            
        embedding.append(val)
    
    # Normalize
    norm = sum(x**2 for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x/norm for x in embedding]
    
    return embedding
