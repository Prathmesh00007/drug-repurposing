"""DRKG-based drug repurposing agent with automated disease resolution."""

import logging
import numpy as np
from typing import List, Dict, Optional
from kg.drkg_loader import DRKGLoader
from kg.automated_disease_mapper import resolve_disease_to_drkg  # NEW IMPORT
from agents.base import cache_manager

logger = logging.getLogger(__name__)
loader = DRKGLoader()

def _compute_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


async def run_drkg_discovery(disease_name: str, existing_candidates: List[str]) -> Dict:
    """
    Discover novel drug candidates using DRKG embeddings.
    
    FIXES Flaw #4: Fully automated disease mapping via NCBI MeSH API.
    
    Returns:
        {
            "hidden_candidates": [...],  # Novel candidates not in existing list
            "validated_candidates": [...]  # Known candidates with DRKG scores
        }
    """
    logger.info(f"🧬 DRKG Discovery: {disease_name}")
    
    cache_key = {"disease": disease_name}
    cached = cache_manager.get("drkg_discovery", cache_key)
    if cached:
        return cached
    
    try:
        # Load DRKG
        loader.load()  # ✅ FIXED

        entity_map = loader.entity_to_id
        embeddings = loader.entity_embeddings
        triples = loader.triples_df
            
        # AUTOMATED: Resolve disease to DRKG ID using APIs
        disease_drkg_id = await resolve_disease_to_drkg(disease_name)
        
        if not disease_drkg_id:
            logger.warning(f"⚠️ Disease '{disease_name}' not found in DRKG (automated resolution failed)")
            return {"hidden_candidates": [], "validated_candidates": []}
        
        if disease_drkg_id not in entity_map:
            logger.warning(f"⚠️ Disease ID '{disease_drkg_id}' not in DRKG entity map")
            return {"hidden_candidates": [], "validated_candidates": []}
        
        disease_idx = entity_map[disease_drkg_id]
        disease_emb = embeddings[disease_idx]
        
        # Find all compounds in DRKG
        compound_entities = {ent: idx for ent, idx in entity_map.items() if ent.startswith("Compound::")}
        logger.info(f"Found {len(compound_entities)} compounds in DRKG")
        
        # Compute similarities
        candidate_scores = []
        for compound_ent, compound_idx in compound_entities.items():
            compound_emb = embeddings[compound_idx]
            similarity = _compute_embedding_similarity(disease_emb, compound_emb)
            
            # Extract compound name (format: "Compound::DB00001" -> "DB00001")
            compound_name = compound_ent.split("::")[-1]
            
            candidate_scores.append({
                "name": compound_name,
                "drkg_entity": compound_ent,
                "drkg_score": similarity
            })
        
        # Sort by similarity (descending)
        candidate_scores.sort(key=lambda x: x["drkg_score"], reverse=True)
        
        # Split into hidden and validated
        existing_set = {c.upper() for c in existing_candidates}
        
        hidden_candidates = []
        validated_candidates = []
        
        for cand in candidate_scores[:100]:  # Top 100
            if cand["drkg_score"] < 0.3:  # Minimum threshold
                continue
            
            if cand["name"].upper() in existing_set:
                validated_candidates.append(cand)
            else:
                hidden_candidates.append(cand)
        
        result = {
            "hidden_candidates": hidden_candidates[:10],  # Top 10 novel
            "validated_candidates": validated_candidates[:10]  # Top 10 validated
        }
        
        logger.info(f"✅ DRKG Discovery: {len(hidden_candidates)} hidden, {len(validated_candidates)} validated")
        
        cache_manager.set("drkg_discovery", cache_key, result)
        return result
        
    except Exception as e:
        logger.error(f"❌ DRKG discovery failed: {e}")
        return {"hidden_candidates": [], "validated_candidates": []}
