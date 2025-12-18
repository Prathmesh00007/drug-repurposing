"""
Pathway-Based Mechanism Validator
Uses Reactome API (FREE) for deterministic pathway overlap calculation.

Replaces LLM-based mechanism validation with graph algorithms.
"""

import httpx
import asyncio
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MechanismDecision(str, Enum):
    KEEP = "KEEP"
    REJECT = "REJECT"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class PathwayOverlapResult:
    decision: MechanismDecision
    confidence: float  # 0.0 to 1.0
    jaccard_similarity: float
    shared_pathways: List[Dict]
    target_pathways: List[Dict]
    disease_pathways: List[Dict]
    reasoning: str


class PathwayMechanismValidator:
    """
    Validates target-disease mechanism using Reactome pathway database.
    
    Algorithm:
    1. Get disease-associated pathways from Reactome
    2. Get target-associated pathways from Reactome
    3. Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
    4. Decision thresholds:
       - Jaccard > 0.3: KEEP (high confidence)
       - Jaccard > 0.15: KEEP (moderate confidence)
       - Jaccard < 0.15: REJECT (low confidence)
    """
    
    REACTOME_API = "https://reactome.org/ContentService"
    TIMEOUT = httpx.Timeout(30.0, connect=10.0)
    
    # Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.3
    MODERATE_CONFIDENCE_THRESHOLD = 0.15
    
    async def validate_mechanism(
        self,
        target_symbol: str,
        disease_context,  # DiseaseContext from disease_resolver_v2
        species: str = "Homo sapiens"
    ) -> PathwayOverlapResult:
        """
        Validate if target has pathway overlap with disease.
        
        Args:
            target_symbol: Gene symbol (e.g., "JAK1")
            disease_context: DiseaseContext with disease info
            species: Species name (default: "Homo sapiens")
        
        Returns:
            PathwayOverlapResult with decision and reasoning
        """
        logger.info(f"🧬 Validating mechanism: {target_symbol} for {disease_context.corrected_name}")
        
        # Fetch pathways in parallel
        target_task = self._get_target_pathways(target_symbol, species)
        disease_task = self._get_disease_pathways(disease_context.corrected_name, species)
        
        target_pathways, disease_pathways = await asyncio.gather(
            target_task, disease_task, return_exceptions=True
        )
        
        # Handle errors
        if isinstance(target_pathways, Exception):
            logger.error(f"Target pathway query failed: {target_pathways}")
            target_pathways = []
        
        if isinstance(disease_pathways, Exception):
            logger.error(f"Disease pathway query failed: {disease_pathways}")
            disease_pathways = []
        
        if not target_pathways or not disease_pathways:
            return PathwayOverlapResult(
                decision=MechanismDecision.UNCERTAIN,
                confidence=0.0,
                jaccard_similarity=0.0,
                shared_pathways=[],
                target_pathways=target_pathways,
                disease_pathways=disease_pathways,
                reasoning=f"Insufficient pathway data (target: {len(target_pathways)}, disease: {len(disease_pathways)})"
            )
        
        # Calculate pathway overlap
        target_ids = {p["stId"] for p in target_pathways}
        disease_ids = {p["stId"] for p in disease_pathways}
        
        intersection = target_ids & disease_ids
        union = target_ids | disease_ids
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Get shared pathway details
        shared_pathways = [
            p for p in target_pathways if p["stId"] in intersection
        ]
        
        # Make decision
        if jaccard >= self.HIGH_CONFIDENCE_THRESHOLD:
            decision = MechanismDecision.KEEP
            confidence = min(jaccard / self.HIGH_CONFIDENCE_THRESHOLD, 1.0)
            reasoning = f"High pathway overlap (Jaccard: {jaccard:.2f}, {len(intersection)} shared pathways)"
        elif jaccard >= self.MODERATE_CONFIDENCE_THRESHOLD:
            decision = MechanismDecision.KEEP
            confidence = jaccard / self.HIGH_CONFIDENCE_THRESHOLD
            reasoning = f"Moderate pathway overlap (Jaccard: {jaccard:.2f}, {len(intersection)} shared pathways)"
        else:
            decision = MechanismDecision.REJECT
            confidence = 0.0
            reasoning = f"Low pathway overlap (Jaccard: {jaccard:.2f}, only {len(intersection)} shared pathways)"
        
        logger.info(f"   Decision: {decision}, Confidence: {confidence:.2f}, Jaccard: {jaccard:.2f}")
        
        return PathwayOverlapResult(
            decision=decision,
            confidence=confidence,
            jaccard_similarity=jaccard,
            shared_pathways=shared_pathways,
            target_pathways=target_pathways,
            disease_pathways=disease_pathways,
            reasoning=reasoning
        )
    
    async def _get_target_pathways(self, gene_symbol: str, species: str) -> List[Dict]:
        """Query Reactome for gene-associated pathways"""
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            try:
                response = await client.get(
                    f"{self.REACTOME_API}/data/pathways/low/entity/{gene_symbol}/allForms",
                    params={"species": species}
                )
                
                if response.status_code == 404:
                    logger.warning(f"Gene {gene_symbol} not found in Reactome")
                    return []
                
                response.raise_for_status()
                pathways = response.json()
                
                return pathways if isinstance(pathways, list) else []
                
            except Exception as e:
                logger.error(f"Reactome target query failed for {gene_symbol}: {e}")
                return []
    
    async def _get_disease_pathways(self, disease_name: str, species: str) -> List[Dict]:
        """
        Query Reactome for disease-associated pathways.
        
        Note: Reactome disease mapping is limited.
        Fallback strategy: search by disease keywords in pathway names.
        """
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            try:
                # Try direct query first
                response = await client.get(
                    f"{self.REACTOME_API}/data/diseases/{disease_name}"
                )
                
                if response.status_code == 200:
                    disease_data = response.json()
                    # Extract pathway IDs from disease
                    # (API structure varies, adapt as needed)
                    return []
                
                # Fallback: Search pathways by disease keywords
                search_response = await client.get(
                    f"{self.REACTOME_API}/search/query",
                    params={
                        "query": disease_name,
                        "species": species,
                        "types": "Pathway"
                    }
                )
                
                if search_response.status_code == 200:
                    search_results = search_response.json()
                    
                    pathway_list = []
                    if "results" in search_results:
                        for result in search_results["results"][:20]:  # Limit to top 20
                            if result.get("type") == "Pathway":
                                pathway_list.append({
                                    "stId": result.get("stId"),
                                    "displayName": result.get("name"),
                                    "species": result.get("species", [{}])[0].get("displayName")
                                })
                    
                    return pathway_list
                
                return []
                
            except Exception as e:
                logger.error(f"Reactome disease query failed for {disease_name}: {e}")
                return []


# Singleton
pathway_validator = PathwayMechanismValidator()
