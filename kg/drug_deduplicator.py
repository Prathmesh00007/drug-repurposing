"""
Drug Deduplicator - Resolve drug synonyms and merge duplicates
Uses ChEMBL API to normalize drug names to canonical ChEMBL IDs
"""

import httpx
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class NormalizedDrug:
    """Canonical drug representation"""
    chembl_id: str
    preferred_name: str
    synonyms: List[str]
    source_names: List[str]  # Original names from different sources


class DrugDeduplicator:
    """
    Deduplicates drugs from multiple sources.
    
    Strategy:
    1. Normalize all drug names to ChEMBL IDs
    2. Merge drugs with same ChEMBL ID
    3. Combine evidence from all sources
    """
    
    CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
    
    def __init__(self):
        self.timeout = httpx.Timeout(30.0)
        self._cache = {}  # Cache name -> ChEMBL ID mappings
    
    async def normalize_drug_name(self, drug_name: str) -> Optional[str]:
        """
        Normalize drug name to ChEMBL ID.
        
        Returns:
            ChEMBL ID (e.g., "CHEMBL123") or None
        """
        # Check cache first
        if drug_name in self._cache:
            return self._cache[drug_name]
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Search ChEMBL by name
                response = await client.get(
                    f"{self.CHEMBL_API}/molecule/search.json",
                    params={
                        "q": drug_name,
                        "limit": 5
                    },
                    headers={"Accept": "application/json"}
                )
                
                if response.status_code != 200:
                    return None
                
                data = response.json()
                molecules = data.get("molecules", [])
                
                if not molecules:
                    return None
                
                # Get top match
                top_match = molecules[0]
                chembl_id = top_match.get("molecule_chembl_id")
                
                # Cache result
                self._cache[drug_name] = chembl_id
                
                return chembl_id
                
            except Exception as e:
                logger.error(f"ChEMBL search failed for {drug_name}: {e}")
                return None
    
    async def get_drug_synonyms(self, chembl_id: str) -> List[str]:
        """Get all synonyms for a ChEMBL ID"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.CHEMBL_API}/molecule/{chembl_id}.json",
                    headers={"Accept": "application/json"}
                )
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                
                synonyms = []
                
                # Preferred name
                pref_name = data.get("pref_name")
                if pref_name:
                    synonyms.append(pref_name)
                
                # Trade names
                trade_names = data.get("molecule_synonyms", [])
                for syn in trade_names:
                    syn_text = syn.get("molecule_synonym")
                    if syn_text and syn_text not in synonyms:
                        synonyms.append(syn_text)
                
                return synonyms
                
            except Exception as e:
                logger.error(f"Failed to get synonyms for {chembl_id}: {e}")
                return []
    
    async def deduplicate_candidates(
        self,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Deduplicate candidate list by normalizing to ChEMBL IDs.
        
        Args:
            candidates: List of drug candidates with 'name' field
        
        Returns:
            Deduplicated list with merged evidence
        """
        logger.info(f"📦 Deduplicating {len(candidates)} candidates...")
        
        # Normalize all to ChEMBL IDs
        normalized_map = {}  # chembl_id -> list of original candidates
        
        for candidate in candidates:
            drug_name = candidate.get("name", "")
            
            # Try to get ChEMBL ID
            chembl_id = await self.normalize_drug_name(drug_name)
            
            if chembl_id:
                if chembl_id not in normalized_map:
                    normalized_map[chembl_id] = []
                normalized_map[chembl_id].append(candidate)
            else:
                # Couldn't normalize - keep as-is with synthetic ID
                synthetic_id = f"UNKNOWN_{drug_name.replace(' ', '_')}"
                normalized_map[synthetic_id] = [candidate]
        
        # Merge duplicates
        deduplicated = []
        
        for chembl_id, candidate_list in normalized_map.items():
            if len(candidate_list) == 1:
                # No duplicates
                merged = candidate_list[0]
                merged["chembl_id"] = chembl_id
            else:
                # Merge duplicates
                logger.info(f"   Merging {len(candidate_list)} entries for {chembl_id}")
                merged = self._merge_candidates(chembl_id, candidate_list)
            
            deduplicated.append(merged)
        
        logger.info(f"   ✓ Deduplicated to {len(deduplicated)} unique drugs (removed {len(candidates) - len(deduplicated)} duplicates)")
        
        return deduplicated
    
    def _merge_candidates(
        self,
        chembl_id: str,
        candidate_list: List[Dict]
    ) -> Dict:
        """Merge multiple candidate entries for same drug"""
        # Use first candidate as base
        merged = candidate_list[0].copy()
        merged["chembl_id"] = chembl_id
        
        # Collect all source names
        source_names = [c.get("name", "") for c in candidate_list]
        merged["source_names"] = source_names
        merged["name"] = source_names[0]  # Use first as primary
        
        # Merge scores (take maximum)
        scores = [c.get("score", 0.0) for c in candidate_list]
        merged["score"] = max(scores)
        
        # Merge targets (union)
        all_targets = set()
        for c in candidate_list:
            targets = c.get("targets", [])
            all_targets.update(targets)
        merged["targets"] = list(all_targets)
        
        # Merge evidence citations (union)
        all_citations = []
        for c in candidate_list:
            citations = c.get("evidence_citations", [])
            all_citations.extend(citations)
        merged["evidence_citations"] = all_citations
        
        # Add deduplication metadata
        merged["is_deduplicated"] = True
        merged["source_count"] = len(candidate_list)
        
        return merged


# Singleton
drug_deduplicator = DrugDeduplicator()
