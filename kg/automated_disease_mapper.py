"""Automated DRKG disease mapping using NCBI MeSH and EBI OLS APIs."""

import logging
import httpx
import re
from typing import Optional, Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd

logger = logging.getLogger(__name__)


class AutomatedDRKGMapper:
    """Automated disease name to DRKG entity mapper."""
    
    def __init__(self, drkg_entities_path: str = "data/drkg/entities.tsv"):
        """Initialize with DRKG entities file."""
        self.drkg_entities_path = drkg_entities_path
        self.disease_entities = {}
        self.disease_name_index = {}
        self._load_disease_entities()
    
    def _load_disease_entities(self):
        """Load all disease entities from DRKG and build indexes."""
        try:
            df = pd.read_csv(self.drkg_entities_path, sep='\t', header=None, names=['entity_id', 'entity_type'])
            disease_df = df[df['entity_id'].str.startswith('Disease::')]
            
            # Build multiple indexes for flexible matching
            for _, row in disease_df.iterrows():
                entity_id = row['entity_id']
                
                # Index 1: By MeSH ID (e.g., "Disease::MESH:D015658" -> store with key "D015658")
                if "MESH:" in entity_id:
                    mesh_id = entity_id.split("MESH:")[1]
                    self.disease_entities[mesh_id] = entity_id
                
                # Index 2: By DOID if present
                if "DOID:" in entity_id:
                    doid = entity_id.split("DOID:")[1]
                    self.disease_entities[doid] = entity_id
                
                # Index 3: By full entity for exact lookups
                self.disease_entities[entity_id] = entity_id
                
                # Index 4: By disease name (lowercase for fuzzy matching)
                entity_lower = entity_id.lower()
                self.disease_name_index[entity_lower] = entity_id
            
            logger.info(f"✓ Loaded {len(disease_df)} disease entities from DRKG")
            logger.debug(f"  Sample MeSH mappings: {list(self.disease_entities.keys())[:5]}")
            
        except Exception as e:
            logger.warning(f"Failed to load DRKG entities: {e}")
            self.disease_entities = {}
            self.disease_name_index = {}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _query_ncbi_mesh_simple(self, disease_name: str) -> Optional[Dict[str, str]]:
        """
        Query NCBI MeSH with SIMPLE query (no filters).
        
        Returns: {"mesh_id": "D015658", "name": "HIV Infections"} or None
        """
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            
            # SIMPLIFIED: Just search the disease name without restrictive filters
            params = {
                "db": "mesh",
                "term": disease_name,
                "retmode": "json",
                "retmax": 10
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                
                if response.status_code != 200:
                    return None
                
                data = response.json()
                mesh_ids = data.get("esearchresult", {}).get("idlist", [])
                
                if not mesh_ids:
                    logger.debug(f"No MeSH results for '{disease_name}'")
                    return None
                
                # Get detailed info for all results
                summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                summary_params = {
                    "db": "mesh",
                    "id": ",".join(mesh_ids),
                    "retmode": "json"
                }
                
                summary_resp = await client.get(summary_url, params=summary_params)
                
                if summary_resp.status_code != 200:
                    return None
                
                summary_data = summary_resp.json()
                results = summary_data.get("result", {})
                
                # Filter results to find the actual disease (not drugs/treatments)
                candidates = []
                for mesh_id in mesh_ids:
                    if mesh_id not in results:
                        continue
                    
                    result = results[mesh_id]
                    mesh_ui = result.get("ds_meshui", "")
                    terms = result.get("ds_meshterms", [])
                    name = terms[0] if terms else ""
                    
                    # Skip if it's clearly a drug/inhibitor/treatment
                    skip_terms = ["inhibitor", "antagonist", "agonist", "antibodies", "vaccine"]
                    if any(term.lower() in name.lower() for term in skip_terms):
                        continue
                    
                    candidates.append({
                        "mesh_id": mesh_ui,
                        "name": name,
                        "relevance": self._calculate_name_similarity(disease_name, name)
                    })
                
                if not candidates:
                    # Use first result even if it might be a drug
                    first_id = mesh_ids[0]
                    if first_id in results:
                        result = results[first_id]
                        mesh_ui = result.get("ds_meshui", "")
                        name = result.get("ds_meshterms", [""])[0]
                        logger.info(f"✓ NCBI MeSH: '{disease_name}' -> {mesh_ui} ('{name}')")
                        return {"mesh_id": mesh_ui, "name": name}
                    return None
                
                # Return best match
                best = max(candidates, key=lambda x: x["relevance"])
                logger.info(f"✓ NCBI MeSH: '{disease_name}' -> {best['mesh_id']} ('{best['name']}')")
                return {"mesh_id": best["mesh_id"], "name": best["name"]}
                
        except Exception as e:
            logger.debug(f"NCBI MeSH query failed: {e}")
            return None
    
    def _calculate_name_similarity(self, query: str, candidate: str) -> float:
        """Calculate similarity between query and candidate disease names."""
        query_lower = query.lower()
        candidate_lower = candidate.lower()
        
        # Exact match
        if query_lower == candidate_lower:
            return 1.0
        
        # Substring match
        if query_lower in candidate_lower or candidate_lower in query_lower:
            return 0.8
        
        # Word overlap
        query_words = set(query_lower.split())
        candidate_words = set(candidate_lower.split())
        
        if query_words & candidate_words:
            overlap = len(query_words & candidate_words)
            return 0.5 + (overlap / max(len(query_words), len(candidate_words))) * 0.3
        
        return 0.0
    
    def _search_drkg_by_mesh(self, mesh_id: str) -> Optional[str]:
        """Search DRKG entities by MeSH ID."""
        if mesh_id in self.disease_entities:
            logger.debug(f"Found in DRKG: {mesh_id} -> {self.disease_entities[mesh_id]}")
            return self.disease_entities[mesh_id]
        logger.debug(f"Not found in DRKG: {mesh_id}")
        return None
    
    def _fuzzy_match_drkg_entities(self, disease_name: str) -> Optional[str]:
        """
        Fuzzy match disease name against DRKG entities.
        
        Returns: DRKG entity ID or None
        """
        disease_lower = disease_name.lower()
        
        # Try exact substring match first
        for entity_lower, entity_id in self.disease_name_index.items():
            if disease_lower in entity_lower or entity_lower in disease_lower:
                logger.info(f"✓ Fuzzy match: '{disease_name}' -> {entity_id}")
                return entity_id
        
        # Try word-level matching
        disease_words = set(disease_lower.split())
        best_match = None
        best_score = 0.0
        
        for entity_lower, entity_id in list(self.disease_name_index.items())[:1000]:
            entity_words = set(entity_lower.split('::')[-1].lower().split())
            overlap = len(disease_words & entity_words)
            if overlap > 0:
                score = overlap / max(len(disease_words), len(entity_words))
                if score > best_score:
                    best_score = score
                    best_match = entity_id
        
        if best_match and best_score > 0.5:
            logger.info(f"✓ Fuzzy match: '{disease_name}' -> {best_match} (score: {best_score:.2f})")
            return best_match
        
        return None
    
    async def resolve(self, disease_name: str) -> Optional[str]:
        """
        Resolve disease name to DRKG entity ID using automated APIs.
        
        Returns: DRKG entity ID (e.g., "Disease::MESH:D015658") or None
        """
        logger.info(f"🔍 Resolving disease '{disease_name}' to DRKG entity (automated)")
        
        # Method 1: Query NCBI MeSH (SIMPLE VERSION)
        mesh_result = await self._query_ncbi_mesh_simple(disease_name)
        if mesh_result:
            mesh_id = mesh_result["mesh_id"]
            drkg_entity = self._search_drkg_by_mesh(mesh_id)
            if drkg_entity:
                logger.info(f"✓ Resolved via NCBI: '{disease_name}' -> {drkg_entity}")
                return drkg_entity
            else:
                logger.debug(f"MeSH ID {mesh_id} found in NCBI but not in DRKG")
        
        # Method 2: Fuzzy matching in DRKG (fallback)
        drkg_entity = self._fuzzy_match_drkg_entities(disease_name)
        if drkg_entity:
            return drkg_entity
        
        logger.warning(f"⚠️ Could not resolve '{disease_name}' to DRKG entity")
        if mesh_result:
            logger.warning(f"   Found MeSH ID {mesh_result['mesh_id']} but not in DRKG")
        return None


# Global instance
_automated_mapper = None


def get_automated_mapper() -> AutomatedDRKGMapper:
    """Get singleton instance of automated mapper."""
    global _automated_mapper
    if _automated_mapper is None:
        _automated_mapper = AutomatedDRKGMapper()
    return _automated_mapper


async def resolve_disease_to_drkg(disease_name: str) -> Optional[str]:
    """
    Public API: Resolve disease name to DRKG entity ID.
    
    Usage:
        drkg_entity = await resolve_disease_to_drkg("HIV")
        # Returns: "Disease::MESH:D015658"
    """
    mapper = get_automated_mapper()
    return await mapper.resolve(disease_name)
