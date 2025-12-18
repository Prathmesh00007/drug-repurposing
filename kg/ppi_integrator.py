# kg/ppi_integrator.py
"""
Integrate protein-protein interaction data for network analysis.
"""

import httpx
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PPIIntegrator:
    """
    Aggregate PPI data from:
    - STRING: Functional associations
    - BioGRID: Physical interactions
    - IntAct: Curated interactions
    """
    
    def __init__(self):
        self.string_url = "https://string-db.org/api"
        self.biogrid_url = "https://webservice.thebiogrid.org"
    
    async def get_protein_interactions(
        self, 
        gene_symbol: str,
        confidence_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Get high-confidence protein interactions.
        
        Returns:
        [
            {
                "partner": "APOA1",
                "score": 0.95,
                "evidence": ["coexpression", "experiments"],
                "source": "STRING"
            },
            ...
        ]
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.string_url}/json/network",
                    data={
                        "identifiers": gene_symbol,
                        "species": 9606,  # Homo sapiens
                        "required_score": int(confidence_threshold * 1000)
                    }
                )
                
                if response.status_code != 200:
                    return []
                
                interactions = response.json()
                
                result = []
                for interaction in interactions:
                    partner = interaction.get("preferredName_B")
                    if partner != gene_symbol:
                        result.append({
                            "partner": partner,
                            "score": interaction.get("score", 0) / 1000.0,
                            "evidence": self._parse_string_evidence(interaction),
                            "source": "STRING"
                        })
                
                logger.info(f"✓ Found {len(result)} interactions for {gene_symbol}")
                return result
                
        except Exception as e:
            logger.warning(f"STRING query failed: {e}")
            return []
    
    def _parse_string_evidence(self, interaction: Dict) -> List[str]:
        """Extract evidence types from STRING interaction."""
        evidence = []
        evidence_types = [
            "neighborhood", "fusion", "cooccurence", "coexpression",
            "experiments", "database", "textmining"
        ]
        
        for ev_type in evidence_types:
            if interaction.get(f"score_{ev_type}", 0) > 0:
                evidence.append(ev_type)
        
        return evidence
    
    async def find_common_interactors(
        self,
        gene_list: List[str]
    ) -> List[Dict]:
        """
        Find proteins that interact with multiple targets in the list.
        These are potential combination therapy targets or biomarkers.
        """
        all_interactions = {}
        
        for gene in gene_list:
            interactions = await self.get_protein_interactions(gene)
            for interaction in interactions:
                partner = interaction["partner"]
                if partner not in all_interactions:
                    all_interactions[partner] = []
                all_interactions[partner].append({
                    "source_target": gene,
                    "score": interaction["score"]
                })
        
        # Filter for proteins interacting with 2+ targets
        common = [
            {
                "protein": partner,
                "interacts_with": [i["source_target"] for i in interactions],
                "count": len(interactions),
                "avg_score": sum(i["score"] for i in interactions) / len(interactions)
            }
            for partner, interactions in all_interactions.items()
            if len(interactions) >= 2
        ]
        
        # Sort by count and score
        common.sort(key=lambda x: (x["count"], x["avg_score"]), reverse=True)
        
        logger.info(f"✓ Found {len(common)} common interactors across {len(gene_list)} targets")
        return common
