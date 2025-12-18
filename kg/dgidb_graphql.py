"""
DGIdb drug-gene interaction fetcher using GraphQL API (v5.0).
The REST API v2 is DEPRECATED - must use GraphQL now.
"""

import logging
import httpx
import json
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class DGIdbGraphQLClient:
    """
    DGIdb GraphQL API client.
    
    API Documentation: https://dgidb.org/api
    GraphQL Endpoint: https://dgidb.org/api/graphql
    
    NOTE: The old REST API (v2) is DEPRECATED and returns HTML.
    Must use GraphQL for DGIdb 5.0+
    """
    
    GRAPHQL_ENDPOINT = "https://dgidb.org/api/graphql"
    
    # GraphQL query for gene-drug interactions
    GENE_INTERACTIONS_QUERY = """
    query GeneInteractions($geneNames: [String!]!) {
      genes(names: $geneNames) {
        nodes {
          name
          conceptId
          interactions {
            interactionScore
            interactionTypes {
              type
              directionality
            }
            drug {
              name
              conceptId
              approved
              drugApplications {
                appNo
              }
              drugAttributes {
                name
                value
              }
            }
            publications {
              pmid
            }
            interactionClaims {
              source {
                sourceDbName
                license
              }
            }
          }
        }
      }
    }
    """
    
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def query_genes(self, gene_symbols: List[str]) -> Dict:
        """
        Query DGIdb GraphQL API for gene-drug interactions.
        
        Args:
            gene_symbols: List of gene symbols (e.g., ["EGFR", "BRAF"])
        
        Returns:
            GraphQL response data
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.GRAPHQL_ENDPOINT,
                    json={
                        "query": self.GENE_INTERACTIONS_QUERY,
                        "variables": {"geneNames": gene_symbols}
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                # Check HTTP status
                if response.status_code != 200:
                    logger.error(f"DGIdb GraphQL returned HTTP {response.status_code}")
                    logger.error(f"Response: {response.text[:500]}")
                    return {}
                
                # Parse JSON
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"DGIdb returned non-JSON: {e}")
                    logger.error(f"Content-Type: {response.headers.get('content-type')}")
                    logger.error(f"Response preview: {response.text[:500]}")
                    return {}
                
                # Check for GraphQL errors
                if "errors" in data:
                    logger.error(f"DGIdb GraphQL errors: {data['errors']}")
                    return {}
                
                return data.get("data", {})
                
        except Exception as e:
            logger.error(f"DGIdb GraphQL query failed: {e}")
            return {}
    
    async def get_drug_interactions(
        self,
        gene_symbols: List[str],
        min_phase: int = 1,
        approved_only: bool = False
    ) -> List[Dict]:
        """
        Get drug-gene interactions from DGIdb.
        
        Args:
            gene_symbols: List of gene symbols
            min_phase: Minimum clinical trial phase (1-4)
            approved_only: Only return FDA-approved drugs
        
        Returns:
            List of drug interactions
        """

        if min_phase is None:
            min_phase = 1
            
        # Query in batches of 10 to avoid URL length limits
        all_interactions = []
        
        for i in range(0, len(gene_symbols), 10):
            batch = gene_symbols[i:i+10]
            
            logger.info(f"📊 Querying DGIdb for {len(batch)} genes (batch {i//10 + 1})...")
            data = await self.query_genes(batch)
            
            genes = data.get("genes", {}).get("nodes", [])
            
            for gene in genes:
                gene_name = gene.get("name", "UNKNOWN")
                interactions = gene.get("interactions", [])
                
                for interaction in interactions:
                    drug = interaction.get("drug", {})
                    drug_name = drug.get("name", "")
                    is_approved = drug.get("approved", False)
                    
                    # Filter by approval status
                    if approved_only and not is_approved:
                        continue
                    
                    # Estimate phase from approval status and applications
                    applications = drug.get("drugApplications", [])
                    phase = 4 if is_approved else (2 if applications else 1)
                    
                    if phase < min_phase:
                        continue
                    
                    # Extract interaction types
                    interaction_types = interaction.get("interactionTypes", [])
                    mechanism = ", ".join([
                        it.get("type", "") for it in interaction_types
                    ]) or "Unknown"
                    
                    # Extract sources
                    claims = interaction.get("interactionClaims", [])
                    sources = [
                        claim.get("source", {}).get("sourceDbName", "")
                        for claim in claims
                    ]
                    
                    # Extract PMIDs
                    publications = interaction.get("publications", [])
                    pmids = [pub.get("pmid") for pub in publications if pub.get("pmid")]
                    
                    all_interactions.append({
                        "source": "dgidb",
                        "target_symbol": gene_name,
                        "drug_name": drug_name,
                        "drug_id": drug.get("conceptId", ""),
                        "mechanism": mechanism,
                        "phase": phase,
                        "approved": is_approved,
                        "sources": sources,
                        "pmids": pmids,
                        "interaction_score": interaction.get("interactionScore")
                    })
        
        logger.info(f"✅ DGIdb GraphQL: Found {len(all_interactions)} interactions from {len(gene_symbols)} genes")
        return all_interactions


# Global instance
_dgidb_client = None

def get_dgidb_client() -> DGIdbGraphQLClient:
    """Get singleton DGIdb client."""
    global _dgidb_client
    if _dgidb_client is None:
        _dgidb_client = DGIdbGraphQLClient()
    return _dgidb_client

async def fetch_dgidb_drugs_graphql(
    gene_symbols: List[str],
    min_phase: int = 1
) -> List[Dict]:
    """
    Public API: Fetch drug-gene interactions from DGIdb using GraphQL.
    
    Usage:
        drugs = await fetch_dgidb_drugs_graphql(["EGFR", "BRAF"], min_phase=2)
    """
    client = get_dgidb_client()
    return await client.get_drug_interactions(gene_symbols, min_phase)
