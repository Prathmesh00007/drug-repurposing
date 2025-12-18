"""
Direct Disease-Drug Discovery from Open Targets Platform.

This module implements the MISSING PIECE from the reference system:
directly querying for drugs that have clinical evidence for a disease,
rather than only finding drugs through target associations.

Key advantages:
- Finds drugs with actual clinical trial data for the disease
- Not limited by target validation filters
- Captures drugs with unknown/indirect mechanisms
- Matches reference system's high recall rate
"""

import logging
import httpx
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DirectDiseaseError(Exception):
    """Custom exception for direct disease-drug query errors."""
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def fetch_known_drugs_for_disease(
    disease_id: str,
    min_phase: int = 1,
    max_results: int = 10000
) -> List[Dict]:
    """
    Fetch drugs with clinical evidence for disease using Open Targets knownDrugs API.
    
    This is the core function that implements the reference system's approach.
    It queries the disease → knownDrugs relationship directly, which captures:
    - Drugs in clinical trials for this disease
    - Approved drugs for this disease
    - Drugs with known efficacy data
    
    Args:
        disease_id: EFO/MONDO disease identifier (e.g., "EFO_0000249")
        min_phase: Minimum clinical trial phase (1-4)
            1 = Phase 1 and above
            2 = Phase 2 and above
            3 = Phase 3 and above
            4 = Approved drugs only
        max_results: Maximum number of drugs to return (default 10000)
        
    Returns:
        List of drug dictionaries with structure:
        {
            "drug_id": str,           # ChEMBL ID
            "drug_name": str,         # Preferred name
            "target_symbol": str,     # Primary target gene symbol
            "target_name": str,       # Primary target name
            "drug_type": str,         # "Small molecule", "Antibody", etc.
            "phase": int,             # Clinical trial phase (0-4)
            "ct_ids": List[str],      # Clinical trial identifiers
            "source": str,            # "opentargets_known_drugs"
            "has_clinical_evidence": bool  # Always True for this source
        }
        
    Raises:
        DirectDiseaseError: If API calls fail after retries
    """
    logger.info(f"🎯 Direct disease-drug query for {disease_id} (phase >= {min_phase})...")
    
    # =========================================================================
    # STEP 1: Get count of known drugs for this disease
    # =========================================================================
    count_query = """
    query GetDrugCount($efoId: String!) {
        disease(efoId: $efoId) {
            id
            name
            knownDrugs {
                count
            }
        }
    }
    """
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Query for count
            response = await client.post(
                "https://api.platform.opentargets.org/api/v4/graphql",
                json={"query": count_query, "variables": {"efoId": disease_id}},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise DirectDiseaseError(
                    f"Failed to get drug count: HTTP {response.status_code}"
                )
            
            data = response.json()
            disease_data = data.get("data", {}).get("disease")
            
            if not disease_data:
                logger.warning(f"Disease {disease_id} not found in Open Targets")
                return []
            
            count = disease_data.get("knownDrugs", {}).get("count", 0)
            disease_name = disease_data.get("name", "Unknown")
            
            if count == 0:
                logger.info(f"No known drugs found for {disease_id} ({disease_name})")
                return []
            
            # Limit to max_results
            if count > max_results:
                logger.warning(
                    f"Found {count} drugs for {disease_id}, limiting to {max_results}"
                )
                count = max_results
            
            logger.info(f"Found {count} total drugs for {disease_name}, fetching...")
            
            # =====================================================================
            # STEP 2: Fetch full drug list
            # =====================================================================
            drugs_query = """
            query GetKnownDrugs($efoId: String!, $size: Int!) {
                disease(efoId: $efoId) {
                    id
                    name
                    knownDrugs(size: $size) {
                        count
                        rows {
                            approvedSymbol
                            approvedName
                            prefName
                            drugType
                            drugId
                            phase
                            ctIds
                        }
                    }
                }
            }
            """
            
            response = await client.post(
                "https://api.platform.opentargets.org/api/v4/graphql",
                json={
                    "query": drugs_query,
                    "variables": {"efoId": disease_id, "size": count}
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise DirectDiseaseError(
                    f"Failed to get drugs: HTTP {response.status_code}"
                )
            
            data = response.json()
            rows = data.get("data", {}).get("disease", {}).get("knownDrugs", {}).get("rows", [])
            
            # =====================================================================
            # STEP 3: Filter and format results
            # =====================================================================
            filtered = []
            phase_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            for row in rows:
                # Normalize phase
                phase = row.get("phase", 0)
                if not isinstance(phase, (int, float)):
                    phase = 0
                phase = int(phase)
                
                # Track phase distribution
                if phase in phase_counts:
                    phase_counts[phase] += 1
                
                # Filter by minimum phase
                if phase < min_phase:
                    continue
                
                # Build standardized drug object
                drug_obj = {
                    "drug_id": row.get("drugId", ""),
                    "drug_name": row.get("prefName", row.get("drugId", "Unknown")),
                    "target_symbol": row.get("approvedSymbol", ""),
                    "target_name": row.get("approvedName", ""),
                    "drug_type": row.get("drugType", "Unknown"),
                    "phase": phase,
                    "ct_ids": row.get("ctIds", []),
                    "source": "opentargets_known_drugs",
                    "has_clinical_evidence": True,  # Key flag
                    "disease_id": disease_id,
                    "disease_name": disease_name
                }
                
                filtered.append(drug_obj)
            
            # Log statistics
            logger.info(
                f"✅ Direct query complete: {len(filtered)}/{len(rows)} drugs passed filter "
                f"(phase >= {min_phase})"
            )
            logger.info(
                f"   Phase distribution: "
                f"Phase 0: {phase_counts[0]}, "
                f"Phase 1: {phase_counts[1]}, "
                f"Phase 2: {phase_counts[2]}, "
                f"Phase 3: {phase_counts[3]}, "
                f"Approved: {phase_counts[4]}"
            )
            
            return filtered
            
    except httpx.HTTPError as e:
        raise DirectDiseaseError(f"HTTP error during direct disease query: {e}")
    except Exception as e:
        raise DirectDiseaseError(f"Unexpected error during direct disease query: {e}")


async def fetch_known_drugs_batch(
    disease_ids: List[str],
    min_phase: int = 1
) -> Dict[str, List[Dict]]:
    """
    Batch fetch known drugs for multiple diseases.
    
    Args:
        disease_ids: List of disease identifiers
        min_phase: Minimum clinical trial phase
        
    Returns:
        Dictionary mapping disease_id -> list of drugs
    """
    results = {}
    
    for disease_id in disease_ids:
        try:
            drugs = await fetch_known_drugs_for_disease(disease_id, min_phase)
            results[disease_id] = drugs
        except DirectDiseaseError as e:
            logger.error(f"Failed to fetch drugs for {disease_id}: {e}")
            results[disease_id] = []
    
    return results


def get_drug_statistics(drugs: List[Dict]) -> Dict:
    """
    Calculate statistics for a list of drugs.
    
    Args:
        drugs: List of drug dictionaries
        
    Returns:
        Statistics dictionary
    """
    if not drugs:
        return {
            "total_drugs": 0,
            "unique_drugs": 0,
            "unique_targets": 0,
            "phase_distribution": {},
            "drug_types": {}
        }
    
    # Count phases
    phase_dist = {}
    for drug in drugs:
        phase = drug.get("phase", 0)
        phase_dist[phase] = phase_dist.get(phase, 0) + 1
    
    # Count drug types
    type_dist = {}
    for drug in drugs:
        drug_type = drug.get("drug_type", "Unknown")
        type_dist[drug_type] = type_dist.get(drug_type, 0) + 1
    
    # Count unique drugs and targets
    unique_drugs = len(set(d.get("drug_id") for d in drugs if d.get("drug_id")))
    unique_targets = len(set(
        d.get("target_symbol") for d in drugs 
        if d.get("target_symbol")
    ))
    
    return {
        "total_drugs": len(drugs),
        "unique_drugs": unique_drugs,
        "unique_targets": unique_targets,
        "phase_distribution": phase_dist,
        "drug_types": type_dist
    }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test with Alzheimer's disease
        drugs = await fetch_known_drugs_for_disease("EFO_0000249", min_phase=1)
        stats = get_drug_statistics(drugs)
        
        print(f"\n{'='*60}")
        print(f"DIRECT DISEASE-DRUG QUERY TEST")
        print(f"{'='*60}")
        print(f"Disease: EFO_0000249 (Alzheimer's)")
        print(f"Total drugs found: {stats['total_drugs']}")
        print(f"Unique drugs: {stats['unique_drugs']}")
        print(f"Unique targets: {stats['unique_targets']}")
        print(f"\nPhase distribution:")
        for phase, count in sorted(stats['phase_distribution'].items()):
            phase_name = "Approved" if phase == 4 else f"Phase {phase}"
            print(f"  {phase_name}: {count}")
        print(f"\nTop 5 drugs:")
        for i, drug in enumerate(drugs[:5], 1):
            print(f"  {i}. {drug['drug_name']} (Phase {drug['phase']})")
    
    asyncio.run(test())
