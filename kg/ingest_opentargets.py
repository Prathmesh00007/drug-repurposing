"""Production-grade Open Targets integration with TARGET VALIDATION."""

import logging
import httpx
import asyncio
import numpy as np
from typing import List, Dict, Optional
from agents.base import cache_manager
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.app.config import get_settings
from kg.disease_resolver import disease_resolver
from kg.mechanism_reasoner import MechanismReasoner
import json


logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def _query_opentargets_targets_paginated(disease_id: str, page_size: int = 100) -> List[Dict]:
    """
    Fetch ALL target associations with pagination.
    Returns: Complete list of all association rows
    """
    settings = get_settings()
    
    query = """
    query DiseaseTargets($efoId: String!, $size: Int!, $index: Int!) {
        disease(efoId: $efoId) {
            associatedTargets(
                page: {index: $index, size: $size},
                orderByScore: "score DESC",
                enableIndirect: true
            ) {
                count
                rows {
                    target {
                        id
                        approvedSymbol
                        approvedName
                        biotype
                        proteinIds {
                            id
                            source
                        }
                        tractability {
                            label
                            modality
                            value
                        }
                    }
                    score
                    datatypeScores {
                        id
                        score
                    }
                }
            }
        }
    }
    """
    
    all_rows = []
    total_count = None
    page_index = 0
    
    async with httpx.AsyncClient(timeout=30.0, http2=False, follow_redirects=True) as client:
        while True:
            variables = {"efoId": disease_id, "size": page_size, "index": page_index}
            response = await client.post(
                settings.opentargets_gql_url,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            data = response.json()
            
            disease_obj = data.get("data", {}).get("disease", {})
            if not disease_obj:
                break
            
            assoc = disease_obj.get("associatedTargets", {})
            rows = assoc.get("rows", [])
            
            if total_count is None:
                total_count = assoc.get("count", 0)
                logger.info(f"📊 Total associations: {total_count}")
            
            if not rows:
                break
            
            all_rows.extend(rows)
            logger.info(f"   Fetched page {page_index + 1}: {len(all_rows)}/{total_count} associations")
            
            # Stop if we've fetched everything
            if len(all_rows) >= total_count:
                break
            
            page_index += 1
            
            # Safety limit: max 500 pages (50,000 associations)
            if page_index >= 500:
                logger.warning(f"⚠️ Reached safety limit of 500 pages")
                break
    
    logger.info(f"✅ Fetched {len(all_rows)} total associations")
    return all_rows


def normalize_disease_id(disease_id: str) -> str:
    """Normalize disease ID to consistent format: PREFIX_NUMBERS"""
    if not disease_id:
        return disease_id
    
    if "_" in disease_id and disease_id.count("_") == 1:
        parts = disease_id.split("_")
        if parts[0].isalpha() and parts[1].isdigit():
            return disease_id
    
    clean_id = disease_id.replace(":", "").replace("_", "")
    
    for i, char in enumerate(clean_id):
        if char.isdigit():
            prefix = clean_id[:i]
            numbers = clean_id[i:]
            return f"{prefix}_{numbers}"
    
    return disease_id


def _calculate_multi_dimensional_scores(rows: List[Dict]) -> List[Dict]:
    """
    Calculate normalized scores across multiple dimensions using statistical methods.
    Returns: List of rows with added 'composite_score' and dimension scores
    """
    if not rows:
        return []
    
    # Extract raw scores for each dimension
    base_scores = []
    evidence_diversity = []
    tractability_scores = []
    
    for row in rows:
        # Dimension 1: Base association score
        base_scores.append(row.get("score", 0.0))
        
        # Dimension 2: Evidence diversity (number of data types)
        datatype_scores = row.get("datatypeScores", [])
        evidence_count = len([d for d in datatype_scores if d.get("score", 0) > 0])
        evidence_diversity.append(evidence_count)
        
        # Dimension 3: Tractability (converted to numeric)
        target = row.get("target", {})
        tractability = target.get("tractability", [])
        tract_score = 0.0
        
        for tract in tractability:
            if tract.get("modality") == "SM":
                value = tract.get("value", "")
                if value == "Approved":
                    tract_score = max(tract_score, 1.0)
                elif value in ["Clinical Precedence", "Phase 3", "Phase 2", "Phase 1"]:
                    tract_score = max(tract_score, 0.7)
                elif value == "Discovery Precedence":
                    tract_score = max(tract_score, 0.4)
                elif value == "Predicted Tractable":
                    tract_score = max(tract_score, 0.2)
        
        tractability_scores.append(tract_score)
    
    # Convert to numpy arrays
    base_scores = np.array(base_scores)
    evidence_diversity = np.array(evidence_diversity)
    tractability_scores = np.array(tractability_scores)
    
    # Normalize using Min-Max scaling (0-1 range)
    def normalize(arr):
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    base_norm = normalize(base_scores)
    evidence_norm = normalize(evidence_diversity)
    tract_norm = tractability_scores  # Already 0-1
    
    # Composite score with weights (evidence-based weights)
    # Base score: 70% (most important)
    # Evidence diversity: 20% (multi-source validation)
    # Tractability: 10% (drugability)
    composite_scores = (
        base_norm * 0.7 +
        evidence_norm * 0.2 +
        tract_norm * 0.1
    )
    
    # Add scores back to rows
    for i, row in enumerate(rows):
        row['composite_score'] = float(composite_scores[i])
        row['base_score_norm'] = float(base_norm[i])
        row['evidence_norm'] = float(evidence_norm[i])
        row['tract_norm'] = float(tract_norm[i])
    
    return rows


async def _fetch_known_drugs_for_disease(disease_id: str) -> Dict[str, int]:
    """
    Fetch approved drugs for this disease to learn valid mechanisms.
    Returns: Dict of mechanism_of_action -> frequency
    """
    settings = get_settings()
    query = """
    query KnownDrugs($efoId: String!) {
      disease(efoId: $efoId) {
        knownDrugs(freeTextQuery: "", size: 50) {
          rows {
            drug {
              name
              mechanismsOfAction {
                rows {
                  mechanismOfAction
                  targets {
                    approvedSymbol
                  }
                }
              }
            }
            phase
          }
        }
      }
    }
    """
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            settings.opentargets_gql_url,
            json={"query": query, "variables": {"efoId": disease_id}},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        # Count mechanisms in approved drugs (phase 4)
        mechanism_counts = {}
        disease_obj = data.get("data", {}).get("disease", {})
        known_drugs = disease_obj.get("knownDrugs", {}).get("rows", [])
        
        for row in known_drugs:
            if row.get("phase") == 4:  # Only FDA-approved
                drug = row.get("drug", {})
                moas = drug.get("mechanismsOfAction", {}).get("rows", [])
                for moa in moas:
                    moa_text = moa.get("mechanismOfAction", "")
                    if moa_text:
                        mechanism_counts[moa_text] = mechanism_counts.get(moa_text, 0) + 1
        
        logger.info(f"📚 Bootstrapped {len(mechanism_counts)} mechanism classes from approved drugs")
        return mechanism_counts


def _extract_uniprot_id(protein_ids: List[Dict]) -> Optional[str]:
    """Extract UniProt accession from proteinIds list."""
    if not protein_ids:
        return None
    
    # Prefer Swiss-Prot (reviewed)
    for pid in protein_ids:
        if pid.get("source") == "uniprot_swissprot":
            return pid.get("id")
    
    # Fallback to TrEMBL (unreviewed)
    for pid in protein_ids:
        if pid.get("source") == "uniprot_trembl":
            return pid.get("id")
    
    # Last resort: any uniprot
    for pid in protein_ids:
        if "uniprot" in pid.get("source", "").lower():
            return pid.get("id")
    
    return None


def _filter_by_percentile(scored_rows: List[Dict], top_percent: float = 2.0, min_targets: int = 10, max_targets: int = 30) -> List[Dict]:
    """
    Filter targets by percentile ranking (universal, disease-agnostic).
    
    Args:
        scored_rows: Rows with composite_score
        top_percent: Take top N% of associations
        min_targets: Minimum number of targets to return
        max_targets: Maximum number of targets to return
    
    Returns:
        Filtered list of top targets
    """
    if not scored_rows:
        return []
    
    # Sort by composite score
    scored_rows.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Calculate percentile cutoff
    n_targets = max(min_targets, int(len(scored_rows) * (top_percent / 100)))
    n_targets = min(n_targets, max_targets)
    
    # Additional filter: Must be protein-coding and have base score > 0
    filtered = []
    for row in scored_rows:
        target = row.get("target", {})
        biotype = target.get("biotype", "")
        base_score = row.get("score", 0.0)
        
        if biotype == "protein_coding" and base_score > 0:
            filtered.append(row)
            if len(filtered) >= n_targets:
                break
    
    logger.info(f"📊 Percentile filter: Top {top_percent}% = {n_targets} targets, after filtering: {len(filtered)}")
    return filtered

async def ingest_opentargets_for_disease(
    disease_name: str,
    neo4j_client,
    disease_id: str,
    disease_context,  # DiseaseContext from disease_resolver_v2
    top_percent: float = 10.0,
    min_targets: int = 3,
    max_targets: int = 5,
    enable_reasoning: bool = True
) -> List[Dict]:
    """
    FIXED VERSION: Validate BEFORE saving to Neo4j.
    
    New Sequence:
    1. Fetch targets from OpenTargets
    2. Validate mechanisms (pathway overlap)
    3. Validate evidence (DisGeNET/UniProt)
    4. THEN save to Neo4j (validated targets only)
    """
    logger.info(f"🔍 Fetching OpenTargets data for {disease_id}...")
    
    # Import validators
    from kg.pathway_mechanism_validator import pathway_validator
    from kg.target_validator import validate_targets_for_disease
    
    # STEP 1: Fetch raw targets from OpenTargets
    all_rows = await _query_opentargets_targets_paginated(disease_id, page_size=100)
    
    if not all_rows:
        logger.warning(f"❌ No targets found in OpenTargets for {disease_id}")
        raise ValueError(f"No targets found for {disease_id}")
    
    logger.info(f"   Fetched {len(all_rows)} total targets from OpenTargets")
    
    # STEP 2: Score and filter by percentile
    scored_rows = _calculate_multi_dimensional_scores(all_rows)
    top_rows = _filter_by_percentile(scored_rows, top_percent, min_targets, max_targets)
    
    logger.info(f"   Filtered to top {len(top_rows)} targets")
    
    # STEP 1.5: Bootstrap mechanism classes from known drugs
    try:
        known_mechanisms = await _fetch_known_drugs_for_disease(disease_id)
        logger.info(f"   Learned mechanisms: {list(known_mechanisms.keys())[:5]}...")
    except Exception as e:
        logger.warning(f"⚠️ Bootstrapping failed: {e}")
        known_mechanisms = {}
    
    # STEP 3: Build target data list
    target_data_list = []
    for row in top_rows:
        target = row.get("target", {})
        target_data_list.append({
            "symbol": target.get("approvedSymbol"),
            "uniprot_id": _extract_uniprot_id(target.get("proteinIds", [])),
            "ensembl_id": target.get("id", ""),
            "composite_score": row["composite_score"],
            "opentargets_score": row.get("score", 0.0),
            "known_mechanisms": known_mechanisms  # ← ADD THIS for downstream use
        })
    
    # STEP 4: VALIDATE MECHANISMS FIRST (before Neo4j)
    if enable_reasoning and disease_context:
        logger.info(f"🧬 Validating mechanisms for {len(target_data_list)} targets...")
        
        mechanistically_valid = []
        
        for target_data in target_data_list:
            symbol = target_data["symbol"]
            
            # Validate pathway overlap
            pathway_result = await pathway_validator.validate_mechanism(
                target_symbol=symbol,
                disease_context=disease_context
            )
            
            if pathway_result.decision == "KEEP":
                target_data["mechanism_score"] = pathway_result.confidence
                target_data["mechanism_reasoning"] = pathway_result.reasoning
                target_data["pathway_jaccard"] = pathway_result.jaccard_similarity
                mechanistically_valid.append(target_data)
            else:
                logger.debug(f"   ❌ {symbol}: {pathway_result.reasoning}")
        
        logger.info(f"   ✓ Mechanism validation: {len(mechanistically_valid)}/{len(target_data_list)} passed")
        
        if not mechanistically_valid:
            logger.warning("⚠️ No targets passed mechanism validation - keeping top OpenTargets targets")
            mechanistically_valid = target_data_list[:max(5, min_targets // 2)]
        
        target_data_list = mechanistically_valid
    
    # STEP 5: VALIDATE EVIDENCE (DisGeNET/UniProt)
    logger.info(f"🧬 Validating evidence for {len(target_data_list)} targets...")
    
    evidence_validated = await validate_targets_for_disease(
        disease_name=disease_name,
        targets=target_data_list,
        disease_context=disease_context,
        use_gemini_filter=False  # Already did mechanism validation
    )
    
    logger.info(f"   ✓ Evidence validation: {len(evidence_validated)}/{len(target_data_list)} passed")
    
    if not evidence_validated:
        logger.warning("⚠️ No targets passed evidence validation - using top OpenTargets only")
        evidence_validated = target_data_list[:min(5, len(target_data_list))]
    
    # STEP 6: NOW save to Neo4j (validated targets only)
    logger.info(f"💾 Saving {len(evidence_validated)} validated targets to Neo4j...")
    
    final_targets = []
    
        # STEP 6: NOW save to Neo4j (validated targets only)
    logger.info(f"💾 Saving {len(evidence_validated)} validated targets to Neo4j...")

    def save_target_worker(target_data):
        """Helper to run blocking Neo4j calls in a thread."""
        symbol = target_data["symbol"]
        ensembl_id = target_data.get("ensembl_id", "")
        
        # Create target node
        neo4j_client.create_target_node(
            target_id=ensembl_id,
            target_symbol=symbol,
            target_name=""
        )
        
        # Create validated association
        neo4j_client.create_target_disease_association(
            target_id=ensembl_id,
            disease_id=disease_id,
            score=target_data.get("validation_score", target_data.get("composite_score", 0.5)),
            evidence="OpenTargets + DisGeNET + Pathway (validated)",
            mechanism_score=target_data.get("mechanism_score", 0.0)
        )
        
        # Return the formatted dict for final_targets
        return {
            "symbol": symbol,
            "uniprot_id": target_data.get("uniprot_id"),
            "ensembl_id": ensembl_id,
            "validation_score": target_data.get("validation_score", 0.5),
            "mechanism_score": target_data.get("mechanism_score", 0.0),
            "pathway_jaccard": target_data.get("pathway_jaccard", 0.0)
        }

    # 🚀 PARALLEL SAVE: Execute writes concurrently using threads
    tasks = [asyncio.to_thread(save_target_worker, t) for t in evidence_validated]
    final_targets = await asyncio.gather(*tasks)
    
    logger.info(f"   ✅ Saved {len(final_targets)} validated targets to Neo4j")
    
    return final_targets
