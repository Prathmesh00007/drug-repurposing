import logging
from typing import List, Dict, Optional

from kg.disease_resolver_v2 import DiseaseContext
from kg.direct_disease_drugs import fetch_known_drugs_for_disease
from kg.ingest_opentargets import ingest_opentargets_for_disease
from kg.ingest_chembl import ingest_chembl_candidates
from kg.evidence_validator import EvidenceValidator, ValidationDecision
from kg.scoring_engine import ScoringEngine
from kg.candidate_ranker import CandidateRanker, RankingStrategy
from kg.pathway_integrator import PathwayIntegrator
from kg.ppi_integrator import PPIIntegrator
from kg.clinical_trial_parser import trial_parser
from kg.neo4j_client import Neo4jClient
from kg.utils import normalize_phase, normalize_drug_id  # ← NEW IMPORT

logger = logging.getLogger(__name__)

class HybridDrugDiscoveryV2:
    """
    Production-grade hybrid discovery system.
    Philosophy:
    - Lenient validation (low rejection rate)
    - Multi-source enrichment (non-blocking)
    - Transparent scoring (explainable)
    - Clinical-friendly output
    - Unified validation using EvidenceValidator only
    """

    def __init__(self):
        # Unified validator (replaces moa_validator, safety_validator, pathway_mechanism_validator)
        self.validator = EvidenceValidator(
            min_score=0.2,  # Lenient
            min_evidence_sources=1,
            enable_pathway_check=False,  # Optional
            enable_literature_check=False  # Optional
        )
        self.scorer = ScoringEngine()
        self.ranker = CandidateRanker(strategy=RankingStrategy.BALANCED)
        # Enrichment integrators
        self.pathway_integrator = PathwayIntegrator()
        self.ppi_integrator = PPIIntegrator()

    # kg.py - MODIFIED SECTIONS ONLY (keep all other code unchanged)

    async def discover_for_disease(
        self,
        disease_id: str,
        disease_context: DiseaseContext,
        min_phase: int = 1,
        top_n_candidates: int = 3,
        enable_enrichment: bool = False
    ) -> Dict:
        """
        Full REPURPOSING pipeline (CONVERTED FROM DISCOVERY).
        
        Key changes:
        - Track 1 drugs used as EXCLUSION list
        - Only Track 2 (target-based) drugs scored
        - Novelty scoring added (drugs NOT treating query disease)
        """
        logger.info(f"🚀 HYBRID Discovery: {disease_context.corrected_name}")
        logger.info(f"✅ Using disease context from orchestrator")
        logger.info(f"   EFO: {disease_context.efo_id}, MONDO: {disease_context.mondo_id}")
        logger.info(f"   Therapeutic Area: {disease_context.therapeutic_area}")

        # =================================================================
        # PHASE 1: DUAL-TRACK DISCOVERY
        # =================================================================
        logger.info("📊 PHASE 1: Dual-Track Discovery")
        
        # ✅ REPURPOSING CHANGE #1: Track 1 builds EXCLUSION list
        logger.info(" 🔒 Track 1: Building exclusion list (drugs already treating disease)...")
        direct_drugs = await fetch_known_drugs_for_disease(
            disease_id=disease_id,
            min_phase=min_phase
        )
        
        # Build exclusion set (drugs already approved/tested for this disease)
        exclusion_set = set()
        for drug in direct_drugs:
            exclusion_set.add(drug["drug_id"])
            exclusion_set.add(drug["drug_name"].lower())  # Also match by name
        
        logger.info(f" ✓ Exclusion list: {len(exclusion_set)} drugs already treat {disease_context.corrected_name}")
        logger.info(f"   These will be filtered OUT (not true repurposing candidates)")

        # Track 2: Target-based discovery (PRIMARY SOURCE for repurposing)
        logger.info(" 🎯 Track 2: Target-based discovery (PRIMARY for repurposing)...")
        disease_targets = []
        neo4j = None
        try:
            neo4j = Neo4jClient()
            disease_targets = await ingest_opentargets_for_disease(
                disease_name=disease_context.corrected_name,
                neo4j_client=neo4j,
                disease_id=disease_id,
                disease_context=disease_context,
                top_percent=10.0,
                min_targets=20,
                max_targets=50,
                enable_reasoning=False
            )
            logger.info(f" ✓ Found {len(disease_targets)} targets")
            
            target_drugs = await ingest_chembl_candidates(
                targets=disease_targets,
                neo4j_client=neo4j,
                disease_name=disease_context.corrected_name,
                include_clinical_candidates=True,
                min_phase=min_phase
            )
            logger.info(f" ✓ Found {len(target_drugs)} target-based drugs (BEFORE repurposing filter)")
        except Exception as e:
            logger.error(f"❌ Target-based discovery failed: {e}")
            target_drugs = []
        finally:
            if neo4j:
                try:
                    neo4j.close()
                except Exception as close_error:
                    logger.warning(f"Failed to close Neo4j connection: {close_error}")
        # Track 2: MECHANISTIC REPURPOSING (PRIMARY SOURCE)
        logger.info(" 🔬 Track 2: Mechanistic repurposing (PRIMARY)...")

        from kg.mechanistic_repurposing import MechanisticRepurposingEngine

        repurposing_engine = MechanisticRepurposingEngine()


        try:
            # Get disease pathways for mechanistic analysis
            disease_pathway_data = await self.pathway_integrator.get_disease_pathways(
                disease_targets=disease_targets[:20]
            )

            disease_pathway_ids = []

            if disease_pathway_data and isinstance(disease_pathway_data[0], dict):
                disease_pathway_ids = [p["pathway_id"] for p in disease_pathway_data]
            else:
                disease_pathway_ids = disease_pathway_data 
                        
            logger.info(f"   Found {len(disease_pathway_ids)} disease pathways for mechanistic analysis")
            
            # Run mechanistic repurposing
            repurposing_candidates = await repurposing_engine.find_repurposing_candidates(
                disease_name=disease_context.corrected_name,
                disease_id=disease_id,
                disease_targets=disease_targets,
                disease_pathways=disease_pathway_ids,
                therapeutic_area=disease_context.therapeutic_area,
                min_phase=min_phase,
                top_n=50
            )
            
            logger.info(f" ✓ Found {len(repurposing_candidates)} mechanistic repurposing candidates")
            
            # Convert to standard drug format
            target_drugs = []
            for candidate in repurposing_candidates:
                target_drugs.append({
                    "chembl_id": candidate.drug_id,
                    "drug_name": candidate.drug_name,
                    "phase": candidate.phase,
                    "target": candidate.molecular_target,
                    "target_name": candidate.target_protein,
                    "drug_type": candidate.drug_type,
                    "score": candidate.opentargets_score,
                    
                    # ✅ REPURPOSING-SPECIFIC FIELDS
                    "original_indication": candidate.original_indication,
                    "repurposing_novelty": candidate.novelty_score,
                    "mechanistic_confidence": candidate.mechanistic_confidence,
                    "repurposing_rationale": candidate.disease_pathway_link,
                    "shared_pathways": candidate.shared_pathways,
                    "pathway_overlap": candidate.pathway_overlap_score,
                    
                    # ✅ EXPERIMENTAL VALIDATION
                    "in_vitro_experiments": candidate.in_vitro_experiments,
                    "in_vivo_experiments": candidate.in_vivo_experiments,
                    "biomarkers": candidate.biomarkers_to_measure,
                    
                    # ✅ SAFETY
                    "safety_concerns": candidate.safety_concerns,
                    "contraindications": candidate.contraindications,
                    "feasibility": candidate.repurposing_feasibility,
                    
                    "is_repurposing_candidate": True
                })
            
            logger.info(f" ✓ Converted to {len(target_drugs)} drug candidates")
            
        except Exception as e:
            logger.error(f"❌ Mechanistic repurposing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            target_drugs = []


        # =================================================================
        # PHASE 2: MERGE & DEDUPLICATE (REPURPOSING MODE)
        # =================================================================
        logger.info("📊 PHASE 2: Filter for Repurposing (Exclude existing treatments)")
        
        all_drugs = {}
        repurposing_filtered = 0
        
        # ✅ REPURPOSING CHANGE #2: ONLY add target-based drugs NOT in exclusion set
        for drug in target_drugs:
            drug_id = drug.get("chembl_id") or drug.get("drug_id")
            drug_name = (drug.get("drug_name") or drug.get("name", "")).lower()
            
            # Check if drug is in exclusion set
            if drug_id in exclusion_set or drug_name in exclusion_set:
                repurposing_filtered += 1
                logger.debug(f" ❌ Filtered OUT (already treats disease): {drug.get('drug_name')}")
                continue  # Skip this drug
            
            # This is a true repurposing candidate!
            if drug_id not in all_drugs:
                # ✅ REPURPOSING CHANGE #3: Add indication field and novelty flag
                indication = drug.get("indication", "")
                all_drugs[drug_id] = {
                    "drug_id": drug_id,
                    "drug_name": drug.get("drug_name") or drug.get("name"),
                    "phase": drug.get("phase", 1),
                    "has_clinical_evidence": False,  # Not for query disease
                    "opentargets_score": drug.get("score", 0.5),
                    "evidence_count": 2,
                    "source": "target_based",
                    "target_symbol": drug.get("target"),
                    "target_name": drug.get("target_name", ""),
                    "drug_type": drug.get("drug_type", "Unknown"),
                    # ✅ NEW FIELDS FOR REPURPOSING
                    "original_indication": indication,  # What it's approved for
                    "repurposing_novelty": 100.0,  # High novelty (new use)
                    "is_repurposing_candidate": True
                }
        
        candidates = list(all_drugs.values())
        logger.info(f"✅ Repurposing candidates: {len(candidates)} drugs (filtered out {repurposing_filtered} existing treatments)")
        
        if len(candidates) == 0:
            logger.warning("⚠️ NO REPURPOSING CANDIDATES FOUND! All target-based drugs already treat this disease.")
            logger.warning("   This suggests the query disease is well-studied with many approved drugs.")
            # Return empty result but don't crash
            return {
                "disease_context": {
                    "original_query": disease_context.original_query,
                    "corrected_name": disease_context.corrected_name,
                    "efo_id": disease_context.efo_id,
                    "mondo_id": disease_context.mondo_id,
                    "therapeutic_area": disease_context.therapeutic_area,
                    "is_cancer": disease_context.is_cancer,
                    "is_autoimmune": disease_context.is_autoimmune
                },
                "candidates": [],
                "ranked_metadata": [],
                "stats": {
                    "total_discovered": len(target_drugs),
                    "validated": 0,
                    "rejected": 0,
                    "final_count": 0,
                    "direct_drugs": len(direct_drugs),
                    "target_based_drugs": len(target_drugs),
                    "repurposing_filtered": repurposing_filtered,
                    "repurposing_candidates": 0
                }
            }

        # =================================================================
        # PHASE 3: ENRICHMENT (Same as before, non-blocking)
        # =================================================================
        if enable_enrichment:
            logger.info("📊 PHASE 3: Enrichment (non-blocking)")
            for candidate in candidates:
                # Pathway enrichment
                try:
                    disease_pathways = await self.pathway_integrator.get_disease_pathways(
                        disease_targets=disease_targets
                    )
                    disease_pathway_ids = [p["pathway_id"] for p in disease_pathways]
                    
                    target_symbol = candidate.get("target_symbol")
                    if target_symbol:
                        target_pathways = await self.pathway_integrator.get_target_pathways(
                            gene_symbol=target_symbol
                        )
                        target_pathway_ids = [p["pathway_id"] for p in target_pathways]
                        
                        overlap_result = await self.pathway_integrator.find_pathway_overlap(
                            disease_pathways=disease_pathway_ids,
                            target_pathways=target_pathway_ids
                        )
                        candidate["pathway_overlap"] = overlap_result.get("jaccard_similarity", 0.0)
                    else:
                        candidate["pathway_overlap"] = None
                except Exception as e:
                    logger.warning(f"Pathway enrichment failed for {candidate['drug_name']}: {e}")
                    candidate["pathway_overlap"] = None

                # PPI enrichment
                try:
                    target_symbol = candidate.get("target_symbol")
                    if target_symbol:
                        ppi_data = await self.ppi_integrator.get_protein_interactions(
                            gene_symbol=target_symbol,
                            confidence_threshold=0.7
                        )
                        candidate["ppi_confidence"] = len(ppi_data) / 10.0 if ppi_data else 0.0
                        candidate["ppi_partners"] = [p["partner"] for p in ppi_data[:5]]
                    else:
                        candidate["ppi_confidence"] = None
                        candidate["ppi_partners"] = []
                except Exception as e:
                    logger.warning(f"PPI enrichment failed for {candidate['drug_name']}: {e}")
                    candidate["ppi_confidence"] = None
                    candidate["ppi_partners"] = []

                # Mechanism flags
                candidate["mechanism_known"] = bool(
                    candidate.get("target_symbol") and
                    candidate.get("pathway_overlap") is not None
                )

                # Safety flags (simplified heuristic)
                safety_flags = []
                if candidate["phase"] == 0:
                    safety_flags.append("untested_in_humans")
                if not candidate["has_clinical_evidence"]:
                    safety_flags.append("no_disease_specific_data")
                
                candidate["safety_flags"] = safety_flags
                candidate["has_safety_concerns"] = len(safety_flags) > 0
                
                # MOA compatibility fields
                candidate["moa_appropriate"] = candidate["mechanism_known"]
                candidate["moa_confidence"] = 0.8 if candidate["mechanism_known"] else 0.5

        # =================================================================
        # PHASE 4: VALIDATION (Same as before)
        # =================================================================
        logger.info("📊 PHASE 4: Validation (unified evidence validator)")
        validated_candidates = []
        rejected_count = 0
        
        for candidate in candidates:
            result = self.validator.validate_drug(
                drug_name=candidate["drug_name"],
                phase=candidate["phase"],
                has_clinical_evidence=candidate["has_clinical_evidence"],
                mechanism_known=candidate.get("mechanism_known", True),
                safety_flags=candidate.get("safety_flags", [])
            )
            
            if result.decision != ValidationDecision.REJECT:
                candidate["validation_result"] = {
                    "decision": result.decision.value,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "evidence_scores": result.evidence_scores,
                    "flags": result.flags
                }
                validated_candidates.append(candidate)
            else:
                rejected_count += 1
                logger.debug(f"  Rejected: {candidate['drug_name']} - {result.reasoning}")
        
        logger.info(f"✅ Validated: {len(validated_candidates)} kept, {rejected_count} rejected")

        # =================================================================
        # PHASE 5: SCORING (WITH NOVELTY)
        # =================================================================
        logger.info("📊 PHASE 5: Scoring (WITH REPURPOSING NOVELTY)")
        scored_candidates = []
        
        for candidate in validated_candidates:
            score_breakdown = self.scorer.calculate_composite_score({
                "phase": candidate["phase"],
                "has_clinical_evidence": candidate["has_clinical_evidence"],
                "opentargets_score": candidate["opentargets_score"],
                "evidence_count": candidate["evidence_count"],
                "pathway_overlap": candidate.get("pathway_overlap"),
                "has_known_mechanism": candidate.get("moa_appropriate", True),
                "has_black_box_warning": "black_box" in candidate.get("safety_flags", []),
                "has_serious_adverse_events": "serious_ae" in candidate.get("safety_flags", []),
                "withdrawal_history": "withdrawn" in candidate.get("safety_flags", []),
                # ✅ NEW: Pass novelty score
                "repurposing_novelty": candidate.get("repurposing_novelty", 100.0)
            })
            
            candidate["score_breakdown"] = {
                "composite_score": score_breakdown.composite_score,
                "clinical_phase_score": score_breakdown.clinical_phase_score,
                "evidence_score": score_breakdown.evidence_score,
                "mechanism_score": score_breakdown.mechanism_score,
                "safety_score": score_breakdown.safety_score,
                "novelty_score": score_breakdown.novelty_score,  # ✅ NEW
                "confidence": score_breakdown.confidence
            }
            scored_candidates.append(candidate)
        
        logger.info(f"✅ Scored {len(scored_candidates)} candidates (novelty-weighted)")

        # =================================================================
        # PHASE 6: RANKING (Same as before)
        # =================================================================
        logger.info("📊 PHASE 6: Ranking")
        ranked_candidates = self.ranker.rank_candidates(
            scored_candidates,
            top_n=top_n_candidates
        )
        logger.info(f"✅ Ranked top {len(ranked_candidates)} candidates")

        # =================================================================
        # OUTPUT
        # =================================================================
        return {
            "disease_context": {
                "original_query": disease_context.original_query,
                "corrected_name": disease_context.corrected_name,
                "efo_id": disease_context.efo_id,
                "mondo_id": disease_context.mondo_id,
                "therapeutic_area": disease_context.therapeutic_area,
                "is_cancer": disease_context.is_cancer,
                "is_autoimmune": disease_context.is_autoimmune
            },
            "candidates": scored_candidates[:top_n_candidates],
            "ranked_metadata": [
                {
                    "drug_id": rc.drug_id,
                    "drug_name": rc.drug_name,
                    "rank": rc.rank,
                    "final_score": rc.final_score,
                    "tier": rc.tier,
                    "recommendation": rc.recommendation,
                    "original_indication": scored_candidates[i].get("original_indication", "Unknown")  # ✅ NEW
                }
                for i, rc in enumerate(ranked_candidates)
            ],
            "stats": {
                "total_discovered": len(target_drugs),
                "validated": len(validated_candidates),
                "rejected": rejected_count,
                "final_count": len(scored_candidates[:top_n_candidates]),
                "direct_drugs": len(direct_drugs),
                "target_based_drugs": len(target_drugs),
                "repurposing_filtered": repurposing_filtered,  # ✅ NEW
                "repurposing_candidates": len(scored_candidates)  # ✅ NEW
            }
        }


# Singleton instance
hybrid_discovery = HybridDrugDiscoveryV2()


# Backward compatibility wrapper for existing code
async def run_hybrid_discovery(
    disease_id: str,
    disease_context: DiseaseContext,
    min_phase: int = 1,
    top_n: int = 50
) -> Dict:
    """Wrapper function for backward compatibility."""
    return await hybrid_discovery.discover_for_disease(
        disease_id=disease_id,
        disease_context=disease_context,
        min_phase=min_phase,
        top_n_candidates=top_n
    )