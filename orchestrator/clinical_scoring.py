"""
Clinical Scoring System - Evidence-based, clinically meaningful scoring

Replaces the simplistic linear scoring in orchestrator/scoring.py

Evidence Hierarchy (from strongest to weakest):
1. Phase 3+ completed positive trial: 50 points
2. Phase 3 active trial: 40 points  
3. Phase 2 completed positive: 35 points
4. GWAS genome-wide significant: 25 points
5. Pathway overlap >0.3: 20 points
6. Phase 2 active: 20 points
7. DisGeNET score >0.7: 15 points
8. Pathway overlap >0.15: 15 points
9. Phase 1: 10 points
10. Literature only: 5 points
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClinicalConfidence(str, Enum):
    VERY_HIGH = "very_high"  # ≥70 points
    HIGH = "high"            # 50-69 points
    MODERATE = "moderate"    # 30-49 points
    LOW = "low"             # 15-29 points
    INSUFFICIENT = "insufficient"  # <15 points


@dataclass
class ClinicalScore:
    """Clinical evidence score for a drug candidate"""
    total_score: float  # 0-100
    confidence_level: ClinicalConfidence
    
    # Evidence breakdown
    trial_score: float = 0.0
    genetic_score: float = 0.0
    pathway_score: float = 0.0
    literature_score: float = 0.0
    
    # Evidence details
    evidence_items: List[str] = None
    red_flags: List[str] = None
    
    # Business scores (separate from clinical)
    patent_risk_score: float = 0.0  # 0-20
    supply_chain_score: float = 0.0  # 0-10
    
    def __post_init__(self):
        if self.evidence_items is None:
            self.evidence_items = []
        if self.red_flags is None:
            self.red_flags = []


class ClinicalScoringSystem:
    """
    Evidence-based clinical scoring system.
    
    Separates:
    - Clinical evidence (what we score)
    - Business factors (patent, supply) - tracked separately
    - Safety (gate, not score)
    """
    
    # Trial evidence weights
    PHASE_3_COMPLETED_POSITIVE = 50
    PHASE_3_ACTIVE = 40
    PHASE_2_COMPLETED_POSITIVE = 35
    PHASE_2_ACTIVE = 20
    PHASE_1 = 10
    
    # Genetic evidence weights
    GWAS_SIGNIFICANT = 25  # p < 5e-8
    DISGENET_HIGH = 15     # score > 0.7
    
    # Pathway weights
    PATHWAY_HIGH = 20      # Jaccard > 0.3
    PATHWAY_MODERATE = 15  # Jaccard > 0.15
    
    # Literature
    LITERATURE_ONLY = 5
    
    def calculate_clinical_score(
        self,
        candidate: Dict,
        trial_outcomes: List,  # List[TrialOutcome]
        pathway_result: Optional[Dict] = None,
        genetic_evidence: Optional[Dict] = None,
        literature_evidence: Optional[Dict] = None
    ) -> ClinicalScore:
        """
        Calculate evidence-based clinical score.
        
        Args:
            candidate: Drug candidate dict
            trial_outcomes: List of TrialOutcome from clinical_trial_parser
            pathway_result: PathwayOverlapResult from pathway_mechanism_validator
            genetic_evidence: DisGeNET/GWAS data
            literature_evidence: Literature data
        
        Returns:
            ClinicalScore with breakdown
        """
        logger.info(f"🎯 Scoring {candidate.get('name', 'Unknown')}")
        
        evidence_items = []
        red_flags = []
        
        # Score 1: Clinical Trials (STRONGEST evidence)
        trial_score = self._score_trials(trial_outcomes, evidence_items, red_flags)
        
        # Score 2: Genetic Evidence
        genetic_score = self._score_genetic(genetic_evidence, evidence_items)
        
        # Score 3: Pathway Overlap
        pathway_score = self._score_pathway(pathway_result, evidence_items)
        
        # Score 4: Literature (WEAKEST evidence)
        literature_score = self._score_literature(
            literature_evidence,
            evidence_items,
            has_clinical=trial_score > 0
        )
        
        # Total clinical score
        total = trial_score + genetic_score + pathway_score + literature_score
        
        # Determine confidence level
        if total >= 70:
            confidence = ClinicalConfidence.VERY_HIGH
        elif total >= 50:
            confidence = ClinicalConfidence.HIGH
        elif total >= 30:
            confidence = ClinicalConfidence.MODERATE
        elif total >= 15:
            confidence = ClinicalConfidence.LOW
        else:
            confidence = ClinicalConfidence.INSUFFICIENT
        
        logger.info(f"   Total: {total:.1f}, Confidence: {confidence.value}")
        
        return ClinicalScore(
            total_score=total,
            confidence_level=confidence,
            trial_score=trial_score,
            genetic_score=genetic_score,
            pathway_score=pathway_score,
            literature_score=literature_score,
            evidence_items=evidence_items,
            red_flags=red_flags
        )
    
    def _score_trials(
        self,
        trial_outcomes: List,
        evidence_items: List[str],
        red_flags: List[str]
    ) -> float:
        """Score clinical trial evidence"""
        if not trial_outcomes:
            return 0.0
        
        max_score = 0.0
        
        for trial in trial_outcomes:
            score = 0.0
            
            # Phase 3
            if trial.phase.value == "Phase 3":
                if trial.status.value == "completed_positive":
                    score = self.PHASE_3_COMPLETED_POSITIVE
                    evidence_items.append(
                        f"Phase 3 completed with positive outcome ({trial.nct_id})"
                    )
                elif trial.status.value == "active" or trial.status.value == "recruiting":
                    score = self.PHASE_3_ACTIVE
                    evidence_items.append(f"Active Phase 3 trial ({trial.nct_id})")
                elif trial.status.value == "completed_negative":
                    score = -20  # Negative evidence
                    red_flags.append(f"Phase 3 trial failed ({trial.nct_id})")
                elif trial.status.value == "terminated":
                    score = -30  # Strong negative evidence
                    red_flags.append(
                        f"Phase 3 terminated: {trial.termination_reason} ({trial.nct_id})"
                    )
            
            # Phase 2
            elif trial.phase.value == "Phase 2":
                if trial.status.value == "completed_positive":
                    score = self.PHASE_2_COMPLETED_POSITIVE
                    evidence_items.append(f"Phase 2 completed positive ({trial.nct_id})")
                elif trial.status.value == "active" or trial.status.value == "recruiting":
                    score = self.PHASE_2_ACTIVE
                    evidence_items.append(f"Active Phase 2 trial ({trial.nct_id})")
                elif trial.status.value == "terminated":
                    score = -15
                    red_flags.append(f"Phase 2 terminated ({trial.nct_id})")
            
            # Phase 1
            elif trial.phase.value == "Phase 1":
                if trial.status.value in ["completed_positive", "completed_unknown", "active"]:
                    score = self.PHASE_1
                    evidence_items.append(f"Phase 1 trial ({trial.nct_id})")
            
            max_score = max(max_score, score)
        
        return max(max_score, 0.0)  # Don't allow negative total
    
    def _score_genetic(
        self,
        genetic_evidence: Optional[Dict],
        evidence_items: List[str]
    ) -> float:
        """Score genetic evidence (GWAS, DisGeNET)"""
        if not genetic_evidence:
            return 0.0
        
        score = 0.0
        
        # GWAS evidence
        gwas_p = genetic_evidence.get("gwas_p_value")
        if gwas_p and gwas_p < 5e-8:
            score += self.GWAS_SIGNIFICANT
            evidence_items.append(f"Genome-wide significant GWAS (p={gwas_p:.2e})")
        
        # DisGeNET evidence
        disgenet_score = genetic_evidence.get("disgenet_score", 0)
        if disgenet_score > 0.7:
            score += self.DISGENET_HIGH
            evidence_items.append(f"Strong gene-disease association (DisGeNET: {disgenet_score:.2f})")
        
        return score
    
    def _score_pathway(
        self,
        pathway_result: Optional[Dict],
        evidence_items: List[str]
    ) -> float:
        """Score pathway overlap"""
        if not pathway_result:
            return 0.0
        
        jaccard = pathway_result.get("jaccard_similarity", 0)
        
        if jaccard >= 0.3:
            evidence_items.append(f"High pathway overlap (Jaccard: {jaccard:.2f})")
            return self.PATHWAY_HIGH
        elif jaccard >= 0.15:
            evidence_items.append(f"Moderate pathway overlap (Jaccard: {jaccard:.2f})")
            return self.PATHWAY_MODERATE
        else:
            return 0.0
    
    def _score_literature(
        self,
        literature_evidence: Optional[Dict],
        evidence_items: List[str],
        has_clinical: bool
    ) -> float:
        """Score literature evidence (weakest)"""
        if not literature_evidence:
            return 0.0
        
        # Only give literature score if NO clinical evidence
        if has_clinical:
            return 0.0
        
        evidence_items.append("Literature evidence only (weak)")
        return self.LITERATURE_ONLY


# Singleton
clinical_scorer = ClinicalScoringSystem()
