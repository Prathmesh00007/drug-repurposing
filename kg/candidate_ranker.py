"""
Candidate Ranking System.

Ranks and prioritizes drug repurposing candidates based on:
1. Composite scores from scoring engine
2. Novelty (prioritize unexpected findings)
3. Clinical feasibility
4. Repurposing potential

Outputs ranked, filtered list ready for clinical review.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Different ranking strategies."""
    SCORE_ONLY = "score_only"  # Pure score-based
    BALANCED = "balanced"  # Score + novelty + feasibility
    NOVELTY_FOCUSED = "novelty_focused"  # Prioritize unexpected findings
    CLINICAL_FOCUSED = "clinical_focused"  # Prioritize clinical readiness


@dataclass
class RankedCandidate:
    """Candidate with ranking information."""
    drug_id: str
    drug_name: str
    rank: int
    composite_score: float
    novelty_score: float
    feasibility_score: float
    final_score: float
    tier: str  # "High Priority", "Medium Priority", "Low Priority"
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            "drug_id": self.drug_id,
            "drug_name": self.drug_name,
            "rank": self.rank,
            "composite_score": self.composite_score,
            "novelty_score": self.novelty_score,
            "feasibility_score": self.feasibility_score,
            "final_score": self.final_score,
            "tier": self.tier,
            "recommendation": self.recommendation
        }


class CandidateRanker:
    """
    Ranks drug repurposing candidates using multiple strategies.
    """
    
    def __init__(self, strategy: RankingStrategy = RankingStrategy.BALANCED):
        """
        Initialize ranker.
        
        Args:
            strategy: Ranking strategy to use
        """
        self.strategy = strategy
    
    def calculate_novelty_score(
        self,
        candidate: Dict,
        known_drugs_for_disease: Optional[List[str]] = None
    ) -> float:
        """
        Calculate novelty score (0-100).
        
        Novelty indicators:
        - Drug from different therapeutic area (+40)
        - No prior clinical trials for this disease (+30)
        - Unexpected mechanism (+20)
        - Recent discovery (<5 years) (+10)
        
        Args:
            candidate: Candidate dictionary
            known_drugs_for_disease: List of drug IDs already used for disease
            
        Returns:
            Novelty score 0-100
        """
        score = 0.0
        
        # Different therapeutic area
        if candidate.get("therapeutic_area_match") == False:
            score += 40
        
        # No prior clinical trials for this disease
        if not candidate.get("has_clinical_evidence"):
            score += 30
        
        # Not in known drugs list
        if known_drugs_for_disease and candidate.get("drug_id") not in known_drugs_for_disease:
            score += 20
        
        # Unexpected mechanism
        if candidate.get("mechanism_unexpected", False):
            score += 20
        
        # Recent discovery
        years_since_approval = candidate.get("years_on_market", 100)
        if years_since_approval < 5:
            score += 10
        
        return min(score, 100)
    
    def calculate_feasibility_score(
        self,
        candidate: Dict
    ) -> float:
        """
        Calculate clinical feasibility score (0-100).
        
        Feasibility indicators:
        - Already approved (Phase 4): +40
        - Oral formulation available: +20
        - Good safety profile: +20
        - Patent expired (low cost): +10
        - Known dosing regimen: +10
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Feasibility score 0-100
        """
        score = 0.0
        
        # Approved drug
        if candidate.get("phase", 0) == 4:
            score += 40
        elif candidate.get("phase", 0) >= 3:
            score += 30
        elif candidate.get("phase", 0) >= 2:
            score += 20
        
        # Oral formulation
        if candidate.get("is_oral", False):
            score += 20
        
        # Safety profile
        safety_score = candidate.get("score_breakdown", {}).get("safety_score", 50)
        if safety_score >= 90:
            score += 20
        elif safety_score >= 70:
            score += 15
        elif safety_score >= 50:
            score += 10
        
        # Patent status
        if candidate.get("patent_expired", False):
            score += 10
        
        # Known dosing
        if candidate.get("has_known_dosing", True):
            score += 10
        
        return min(score, 100)
    
    def calculate_final_score(
        self,
        composite_score: float,
        novelty_score: float,
        feasibility_score: float
    ) -> float:
        """
        Calculate final ranking score based on strategy.
        
        Args:
            composite_score: Base composite score (0-100)
            novelty_score: Novelty score (0-100)
            feasibility_score: Feasibility score (0-100)
            
        Returns:
            Final score (0-100)
        """
        if self.strategy == RankingStrategy.SCORE_ONLY:
            return composite_score
        
        elif self.strategy == RankingStrategy.BALANCED:
            # 60% composite, 20% novelty, 20% feasibility
            return (
                composite_score * 0.6 +
                novelty_score * 0.2 +
                feasibility_score * 0.2
            )
        
        elif self.strategy == RankingStrategy.NOVELTY_FOCUSED:
            # 40% composite, 40% novelty, 20% feasibility
            return (
                composite_score * 0.4 +
                novelty_score * 0.4 +
                feasibility_score * 0.2
            )
        
        elif self.strategy == RankingStrategy.CLINICAL_FOCUSED:
            # 50% composite, 10% novelty, 40% feasibility
            return (
                composite_score * 0.5 +
                novelty_score * 0.1 +
                feasibility_score * 0.4
            )
        
        else:
            return composite_score
    
    def assign_tier(
        self,
        final_score: float,
        phase: int,
        has_clinical_evidence: bool
    ) -> str:
        """
        Assign priority tier based on final score and clinical status.
        
        Tiers:
        - High Priority: Score >= 70 OR (Approved + Clinical Evidence)
        - Medium Priority: Score >= 50
        - Low Priority: Score < 50
        
        Args:
            final_score: Final ranking score
            phase: Clinical phase
            has_clinical_evidence: Has trial data for disease
            
        Returns:
            Tier string
        """
        # High priority criteria
        if final_score >= 70:
            return "High Priority"
        
        if phase == 4 and has_clinical_evidence:
            return "High Priority"
        
        # Medium priority
        if final_score >= 50:
            return "Medium Priority"
        
        if phase >= 3:
            return "Medium Priority"
        
        # Low priority
        return "Low Priority"
    
    def generate_recommendation(
        self,
        candidate: Dict,
        tier: str,
        novelty_score: float,
        feasibility_score: float
    ) -> str:
        """
        Generate human-readable recommendation.
        
        Args:
            candidate: Candidate dictionary
            tier: Priority tier
            novelty_score: Novelty score
            feasibility_score: Feasibility score
            
        Returns:
            Recommendation string
        """
        drug_name = candidate.get("drug_name", "Unknown")
        phase = candidate.get("phase", 0)
        
        if tier == "High Priority":
            if phase == 4:
                return f"{drug_name}: Strong repurposing candidate (approved drug). Recommend literature review and pilot study design."
            else:
                return f"{drug_name}: High-confidence candidate. Recommend detailed mechanism investigation and feasibility assessment."
        
        elif tier == "Medium Priority":
            if novelty_score >= 70:
                return f"{drug_name}: Novel candidate with interesting mechanism. Recommend pathway analysis and computational validation."
            else:
                return f"{drug_name}: Moderate evidence. Recommend additional validation before clinical consideration."
        
        else:  # Low Priority
            if feasibility_score < 30:
                return f"{drug_name}: Low feasibility for repurposing. Consider for basic research only."
            else:
                return f"{drug_name}: Insufficient evidence at this time. Monitor for emerging data."
    
    def rank_candidates(
        self,
        candidates: List[Dict],
        known_drugs_for_disease: Optional[List[str]] = None,
        top_n: Optional[int] = None
    ) -> List[RankedCandidate]:
        """
        Rank all candidates and return sorted list.
        
        Args:
            candidates: List of candidate dictionaries (must have 'score_breakdown')
            known_drugs_for_disease: List of drug IDs already used for disease
            top_n: Return only top N candidates (None = return all)
            
        Returns:
            Sorted list of RankedCandidate objects
        """
        logger.info(f"Ranking {len(candidates)} candidates using {self.strategy.value} strategy...")
        
        ranked = []
        
        for candidate in candidates:
            # Get composite score from score_breakdown
            score_breakdown = candidate.get("score_breakdown", {})
            composite_score = score_breakdown.get("composite_score", 0)
            
            # Calculate novelty and feasibility
            novelty_score = self.calculate_novelty_score(
                candidate,
                known_drugs_for_disease
            )
            
            feasibility_score = self.calculate_feasibility_score(candidate)
            
            # Calculate final score
            final_score = self.calculate_final_score(
                composite_score,
                novelty_score,
                feasibility_score
            )
            
            # Assign tier
            tier = self.assign_tier(
                final_score,
                candidate.get("phase", 0),
                candidate.get("has_clinical_evidence", False)
            )
            
            # Generate recommendation
            recommendation = self.generate_recommendation(
                candidate,
                tier,
                novelty_score,
                feasibility_score
            )
            
            ranked.append(RankedCandidate(
                drug_id=candidate.get("drug_id", ""),
                drug_name=candidate.get("drug_name", ""),
                rank=0,  # Will be assigned after sorting
                composite_score=composite_score,
                novelty_score=novelty_score,
                feasibility_score=feasibility_score,
                final_score=final_score,
                tier=tier,
                recommendation=recommendation
            ))
        
        # Sort by final score (descending)
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for i, candidate in enumerate(ranked, 1):
            candidate.rank = i
        
        # Filter to top N if requested
        if top_n:
            ranked = ranked[:top_n]
        
        # Log statistics
        tier_counts = {"High Priority": 0, "Medium Priority": 0, "Low Priority": 0}
        for candidate in ranked:
            tier_counts[candidate.tier] += 1
        
        logger.info(
            f"✅ Ranking complete: {len(ranked)} candidates "
            f"(High: {tier_counts['High Priority']}, "
            f"Medium: {tier_counts['Medium Priority']}, "
            f"Low: {tier_counts['Low Priority']})"
        )
        
        return ranked
    
    def filter_by_tier(
        self,
        ranked_candidates: List[RankedCandidate],
        tiers: List[str]
    ) -> List[RankedCandidate]:
        """
        Filter ranked candidates by tier.
        
        Args:
            ranked_candidates: List of ranked candidates
            tiers: List of tiers to keep (e.g., ["High Priority", "Medium Priority"])
            
        Returns:
            Filtered list
        """
        filtered = [c for c in ranked_candidates if c.tier in tiers]
        logger.info(f"Filtered to {len(filtered)} candidates in tiers: {tiers}")
        return filtered


# Example usage
if __name__ == "__main__":
    from kg.scoring_engine import ScoringEngine
    
    # Mock candidates
    candidates = [
        {
            "drug_id": "CHEMBL1",
            "drug_name": "Drug A",
            "phase": 4,
            "has_clinical_evidence": True,
            "opentargets_score": 0.85,
            "evidence_count": 5,
            "is_oral": True,
            "patent_expired": True
        },
        {
            "drug_id": "CHEMBL2",
            "drug_name": "Drug B",
            "phase": 2,
            "has_clinical_evidence": False,
            "opentargets_score": 0.60,
            "evidence_count": 3,
            "therapeutic_area_match": False
        }
    ]
    
    # Score candidates
    engine = ScoringEngine()
    scored = engine.batch_score(candidates)
    
    # Rank candidates
    ranker = CandidateRanker(strategy=RankingStrategy.BALANCED)
    ranked = ranker.rank_candidates(scored)
    
    print(f"\n{'='*60}")
    print("CANDIDATE RANKING TEST")
    print(f"{'='*60}")
    for candidate in ranked:
        print(f"\nRank {candidate.rank}: {candidate.drug_name}")
        print(f"  Final Score: {candidate.final_score:.1f}")
        print(f"  Tier: {candidate.tier}")
        print(f"  Recommendation: {candidate.recommendation}")
