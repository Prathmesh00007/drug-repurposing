import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class ScoringWeights:
    """Configurable weights for scoring components."""
    clinical_phase: float = 0.35
    evidence_strength: float = 0.25
    mechanism_overlap: float = 0.20
    safety_profile: float = 0.10
    novelty: float = 0.10

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (
            self.clinical_phase +
            self.evidence_strength +
            self.mechanism_overlap +
            self.safety_profile +
            self.novelty
        )
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

# scoring_engine.py - UPDATE ScoreBreakdown dataclass

@dataclass
class ScoreBreakdown:
    """Detailed breakdown of composite score."""
    composite_score: float  # 0-100
    novelty_score: float    # ✅ NEW
    clinical_phase_score: float  # 0-100
    evidence_score: float   # 0-100
    mechanism_score: float  # 0-100
    safety_score: float     # 0-100
    confidence: float       # 0-1
    reasoning: str
    flags: List[str]



class ScoringEngine:
    """
    Multi-factor scoring engine for drug repurposing candidates.

    Design principles:
    - Transparent: Each component is clearly defined
    - Configurable: Weights can be adjusted
    - Interpretable: Provides reasoning for scores
    - Fast: <50ms per candidate
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize scoring engine.

        Args:
            weights: Custom scoring weights (uses defaults if None)
        """
        self.weights = weights or ScoringWeights()

    def score_clinical_phase(self, phase: int) -> float:
        """
        Score based on clinical trial phase.

        Scoring:
        - Phase 0/Preclinical: 10
        - Phase 1: 30
        - Phase 2: 50
        - Phase 3: 70
        - Phase 4/Approved: 100

        Args:
            phase: Clinical trial phase (0-4)

        Returns:
            Score 0-100
        """
        phase_scores = {
            0: 10,   # Preclinical/unknown
            1: 30,   # Safety established
            2: 50,   # Efficacy signals
            3: 70,   # Strong efficacy data
            4: 100   # Approved, proven efficacy
        }
        return phase_scores.get(phase, 10)

    def score_evidence_strength(
        self,
        has_clinical_evidence: bool,
        opentargets_score: float,
        evidence_count: int,
        literature_count: Optional[int] = None
    ) -> float:
        """
        Score based on evidence strength and diversity.

        Components:
        1. Clinical evidence (40 points)
        2. Open Targets score (30 points)
        3. Evidence diversity (20 points)
        4. Literature support (10 points)

        Args:
            has_clinical_evidence: Has trial data for disease
            opentargets_score: Association score (0-1)
            evidence_count: Number of evidence sources
            literature_count: Number of supporting papers

        Returns:
            Score 0-100
        """
        score = 0.0

        # Component 1: Clinical evidence (0-40 points)
        if has_clinical_evidence:
            score += 40

        # Component 2: Open Targets score (0-30 points)
        score += opentargets_score * 30

        # Component 3: Evidence diversity (0-20 points)
        # 1 source = 5, 2 sources = 10, 3 sources = 15, 4+ sources = 20
        evidence_points = min(evidence_count * 5, 20)
        score += evidence_points

        # Component 4: Literature support (0-10 points)
        if literature_count is not None:
            if literature_count >= 100:
                score += 10
            elif literature_count >= 50:
                score += 8
            elif literature_count >= 20:
                score += 6
            elif literature_count >= 10:
                score += 4
            elif literature_count >= 5:
                score += 2
            # else 0 points

        return min(score, 100)

    def score_mechanism_overlap(
        self,
        opentargets_score: float = 0.0,  # ← NEW: Primary signal
        pathway_overlap: Optional[float] = None,  # ← Downgraded to validation
        has_known_mechanism: bool = True,
        target_druggability: Optional[str] = None
    ) -> float:
        """
        Score based on target-disease association and mechanism.
        
        NEW: Uses OpenTargets association score as primary signal.

        Components:
        1. OpenTargets Association Score (0-40 points) ← NEW PRIMARY
        2. Pathway overlap validation (0-30 points) ← DOWNGRADED
        3. Known mechanism (0-15 points) ← REDUCED
        4. Target druggability (0-15 points) ← REDUCED

        Args:
            opentargets_score: OpenTargets association score (0-1)
            pathway_overlap: Jaccard similarity of pathways (0-1)
            has_known_mechanism: Mechanism of action is known
            target_druggability: Druggability class (e.g., "Tier 1")

        Returns:
            Score 0-100
        """
        score = 0.0

        # Component 1: OpenTargets Association Score (0-40 points) ← NEW PRIMARY
        # This is pre-computed by OpenTargets using genetics, literature, pathways, etc.
        score += opentargets_score * 40

        # Component 2: Pathway overlap validation (0-30 points) ← DOWNGRADED
        # Only used as secondary confirmation
        if pathway_overlap is not None:
            if pathway_overlap > 0.15:  # At least 15% overlap
                score += pathway_overlap * 30
            else:
                score += 5  # Minimal score for low overlap
        else:
            score += 10  # Missing data gets conservative score

        # Component 3: Known mechanism (0-15 points) ← REDUCED
        if has_known_mechanism:
            score += 15

        # Component 4: Target druggability (0-15 points) ← REDUCED
        if target_druggability:
            druggability_scores = {
                "Tier 1": 15,
                "Tier 2": 10,
                "Tier 3": 5,
                "Unknown": 2
            }
            score += druggability_scores.get(target_druggability, 2)

        return min(score, 100)

    def score_safety_profile(
        self,
        has_black_box_warning: bool = False,
        has_serious_adverse_events: bool = False,
        withdrawal_history: bool = False,
        years_on_market: Optional[int] = None
    ) -> float:
        """
        Score based on safety profile.

        Philosophy: Start with 100, subtract for red flags.

        Penalties:
        - Black box warning: -30
        - Serious adverse events: -20
        - Withdrawal history: -40

        Bonuses:
        - Long market history: +10

        Args:
            has_black_box_warning: FDA black box warning
            has_serious_adverse_events: Serious AEs reported
            withdrawal_history: Ever withdrawn from market
            years_on_market: Years since approval

        Returns:
            Score 0-100
        """
        score = 100.0

        # Apply penalties
        if has_black_box_warning:
            score -= 30
        if has_serious_adverse_events:
            score -= 20
        if withdrawal_history:
            score -= 40

        # Bonus for long market history (indicates safety)
        if years_on_market and years_on_market >= 10:
            score = min(score + 10, 100)

        return max(score, 0)
        # scoring_engine.py - ADD this new method to ScoringEngine class

    def score_repurposing_novelty(
        self,
        repurposing_novelty: Optional[float] = None,
        original_indication: Optional[str] = None
    ) -> float:
        """
        Score based on repurposing novelty (0-100).
        
        High novelty = drug approved for different disease (true repurposing)
        Low novelty = drug already treats query disease (not repurposing)
        
        Args:
            repurposing_novelty: Pre-calculated novelty score (0-100)
            original_indication: What the drug is approved for
        
        Returns:
            Score 0-100
        """
        if repurposing_novelty is not None:
            return min(repurposing_novelty, 100)
        
        # Fallback: If we have indication data, infer novelty
        if original_indication:
            # Drug has known indication = some novelty
            return 70.0
        
        # No data = moderate novelty
        return 50.0


    # scoring_engine.py - UPDATE calculate_composite_score method

    def calculate_composite_score(
        self,
        candidate: Dict
    ) -> ScoreBreakdown:
        """
        Calculate composite score for a drug candidate (REPURPOSING MODE).
        
        Args:
            candidate: Dictionary with keys:
                - phase: int
                - has_clinical_evidence: bool
                - opentargets_score: float
                - evidence_count: int
                - literature_count: Optional[int]
                - pathway_overlap: Optional[float]
                - has_known_mechanism: bool
                - target_druggability: Optional[str]
                - has_black_box_warning: bool
                - has_serious_adverse_events: bool
                - withdrawal_history: bool
                - years_on_market: Optional[int]
                - repurposing_novelty: Optional[float]  # ✅ NEW
                - original_indication: Optional[str]    # ✅ NEW
        
        Returns:
            ScoreBreakdown with composite score and component scores
        """
        # ✅ Calculate novelty score FIRST
        novelty_score = self.score_repurposing_novelty(
            repurposing_novelty=candidate.get("repurposing_novelty"),
            original_indication=candidate.get("original_indication")
        )
        
        # Calculate other component scores (same as before)
        clinical_score = self.score_clinical_phase(
            candidate.get("phase", 0)
        )
        
        evidence_score = self.score_evidence_strength(
            has_clinical_evidence=candidate.get("has_clinical_evidence", False),
            opentargets_score=candidate.get("opentargets_score", 0.0),
            evidence_count=candidate.get("evidence_count", 0),
            literature_count=candidate.get("literature_count")
        )
        
        mechanism_score = self.score_mechanism_overlap(
            opentargets_score=candidate.get("opentargets_score", 0.0),  # ← ADD THIS
            pathway_overlap=candidate.get("pathway_overlap"),
            has_known_mechanism=candidate.get("has_known_mechanism", True),
            target_druggability=candidate.get("target_druggability")
        )
        
        safety_score = self.score_safety_profile(
            has_black_box_warning=candidate.get("has_black_box_warning", False),
            has_serious_adverse_events=candidate.get("has_serious_adverse_events", False),
            withdrawal_history=candidate.get("withdrawal_history", False),
            years_on_market=candidate.get("years_on_market")
        )
        
        # ✅ Calculate weighted composite score (REPURPOSING WEIGHTS)
        composite = (
            novelty_score * self.weights.novelty +
            clinical_score * self.weights.clinical_phase +
            mechanism_score * self.weights.mechanism_overlap +
            evidence_score * self.weights.evidence_strength +
            safety_score * self.weights.safety_profile
        )
        
        # Calculate confidence (based on data completeness)
        data_completeness = sum([
            1 if candidate.get("has_clinical_evidence") else 0,
            1 if candidate.get("pathway_overlap") is not None else 0,
            1 if candidate.get("literature_count") is not None else 0,
            1 if candidate.get("target_druggability") is not None else 0,
            1 if candidate.get("repurposing_novelty") is not None else 0  # ✅ NEW
        ]) / 5.0  # ✅ CHANGED: Divide by 5 now
        
        confidence = 0.5 + (data_completeness * 0.5)  # Range: 0.5-1.0
        
        # Generate reasoning
        reasoning_parts = []
        if novelty_score >= 80:
            reasoning_parts.append("high repurposing novelty")
        if clinical_score >= 70:
            reasoning_parts.append("strong clinical data")
        if mechanism_score >= 60:
            reasoning_parts.append("good mechanistic rationale")
        if evidence_score >= 70:
            reasoning_parts.append("robust evidence")
        if safety_score < 70:
            reasoning_parts.append("some safety concerns")
        
        reasoning = f"Composite score {composite:.1f}/100 based on: " + ", ".join(reasoning_parts) if reasoning_parts else f"Composite score {composite:.1f}/100"
        
        # Generate flags
        flags = []
        if novelty_score < 50:
            flags.append("low_novelty")
        if clinical_score < 30:
            flags.append("early_stage")
        if evidence_score < 40:
            flags.append("weak_evidence")
        if safety_score < 60:
            flags.append("safety_concerns")
        if confidence < 0.7:
            flags.append("incomplete_data")
        
        return ScoreBreakdown(
            composite_score=composite,
            novelty_score=novelty_score,  # ✅ NEW
            clinical_phase_score=clinical_score,
            evidence_score=evidence_score,
            mechanism_score=mechanism_score,
            safety_score=safety_score,
            confidence=confidence,
            reasoning=reasoning,
            flags=flags
        )


    def batch_score(
        self,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Score a batch of candidates.

        Args:
            candidates: List of candidate dictionaries

        Returns:
            List of candidates with added 'score_breakdown' field
        """
        scored_candidates = []

        for candidate in candidates:
            score_breakdown = self.calculate_composite_score(candidate)

            # Add score breakdown to candidate
            candidate["score_breakdown"] = {
                "composite_score": score_breakdown.composite_score,
                "clinical_phase_score": score_breakdown.clinical_phase_score,
                "evidence_score": score_breakdown.evidence_score,
                "mechanism_score": score_breakdown.mechanism_score,
                "safety_score": score_breakdown.safety_score,
                "confidence": score_breakdown.confidence,
                "reasoning": score_breakdown.reasoning,
                "flags": score_breakdown.flags
            }
            scored_candidates.append(candidate)

        logger.info(f"Scored {len(candidates)} candidates")
        return scored_candidates