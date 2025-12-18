"""
Deterministic scoring logic for Route A.

FIXED VERSION: Uses kg.py scores as base, adds business modifiers only.
NO double scoring - respects the multi-factor clinical scoring from kg.py.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

from backend.app.schemas import (
    Candidate, ScoredCandidate, PatentRiskTier, SourcingSignal,
    TrialsOutput, PatentOutput, EximOutput
)

logger = logging.getLogger(__name__)


def rank_candidates(
    candidates: List[Candidate],
    trials_output: TrialsOutput,
    patent_outputs: Dict[str, PatentOutput],
    exim_outputs: Dict[str, EximOutput],
    unmet_need_count: int,
    strict_fto: bool = False,
    disease_context = None,  # DiseaseContext
    pathway_results: Dict[str, Any] = None,  # pathway results by target
    safety_results: Dict[str, Any] = None  # safety results by drug
) -> tuple[List[ScoredCandidate], List[ScoredCandidate]]:
    """
    ✅ FIXED VERSION: Business-focused ranking that respects kg.py clinical scoring.

    Architecture:
    1. Start with kg.py composite score (0-100) - comprehensive clinical evidence
    2. Apply business modifiers ONLY:
       - Patent risk: -20 to 0 points
       - Trial competition: -15 to 0 points
       - Supply chain: 0 to +10 points
       - Unmet need: 0 to +10 points
    3. Clamp final score to [0, 100]

    This eliminates double scoring while preserving business intelligence layer.

    Args:
        candidates: List of Candidate objects (with .score from kg.py, 0-100 scale)
        trials_output: Clinical trials data
        patent_outputs: Patent landscape by candidate
        exim_outputs: Supply chain data by candidate
        unmet_need_count: Number of unmet needs identified
        strict_fto: If True, exclude high patent risk candidates
        disease_context: DiseaseContext (optional, for future use)
        pathway_results: Pathway data (optional, for future use)
        safety_results: Safety data (optional, for future use)

    Returns:
        Tuple of (top_3_candidates, alternate_candidates)
    """
    logger.info(f"📊 Ranking {len(candidates)} candidates (business modifiers only)...")

    scored_candidates = []

    for candidate in candidates:
        candidate_name = candidate.name

        # ═══════════════════════════════════════════════════════════════
        # START WITH KG.PY BASE SCORE (0-100)
        # ═══════════════════════════════════════════════════════════════
        # This already incorporates:
        # - Clinical phase (40% weight)
        # - Evidence strength (30% weight)
        # - Mechanism/pathway overlap (20% weight)
        # - Safety profile (10% weight)

        base_clinical_score = candidate.score  # ← From kg.py, already 0-100

        logger.debug(f"   {candidate_name}: base_score={base_clinical_score:.1f}")

        # ═══════════════════════════════════════════════════════════════
        # APPLY BUSINESS MODIFIERS (Adjustments in points)
        # ═══════════════════════════════════════════════════════════════

        # Modifier 1: Patent Risk (-20 to 0 points)
        patent_output = patent_outputs.get(candidate_name)
        patent_adjustment = _calculate_patent_adjustment(patent_output, strict_fto)

        # Strict FTO: exclude high-risk candidates
        if strict_fto and patent_adjustment <= -20:
            logger.info(f"   ❌ {candidate_name}: EXCLUDED (high patent risk, strict_fto=True)")
            continue

        # Modifier 2: Trial Competition (-15 to 0 points)
        trial_adjustment = _calculate_trial_competition_adjustment(
            trials_output, 
            candidate_name
        )

        # Modifier 3: Supply Chain (0 to +10 points)
        exim_output = exim_outputs.get(candidate_name)
        supply_adjustment = _calculate_supply_adjustment(exim_output)

        # Modifier 4: Unmet Need Bonus (0 to +10 points)
        unmet_need_adjustment = _calculate_unmet_need_adjustment(unmet_need_count)

        # ═══════════════════════════════════════════════════════════════
        # CALCULATE FINAL SCORE
        # ═══════════════════════════════════════════════════════════════

        total_adjustment = (
            patent_adjustment + 
            trial_adjustment + 
            supply_adjustment + 
            unmet_need_adjustment
        )

        final_score = base_clinical_score + total_adjustment
        final_score = max(0.0, min(100.0, final_score))  # Clamp to [0, 100]

        # Build rationale
        rationale = []
        rationale.append(f"Clinical evidence score: {base_clinical_score:.1f}/100 (from kg.py)")

        if patent_adjustment != 0:
            rationale.append(
                f"Patent risk: {patent_adjustment:+.1f} points "
                f"({patent_output.risk_tier.value if patent_output else 'unknown'})"
            )

        if trial_adjustment != 0:
            trial_count = len(trials_output.candidate_trials.get(candidate_name, []))
            rationale.append(
                f"Trial competition: {trial_adjustment:+.1f} points "
                f"({trial_count} active trials)"
            )

        if supply_adjustment > 0:
            rationale.append(f"Supply chain: {supply_adjustment:+.1f} points")

        if unmet_need_adjustment > 0:
            rationale.append(f"Unmet need: {unmet_need_adjustment:+.1f} points")

        rationale.append(f"Final score: {final_score:.1f}/100")

        # Create scored candidate
        scored = ScoredCandidate(
            candidate=candidate,
            final_score=final_score,
            evidence_score=base_clinical_score / 100.0,  # Normalize to 0-1 for schema
            trial_penalty=abs(trial_adjustment) / 100.0,  # Convert to 0-1 for schema
            patent_penalty=abs(patent_adjustment) / 100.0,  # Convert to 0-1 for schema
            supply_bonus=supply_adjustment / 100.0,  # Convert to 0-1 for schema
            unmet_need_bonus=unmet_need_adjustment / 100.0,  # Convert to 0-1 for schema
            rationale=rationale,
            # Extended fields (may not exist in schema, add if needed)
            literature_score=None,  # Not recalculated
            clinical_score=base_clinical_score,  # Store original
            patent_score=20.0 + patent_adjustment,  # Convert penalty to 0-20 score
            confidence_level="High" if base_clinical_score >= 70 else "Medium"
        )

        scored_candidates.append(scored)

    # Sort by final score (descending)
    scored_candidates.sort(key=lambda x: x.final_score, reverse=True)

    # Split into top recommendations and alternates
    top_3 = scored_candidates[:3]
    alternates = scored_candidates[3:5]

    if top_3:
        logger.info(f"   ✅ Top 3 scores: {[f'{s.candidate.name}: {s.final_score:.1f}' for s in top_3]}")
    else:
        logger.warning(f"   ⚠️ No candidates passed filtering")

    return top_3, alternates


def _calculate_patent_adjustment(
    patent_output: Optional[PatentOutput], 
    strict_fto: bool
) -> float:
    """
    Calculate patent risk adjustment.

    Returns:
        Adjustment in points (-20 to 0)
        - High risk: -20 points
        - Medium risk: -10 points
        - Low/Unknown risk: 0 points
    """
    if not patent_output:
        return 0.0

    risk_tier = patent_output.risk_tier

    if risk_tier == PatentRiskTier.HIGH:
        return -20.0  # Significant penalty
    elif risk_tier == PatentRiskTier.MEDIUM:
        return -10.0  # Moderate penalty
    elif risk_tier == PatentRiskTier.LOW:
        return 0.0  # No penalty
    else:  # UNKNOWN
        return 0.0  # Neutral


def _calculate_trial_competition_adjustment(
    trials_output: TrialsOutput,
    candidate_name: str
) -> float:
    """
    Calculate trial competition adjustment.

    More active trials = more competition = penalty

    Returns:
        Adjustment in points (-15 to 0)
        - 0 trials: 0 points (no competition)
        - 1-9 trials: -5 points (moderate competition)
        - 10-29 trials: -10 points (high competition)
        - 30+ trials: -15 points (very high competition)
    """
    trial_count = len(trials_output.candidate_trials.get(candidate_name, []))

    if trial_count == 0:
        return 0.0  # No competition
    elif trial_count < 10:
        return -5.0  # Moderate competition
    elif trial_count < 30:
        return -10.0  # High competition
    else:
        return -15.0  # Very high competition


def _calculate_supply_adjustment(exim_output: Optional[EximOutput]) -> float:
    """
    Calculate supply chain adjustment.

    Returns:
        Adjustment in points (0 to +10)
        - Strong signal: +10 points
        - Moderate signal: +5 points
        - Weak/Unknown signal: 0 points
    """
    if not exim_output:
        return 0.0

    signal = exim_output.sourcing_signal

    if signal == SourcingSignal.STRONG:
        return 10.0  # Strong supply chain
    elif signal == SourcingSignal.MODERATE:
        return 5.0  # Moderate supply chain
    else:  # WEAK or UNKNOWN
        return 0.0  # No bonus


def _calculate_unmet_need_adjustment(unmet_need_count: int) -> float:
    """
    Calculate unmet need adjustment.

    More unmet needs = higher priority for drug discovery

    Returns:
        Adjustment in points (0 to +10)
        - 0-2 needs: 0 points
        - 3-4 needs: +5 points
        - 5+ needs: +10 points
    """
    if unmet_need_count >= 5:
        return 10.0
    elif unmet_need_count >= 3:
        return 5.0
    else:
        return 0.0


# ═════════════════════════════════════════════════════════════════════
# DEPRECATED: Old scoring function (kept for backward compatibility)
# ═════════════════════════════════════════════════════════════════════

def score_candidate(
    candidate: Candidate,
    trials_output: TrialsOutput,
    patent_output: Optional[PatentOutput],
    exim_output: Optional[EximOutput],
    unmet_need_count: int = 0,
    max_trials_threshold: int = 50,
    strict_fto: bool = False
) -> Tuple[ScoredCandidate, bool]:
    """
    DEPRECATED: Old scoring function.

    This function is no longer used by rank_candidates() but kept for
    backward compatibility with external callers.

    Use rank_candidates() instead for consistent scoring.
    """
    logger.warning(
        "score_candidate() is deprecated. Use rank_candidates() instead."
    )

    # Base evidence score (from KG)
    evidence_score = candidate.score / 100.0  # Normalize to 0-1

    # Trial competition penalty
    candidate_trial_count = len(
        trials_output.candidate_trials.get(candidate.name, [])
    )

    # Normalize: 0 trials = 0.0 (no penalty), 50+ trials = 1.0 (max penalty)
    if candidate_trial_count >= max_trials_threshold:
        trial_penalty = 1.0
    else:
        trial_penalty = candidate_trial_count / max_trials_threshold

    # Patent risk penalty
    patent_penalty = 0.0
    if patent_output:
        if patent_output.risk_tier == PatentRiskTier.HIGH:
            patent_penalty = 0.35
            if strict_fto:
                return None, False  # Kill switch
        elif patent_output.risk_tier == PatentRiskTier.MEDIUM:
            patent_penalty = 0.15
        elif patent_output.risk_tier == PatentRiskTier.LOW:
            patent_penalty = 0.0

    # Supply bonus
    supply_bonus = 0.0
    if exim_output:
        if exim_output.sourcing_signal == SourcingSignal.STRONG:
            supply_bonus = 0.1
        elif exim_output.sourcing_signal == SourcingSignal.MODERATE:
            supply_bonus = 0.05

    # Unmet need bonus
    unmet_need_bonus = 0.0
    if unmet_need_count > 0:
        unmet_need_bonus = min(0.1, unmet_need_count * 0.02)

    # Final score calculation (0-1 scale)
    final_score = (
        evidence_score
        - (trial_penalty * 0.25)
        - (patent_penalty * 0.25)
        + supply_bonus * 0.15
        + unmet_need_bonus * 0.10
    )

    # Clamp to [0, 1]
    final_score = max(0.0, min(1.0, final_score))

    rationale = []
    rationale.append(f"Evidence score: {evidence_score:.2f}")
    if candidate_trial_count > 0:
        rationale.append(
            f"Trial competition penalty: {trial_penalty:.2f} "
            f"({candidate_trial_count} active trials)"
        )
    if patent_output and patent_output.risk_tier != PatentRiskTier.UNKNOWN:
        rationale.append(f"Patent risk ({patent_output.risk_tier}): {patent_penalty:.2f}")
    if supply_bonus > 0:
        rationale.append(f"Supply/EXIM bonus: +{supply_bonus:.2f}")
    if unmet_need_bonus > 0:
        rationale.append(f"Unmet need bonus: +{unmet_need_bonus:.2f}")

    scored = ScoredCandidate(
        candidate=candidate,
        final_score=final_score * 100,  # Convert back to 0-100 for display
        evidence_score=evidence_score,
        trial_penalty=trial_penalty,
        patent_penalty=patent_penalty,
        supply_bonus=supply_bonus,
        unmet_need_bonus=unmet_need_bonus,
        rationale=rationale
    )

    return scored, True