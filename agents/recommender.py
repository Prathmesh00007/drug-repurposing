"""Recommender agent: Score and rank candidates with enhanced evidence weighting."""

import logging
from typing import List, Optional
from backend.app.schemas import (
    Candidate, FinalRecommendation, ScoredCandidate,
    LiteratureOutput, TrialsOutput, PatentOutput
)

logger = logging.getLogger(__name__)


async def run_recommender(
    candidates: List[Candidate],
    literature: Optional[LiteratureOutput] = None,
    trials: Optional[TrialsOutput] = None,
    patents: Optional[PatentOutput] = None
) -> FinalRecommendation:
    """
    Score and rank candidates based on multi-source evidence.
    
    FIXES:
    - Flaw #5: Match on targets[] array, not single target field
    - Flaw #6: More patents = better validation (inverted scoring)
    - Flaw #7: Match trials by target, not drug name
    """
    logger.info(f"Recommender: Scoring {len(candidates)} candidates")
    
    if not candidates:
        return FinalRecommendation(
            ranked_candidates=[],
            rationale="No candidates available for ranking.",
            next_actions=["Expand search criteria", "Consider alternative therapeutic approaches"]
        )
    
    scored_candidates = []
    
    for candidate in candidates:
        # FLAW #5 FIX: Literature score based on targets[] array
        lit_score = 0.0
        if literature and literature.suggested_targets:
            # Match on ALL targets, not just one
            matched_targets = set(candidate.targets) & set(literature.suggested_targets)
            if matched_targets:
                # Proportional scoring: more matched targets = higher score
                match_ratio = len(matched_targets) / max(len(candidate.targets), 1)
                lit_score = 30.0 * match_ratio
                logger.debug(f"{candidate.name}: Matched targets {matched_targets} -> {lit_score:.1f} pts")
            else:
                # Fallback: partial string matching
                for lit_target in literature.suggested_targets:
                    if any(lit_target.lower() in t.lower() for t in candidate.targets):
                        lit_score = 15.0  # Partial credit
                        break
        
        # FLAW #7 FIX: Clinical trials score based on TARGET presence
        clinical_score = 0.0
        if trials and trials.recent_trials:
            # Match on candidate targets, not candidate name
            trial_matches = 0
            for trial in trials.recent_trials:
                title_lower = trial.get('title', '').lower()
                # Check if any target appears in trial title
                if any(target.lower() in title_lower for target in candidate.targets):
                    trial_matches += 1
            
            if trial_matches > 0:
                clinical_score = min(trial_matches * 15.0, 30.0)
                logger.debug(f"{candidate.name}: {trial_matches} trial matches -> {clinical_score:.1f} pts")
        
        # FLAW #6 FIX: Patent score REWARDS validation (more patents = better)
        patent_score = 10.0  # Default if no patent data
        if patents:
            total_patents = patents.total_patents
            if total_patents == 0:
                # No patents = unknown/risky territory
                patent_score = 5.0
            elif total_patents <= 5:
                # Few patents = early-stage target
                patent_score = 12.0
            elif total_patents <= 20:
                # Moderate patents = validated but not crowded
                patent_score = 18.0
            else:
                # Many patents = well-validated target (good for repurposing)
                patent_score = 20.0
            
            logger.debug(f"{candidate.name}: {total_patents} patents -> {patent_score:.1f} pts")
        
        # KG score (baseline connectivity)
        kg_score = 15.0 if candidate.targets else 5.0
        
        # DRKG score (embedding-based novelty)
        drkg_score = (candidate.drkg_score * 25) if candidate.drkg_score else 0.0
        
        # Final score
        final_score = lit_score + clinical_score + patent_score + kg_score + drkg_score
        
        scored_candidates.append(ScoredCandidate(
            candidate=candidate,
            final_score=final_score,
            literature_score=lit_score,
            clinical_score=clinical_score,
            patent_score=patent_score,
            kg_score=kg_score,
            drkg_score=drkg_score
        ))
    
    # Sort by final score
    scored_candidates.sort(key=lambda x: x.final_score, reverse=True)
    
    # Generate rationale
    top_candidate = scored_candidates[0] if scored_candidates else None
    rationale = ""
    if top_candidate:
        rationale = (
            f"{top_candidate.candidate.name} is recommended based on "
            f"comprehensive evidence (score: {top_candidate.final_score:.1f}/100). "
            f"It has literature support ({top_candidate.literature_score:.0f} pts), "
            f"clinical validation ({top_candidate.clinical_score:.0f} pts), "
            f"and validated target ({top_candidate.patent_score:.0f} pts)."
        )
    
    # Next actions
    next_actions = [
        "Conduct detailed mechanism validation studies",
        "Review existing clinical trial data",
        "Assess patent landscape and freedom-to-operate",
        "Evaluate toxicity and safety profiles",
        "Perform pharmacokinetic/pharmacodynamic analysis",
        "Engage with regulatory consultants",
        "Initiate preclinical validation experiments"
    ]
    
    return FinalRecommendation(
        ranked_candidates=scored_candidates[:10],
        rationale=rationale,
        next_actions=next_actions[:7],
        confidence_level="High" if top_candidate and top_candidate.final_score > 60 else "Medium"
    )
