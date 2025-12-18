"""
Unified Evidence Validation Layer.

Replaces 3 separate validators (pathway_mechanism_validator, moa_validator, safety_validator)
with a single, efficient validation layer that balances precision and recall.

Validation criteria:
1. Score threshold (Open Targets association score)
2. Evidence diversity (multiple data sources)
3. Pathway overlap (optional, lenient)
4. Literature support (optional)
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationDecision(Enum):
    """Validation decision outcomes."""
    KEEP = "keep"
    REJECT = "reject"
    REVIEW = "review"  # Flag for manual review


@dataclass
class ValidationResult:
    """Result of evidence validation."""
    decision: ValidationDecision
    confidence: float  # 0.0 to 1.0
    reasoning: str
    evidence_scores: Dict[str, float]
    flags: List[str] = None
    
    def __post_init__(self):
        if self.flags is None:
            self.flags = []


class EvidenceValidator:
    """
    Unified validator that combines multiple evidence types.
    
    Philosophy:
    - Simple is better: Use clear, interpretable rules
    - Lenient by default: Err on the side of including candidates
    - Transparent: Always explain decisions
    - Fast: <100ms per candidate
    """
    
    def __init__(
        self,
        min_score: float = 0.2,  # Lower than reference (0.3) for leniency
        min_evidence_sources: int = 1,  # At least 1 data source
        enable_pathway_check: bool = False,  # Optional, disabled by default
        enable_literature_check: bool = False  # Optional, disabled by default
    ):
        self.min_score = min_score
        self.min_evidence_sources = min_evidence_sources
        self.enable_pathway_check = enable_pathway_check
        self.enable_literature_check = enable_literature_check
    
    def validate_target(
        self,
        target_symbol: str,
        opentargets_score: float,
        evidence_count: int,
        pathway_overlap: Optional[float] = None,
        literature_count: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate a target using multiple evidence types.
        
        Args:
            target_symbol: Gene symbol
            opentargets_score: Open Targets association score (0-1)
            evidence_count: Number of evidence sources
            pathway_overlap: Jaccard similarity of pathways (0-1)
            literature_count: Number of supporting publications
            
        Returns:
            ValidationResult with decision and reasoning
        """
        evidence_scores = {
            "opentargets_score": opentargets_score,
            "evidence_diversity": evidence_count,
        }
        
        flags = []
        
        # =====================================================================
        # RULE 1: Score threshold (PRIMARY FILTER)
        # =====================================================================
        if opentargets_score < self.min_score:
            return ValidationResult(
                decision=ValidationDecision.REJECT,
                confidence=0.9,
                reasoning=f"Open Targets score ({opentargets_score:.3f}) below threshold ({self.min_score})",
                evidence_scores=evidence_scores,
                flags=["low_score"]
            )
        
        # =====================================================================
        # RULE 2: Evidence diversity
        # =====================================================================
        if evidence_count < self.min_evidence_sources:
            flags.append("single_source")
        
        # =====================================================================
        # RULE 3: Pathway overlap (OPTIONAL, LENIENT)
        # =====================================================================
        if self.enable_pathway_check and pathway_overlap is not None:
            evidence_scores["pathway_overlap"] = pathway_overlap
            
            # Very lenient threshold (0.05 = 5% overlap)
            if pathway_overlap < 0.05:
                flags.append("low_pathway_overlap")
                # Don't reject, just flag
        
        # =====================================================================
        # RULE 4: Literature support (OPTIONAL)
        # =====================================================================
        if self.enable_literature_check and literature_count is not None:
            evidence_scores["literature_count"] = literature_count
            
            if literature_count == 0:
                flags.append("no_literature")
                # Don't reject, just flag
        
        # =====================================================================
        # DECISION: Calculate confidence and decide
        # =====================================================================
        # Base confidence from score
        confidence = min(opentargets_score * 1.2, 1.0)
        
        # Boost for multi-source evidence
        if evidence_count >= 3:
            confidence = min(confidence + 0.1, 1.0)
        
        # Boost for pathway overlap
        if pathway_overlap and pathway_overlap > 0.1:
            confidence = min(confidence + 0.1, 1.0)
        
        # Flag for review if low confidence
        if confidence < 0.5:
            decision = ValidationDecision.REVIEW
            reasoning = f"Target {target_symbol} passes filters but has low confidence ({confidence:.2f})"
        else:
            decision = ValidationDecision.KEEP
            reasoning = f"Target {target_symbol} validated with confidence {confidence:.2f}"
        
        return ValidationResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence_scores=evidence_scores,
            flags=flags
        )
    
    def validate_drug(
        self,
        drug_name: str,
        phase: int,
        has_clinical_evidence: bool,
        mechanism_known: bool = True,
        safety_flags: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate a drug candidate.
        
        Args:
            drug_name: Drug name
            phase: Clinical trial phase (0-4)
            has_clinical_evidence: Has trial data for this disease
            mechanism_known: Mechanism of action is known
            safety_flags: List of safety concerns (if any)
            
        Returns:
            ValidationResult
        """
        evidence_scores = {
            "phase": phase,
            "clinical_evidence": 1.0 if has_clinical_evidence else 0.0,
            "mechanism_known": 1.0 if mechanism_known else 0.5
        }
        
        flags = safety_flags or []
        
        # =====================================================================
        # RULE 1: Phase threshold (very lenient)
        # =====================================================================
        if phase < 1 and not has_clinical_evidence:
            return ValidationResult(
                decision=ValidationDecision.REJECT,
                confidence=0.9,
                reasoning=f"Drug {drug_name} is preclinical with no clinical evidence",
                evidence_scores=evidence_scores,
                flags=["preclinical", "no_evidence"]
            )
        
        # =====================================================================
        # RULE 2: Clinical evidence (strongly preferred but not required)
        # =====================================================================
        if not has_clinical_evidence:
            flags.append("no_clinical_evidence")
        
        # =====================================================================
        # RULE 3: Known mechanism (preferred but not required)
        # =====================================================================
        if not mechanism_known:
            flags.append("unknown_mechanism")
        
        # =====================================================================
        # RULE 4: Safety concerns (flag but don't reject)
        # =====================================================================
        if safety_flags:
            # Don't reject based on safety at discovery stage
            # This should be evaluated later in the pipeline
            pass
        
        # =====================================================================
        # DECISION: Calculate confidence
        # =====================================================================
        confidence = 0.5  # Base confidence
        
        # Boost for phase
        confidence += phase * 0.1
        
        # Boost for clinical evidence
        if has_clinical_evidence:
            confidence += 0.2
        
        # Boost for known mechanism
        if mechanism_known:
            confidence += 0.1
        
        # Cap at 1.0
        confidence = min(confidence, 1.0)
        
        # Decide
        if confidence < 0.3:
            decision = ValidationDecision.REJECT
            reasoning = f"Drug {drug_name} has insufficient evidence (confidence {confidence:.2f})"
        elif confidence < 0.6:
            decision = ValidationDecision.REVIEW
            reasoning = f"Drug {drug_name} flagged for review (confidence {confidence:.2f})"
        else:
            decision = ValidationDecision.KEEP
            reasoning = f"Drug {drug_name} validated with confidence {confidence:.2f}"
        
        return ValidationResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence_scores=evidence_scores,
            flags=flags
        )
    
    def batch_validate_targets(
        self,
        targets: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Validate a batch of targets.
        
        Args:
            targets: List of target dictionaries with keys:
                - symbol: str
                - opentargets_score: float
                - evidence_count: int
                - pathway_overlap: Optional[float]
                - literature_count: Optional[int]
        
        Returns:
            Tuple of (kept, rejected, review) target lists
        """
        kept = []
        rejected = []
        review = []
        
        for target in targets:
            result = self.validate_target(
                target_symbol=target.get("symbol", ""),
                opentargets_score=target.get("opentargets_score", 0.0),
                evidence_count=target.get("evidence_count", 0),
                pathway_overlap=target.get("pathway_overlap"),
                literature_count=target.get("literature_count")
            )
            
            # Add validation result to target
            target["validation_result"] = result
            
            # Categorize
            if result.decision == ValidationDecision.KEEP:
                kept.append(target)
            elif result.decision == ValidationDecision.REJECT:
                rejected.append(target)
            else:
                review.append(target)
        
        logger.info(
            f"Batch validation: {len(kept)} kept, {len(rejected)} rejected, "
            f"{len(review)} for review"
        )
        
        return kept, rejected, review


# Example usage
if __name__ == "__main__":
    validator = EvidenceValidator(
        min_score=0.2,
        min_evidence_sources=1,
        enable_pathway_check=False
    )
    
    # Test target validation
    result = validator.validate_target(
        target_symbol="TP53",
        opentargets_score=0.75,
        evidence_count=5,
        pathway_overlap=0.15,
        literature_count=1000
    )
    
    print(f"\n{'='*60}")
    print("EVIDENCE VALIDATION TEST")
    print(f"{'='*60}")
    print(f"Decision: {result.decision.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Evidence scores: {result.evidence_scores}")
    print(f"Flags: {result.flags}")
