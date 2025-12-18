"""
Clinical Trial Outcome Parser
Uses ClinicalTrials.gov API (FREE) to fetch trial status and outcomes.

Distinguishes between:
- Successful trials (positive evidence)
- Failed trials (negative evidence)
- Terminated trials (negative evidence)
"""

import httpx
import logging
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TrialStatus(str, Enum):
    COMPLETED_POSITIVE = "completed_positive"
    COMPLETED_NEGATIVE = "completed_negative"
    COMPLETED_UNKNOWN = "completed_unknown"
    ACTIVE = "active"
    RECRUITING = "recruiting"
    TERMINATED = "terminated"
    WITHDRAWN = "withdrawn"
    SUSPENDED = "suspended"
    UNKNOWN = "unknown"


class TrialPhase(str, Enum):
    PHASE_4 = "Phase 4"
    PHASE_3 = "Phase 3"
    PHASE_2 = "Phase 2"
    PHASE_1 = "Phase 1"
    EARLY_PHASE_1 = "Early Phase 1"
    NOT_APPLICABLE = "Not Applicable"
    UNKNOWN = "Unknown"


@dataclass
class TrialOutcome:
    nct_id: str
    phase: TrialPhase
    status: TrialStatus
    title: str
    completion_date: Optional[str]
    enrollment: int
    
    # Outcome analysis
    is_positive_evidence: bool
    evidence_weight: float  # -1.0 to 1.0 (negative to positive)
    
    # Details
    primary_outcome: Optional[str]
    termination_reason: Optional[str]
    has_results: bool
    results_summary: Optional[str]


class ClinicalTrialParser:
    """
    Parses clinical trial data from ClinicalTrials.gov API v2.
    
    Evidence Weight Scoring:
    - Completed + Positive Results: +1.0
    - Completed + Unknown Results: +0.5
    - Active/Recruiting: +0.3
    - Completed + Negative Results: -0.5
    - Terminated (safety): -1.0
    - Terminated (futility): -0.8
    - Withdrawn: -0.3
    """
    
    API_BASE = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self):
        self.timeout = httpx.Timeout(30.0)
    
    async def parse_trial(self, nct_id: str) -> Optional[TrialOutcome]:
        """Parse single trial by NCT ID"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.API_BASE}/studies/{nct_id}",
                    params={"format": "json"}
                )
                
                if response.status_code == 404:
                    logger.warning(f"Trial {nct_id} not found")
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                return self._parse_trial_data(data)
                
            except Exception as e:
                logger.error(f"Failed to fetch trial {nct_id}: {e}")
                return None
    
    async def search_trials_for_drug_disease(
        self,
        drug_name: str,
        disease_name: str,
        max_results: int = 20
    ) -> List[TrialOutcome]:
        """Search trials for drug-disease combination"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Search query
                query = f"{drug_name} AND {disease_name}"
                
                response = await client.get(
                    f"{self.API_BASE}/studies",
                    params={
                        "query.term": query,
                        "pageSize": max_results,
                        "format": "json"
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                trials = []
                if "studies" in data:
                    for study in data["studies"]:
                        trial = self._parse_trial_data(study)
                        if trial:
                            trials.append(trial)
                
                return trials
                
            except Exception as e:
                logger.error(f"Trial search failed: {e}")
                return []
    
    def _parse_trial_data(self, data: Dict) -> Optional[TrialOutcome]:
        """Parse trial data from API response"""
        try:
            protocol = data.get("protocolSection", {})
            
            # Identification
            id_module = protocol.get("identificationModule", {})
            nct_id = id_module.get("nctId", "")
            title = id_module.get("briefTitle", "")
            
            # Status
            status_module = protocol.get("statusModule", {})
            overall_status = status_module.get("overallStatus", "Unknown")
            completion_date = status_module.get("primaryCompletionDate", {}).get("date")
            
            # Design
            design_module = protocol.get("designModule", {})
            phases = design_module.get("phases", ["Unknown"])
            phase_str = phases[0] if phases else "Unknown"
            
            # Enrollment
            design_info = protocol.get("designModule", {}).get("enrollmentInfo", {})
            enrollment = design_info.get("count", 0)
            
            # Results
            has_results = "resultsSection" in data
            results_summary = None
            
            if has_results:
                results_summary = self._extract_results_summary(data["resultsSection"])
            
            # Termination reason (if applicable)
            termination_reason = status_module.get("whyStopped")
            
            # Primary outcome
            outcomes_module = protocol.get("outcomesModule", {})
            primary_outcomes = outcomes_module.get("primaryOutcomes", [])
            primary_outcome = primary_outcomes[0].get("measure") if primary_outcomes else None
            
            # Determine trial status and evidence weight
            trial_status, evidence_weight, is_positive = self._determine_trial_status(
                overall_status=overall_status,
                has_results=has_results,
                results_summary=results_summary,
                termination_reason=termination_reason
            )
            
            # Map phase
            phase = self._map_phase(phase_str)
            
            return TrialOutcome(
                nct_id=nct_id,
                phase=phase,
                status=trial_status,
                title=title,
                completion_date=completion_date,
                enrollment=enrollment,
                is_positive_evidence=is_positive,
                evidence_weight=evidence_weight,
                primary_outcome=primary_outcome,
                termination_reason=termination_reason,
                has_results=has_results,
                results_summary=results_summary
            )
            
        except Exception as e:
            logger.error(f"Failed to parse trial data: {e}")
            return None
    
    def _determine_trial_status(
        self,
        overall_status: str,
        has_results: bool,
        results_summary: Optional[str],
        termination_reason: Optional[str]
    ) -> tuple[TrialStatus, float, bool]:
        """
        Determine trial status and evidence weight.
        
        Returns:
            (TrialStatus, evidence_weight, is_positive_evidence)
        """
        status_lower = overall_status.lower()
        
        # COMPLETED
        if status_lower == "completed":
            if has_results and results_summary:
                # Try to parse positive/negative from summary
                if any(word in results_summary.lower() for word in ["significant", "improved", "effective", "benefit"]):
                    return (TrialStatus.COMPLETED_POSITIVE, 1.0, True)
                elif any(word in results_summary.lower() for word in ["no significant", "not effective", "no benefit", "failed"]):
                    return (TrialStatus.COMPLETED_NEGATIVE, -0.5, False)
            # Completed but no clear outcome
            return (TrialStatus.COMPLETED_UNKNOWN, 0.5, True)
        
        # ACTIVE / RECRUITING
        elif status_lower in ["active, not recruiting", "enrolling by invitation"]:
            return (TrialStatus.ACTIVE, 0.3, True)
        elif status_lower == "recruiting":
            return (TrialStatus.RECRUITING, 0.3, True)
        
        # TERMINATED
        elif status_lower == "terminated":
            if termination_reason:
                reason_lower = termination_reason.lower()
                if any(word in reason_lower for word in ["safety", "adverse", "toxicity"]):
                    return (TrialStatus.TERMINATED, -1.0, False)  # Strong negative
                elif any(word in reason_lower for word in ["futility", "ineffective", "lack of efficacy"]):
                    return (TrialStatus.TERMINATED, -0.8, False)
            return (TrialStatus.TERMINATED, -0.5, False)
        
        # WITHDRAWN
        elif status_lower == "withdrawn":
            return (TrialStatus.WITHDRAWN, -0.3, False)
        
        # SUSPENDED
        elif status_lower == "suspended":
            return (TrialStatus.SUSPENDED, 0.0, False)
        
        # UNKNOWN
        else:
            return (TrialStatus.UNKNOWN, 0.0, False)
    
    def _extract_results_summary(self, results_section: Dict) -> Optional[str]:
        """Extract summary from results section"""
        try:
            # Try to get primary outcome results
            outcome_measures = results_section.get("outcomeMeasuresModule", {})
            primary_measures = outcome_measures.get("outcomeMeasures", [])
            
            if primary_measures:
                first_measure = primary_measures[0]
                # Get description or title
                description = first_measure.get("description", "")
                title = first_measure.get("title", "")
                return f"{title}: {description}"[:500]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract results summary: {e}")
            return None
    
    def _map_phase(self, phase_str: str) -> TrialPhase:
        """Map phase string to enum"""
        phase_lower = phase_str.lower()
        
        if "phase 4" in phase_lower or "phase iv" in phase_lower:
            return TrialPhase.PHASE_4
        elif "phase 3" in phase_lower or "phase iii" in phase_lower:
            return TrialPhase.PHASE_3
        elif "phase 2" in phase_lower or "phase ii" in phase_lower:
            return TrialPhase.PHASE_2
        elif "phase 1" in phase_lower or "phase i" in phase_lower:
            if "early" in phase_lower:
                return TrialPhase.EARLY_PHASE_1
            return TrialPhase.PHASE_1
        elif "not applicable" in phase_lower or "n/a" in phase_lower:
            return TrialPhase.NOT_APPLICABLE
        else:
            return TrialPhase.UNKNOWN


# Singleton
trial_parser = ClinicalTrialParser()
