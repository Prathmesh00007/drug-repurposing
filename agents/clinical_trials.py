"""Clinical Trials agent: Query ClinicalTrials.gov API v2."""
import logging
import httpx
from typing import List, Dict, Any, Optional
from backend.app.schemas import TrialsOutput, TrialInfo
from agents.base import cache_manager
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.app.config import get_settings
from kg.semantic_router import DiseaseContext

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _query_clinicaltrials_api(
    condition: str = None,
    intervention: str = None,
    status: List[str] = None,
    page_size: int = 100
) -> dict:
    """
    Query ClinicalTrials.gov API v2.
    
    Docs: https://clinicaltrials.gov/data-api/api
    """
    settings = get_settings()
    
    # Build query
    query_parts = []
    if condition:
        query_parts.append(f"AREA[ConditionSearch]{condition}")
    if intervention:
        query_parts.append(f"AREA[InterventionSearch]{intervention}")
    
    query_string = " AND ".join(query_parts) if query_parts else ""
    
    # Build filter
    filter_parts = []
    if status:
        # API v2 uses specific status values
        status_filter = ",".join(status)
        filter_parts.append(f"AREA[OverallStatus]{status_filter}")
    
    filter_string = " AND ".join(filter_parts) if filter_parts else None
    
    params = {
        "format": "json",
        "pageSize": page_size
    }
    
    if query_string:
        params["query.cond"] = condition
    
    if filter_string:
        params["filter.overallStatus"] = ",".join(status) if status else None
    
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        response = await client.get(
            f"{settings.clinicaltrials_base_url}/studies",
            params=params
        )
        response.raise_for_status()
        return response.json()


async def run_clinical_trials(
    disease_name: str,
    candidate_names: List[str],
    geography: str = None,
    disease_context: Optional[any] = None
) -> TrialsOutput:
    """
    Query clinical trials for disease and candidates.
    """

    search_term = disease_name
    if disease_context and hasattr(disease_context, 'corrected_name'):
        search_term = disease_context.corrected_name
        logger.info(f"💡 Using corrected disease name: '{disease_name}' → '{search_term}'")

    logger.info(f"Clinical Trials Agent: {disease_name}")
    
    cache_key = {"disease": disease_name}
    cached = cache_manager.get("clinical_trials", cache_key)
    if cached:
        return TrialsOutput(**cached)
    
    try:
        # Query for disease trials (active/recruiting)
        trials_data = await _query_clinicaltrials_api(
            condition=search_term,
            status=["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"],
            page_size=200
        )
        
        studies = trials_data.get("studies", [])
        
        total_trials = len(studies)
        phase_breakdown = {}
        sponsor_counts = {}
        candidate_trials = {name: [] for name in candidate_names}
        crowding_flags = []
        
        for study in studies:
            protocol = study.get("protocolSection", {})
            
            # Identification
            id_module = protocol.get("identificationModule", {})
            nct_id = id_module.get("nctId", "")
            title = id_module.get("officialTitle", id_module.get("briefTitle", ""))
            
            # Status
            status_module = protocol.get("statusModule", {})
            status = status_module.get("overallStatus", "Unknown")
            
            # Sponsor
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            lead_sponsor = sponsor_module.get("leadSponsor", {})
            sponsor = lead_sponsor.get("name", "Unknown")
            
            # Design/Phase
            design_module = protocol.get("designModule", {})
            phases = design_module.get("phases", [])
            
            # Interventions
            arms_module = protocol.get("armsInterventionsModule", {})
            interventions = arms_module.get("interventions", [])
            intervention_names = [
                interv.get("name", "").lower() 
                for interv in interventions
            ]
            
            # Extract phase
            phase = None
            if phases:
                phase_str = phases[0].replace("PHASE", "").replace("_", "").strip()
                phase = phase_str
                phase_breakdown[phase] = phase_breakdown.get(phase, 0) + 1
            
            # Count sponsors
            sponsor_counts[sponsor] = sponsor_counts.get(sponsor, 0) + 1
            
            # Check if candidate is mentioned
            for candidate_name in candidate_names:
                candidate_lower = candidate_name.lower()
                
                # Check in title or interventions
                if (candidate_lower in title.lower() or 
                    any(candidate_lower in interv_name for interv_name in intervention_names)):
                    
                    candidate_trials[candidate_name].append(TrialInfo(
                        nct_id=nct_id,
                        phase=phase,
                        status=status,
                        sponsor=sponsor,
                        url=f"https://clinicaltrials.gov/study/{nct_id}"
                    ))
        
        # Get top sponsors
        top_sponsors = sorted(
            sponsor_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        top_sponsors = [s[0] for s in top_sponsors]
        
        # Flag if too many trials (competition)
        if total_trials > 50:
            crowding_flags.append({
                "disease": disease_name,
                "flag": "high_competition",
                "trial_count": total_trials
            })
        
        output = TrialsOutput(
            total_trials=total_trials,
            phase_breakdown=phase_breakdown,
            top_sponsors=top_sponsors,
            candidate_trials=candidate_trials,
            crowding_flags=crowding_flags
        )
        
        logger.info(f"Found {total_trials} clinical trials for {disease_name}")
        
        # Cache
        cache_manager.set("clinical_trials", cache_key, output.dict())
        
        return output
        
    except Exception as e:
        logger.error(f"ClinicalTrials.gov query failed: {e}")
        # Return empty output on failure
        return TrialsOutput(
            total_trials=0,
            candidate_trials={name: [] for name in candidate_names}
        )
