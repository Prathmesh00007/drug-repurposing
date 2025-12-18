"""State definition for LangGraph Route A."""
from typing import TypedDict, Optional, List, Dict, Any
from backend.app.schemas import (
    WebIntelOutput, LiteratureOutput, KGQueryOutput,
    TrialsOutput, PatentOutput, EximOutput, FinalRecommendation,
    RunStatus
)
from datetime import datetime


class RouteAStateDict(TypedDict, total=False):
    """State schema for LangGraph workflow."""
    
    # User input
    run_id: str
    indication: str
    geography: str
    min_phase: Optional[int]
    oral_only: bool
    exclude_biologics: bool
    strict_fto: bool
    
    # Normalized
    disease_id: Optional[str]
    disease_synonyms: List[str]
    
    # Agent outputs
    web_intel_output: Optional[WebIntelOutput]
    literature_output: Optional[LiteratureOutput]
    kg_output: Optional[KGQueryOutput]
    trials_output: Optional[TrialsOutput]
    patent_outputs: Dict[str, PatentOutput]
    exim_outputs: Dict[str, EximOutput]
    
    # Final ranking
    recommendation: Optional[FinalRecommendation]
    
    # Metadata
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: RunStatus
    error_message: Optional[str]
    
    # Reporting
    report_path: Optional[str]
    report_url: Optional[str]
