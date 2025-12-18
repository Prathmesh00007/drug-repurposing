"""Pydantic schemas for Route A with full backward compatibility."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from enum import Enum

from enum import Enum

class CandidateStage(str, Enum):
    """Drug development stage."""

    APPROVED = "approved"
    CLINICAL = "clinical"
    PRECLINICAL = "preclinical"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.UNKNOWN

        v = str(value).lower().strip()

        if v in {"approved", "phase_4", "4"}:
            return cls.APPROVED

        if v in {"clinical", "phase_1", "phase_2", "phase_3", "1", "2", "3"}:
            return cls.CLINICAL

        if v in {"preclinical", "phase_0", "0"}:
            return cls.PRECLINICAL

        return cls.UNKNOWN


class PatentRiskTier(str, Enum):
    """Patent landscape risk tier."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"

class SourcingSignal(str, Enum):
    """Supply/sourcing availability signal."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    UNKNOWN = "unknown"

class RunStatus(str, Enum):
    """Run execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class RouteARequest(BaseModel):
    """User input for Route A."""
    indication: str = Field(..., description="Disease/condition name")
    geography: str = Field(..., description="Country/region (e.g., 'US', 'India', 'EU5')")
    min_phase: Optional[int] = Field(
        None,
        description="Minimum clinical trial phase (1, 2, 3, 4)",
        ge=1, le=4
    )
    oral_only: bool = Field(False, description="Filter for oral formulations only")
    exclude_biologics: bool = Field(False, description="Exclude biologic drugs")
    strict_fto: bool = Field(False, description="Strict FTO: drop high-risk patents")

    @validator("indication")
    def indication_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Indication cannot be empty")
        return v.strip()

    @validator("geography")
    def geography_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Geography cannot be empty")
        return v.strip()

class RouteARunResponse(BaseModel):
    """Response from creating a Route A run."""
    run_id: str = Field(..., description="Unique run identifier")
    status: RunStatus = Field(default=RunStatus.QUEUED)
    created_at: datetime
    message: str = "Run queued successfully"

class Citation(BaseModel):
    """Evidence citation."""
    url: str
    title: Optional[str] = None
    source: Optional[str] = None  # "PubMed", "PatentsView", etc.

class SOCDetail(BaseModel):
    """Detailed Standard of Care information."""
    drug_name: str
    line_of_therapy: str = "Unknown"  # "First-Line", "Second-Line", "Adjuvant"
    source_document: str = ""  # URL or title of guideline
    approval_status: Optional[str] = None  # "FDA Approved", "EMA Approved"

class UnmetNeedDetail(BaseModel):
    """Detailed Unmet Need information."""
    description: str  # The summary of the unmet need
    category: str = "General"  # "Efficacy", "Safety", "Subgroup", "Access"
    source_quote: str = ""  # The exact quote from source
    severity: str = "Medium"  # "High", "Medium", "Low"

class WebIntelOutput(BaseModel):
    """Enhanced Web Intelligence Output."""
    standard_of_care: List[SOCDetail] = []
    unmet_needs: List[UnmetNeedDetail] = []
    key_market_players: List[str] = []
    estimated_market_size: Optional[str] = None
    regulatory_landscape: Optional[str] = None
    citations: List[Citation] = []
    # Legacy compatibility
    soc: List[str] = []  # Simple list for backward compatibility
    unmet_need_points: List[str] = []  # Simple list for backward compatibility
    keywords: Dict[str, List[str]] = Field(default_factory=lambda: {"targets": [], "phenotypes": []})

class MechanismClaim(BaseModel):
    """A claimed mechanism or target association."""
    claim: str
    sources: List[Citation] = Field(default_factory=list)

class TargetEvidence(BaseModel):
    """Evidence linking a target to disease."""
    target_name: str
    confidence_score: str  # "High", "Medium", "Low"
    supporting_evidence: str  # Summary from LLM analysis
    source_pmids: List[str] = []
    citation_count: int = 0

class LiteratureOutput(BaseModel):
    """Enhanced Literature Output."""
    pathophysiology_summary: str = ""  # Coherent paragraph from LLM
    validated_targets: List[TargetEvidence] = []
    emerging_targets: List[TargetEvidence] = []
    key_review_articles: List[Citation] = []
    # Legacy compatibility
    suggested_targets: List[str] = []
    mechanism_summary: List[str] = []
    citations: List[Citation] = []
    mechanism_claims: List[MechanismClaim] = []
    pubmed_articles: List[Dict[str, Any]] = []

class Candidate(BaseModel):
    """A drug or clinical candidate."""
    name: str
    stage: CandidateStage = CandidateStage.UNKNOWN
    chembl_id: Optional[str] = None
    targets: List[str] = Field(default_factory=list, description="Target symbols")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Evidence score")
    evidence_citations: List[Citation] = Field(default_factory=list)
    notes: Optional[str] = None
    drkg_score: Optional[float] = None  # NEW: DRKG embedding score
    mechanism : List[str] = Field(default_factory=list, description="Candidate Mechanims")
    indication_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Indication Match score")

class KGQueryOutput(BaseModel):
    """Knowledge Graph agent output."""
    candidates: List[Candidate] = Field(default_factory=list)
    top_targets: List[str] = Field(default_factory=list)
    neo4j_run_id: Optional[str] = None
    drkg_hidden_candidates: List[Dict] = []  # NEW: From DRKG discovery
    drkg_validated_candidates: List[Dict] = []  # NEW: DRKG validation

class TrialInfo(BaseModel):
    """Info about a clinical trial."""
    nct_id: str
    phase: Optional[str] = None
    status: str = "Unknown"
    sponsor: Optional[str] = None
    url: Optional[str] = None

class TrialsOutput(BaseModel):
    """Clinical Trials agent output."""

    total_trials: int = 0
    phase_breakdown: Dict[str, int] = Field(default_factory=dict)
    top_sponsors: List[str] = Field(default_factory=list)
    candidate_trials: Dict[str, List[TrialInfo]] = Field(default_factory=dict)
    crowding_flags: List[Dict[str, Any]] = Field(default_factory=list)

    # Backward / report compatibility
    trials_by_phase: Dict[str, int] = Field(default_factory=dict)
    top_recruiters: List[str] = Field(default_factory=list)

    # NEW: Clinical outcome tracking
    trial_outcomes_by_candidate: Dict[str, List[TrialInfo]] = Field(default_factory=dict)


class PatentHit(BaseModel):
    """A patent search result."""
    patent_id: str
    title: str
    date: Optional[str] = None
    assignee: Optional[str] = None
    url: Optional[str] = None

class PatentOutput(BaseModel):
    """Patent Landscape agent output."""
    candidate: str
    risk_tier: PatentRiskTier = PatentRiskTier.UNKNOWN
    top_assignees: List[str] = Field(default_factory=list)
    key_patents: List[PatentHit] = Field(default_factory=list)
    notes: Optional[str] = None

class EximOutput(BaseModel):
    """EXIM Trends agent output."""
    candidate: str
    sourcing_signal: SourcingSignal = SourcingSignal.UNKNOWN
    top_partner_countries: List[str] = Field(default_factory=list)
    dependency_flags: List[str] = Field(default_factory=list)
    proxy_cogs_usd: Optional[float] = None
    notes: Optional[str] = None

class ScoredCandidate(BaseModel):
    """A candidate with final ranking score and rationale."""
    candidate: Candidate
    final_score: float = Field(ge=0.0, le=100.0)  # Changed to 100 scale
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    trial_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    patent_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    supply_bonus: float = Field(default=0.0, ge=0.0, le=0.1)
    unmet_need_bonus: float = Field(default=0.0, ge=0.0, le=0.1)
    rationale: List[str] = Field(default_factory=list)
    # NEW: For report generator
    literature_score: Optional[float] = 0.0
    clinical_score: Optional[float] = 0.0
    patent_score: Optional[float] = 0.0

class FinalRecommendation(BaseModel):
    """Final recommendation from rank_and_select."""
    ranked_candidates: List[ScoredCandidate] = Field(
        default_factory=list,
        description="Top N candidates ranked by final_score"
    )
    alternate_candidates: List[ScoredCandidate] = Field(
        default_factory=list,
        description="2 backup candidates"
    )
    next_actions: List[str] = Field(default_factory=list)
    rationale: str = ""  # NEW: For report generator
    confidence_level: str = "Medium"  # NEW

# ============================================================================
# STATE SCHEMA (SHARED ACROSS ALL AGENTS)
# ============================================================================

class RouteAState(BaseModel):
    """Shared state for LangGraph Route A orchestrator."""
    # User input
    run_id: str
    indication: str
    geography: str
    min_phase: Optional[int] = None
    oral_only: bool = False
    exclude_biologics: bool = False
    strict_fto: bool = False

    # Normalized
    disease_id: Optional[str] = None
    disease_synonyms: List[str] = Field(default_factory=list)

    # Agent outputs
    web_intel_output: Optional[WebIntelOutput] = None
    literature_output: Optional[LiteratureOutput] = None
    kg_output: Optional[KGQueryOutput] = None
    trials_output: Optional[TrialsOutput] = None
    patent_outputs: Dict[str, PatentOutput] = Field(default_factory=dict)
    exim_outputs: Dict[str, EximOutput] = Field(default_factory=dict)

    # Final ranking
    recommendation: Optional[FinalRecommendation] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RunStatus = RunStatus.QUEUED
    error_message: Optional[str] = None

    # Reporting
    report_path: Optional[str] = None
    report_url: Optional[str] = None

class RunMetadata(BaseModel):
    """Metadata for a completed run."""
    run_id: str
    indication: str
    geography: str
    status: RunStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    report_path: Optional[str] = None
    full_state_path: Optional[str] = None

class RunStatusResponse(BaseModel):
    """Response for GET /route-a/run/{run_id}."""
    run_id: str
    status: RunStatus
    indication: str
    geography: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Partial results (if available)
    web_intel_summary: Optional[str] = None
    candidates_found: int = 0
    trials_count: int = 0

    # Links
    report_url: Optional[str] = None
    error_message: Optional[str] = None
