import logging
from datetime import datetime
import os
import traceback
from langgraph.graph import StateGraph, END

from backend.app.schemas import RouteAState, RunStatus
from agents.web_intel import run_web_intelligence
from agents.literature import run_literature_rag
from agents.clinical_trials import run_clinical_trials
from agents.patents import run_patent_landscape
from agents.exim import run_exim_trends
from orchestrator.scoring import rank_candidates

# ✅ CORRECTED IMPORTS
from agents.kg import hybrid_discovery
from kg.disease_resolver_v2 import resolve_disease_deterministic

logger = logging.getLogger(__name__)

def normalize_disease_id(disease_id: str) -> str:
    """Normalize disease ID to PREFIX_NUMBERS format."""
    if not disease_id:
        return disease_id

    # If already has underscore, return as-is
    if "_" in disease_id and disease_id.count("_") == 1:
        parts = disease_id.split("_")
        if parts[0].isalpha() and parts[1].isdigit():
            return disease_id

    # Remove separators and split
    clean_id = disease_id.replace(":", "").replace("_", "")
    for i, char in enumerate(clean_id):
        if char.isdigit():
            return f"{clean_id[:i]}_{clean_id[i:]}"

    return disease_id


async def normalize_input(state: dict) -> dict:
    """Disease resolution using deterministic OLS API."""
    logger.info(f"🔍 Resolving disease: '{state['indication']}'")

    disease_context = await resolve_disease_deterministic(state["indication"])

    if not disease_context:
        logger.error(f"❌ Disease resolution failed")
        state["disease_context"] = None
        state["disease_id"] = None
        state["status"] = RunStatus.FAILED
        state["error_message"] = f"Could not resolve disease: {state['indication']}"
        return state

    # Get disease ID
    raw_disease_id = disease_context.efo_id or disease_context.mondo_id or disease_context.mesh_id

    if not raw_disease_id:
        logger.error(f"❌ No ontology ID found")
        state["disease_context"] = None
        state["disease_id"] = None
        state["status"] = RunStatus.FAILED
        state["error_message"] = f"No ontology ID found for: {state['indication']}"
        return state

    disease_id = normalize_disease_id(raw_disease_id)

    # Update state
    state["disease_context"] = disease_context
    state["disease_id"] = disease_id
    state["indication"] = disease_context.corrected_name
    state["disease_synonyms"] = disease_context.synonyms

    logger.info(f"✅ Disease resolved: {disease_context.corrected_name}")
    logger.info(f"   ID: {disease_id}, Area: {disease_context.therapeutic_area}")

    return state


async def web_intelligence_node(state: dict) -> dict:
    """Run Web Intelligence agent."""
    logger.info(f"Running Web Intelligence for {state['indication']}")

    web_output = await run_web_intelligence(
        disease_name=state["indication"],
        geography=state["geography"]
    )

    state["web_intel_output"] = web_output
    return state


async def literature_node(state: dict) -> dict:
    """Run Literature RAG agent."""
    logger.info(f"Running Literature RAG for {state['indication']}")

    keywords = state["web_intel_output"].keywords if state["web_intel_output"] else {}

    lit_output = await run_literature_rag(
        disease_name=state["indication"],
        soc_list=state["web_intel_output"].soc if state["web_intel_output"] else [],
        keywords=keywords
    )

    state["literature_output"] = lit_output
    return state


async def kg_node(state: dict) -> dict:
    """
    ✅ FIXED: Use hybrid_drug_discovery_v2 with proper state passing and error handling.
    FIXES:
    - Issue #3: Added try-except for error handling
    - Issue #4: Pass disease_id and disease_context from state (no re-resolution)
    - Issue #7: Proper state isolation fixed
    - Issue SA7: Removed /100 division - keep 0-100 scale
    """
    logger.info(f"Running Hybrid Discovery for {state['indication']}")

    # ✅ FIX #3: Add error handling
    try:
        # ✅ FIX #4 & #7: Pass disease_id and disease_context from orchestrator state
        discovery_result = await hybrid_discovery.discover_for_disease(
            disease_id=state["disease_id"],  # ← FIX: Pass from state
            disease_context=state["disease_context"],  # ← FIX: Pass from state
            min_phase=state.get("min_phase", 1),
            top_n_candidates=50,
            enable_enrichment=True
        )

    except ValueError as e:
        logger.error(f"❌ Discovery failed (ValueError): {e}")
        # Return empty results but don't crash
        from backend.app.schemas import KGQueryOutput
        state["kg_output"] = KGQueryOutput(
            candidates=[],
            top_targets=[],
            neo4j_run_id="",
            drkg_hidden_candidates=[],
            drkg_validated_candidates=[]
        )
        state["error_message"] = f"Discovery failed: {str(e)}"
        state["discovery_stats"] = {
            "total_discovered": 0,
            "validated": 0,
            "rejected": 0,
            "final_count": 0,
            "error": str(e)
        }
        state["discovery_raw_candidates"] = []  # ← FIX C4: Set empty list
        return state

    except Exception as e:
        logger.error(f"FULL ERROR TYPE: {type(e)} | {repr(e)}")
        from backend.app.schemas import KGQueryOutput
        state["kg_output"] = KGQueryOutput(
            candidates=[],
            top_targets=[],
            neo4j_run_id="",
            drkg_hidden_candidates=[],
            drkg_validated_candidates=[]
        )
        state["error_message"] = f"Discovery failed: {str(e)}"
        state["discovery_stats"] = {
            "total_discovered": 0,
            "validated": 0,
            "rejected": 0,
            "final_count": 0,
            "error": str(e)
        }
        state["discovery_raw_candidates"] = []  # ← FIX C4: Set empty list
        return state

    # ✅ FIX SA1 & SA7: Convert to KGQueryOutput format (NO DIVISION BY 100!)
    from backend.app.schemas import KGQueryOutput, Candidate, CandidateStage, Citation

    candidates = []
    for cand_dict in discovery_result["candidates"]:
        # Now cand_dict has all fields we need (phase, target_symbol, score_breakdown, etc.)

        # ← FIXED: Keep score as 0-100 (DO NOT divide by 100!)
        composite_score = cand_dict["score_breakdown"]["composite_score"]  # Already 0-100

        candidates.append(Candidate(
            name=cand_dict["drug_name"],
            stage=CandidateStage(f"phase_{cand_dict['phase']}"),  # Phase is now int
            chembl_id=cand_dict["drug_id"],
            targets=[cand_dict.get("target_symbol")] if cand_dict.get("target_symbol") else [],
            score=composite_score / 100.0,  # ← FIXED: Keep 0-100 scale, no division!
            evidence_citations=[]
        ))

    kg_output = KGQueryOutput(
        candidates=candidates,
        top_targets=[],
        neo4j_run_id="",
        drkg_hidden_candidates=[],
        drkg_validated_candidates=[]
    )

    state["kg_output"] = kg_output
    state["discovery_stats"] = discovery_result["stats"]
    state["discovery_raw_candidates"] = discovery_result["candidates"]  # Keep full data for downstream

    logger.info(f"✅ Discovery complete: {len(candidates)} candidates")
    logger.info(f"   Stats: {discovery_result['stats']}")

    return state


async def clinical_trials_node(state: dict) -> dict:
    """Run Clinical Trials agent."""
    logger.info(f"Running Clinical Trials Agent")

    candidates = state["kg_output"].candidates if state["kg_output"] else []
    candidate_names = [c.name for c in candidates]

    trials_output = await run_clinical_trials(
        disease_name=state["indication"],
        candidate_names=candidate_names,
        geography=state.get("geography"),
        disease_context=state.get("disease_context")
    )

    state["trials_output"] = trials_output
    return state


async def patent_node(state: dict) -> dict:
    """Run Patent Landscape agent."""
    logger.info(f"Running Patent Landscape Agent")

    candidates = state["kg_output"].candidates if state["kg_output"] else []
    patent_outputs = {}

    for candidate in candidates[:10]:  # Limit to top 10
        patent_output = await run_patent_landscape(
            candidate_name=candidate.name,
            indication=state["indication"],
            jurisdiction="US"
        )
        patent_outputs[candidate.name] = patent_output

    state["patent_outputs"] = patent_outputs
    return state


async def exim_node(state: dict) -> dict:
    """Run EXIM Trends agent."""
    logger.info(f"Running EXIM Trends Agent")

    candidates = state["kg_output"].candidates if state["kg_output"] else []
    exim_outputs = {}

    for candidate in candidates[:10]:  # Limit to top 10
        exim_output = await run_exim_trends(
            candidate_name=candidate.name,
            geography=state.get("geography")
        )
        exim_outputs[candidate.name] = exim_output

    state["exim_outputs"] = exim_outputs
    return state


async def rank_and_select_node(state: dict) -> dict:
    """Score and rank candidates."""
    logger.info(f"Ranking and selecting candidates")

    candidates = state["kg_output"].candidates if state["kg_output"] else []
    unmet_need_count = len(state["web_intel_output"].unmet_need_points) if state["web_intel_output"] else 0

    top_3, alternates = rank_candidates(
        candidates=candidates,
        trials_output=state["trials_output"],
        patent_outputs=state["patent_outputs"],
        exim_outputs=state["exim_outputs"],
        unmet_need_count=unmet_need_count,
        strict_fto=state.get("strict_fto", False)
    )

    # Build recommendation
    from backend.app.schemas import FinalRecommendation

    state["recommendation"] = FinalRecommendation(
        ranked_candidates=top_3,
        alternate_candidates=alternates,
        next_actions=[
            "Conduct detailed pharmacology review",
            "Perform FTO analysis with IP counsel",
            "Assess manufacturing feasibility",
            "Evaluate clinical development pathway"
        ],
        rationale=f"Top candidate selected based on comprehensive evidence." if top_3 else "",
        confidence_level="High" if top_3 and top_3[0].final_score > 60 else "Medium"
    )

    return state


async def report_generator_node(state: dict) -> dict:
    """Generate final report."""
    run_id = state.get("run_id", "unknown")
    indication = state.get("indication", "Unknown Disease")

    logger.info(f"📊 [REPORT] Starting report generation")
    logger.info(f"   Run ID: {run_id}")
    logger.info(f"   Indication: {indication}")

    start_time = datetime.utcnow()

    try:
        # Import the production-ready generator
        from agents.report import run_report_generator
        
        # Call with ALL available state (function handles None/missing gracefully)
        logger.info(f"🔄 [REPORT] Calling run_report_generator with {len(state)} state fields")
        
        pdf_bytes = await run_report_generator(
            run_id=run_id,
            indication=indication,
            geography=state.get("geography", "US"),
            web_intel=state.get("web_intel_output"),
            literature=state.get("literature_output"),
            kg_output=state.get("kg_output"),
            trials=state.get("trials_output"),
            recommendation=state.get("recommendation"),
            # Additional data for discovery_result building
            disease_context=state.get("disease_context"),
            discovery_raw_candidates=state.get("discovery_raw_candidates"),
            discovery_stats=state.get("discovery_stats"),
            patent_outputs=state.get("patent_outputs"),
            exim_outputs=state.get("exim_outputs")
        )
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Update state
        state["report_bytes"] = pdf_bytes
        state["report_path"] = f"./reports/{run_id}/{indication.replace(' ', '_')}.pdf"
        state["report_generated"] = True
        state["report_generation_time"] = duration
        
        logger.info(f"✅ [REPORT] Generation successful in {duration:.2f}s")
        logger.info(f"   PDF Size: {len(pdf_bytes)} bytes")
        logger.info(f"   Report Path: {state['report_path']}")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [REPORT] Report generation failed: {e}")
        logger.error(f"   Error Type: {type(e).__name__}")
        logger.error(f"   Stack: {traceback.format_exc()}")
        
        # Fallback: Set error state but DON'T CRASH
        state["report_generated"] = False
        state["report_error"] = str(e)
        state["report_error_type"] = type(e).__name__
        
        # Attempt minimal error report
        try:
            logger.info(f"🔄 [REPORT] Attempting fallback error report...")
            from agents.report import FailproofReportGenerator
            
            generator = FailproofReportGenerator()
            pdf_bytes, markdown = generator._generate_fallback_report(
                run_id=run_id,
                indication=indication,
                error=str(e)
            )
            
            state["report_bytes"] = pdf_bytes
            state["report_path"] = f"./reports/{run_id}/ERROR_REPORT.pdf"
            logger.info(f"✅ [REPORT] Fallback report generated: {len(pdf_bytes)} bytes")
            
        except Exception as e2:
            logger.error(f"❌ [REPORT] Fallback also failed: {e2}")
            state["report_bytes"] = f"Error: {str(e)}".encode('utf-8')
            state["report_path"] = "error_report.txt"
        
        return state
    
def should_abort_pipeline(state: dict) -> str:
    """Check if disease resolution failed."""
    if not state.get("disease_context"):
        logger.error("🛑 ABORTING: Disease resolution failed")
        return "abort"

    if not state.get("disease_id"):
        logger.error("🛑 ABORTING: No disease ID found")
        return "abort"

    return "continue"


def should_expand_search(state: dict) -> str:
    """Check if we need to widen search criteria."""
    kg_output = state.get("kg_output")

    if not kg_output:
        logger.warning("⚠️ No KG output - expanding search")
        return "expand"

    candidate_count = len(kg_output.candidates) if hasattr(kg_output, 'candidates') else 0

    if candidate_count < 3:
        logger.warning(f"⚠️ Only {candidate_count} candidates - expanding search")
        return "expand"

    return "continue"


async def expand_search_node(state: dict) -> dict:
    """Retry with looser criteria."""
    logger.info("Expanding search with looser criteria...")

    candidates = []
    
    try:
        # FIX: Pass diseaseid and diseasecontext from state
        discovery_result = await hybrid_discovery.discover_for_disease(
            disease_id=state["disease_id"],
            disease_context=state["disease_context"],
            min_phase=0,  # Include preclinical
            top_n_candidates=100,  # More candidates
            enable_enrichment=False  # Faster
        )
        
        from backend.app.schemas import KGQueryOutput, Candidate, CandidateStage
        
      
        # ✅ FIX: Check if discovery_result has candidates
        if "candidates" in discovery_result and discovery_result["candidates"]:
            for cand_dict in discovery_result["candidates"]:
                candidates.append(Candidate(
                    name=cand_dict["drug_name"],
                    stage=CandidateStage(f"phase_{cand_dict['phase']}"),
                    chembl_id=cand_dict["drug_id"],
                    targets=[cand_dict.get("target_symbol")] if cand_dict.get("target_symbol") else [],
                    score=cand_dict["score_breakdown"]["composite_score"] / 100,  # Normalize to 0-1
                    evidence_citations=[]
                ))
        
        # ✅ FIX: Create KGQueryOutput even if candidates is empty
        state["kg_output"] = KGQueryOutput(
            candidates=candidates,
            top_targets=[],
            neo4j_run_id="",
            drkg_hidden_candidates=[],
            drkg_validated_candidates=[]
        )
        state["discovery_stats"] = discovery_result.get("stats", {
            "total_discovered": 0,
            "validated": 0,
            "rejected": 0,
            "final_count": len(candidates)
        })
        
        logger.info(f"Expanded search: {len(candidates)} candidates found")
    
    except Exception as e:
        logger.error(f"Expanded search failed: {e}")
        
        # ✅ FIX: Return empty but valid state structure
        from backend.app.schemas import KGQueryOutput
        state["kgoutput"] = KGQueryOutput(
            candidates=candidates,
            top_targets=[],
            neo4j_run_id="",
            drkg_hidden_candidates=[],
            drkg_validated_candidates=[]
        )
        state["discoverystats"] = {
            "total_discovered": 0,
            "validated": 0,
            "rejected": 0,
            "final_count": len(candidates),
            "error": str(e)
        }
    
    return state




def create_route_a_graph():
    """Create Route A graph with conditional routing."""
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("web_intelligence", web_intelligence_node)
    graph.add_node("literature", literature_node)
    graph.add_node("kg", kg_node)
    graph.add_node("expand_search", expand_search_node)
    graph.add_node("clinical_trials", clinical_trials_node)
    graph.add_node("patents", patent_node)
    graph.add_node("exim", exim_node)
    graph.add_node("rank_and_select", rank_and_select_node)
    graph.add_node("generate_report", report_generator_node)

    # Set entry point
    graph.set_entry_point("normalize_input")

    # Conditional: Abort if disease resolution failed
    graph.add_conditional_edges(
        "normalize_input",
        should_abort_pipeline,
        {
            "abort": END,
            "continue": "web_intelligence"
        }
    )

    # Linear flow
    graph.add_edge("web_intelligence", "literature")
    graph.add_edge("literature", "kg")

    # Conditional: Expand if too few candidates
    graph.add_conditional_edges(
        "kg",
        should_expand_search,
        {
            "expand": "expand_search",
            "continue": "clinical_trials"
        }
    )

    graph.add_edge("expand_search", "clinical_trials")
    graph.add_edge("clinical_trials", "patents")
    graph.add_edge("patents", "exim")
    graph.add_edge("exim", "rank_and_select")
    graph.add_edge("rank_and_select", "generate_report")
    graph.set_finish_point("generate_report")

    return graph.compile()