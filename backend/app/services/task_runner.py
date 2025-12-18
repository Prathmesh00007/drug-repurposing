"""Background task execution for Route A workflow."""
import logging
from datetime import datetime
from backend.app.schemas import RouteARequest, RouteAState, RunStatus
from backend.app.services.run_store import RunStore
from orchestrator.route_a_graph import create_route_a_graph

logger = logging.getLogger(__name__)


async def run_route_a_workflow(
    run_id: str,
    request: RouteARequest,
    run_store: RunStore
):
    """
    Execute the full Route A workflow.
    
    This runs in the background and updates the run store as it progresses.
    """
    logger.info(f"DEBUG: Entering workflow for {run_id}")
    try:
        logger.info(f"Starting Route A workflow for run {run_id}")
        
        # Update status to running
        run_store.update_metadata(
            run_id,
            status=RunStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        # Create initial state
        state = RouteAState(
            run_id=run_id,
            indication=request.indication,
            geography=request.geography,
            min_phase=request.min_phase,
            oral_only=request.oral_only,
            exclude_biologics=request.exclude_biologics,
            strict_fto=request.strict_fto,
            status=RunStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        # Create and execute graph
        graph = create_route_a_graph()
        
        # Run the graph workflow
        output_state = await graph.ainvoke(state.dict())
        
        # Persist report bytes (if generated) to disk so it can be served to frontend
        report_bytes = output_state.pop("report_bytes", None)
        if report_bytes:
            # The orchestrator generates a PDF report
            run_store.save_report(run_id, report_bytes, filename="report.pdf")
            # Keep a report_path in state for completeness (served via metadata)
            output_state["report_path"] = str(run_store.get_metadata(run_id).get("report_path"))
        
        # Convert back to state object
        state = RouteAState(**output_state)
        state.completed_at = datetime.utcnow()
        state.status = RunStatus.SUCCEEDED
        
        # Save final state
        run_store.save_state(run_id, state)
        
        # Update metadata
        run_store.update_metadata(
            run_id,
            status=RunStatus.SUCCEEDED,
            completed_at=datetime.utcnow()
        )
        
        logger.info(f"Completed Route A workflow for run {run_id}")
        
    except Exception as e:
        logger.exception(f"Error in Route A workflow for run {run_id}: {str(e)}")
        
        run_store.update_metadata(
            run_id,
            status=RunStatus.FAILED,
            error_message=str(e),
            completed_at=datetime.utcnow()
        )
