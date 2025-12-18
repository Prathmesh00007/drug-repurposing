"""Routes for Route A."""
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.app.schemas import RouteARequest, RouteARunResponse, RunStatusResponse, RunStatus
from backend.app.services.run_store import RunStore
from backend.app.services.task_runner import run_route_a_workflow
from backend.app.config import get_settings
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()
run_store = RunStore()


@router.post("/route-a/run", response_model=RouteARunResponse)
async def create_route_a_run(request: RouteARequest, background_tasks: BackgroundTasks):
    """
    Create and start a Route A analysis run.
    
    Returns immediately with a run_id; execution happens in background.
    """
    settings = get_settings()
    run_id = str(uuid.uuid4())
    
    logger.info(f"Creating Route A run {run_id}: {request.indication} in {request.geography}")
    
    # Store initial metadata
    run_store.create_run(
        run_id=run_id,
        indication=request.indication,
        geography=request.geography,
        status=RunStatus.QUEUED
    )
    
    # Queue background task
    background_tasks.add_task(
        run_route_a_workflow,
        run_id=run_id,
        request=request,
        run_store=run_store
    )
    
    return RouteARunResponse(
        run_id=run_id,
        status=RunStatus.QUEUED,
        created_at=datetime.utcnow(),
        message=f"Run {run_id} queued for {request.indication}"
    )


@router.get("/route-a/run/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get status and partial results of a Route A run."""
    metadata = run_store.get_metadata(run_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Try to load full state if available
    state = run_store.load_state(run_id)
    
    candidates_found = 0
    trials_count = 0
    web_intel_summary = None
    
    if state:
        if state.kg_output:
            candidates_found = len(state.kg_output.candidates)
        if state.trials_output:
            trials_count = state.trials_output.total_trials
        if state.web_intel_output:
            web_intel_summary = f"{len(state.web_intel_output.soc)} SoC options, {len(state.web_intel_output.unmet_need_points)} unmet needs"
    
    return RunStatusResponse(
        run_id=run_id,
        status=metadata["status"],
        indication=metadata["indication"],
        geography=metadata["geography"],
        created_at=metadata["created_at"],
        started_at=metadata.get("started_at"),
        completed_at=metadata.get("completed_at"),
        candidates_found=candidates_found,
        trials_count=trials_count,
        web_intel_summary=web_intel_summary,
        report_url=f"/api/v1/route-a/run/{run_id}/report" if metadata.get("report_path") else None,
        error_message=metadata.get("error_message")
    )


@router.get("/route-a/run/{run_id}/report")
async def download_report(run_id: str):
    """Download the generated report (PPTX)."""
    metadata = run_store.get_metadata(run_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    if not metadata.get("report_path"):
        raise HTTPException(
            status_code=404, 
            detail=f"Report not ready yet for run {run_id}"
        )
    
    import aiofiles
    from fastapi.responses import FileResponse
    
    report_path = metadata["report_path"]
    return FileResponse(
        report_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=f"route_a_{run_id}.pptx"
    )


@router.get("/route-a/run/{run_id}/state")
async def get_full_state(run_id: str):
    """Get full run state as JSON (for debugging)."""
    metadata = run_store.get_metadata(run_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    state = run_store.load_state(run_id)
    
    if not state:
        raise HTTPException(status_code=404, detail=f"State not found for run {run_id}")
    
    return state.dict()
