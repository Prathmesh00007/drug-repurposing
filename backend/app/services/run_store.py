"""Persistent storage for run metadata and state."""
import json
import logging
from pathlib import Path
from datetime import datetime
from backend.app.schemas import RouteAState, RunStatus
from backend.app.config import get_settings

logger = logging.getLogger(__name__)


class RunStore:
    """Manages run metadata and state persistence."""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_dir = Path("/tmp/runs/.data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_run_dir(self, run_id: str) -> Path:
        """Get directory for a specific run."""
        run_dir = self.data_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def create_run(self, run_id: str, indication: str, geography: str, status: RunStatus):
        """Create initial run metadata."""
        run_dir = self._get_run_dir(run_id)
        
        metadata = {
            "run_id": run_id,
            "indication": indication,
            "geography": geography,
            "status": status.value,
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "report_path": None,
            "error_message": None
        }
        
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created run {run_id} in {run_dir}")
    
    def update_metadata(self, run_id: str, **kwargs):
        """Update run metadata."""
        run_dir = self._get_run_dir(run_id)
        metadata_path = run_dir / "metadata.json"
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Handle datetime objects
        for key, value in kwargs.items():
            if isinstance(value, datetime):
                metadata[key] = value.isoformat()
            elif isinstance(value, RunStatus):
                metadata[key] = value.value
            else:
                metadata[key] = value
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.warning(f"Updated run {run_id} metadata: {kwargs.keys()}")
    
    def get_metadata(self, run_id: str) -> dict:
        """Get run metadata."""
        run_dir = self._get_run_dir(run_id)
        metadata_path = run_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def save_state(self, run_id: str, state: RouteAState):
        """Save full run state."""
        run_dir = self._get_run_dir(run_id)
        state_path = run_dir / "state.json"
        
        with open(state_path, "w") as f:
            json.dump(state.dict(), f, indent=2, default=str)
        
        logger.warning(f"Saved state for run {run_id}")
    
    def load_state(self, run_id: str) -> RouteAState:
        """Load full run state."""
        run_dir = self._get_run_dir(run_id)
        state_path = run_dir / "state.json"
        
        if not state_path.exists():
            return None
        
        with open(state_path, "r") as f:
            data = json.load(f)
        
        # Handle datetime strings
        for key in ["created_at", "started_at", "completed_at"]:
            if key in data and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        
        return RouteAState(**data)
    
    def save_report(self, run_id: str, report_bytes: bytes, filename: str = "report.pptx"):
        """Save generated report."""
        run_dir = self._get_run_dir(run_id)
        report_path = run_dir / filename
        
        with open(report_path, "wb") as f:
            f.write(report_bytes)
        
        self.update_metadata(run_id, report_path=str(report_path))
        logger.info(f"Saved report for run {run_id}: {report_path}")
        
        return report_path
