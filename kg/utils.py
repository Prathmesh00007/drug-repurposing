import re
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_phase(phase: Union[int, str, None]) -> int:
    """
    Convert any phase format to integer 0-4.

    Handles:
    - Integers: 0-4 (validated and clamped)
    - Strings: "Phase 1", "phase_2", "PHASE-3", "approved", "preclinical"
    - None: defaults to 0

    Args:
        phase: Clinical phase in any format

    Returns:
        Integer phase (0-4)
    """
    if phase is None:
        return 0

    # Already integer
    if isinstance(phase, int):
        return max(0, min(4, phase))

    # String conversions
    phase_str = str(phase).lower().strip()

    # Extract digit from "Phase 2", "phase_3", "PHASE-1", "phase 2"
    match = re.search(r'(\d)', phase_str)
    if match:
        num = int(match.group(1))
        return max(0, min(4, num))

    # Special cases
    if 'preclinical' in phase_str or 'discovery' in phase_str:
        return 0
    if 'approved' in phase_str or 'marketed' in phase_str or 'launch' in phase_str:
        return 4

    # Default to preclinical if unknown
    logger.warning(f"Unknown phase format: {phase}, defaulting to 0")
    return 0


def normalize_drug_id(drug_id: Union[str, None], fallback_id: Optional[str] = None) -> Optional[str]:
    """
    Normalize drug ID, handling multiple ID formats.

    Args:
        drug_id: Primary drug ID (e.g., "CHEMBL123")
        fallback_id: Fallback ID if primary is None

    Returns:
        Normalized drug ID or None
    """
    if drug_id:
        return str(drug_id).strip()
    if fallback_id:
        return str(fallback_id).strip()
    return None