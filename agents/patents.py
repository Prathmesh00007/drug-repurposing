"""Patent Landscape agent: Search via Web Intelligence (DuckDuckGo)."""
import logging
import re
from typing import List
from backend.app.schemas import PatentOutput, PatentRiskTier, PatentHit
from agents.base import cache_manager
from agents.web_intel import _search_duckduckgo
from datetime import datetime

logger = logging.getLogger(__name__)

async def _search_patents_web(candidate_name: str) -> dict:
    """
    Search for patent information using DuckDuckGo.
    """
    # 1. Search for total patent count/landscape
    # Query: "drug_name patent expiry expiration date"
    expiry_query = f"{candidate_name} patent expiry expiration date"
    expiry_results = await _search_duckduckgo(expiry_query, max_results=5)
    
    # 2. Search for recent filings
    # Query: "drug_name patent application 2024 2025"
    recent_query = f"{candidate_name} patent application 2024 2025"
    recent_results = await _search_duckduckgo(recent_query, max_results=5)
    
    return {
        "expiry": expiry_results,
        "recent": recent_results
    }

async def run_patent_landscape(
    candidate_name: str,
    indication: str,
    jurisdiction: str = "US"
) -> PatentOutput:
    """
    Assess patent landscape and FTO risk using web search.
    """
    logger.info(f"Patent Landscape: {candidate_name} for {indication}")
    
    cache_key = {"candidate": candidate_name, "indication": indication}
    cached = cache_manager.get("patents", cache_key)
    if cached:
        return PatentOutput(**cached)
    
    try:
        search_data = await _search_patents_web(candidate_name)
        expiry_hits = search_data.get("expiry", [])
        recent_hits = search_data.get("recent", [])
        
        # Analyze results to infer risk
        risk_tier = PatentRiskTier.LOW
        notes = []
        key_patents = []
        
        # 1. check for "expired" keyword
        is_expired = any("expired" in h.get("snippet", "").lower() for h in expiry_hits)
        if is_expired:
            notes.append("Patents likely expired (low risk)")
            risk_tier = PatentRiskTier.LOW
        else:
            # Check for future dates in snippets
            future_years = [str(y) for y in range(datetime.now().year + 1, 2040)]
            found_future = any(y in h.get("snippet", "") for h in expiry_hits for y in future_years)
            if found_future:
                notes.append("Found future expiration dates (medium/high risk)")
                risk_tier = PatentRiskTier.MEDIUM
        
        # 2. Check for recent activity
        if recent_hits:
            risk_tier = PatentRiskTier.HIGH if risk_tier != PatentRiskTier.LOW else PatentRiskTier.MEDIUM
            notes.append(f"Found {len(recent_hits)} recent patent mentions")
            
            for hit in recent_hits[:3]:
                key_patents.append(PatentHit(
                    patent_id="Web-Result",
                    title=hit.get("title", ""),
                    date=None,
                    assignee="Unknown",
                    url=hit.get("link", "")
                ))

        # Default fallback if nothing found
        if not notes:
            notes.append("No specific patent data found (assuming low/unknown)")
            risk_tier = PatentRiskTier.UNKNOWN

        output = PatentOutput(
            candidate=candidate_name,
            risk_tier=risk_tier,
            top_assignees=[], # Hard to extract reliably from web snippets
            key_patents=key_patents,
            notes="; ".join(notes)
        )
        
        cache_manager.set("patents", cache_key, output.dict())
        return output

    except Exception as e:
        logger.error(f"Patent web search failed: {e}")
        return PatentOutput(
            candidate=candidate_name,
            risk_tier=PatentRiskTier.UNKNOWN,
            notes=f"Search failed: {str(e)}"
        )
"""Patent Landscape agent: Search via Web Intelligence (DuckDuckGo)."""
import logging
import re
from typing import List
from backend.app.schemas import PatentOutput, PatentRiskTier, PatentHit
from agents.base import cache_manager
from agents.web_intel import _search_duckduckgo
from datetime import datetime

logger = logging.getLogger(__name__)

async def _search_patents_web(candidate_name: str) -> dict:
    """
    Search for patent information using DuckDuckGo.
    """
    # 1. Search for total patent count/landscape
    # Query: "drug_name patent expiry expiration date"
    expiry_query = f"{candidate_name} patent expiry expiration date"
    expiry_results = await _search_duckduckgo(expiry_query, max_results=5)
    
    # 2. Search for recent filings
    # Query: "drug_name patent application 2024 2025"
    recent_query = f"{candidate_name} patent application 2024 2025"
    recent_results = await _search_duckduckgo(recent_query, max_results=5)
    
    return {
        "expiry": expiry_results,
        "recent": recent_results
    }

async def run_patent_landscape(
    candidate_name: str,
    indication: str,
    jurisdiction: str = "US"
) -> PatentOutput:
    """
    Assess patent landscape and FTO risk using web search.
    """
    logger.info(f"Patent Landscape: {candidate_name} for {indication}")
    
    cache_key = {"candidate": candidate_name, "indication": indication}
    cached = cache_manager.get("patents", cache_key)
    if cached:
        return PatentOutput(**cached)
    
    try:
        search_data = await _search_patents_web(candidate_name)
        expiry_hits = search_data.get("expiry", [])
        recent_hits = search_data.get("recent", [])
        
        # Analyze results to infer risk
        risk_tier = PatentRiskTier.LOW
        notes = []
        key_patents = []
        
        # 1. check for "expired" keyword
        is_expired = any("expired" in h.get("snippet", "").lower() for h in expiry_hits)
        if is_expired:
            notes.append("Patents likely expired (low risk)")
            risk_tier = PatentRiskTier.LOW
        else:
            # Check for future dates in snippets
            future_years = [str(y) for y in range(datetime.now().year + 1, 2040)]
            found_future = any(y in h.get("snippet", "") for h in expiry_hits for y in future_years)
            if found_future:
                notes.append("Found future expiration dates (medium/high risk)")
                risk_tier = PatentRiskTier.MEDIUM
        
        # 2. Check for recent activity
        if recent_hits:
            risk_tier = PatentRiskTier.HIGH if risk_tier != PatentRiskTier.LOW else PatentRiskTier.MEDIUM
            notes.append(f"Found {len(recent_hits)} recent patent mentions")
            
            for hit in recent_hits[:3]:
                key_patents.append(PatentHit(
                    patent_id="Web-Result",
                    title=hit.get("title", ""),
                    date=None,
                    assignee="Unknown",
                    url=hit.get("link", "")
                ))

        # Default fallback if nothing found
        if not notes:
            notes.append("No specific patent data found (assuming low/unknown)")
            risk_tier = PatentRiskTier.UNKNOWN

        output = PatentOutput(
            candidate=candidate_name,
            risk_tier=risk_tier,
            top_assignees=[], # Hard to extract reliably from web snippets
            key_patents=key_patents,
            notes="; ".join(notes)
        )
        
        cache_manager.set("patents", cache_key, output.dict())
        return output

    except Exception as e:
        logger.error(f"Patent web search failed: {e}")
        return PatentOutput(
            candidate=candidate_name,
            risk_tier=PatentRiskTier.UNKNOWN,
            notes=f"Search failed: {str(e)}"
        )
