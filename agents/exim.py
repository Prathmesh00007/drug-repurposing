"""EXIM Trends agent: Search via Web Intelligence."""
import logging
from backend.app.schemas import EximOutput, SourcingSignal
from agents.base import cache_manager
from agents.web_intel import _search_duckduckgo

logger = logging.getLogger(__name__)

async def run_exim_trends(
    candidate_name: str,
    geography: str
) -> EximOutput:
    """
    Assess EXIM trends and sourcing signals using web search.
    """
    logger.info(f"EXIM Trends: {candidate_name} in {geography}")
    
    cache_key = {"candidate": candidate_name}
    cached = cache_manager.get("exim", cache_key)
    if cached:
        return EximOutput(**cached)
    
    try:
        # Query: "drug_name manufacturers suppliers India China"
        # This helps identify major sourcing hubs
        query = f"{candidate_name} API manufacturers suppliers India China export"
        results = await _search_duckduckgo(query, max_results=7)
        
        # Analyze snippets for country mentions
        countries_found = {}
        target_countries = ["China", "India", "USA", "Europe", "Germany", "Italy"]
        
        for res in results:
            snippet = res.get("snippet", "").lower()
            title = res.get("title", "").lower()
            text = snippet + " " + title
            
            for country in target_countries:
                if country.lower() in text:
                    countries_found[country] = countries_found.get(country, 0) + 1
        
        # Sort top countries
        top_partners = sorted(countries_found.items(), key=lambda x: x[1], reverse=True)
        top_partners_list = [p[0] for p in top_partners[:5]]
        
        # Infer signal
        sourcing_signal = SourcingSignal.WEAK
        notes = []
        
        if len(results) > 0:
            sourcing_signal = SourcingSignal.MODERATE
            if "China" in top_partners_list or "India" in top_partners_list:
                 sourcing_signal = SourcingSignal.STRONG
                 notes.append("Strong presence in major API manufacturing hubs")
        
        if not top_partners_list:
            notes.append("No specific sourcing countries identified")
            sourcing_signal = SourcingSignal.UNKNOWN

        output = EximOutput(
            candidate=candidate_name,
            sourcing_signal=sourcing_signal,
            top_partner_countries=top_partners_list,
            dependency_flags=["Potential reliance on Asian markets"] if "China" in top_partners_list else [],
            proxy_cogs_usd=None,
            notes="; ".join(notes)
        )
        
        cache_manager.set("exim", cache_key, output.dict())
        return output
        
    except Exception as e:
        logger.error(f"EXIM web search failed: {e}")
        return EximOutput(
            candidate=candidate_name,
            sourcing_signal=SourcingSignal.UNKNOWN,
            notes=f"Search failed: {str(e)}"
        )
