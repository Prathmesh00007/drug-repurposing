import logging
import httpx
import re
import asyncio
import json
import io
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from backend.app.schemas import (
    WebIntelOutput, SOCDetail, UnmetNeedDetail, Citation
)
from agents.base import cache_manager
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import cerebras_llm as llm

logger = logging.getLogger(__name__)

# Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080/search")
GEMINI_API_KEY = ""

# Try to import PDF parsing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyPDF2 not installed, PDF parsing disabled")

# Drug extraction patterns
DRUG_SUFFIXES = [
    "mab", "nib", "gib", "mib", "limus", "statin", "prazole", "vir",
    "tinib", "zumab", "ximab", "gliflozin", "gliptin", "cept", "ciclib"
]

DRUG_BLACKLIST = {
    "placebo", "inhibitor", "receptor", "agonist", "antagonist",
    "treatment", "therapy", "medicine", "available", "online"
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _search_web(query: str, max_results: int = 5) -> List[dict]:
    """Perform web search via SearXNG."""
    params = {
        "q": query,
        "format": "json",
        "engines": "google,bing,wikipedia",
        "language": "en-US",
        "safesearch": 0
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            response = await client.get(SEARXNG_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")
                }
                for r in data.get("results", [])[:max_results]
            ]
    except Exception as e:
        logger.error(f"Search failed for '{query}': {e}")
        return []

async def _scrape_webpage(url: str, max_length: int = 20000) -> str:
    """Download and extract text from webpage or PDF."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return ""

            content_type = resp.headers.get('content-type', '').lower()

            # PDF Handling
            if 'application/pdf' in content_type and PDF_SUPPORT:
                try:
                    pdf_file = io.BytesIO(resp.content)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages[:10]:  # First 10 pages
                        text += page.extract_text() + "\n"
                    return text[:max_length]
                except Exception as e:
                    logger.warning(f"PDF parsing failed for {url}: {e}")
                    return ""

            # HTML Handling
            elif 'text/html' in content_type:
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Remove unwanted elements
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
                return text[:max_length]

    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return ""

async def _analyze_with_llm(prompt: str, text: str, model: str = "gemini-2.5-flash") -> Dict:
    """Use Gemini to analyze text and extract structured data."""
    if not GEMINI_API_KEY:
        logger.warning("No Gemini API key found, skipping LLM analysis")
        return {}

    try:
        full_prompt = (
            "You are a drug repurposing research analyst specializing in mechanistic biology. "
            "Respond only with valid JSON.\n\n"
            f"{prompt}\n\nText:\n{text[:8000]}"
        )

        # Use the cerebras compat wrapper
        response = await llm.generate(full_prompt, temperature=0.3, stream=False)
        raw = response.text if hasattr(response, "text") else str(response)

        # Try direct JSON parsing
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            # Try to extract JSON from text
            m = re.search(r"(\{(?:.|\n)*\})", raw)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    return parsed
                except json.JSONDecodeError:
                    logger.debug("Found braces but failed to parse JSON")
            return {"raw": raw}

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return {}

def _extract_drugs_basic(text: str) -> List[str]:
    """Fallback: Basic regex drug extraction."""
    text = text.lower()
    found = set()
    pattern = r'\b[a-z]{4,}(?:' + '|'.join(DRUG_SUFFIXES) + r')\b'
    words = re.findall(pattern, text)
    for w in words:
        if w not in DRUG_BLACKLIST and len(w) > 5:
            found.add(w.title())
    return list(found)[:15]


async def run_web_intelligence(disease_name: str, geography: str) -> WebIntelOutput:
    """
    🔬 DRUG REPURPOSING INTELLIGENCE AGENT

    Mission: Extract mechanistic, pathway, and cross-indication intelligence
    to fuel repurposing discovery (NOT just current treatments).

    Key Outputs:
    1. Related diseases with approved drugs (indication expansion)
    2. Molecular pathways and therapeutic targets
    3. Off-label use evidence and case studies
    4. Mechanistic hypotheses from literature
    5. Competing repurposing trials
    6. Biomarkers for patient stratification
    """
    logger.info(f"🔬 Repurposing Intel: {disease_name} in {geography}")

    # Check cache
    cache_key = {"disease": disease_name, "geography": geography}
    cached = cache_manager.get("web_intel", cache_key)
    if cached:
        return WebIntelOutput(**cached)

    citations = []
    soc_details = []
    unmet_details = []
    market_players = []

    # NEW: Repurposing-specific data structures
    related_diseases = []  # Similar diseases for cross-indication analysis
    molecular_targets = []  # Key targets for mechanistic search
    pathway_insights = []  # Pathway-level intelligence
    offlabel_evidence = []  # Off-label use cases
    repurposing_trials = []  # Competing repurposing clinical trials

    # =========================================================================
    # PHASE 1: PATHOPHYSIOLOGY & MOLECULAR TARGETS (REPURPOSING FOUNDATION)
    # =========================================================================
    logger.info("Phase 1: Extracting pathophysiology and molecular targets...")

    pathway_queries = [
        f"{disease_name} molecular pathogenesis pathways mechanisms 2024",
        f"{disease_name} therapeutic targets biomarkers",
        f"site:nature.com OR site:science.org {disease_name} pathophysiology review"
    ]

    pathway_texts = []
    for query in pathway_queries:
        results = await _search_web(query, max_results=3)
        for res in results[:2]:
            citations.append(Citation(
                url=res["url"],
                title=res["title"],
                source="Pathway Analysis"
            ))
            page_text = await _scrape_webpage(res["url"])
            if page_text:
                pathway_texts.append(page_text)

    # LLM Analysis: Extract molecular targets and pathways
    if pathway_texts and GEMINI_API_KEY:
        combined_text = "\n\n".join(pathway_texts[:3])
        pathway_prompt = f"""
Analyze this {disease_name} pathophysiology text for drug repurposing insights.

Extract:
1. **Molecular Targets**: Proteins, receptors, enzymes involved in disease
2. **Signaling Pathways**: Key pathways (e.g., JAK/STAT, PI3K/AKT, MAPK)
3. **Biomarkers**: Measurable indicators for patient stratification
4. **Druggable Mechanisms**: Mechanisms that could be targeted by existing drugs

Return JSON:
{{
  "molecular_targets": [
    {{"name": "...", "role": "...", "druggability": "High|Medium|Low"}}
  ],
  "pathways": [
    {{"pathway_name": "...", "relevance": "...", "drugs_known": ["..."]}}
  ],
  "biomarkers": ["..."],
  "mechanistic_insights": ["..."]
}}
"""
        pathway_analysis = await _analyze_with_llm(pathway_prompt, combined_text)

        if pathway_analysis.get("molecular_targets"):
            molecular_targets = pathway_analysis["molecular_targets"]
            logger.info(f"  ✓ Found {len(molecular_targets)} molecular targets")

        if pathway_analysis.get("pathways"):
            pathway_insights = pathway_analysis["pathways"]
            logger.info(f"  ✓ Found {len(pathway_insights)} key pathways")

    # =========================================================================
    # PHASE 2: RELATED DISEASES (CROSS-INDICATION OPPORTUNITIES)
    # =========================================================================
    logger.info("Phase 2: Identifying related diseases for cross-indication analysis...")

    related_disease_queries = [
        f"{disease_name} similar diseases shared pathogenesis",
        f"{disease_name} comorbidities overlapping mechanisms",
        f"{disease_name} disease family therapeutic area"
    ]

    related_texts = []
    for query in related_disease_queries:
        results = await _search_web(query, max_results=3)
        for res in results[:2]:
            citations.append(Citation(
                url=res["url"],
                title=res["title"],
                source="Cross-Indication"
            ))
            page_text = await _scrape_webpage(res["url"])
            if page_text:
                related_texts.append(page_text)

    # LLM Analysis: Find related diseases
    if related_texts and GEMINI_API_KEY:
        combined_related = "\n\n".join(related_texts[:3])
        related_prompt = f"""
Analyze diseases related to {disease_name} for drug repurposing opportunities.

Extract:
1. **Related Diseases**: Diseases with shared mechanisms/pathways
2. **Approved Drugs**: Drugs approved for related diseases that might work for {disease_name}
3. **Shared Mechanisms**: Common biological pathways

Return JSON:
{{
  "related_diseases": [
    {{
      "disease_name": "...",
      "relationship": "shared pathway|comorbidity|same family",
      "approved_drugs": ["..."],
      "shared_mechanisms": ["..."]
    }}
  ]
}}
"""
        related_analysis = await _analyze_with_llm(related_prompt, combined_related)

        if related_analysis.get("related_diseases"):
            related_diseases = related_analysis["related_diseases"]
            logger.info(f"  ✓ Found {len(related_diseases)} related diseases")

            # Extract drugs from related diseases as repurposing candidates
            for rel_disease in related_diseases[:5]:
                for drug in rel_disease.get("approved_drugs", [])[:3]:
                    soc_details.append(SOCDetail(
                        drug_name=drug,
                        line_of_therapy="Repurposing Candidate",
                        source_document=f"Approved for {rel_disease['disease_name']}",
                        approval_status=f"Cross-indication from {rel_disease['disease_name']}"
                    ))

    # =========================================================================
    # PHASE 3: OFF-LABEL USE & CASE STUDIES (REAL-WORLD EVIDENCE)
    # =========================================================================
    logger.info("Phase 3: Searching for off-label use evidence...")

    offlabel_queries = [
        f"{disease_name} off-label drug use case reports",
        f"{disease_name} repurposed drugs clinical experience",
        f'"{disease_name}" "off-label" OR "compassionate use"'
    ]

    offlabel_texts = []
    for query in offlabel_queries:
        results = await _search_web(query, max_results=3)
        for res in results[:2]:
            citations.append(Citation(
                url=res["url"],
                title=res["title"],
                source="Off-Label Evidence"
            ))
            page_text = await _scrape_webpage(res["url"])
            if page_text:
                offlabel_texts.append(page_text)

    # LLM Analysis: Extract off-label evidence
    if offlabel_texts and GEMINI_API_KEY:
        combined_offlabel = "\n\n".join(offlabel_texts[:3])
        offlabel_prompt = f"""
Analyze off-label drug use for {disease_name}.

Extract:
1. **Drugs**: Drugs used off-label with evidence of efficacy
2. **Clinical Context**: Patient populations, dosing, outcomes
3. **Evidence Strength**: Case report, case series, retrospective study, etc.

Return JSON:
{{
  "offlabel_drugs": [
    {{
      "drug_name": "...",
      "original_indication": "...",
      "evidence_type": "...",
      "outcome_summary": "...",
      "citation": "..."
    }}
  ]
}}
"""
        offlabel_analysis = await _analyze_with_llm(offlabel_prompt, combined_offlabel)

        if offlabel_analysis.get("offlabel_drugs"):
            offlabel_evidence = offlabel_analysis["offlabel_drugs"]
            logger.info(f"  ✓ Found {len(offlabel_evidence)} off-label use cases")

            # Add to SOC list with special annotation
            for evidence in offlabel_evidence[:5]:
                soc_details.append(SOCDetail(
                    drug_name=evidence.get("drug_name", "Unknown"),
                    line_of_therapy="Off-Label/Repurposing",
                    source_document=evidence.get("citation", "Case study"),
                    approval_status=f"Off-label from {evidence.get('original_indication', 'Unknown')}"
                ))

    # =========================================================================
    # PHASE 4: REPURPOSING CLINICAL TRIALS (COMPETITIVE INTELLIGENCE)
    # =========================================================================
    logger.info("Phase 4: Searching for repurposing clinical trials...")

    trial_queries = [
        f'site:clinicaltrials.gov "{disease_name}" drug repurposing',
        f"{disease_name} clinical trials repositioning 2023 2024 2025",
        f"{disease_name} phase 2 phase 3 trials"
    ]

    trial_texts = []
    for query in trial_queries:
        results = await _search_web(query, max_results=3)
        for res in results[:2]:
            citations.append(Citation(
                url=res["url"],
                title=res["title"],
                source="Clinical Trials"
            ))
            page_text = await _scrape_webpage(res["url"])
            if page_text:
                trial_texts.append(page_text)

    # LLM Analysis: Extract trial information
    if trial_texts and GEMINI_API_KEY:
        combined_trials = "\n\n".join(trial_texts[:3])
        trial_prompt = f"""
Analyze clinical trial information for {disease_name}.

Focus on:
1. Drug repurposing trials (drugs approved for other indications)
2. Novel mechanism trials
3. Trial phase, status, sponsor

Return JSON:
{{
  "repurposing_trials": [
    {{
      "drug_name": "...",
      "phase": "...",
      "status": "...",
      "sponsor": "...",
      "mechanism": "...",
      "nct_id": "..."
    }}
  ]
}}
"""
        trial_analysis = await _analyze_with_llm(trial_prompt, combined_trials)

        if trial_analysis.get("repurposing_trials"):
            repurposing_trials = trial_analysis["repurposing_trials"]
            logger.info(f"  ✓ Found {len(repurposing_trials)} repurposing trials")

    # =========================================================================
    # PHASE 5: STANDARD OF CARE (BASELINE FOR COMPARISON)
    # =========================================================================
    logger.info("Phase 5: Establishing standard of care baseline...")

    soc_queries = [
        f"{disease_name} treatment guidelines {geography} 2024 2025",
        f"{disease_name} first-line therapy FDA approved"
    ]

    soc_texts = []
    for query in soc_queries:
        results = await _search_web(query, max_results=2)
        for res in results[:2]:
            citations.append(Citation(
                url=res["url"],
                title=res["title"],
                source="Guidelines"
            ))
            page_text = await _scrape_webpage(res["url"])
            if page_text:
                soc_texts.append(page_text)

    # LLM Analysis of SOC (brief, since this is less critical for repurposing)
    if soc_texts and GEMINI_API_KEY:
        combined_soc = "\n\n".join(soc_texts[:2])
        soc_prompt = f"""
Extract ONLY the current standard of care drugs for {disease_name}.
Be concise - we need this as a baseline, not the main focus.

Return JSON:
{{
  "current_drugs": [
    {{"drug_name": "...", "line_of_therapy": "First-Line|Second-Line"}}
  ]
}}
"""
        soc_analysis = await _analyze_with_llm(soc_prompt, combined_soc)

        if soc_analysis.get("current_drugs"):
            for drug in soc_analysis["current_drugs"][:5]:
                # Only add if not already in list
                if not any(s.drug_name == drug.get("drug_name") for s in soc_details):
                    soc_details.append(SOCDetail(
                        drug_name=drug.get("drug_name", "Unknown"),
                        line_of_therapy=drug.get("line_of_therapy", "Unknown"),
                        source_document="Current SOC",
                        approval_status="FDA Approved"
                    ))

    # =========================================================================
    # PHASE 6: UNMET NEEDS (FOCUS ON REPURPOSING OPPORTUNITIES)
    # =========================================================================
    logger.info("Phase 6: Identifying unmet needs as repurposing opportunities...")

    unmet_queries = [
        f"{disease_name} unmet medical needs treatment gaps 2024",
        f"{disease_name} treatment resistance refractory patients",
        f"{disease_name} subpopulations poor outcomes"
    ]

    unmet_texts = []
    for query in unmet_queries:
        results = await _search_web(query, max_results=3)
        for res in results[:2]:
            citations.append(Citation(
                url=res["url"],
                title=res["title"],
                source="Unmet Needs"
            ))
            page_text = await _scrape_webpage(res["url"])
            if page_text:
                unmet_texts.append(page_text)

    # LLM Analysis of Unmet Needs (with repurposing angle)
    if unmet_texts and GEMINI_API_KEY:
        combined_unmet = "\n\n".join(unmet_texts[:3])
        unmet_prompt = f"""
Analyze unmet medical needs for {disease_name} from a drug repurposing perspective.

Focus on:
1. Patient subgroups with inadequate treatment options
2. Mechanisms not addressed by current therapies
3. Treatment resistance mechanisms
4. Repurposing opportunities these gaps create

Return JSON:
{{
  "unmet_needs": [
    {{
      "description": "...",
      "category": "Efficacy|Safety|Access|Subgroup|Mechanism",
      "repurposing_opportunity": "...",
      "potential_mechanisms": ["..."],
      "severity": "High|Medium|Low"
    }}
  ]
}}
"""
        unmet_analysis = await _analyze_with_llm(unmet_prompt, combined_unmet)

        if unmet_analysis.get("unmet_needs"):
            for need in unmet_analysis["unmet_needs"][:10]:
                unmet_details.append(UnmetNeedDetail(
                    description=need.get("description", ""),
                    category=need.get("category", "General"),
                    source_quote=need.get("repurposing_opportunity", "")[:300],
                    severity=need.get("severity", "Medium")
                ))

    # =========================================================================
    # PHASE 7: COMPETITIVE LANDSCAPE (MARKET INTELLIGENCE)
    # =========================================================================
    logger.info("Phase 7: Analyzing competitive landscape...")

    market_query = f"{disease_name} pharmaceutical companies pipeline drugs"
    market_results = await _search_web(market_query, max_results=2)

    for res in market_results:
        text = res["snippet"]
        # Extract company names
        companies = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Ltd|Corp|Pharma|Therapeutics|Biosciences)',
            text
        )
        market_players.extend(companies[:5])

    # =========================================================================
    # OUTPUT: Repurposing-Enhanced Intelligence
    # =========================================================================

    # Build enhanced output with repurposing context
    output = WebIntelOutput(
        standard_of_care=soc_details[:15],  # Increased to include cross-indication drugs
        unmet_needs=unmet_details[:10],
        key_market_players=list(set(market_players))[:5],
        citations=citations[:20],  # More citations for deeper analysis

        # Legacy fields for backward compatibility
        soc=[s.drug_name for s in soc_details],
        unmet_need_points=[u.description for u in unmet_details],

        # NEW: Repurposing-specific intelligence (stored in keywords for now)
        keywords={
            "molecular_targets": [t.get("name", "") for t in molecular_targets[:10]],
            "pathways": [p.get("pathway_name", "") for p in pathway_insights[:10]],
            "related_diseases": [d.get("disease_name", "") for d in related_diseases[:5]],
            "offlabel_candidates": [e.get("drug_name", "") for e in offlabel_evidence[:10]],
            "repurposing_trials": [t.get("drug_name", "") for t in repurposing_trials[:10]],
            "biomarkers": pathway_analysis.get("biomarkers", [])[:5] if 'pathway_analysis' in locals() else []
        }
    )

    # Cache
    cache_manager.set("web_intel", cache_key, output.dict())

    logger.info(f"✅ Repurposing Intel Complete:")
    logger.info(f"   • {len(soc_details)} drugs (SOC + cross-indication + off-label)")
    logger.info(f"   • {len(unmet_details)} unmet needs")
    logger.info(f"   • {len(molecular_targets)} molecular targets")
    logger.info(f"   • {len(pathway_insights)} pathways")
    logger.info(f"   • {len(related_diseases)} related diseases")
    logger.info(f"   • {len(offlabel_evidence)} off-label cases")
    logger.info(f"   • {len(repurposing_trials)} repurposing trials")

    return output

_search_duckduckgo = _search_web