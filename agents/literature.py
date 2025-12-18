"""Enhanced Literature Agent: Citation-weighted, LLM-synthesized evidence.
Adapted to use the local `cerebras_llm` compat wrapper for generation.
"""
import logging
import httpx
import asyncio
import json
import re
from typing import List, Dict, Optional
from xml.etree import ElementTree as ET
from backend.app.schemas import (
    LiteratureOutput, TargetEvidence, Citation
)
from agents.base import cache_manager
from tenacity import retry, stop_after_attempt, wait_exponential
import cerebras_llm as llm
import os

logger = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ROBUST gene symbol validation
MEDICAL_ABBREVIATIONS = {
    'ICI', 'RAI', 'ATID', 'AITD', 'TSH', 'T3', 'T4', 'FT3', 'FT4',
    'FDA', 'EMA', 'USA', 'DNA', 'RNA', 'ATP', 'ADP', 'HIV', 'AIDS',
    'BMI', 'ECG', 'MRI', 'CT', 'PET', 'COPD', 'NSAID', 'ACE', 'ARB'
}


def _is_valid_gene_symbol(symbol: str) -> bool:
    """Validate gene symbol before adding as target."""
    # Reject medical abbreviations
    if symbol.upper() in MEDICAL_ABBREVIATIONS:
        return False

    # Reject if too short
    if len(symbol) < 3:
        return False

    # Reject if all uppercase and 2-3 letters (likely abbreviation)
    if len(symbol) <= 3 and symbol.isupper():
        return False

    # Must start with letter
    if not symbol[0].isalpha():
        return False

    # Should be mostly uppercase (gene symbols convention)
    uppercase_ratio = sum(1 for c in symbol if c.isupper()) / len(symbol)
    if uppercase_ratio < 0.5:
        return False

    return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _pubmed_search(query: str, max_results: int = 10) -> List[str]:
    """Search PubMed and return PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
            resp.raise_for_status()
            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"PubMed search: '{query}' -> {len(pmids)} articles")
            return pmids
    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """Fetch full metadata + abstracts for PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids[:50]),  # Limit batch size
        "retmode": "xml"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{PUBMED_BASE}/efetch.fcgi", params=params)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            articles = []

            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                title_elem = article.find(".//ArticleTitle")
                # Abstract can have multiple AbstractText nodes; join them if present
                abstract_elems = article.findall(".//AbstractText")
                year_elem = article.find(".//PubDate/Year")

                abstract_text = ""
                if abstract_elems:
                    parts = [ET.tostring(a, encoding='unicode', method='text').strip() for a in abstract_elems]
                    abstract_text = "\n".join(parts)

                if pmid_elem is not None and title_elem is not None:
                    articles.append({
                        "pmid": pmid_elem.text,
                        "title": title_elem.text or "",
                        "abstract": abstract_text or "",
                        "year": year_elem.text if year_elem is not None else "Unknown"
                    })

            return articles
    except Exception as e:
        logger.error(f"Failed to fetch abstracts: {e}")
        return []


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
async def _get_citation_count(pmid: str) -> int:
    """Get citation count for a PMID using elink."""
    params = {
        "dbfrom": "pubmed",
        "id": pmid,
        "cmd": "neighbor",
        "linkname": "pubmed_pubmed_citedin"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{PUBMED_BASE}/elink.fcgi", params=params)
            root = ET.fromstring(resp.content)
            links = root.findall(".//Link")
            return len(links)
    except:
        return 0


async def _analyze_with_llm(prompt: str, abstracts: List[Dict]) -> Dict:
    """Use the cerebras_llm compat wrapper to synthesize literature and return parsed JSON.

    The function is robust to: plain text responses, JSON-wrapped responses, and
    extra commentary before/after a JSON payload.
    """
    # Prepare abstract text (truncate to keep prompt size reasonable)
    text = "\n\n".join([
        f"PMID {a['pmid']} ({a.get('year', 'Unknown')}): {a['title']}\nAbstract: {a['abstract'][:800]}"
        for a in abstracts[:6]
    ])

    full_prompt = f"{prompt}\n\n{text}"

    try:
        # Use the cerebras compat wrapper. It returns an object with .text
        response = await llm.generate(full_prompt, temperature=0.3, stream=False)
        raw = response.text if hasattr(response, "text") else str(response)

        # First try direct JSON parsing
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            # Try to extract the first JSON object in the text
            m = re.search(r"(\{(?:.|\n)*\})", raw)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    return parsed
                except json.JSONDecodeError:
                    logger.debug("Found braces but failed to parse JSON from LLM output")

            # Fallback: return the raw text under a key so callers can still see output
            return {"raw": raw}

    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        return {}


async def run_literature_rag(
    disease_name: str,
    soc_list: List[str],
    keywords: dict
) -> LiteratureOutput:
    """
    Enhanced Literature Analysis with tiered search + LLM synthesis.
    """
    logger.info(f"Enhanced Literature RAG: {disease_name}")

    cache_key = {"disease": disease_name}
    cached = cache_manager.get("literature", cache_key)
    if cached:
        return LiteratureOutput(**cached)

    citations = []
    all_articles = []

    # ===== TIER 1: META-ANALYSES & SYSTEMATIC REVIEWS =====
    logger.info("Tier 1: Searching for meta-analyses...")
    tier1_query = f"{disease_name} AND (meta-analysis[Publication Type] OR systematic review[Publication Type])"
    tier1_pmids = await _pubmed_search(tier1_query, max_results=5)

    if tier1_pmids:
        tier1_articles = await _fetch_abstracts(tier1_pmids)
        all_articles.extend(tier1_articles)

        for article in tier1_articles:
            citations.append(Citation(
                url=f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/",
                title=article['title'],
                source="PubMed (Meta-Analysis)"
            ))

    # ===== TIER 2: RECENT REVIEWS (Last 3 years) =====
    logger.info("Tier 2: Searching for recent reviews...")
    tier2_query = f'{disease_name} AND review[Publication Type] AND ("2022"[Date - Publication] : "2025"[Date - Publication])'
    tier2_pmids = await _pubmed_search(tier2_query, max_results=8)

    if tier2_pmids:
        tier2_articles = await _fetch_abstracts(tier2_pmids)
        all_articles.extend(tier2_articles)

        for article in tier2_articles[:5]:
            citations.append(Citation(
                url=f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/",
                title=article['title'],
                source="PubMed (Recent Review)"
            ))

    # ===== TIER 3: MECHANISTIC STUDIES =====
    logger.info("Tier 3: Searching for mechanism studies...")
    tier3_query = f"{disease_name} pathophysiology mechanism molecular targets"
    tier3_pmids = await _pubmed_search(tier3_query, max_results=10)
    tier3_articles = await _fetch_abstracts(tier3_pmids)
    all_articles.extend(tier3_articles)

    # ===== CITATION WEIGHTING =====
    logger.info("Calculating citation counts...")
    citation_tasks = [_get_citation_count(a['pmid']) for a in all_articles[:10]]
    citation_counts = await asyncio.gather(*citation_tasks)

    for i, article in enumerate(all_articles[:10]):
        article['citation_count'] = citation_counts[i] if i < len(citation_counts) else 0

    # Sort by citations
    all_articles.sort(key=lambda x: x.get('citation_count', 0), reverse=True)

    # ===== LLM SYNTHESIS =====
    pathophysiology_summary = ""
    validated_targets = []

    if all_articles:
        # Pathophysiology Summary
        patho_prompt = f"""
Analyze these scientific abstracts about {disease_name}.
Synthesize a comprehensive 2-3 paragraph summary explaining:
1. The core molecular pathophysiology
2. Key cellular processes and pathways involved
3. Main cell types and tissues affected

Return JSON:
{{
    "summary": "Your 2-3 paragraph synthesis here"
}}
"""
        patho_result = await _analyze_with_llm(patho_prompt, all_articles[:5])
        pathophysiology_summary = patho_result.get("summary", "") if isinstance(patho_result, dict) else ""

        # Target Validation
        target_prompt = f"""
Analyze these abstracts about {disease_name}.
Identify the top 8 therapeutic targets (genes/proteins) with strongest evidence.
For each:
- target_name: Gene symbol (e.g., "TNF", "IL6", "VEGFA")
- confidence_score: "High" (mentioned in multiple high-impact studies), "Medium", or "Low"
- supporting_evidence: 2-3 sentence summary of the evidence and mechanism

Return JSON:
{{
    "targets": [
        {{"target_name": "...", "confidence_score": "...", "supporting_evidence": "..."}},
        ...
    ]
}}
"""
        target_result = await _analyze_with_llm(target_prompt, all_articles[:10])

        if isinstance(target_result, dict) and target_result.get("targets"):
            for tgt in target_result["targets"][:10]:
                validated_targets.append(TargetEvidence(
                    target_name=tgt.get("target_name", "Unknown"),
                    confidence_score=tgt.get("confidence_score", "Low"),
                    supporting_evidence=tgt.get("supporting_evidence", ""),
                    source_pmids=[a['pmid'] for a in all_articles[:3]],
                    citation_count=all_articles[0].get('citation_count', 0) if all_articles else 0
                ))

    # Fallback: Use basic extraction
    if not validated_targets:
        for article in all_articles[:5]:
            text = article['title'] + " " + article['abstract']
            genes = re.findall(r'\b[A-Z][A-Z0-9]{2,8}\b', text)  # 3-9 chars

            for gene in set(genes):
                if _is_valid_gene_symbol(gene):
                    validated_targets.append(TargetEvidence(
                        target_name=gene,
                        confidence_score="Low",
                        supporting_evidence=f"Mentioned in {article['title'][:50]}...",
                        source_pmids=[article['pmid']]
                    ))

            if len(validated_targets) >= 5:
                break

    # Build output with legacy compatibility
    output = LiteratureOutput(
        pathophysiology_summary=pathophysiology_summary,
        validated_targets=validated_targets[:12],
        emerging_targets=[],
        key_review_articles=citations[:6],
        # Legacy fields
        suggested_targets=[t.target_name for t in validated_targets[:15]],
        mechanism_summary=[pathophysiology_summary] if pathophysiology_summary else [],
        citations=citations[:10],
        pubmed_articles=[
            {
                "pmid": a.get("pmid", ""),
                "title": a.get("title", ""),
                "abstract": a.get("abstract", "")[:300] + "...",
                "year": a.get("year", "Unknown")
            }
            for a in all_articles[:15]
        ]
    )

    cache_manager.set("literature", cache_key, output.dict())

    logger.info(f"Literature complete: {len(validated_targets)} targets, {len(all_articles)} articles")
    return output
