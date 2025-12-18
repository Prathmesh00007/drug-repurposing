"""Automated target validation using multiple biomedical evidence sources."""

import logging
import httpx
import json
import os  # ✅ ADDED
import asyncio
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from kg.semantic_router import DiseaseContext
import cerebras_llm as llm  # Cerebras-hosted LLM wrapper

# ✅ ADDED: Import Gemini SDK
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("⚠️ google-generativeai not installed. Gemini filtering disabled.")

logger = logging.getLogger(__name__)


class AutomatedTargetValidator:
    """Validate drug targets using DisGeNET, UniProt, and NCBI APIs."""
    
    def __init__(self):
        self.timeout = 10.0
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _query_disgenet(self, gene_symbol: str, disease_name: str) -> Optional[float]:
        """
        Query DisGeNET for gene-disease association.
        ✅ FIX #3: Updated to use correct DisGeNET API endpoint
        """
        try:
            url = "https://www.disgenet.com/api/gda/gene"
            params = {
                "gene": gene_symbol,
                "disease": disease_name,
                "source": "ALL",
                "format": "json"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, params=params)
                
                if response.status_code != 200:
                    logger.debug(f"DisGeNET returned status {response.status_code} for {gene_symbol}")
                    return None
                
                content_type = response.headers.get("content-type", "")
                if "json" not in content_type.lower():
                    logger.debug(f"DisGeNET returned non-JSON content-type: {content_type}")
                    return None
                
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    logger.debug(f"DisGeNET returned invalid JSON for {gene_symbol}")
                    return None
                
                # Handle both list and dict responses
                if isinstance(data, list) and len(data) > 0:
                    scores = [float(item.get("score", 0)) for item in data if "score" in item]
                    if scores:
                        max_score = max(scores)
                        logger.debug(f"DisGeNET: {gene_symbol} - {disease_name} = {max_score:.3f}")
                        return max_score
                elif isinstance(data, dict):
                    results = data.get("results", data.get("data", []))
                    if isinstance(results, list) and len(results) > 0:
                        scores = [float(item.get("score", 0)) for item in results if "score" in item]
                        if scores:
                            return max(scores)
                
                return None
                
        except Exception as e:
            logger.debug(f"DisGeNET query failed for {gene_symbol}: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _query_uniprot(self, uniprot_id: str) -> Dict:
        """Query UniProt for protein function annotations."""
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                
                if response.status_code != 200:
                    return {}
                
                data = response.json()
                
                result = {
                    "has_function": False,
                    "has_disease_involvement": False,
                    "is_reviewed": data.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"
                }
                
                # Check for function comments
                comments = data.get("comments", [])
                for comment in comments:
                    if comment.get("commentType") == "FUNCTION":
                        result["has_function"] = True
                    if comment.get("commentType") == "DISEASE":
                        result["has_disease_involvement"] = True
                
                return result
                
        except Exception as e:
            logger.debug(f"UniProt query failed for {uniprot_id}: {e}")
            return {}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _query_ncbi_gene(self, gene_symbol: str) -> Dict:
        """Query NCBI Gene database for gene information."""
        try:
            # Step 1: Search for gene
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "gene",
                "term": f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism]",
                "retmode": "json",
                "retmax": 1
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                search_response = await client.get(search_url, params=search_params)
                
                if search_response.status_code != 200:
                    return {}
                
                search_data = search_response.json()
                gene_ids = search_data.get("esearchresult", {}).get("idlist", [])
                
                if not gene_ids:
                    return {}
                
                # Step 2: Get gene summary
                summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                summary_params = {
                    "db": "gene",
                    "id": gene_ids[0],
                    "retmode": "json"
                }
                
                summary_response = await client.get(summary_url, params=summary_params)
                
                if summary_response.status_code != 200:
                    return {}
                
                summary_data = summary_response.json()
                result = summary_data.get("result", {}).get(gene_ids[0], {})
                
                return {
                    "gene_id": gene_ids[0],
                    "description": result.get("description", ""),
                    "summary": result.get("summary", ""),
                    "has_summary": bool(result.get("summary"))
                }
                
        except Exception as e:
            logger.debug(f"NCBI Gene query failed for {gene_symbol}: {e}")
            return {}
    
    async def validate_target(
        self,
        gene_symbol: str,
        disease_name: str,
        uniprot_id: Optional[str] = None
    ) -> Dict:
        """
        Validate a single target using multiple evidence sources.
        
        PRODUCTION-GRADE SCORING:
        - DisGeNET: 40% weight (direct gene-disease evidence)
        - UniProt: 30% weight (protein function/disease involvement)
        - NCBI Gene: 30% weight (gene characterization)
        
        Returns:
            {
                "symbol": str,
                "is_valid": bool,
                "score": float (0-1),
                "evidence": {"disgenet": float, "uniprot": float, "ncbi": float}
            }
        """
        evidence = {
            "disgenet": 0.0,
            "uniprot": 0.0,
            "ncbi": 0.0
        }
        
        # Evidence 1: DisGeNET (gene-disease association)
        disgenet_score = await self._query_disgenet(gene_symbol, disease_name)
        if disgenet_score is not None:
            evidence["disgenet"] = min(1.0, disgenet_score)
        
        # Evidence 2: UniProt (protein annotation quality)
        if uniprot_id:
            uniprot_data = await self._query_uniprot(uniprot_id)
            uniprot_score = 0.0
            if uniprot_data.get("is_reviewed"):
                uniprot_score += 0.4
            if uniprot_data.get("has_function"):
                uniprot_score += 0.3
            if uniprot_data.get("has_disease_involvement"):
                uniprot_score += 0.3
            evidence["uniprot"] = uniprot_score
        
        # Evidence 3: NCBI Gene (gene characterization)
        ncbi_data = await self._query_ncbi_gene(gene_symbol)
        ncbi_score = 0.0
        if ncbi_data.get("gene_id"):
            ncbi_score += 0.5
        if ncbi_data.get("has_summary"):
            ncbi_score += 0.5
        evidence["ncbi"] = ncbi_score
        
        # Composite score with weights
        composite_score = (
            evidence["disgenet"] * 0.40 +
            evidence["uniprot"] * 0.30 +
            evidence["ncbi"] * 0.30
        )
        
        # LENIENT THRESHOLD: Accept if ANY of the following:
        # 1. DisGeNET score > 0 (direct evidence)
        # 2. Composite score >= 0.20 (20% threshold - VERY lenient)
        # 3. Both UniProt and NCBI have evidence (well-characterized gene)
        is_valid = (
            evidence["disgenet"] > 0 or
            composite_score >= 0.20 or
            (evidence["uniprot"] > 0.3 and evidence["ncbi"] > 0.3)
        )
        
        return {
            "symbol": gene_symbol,
            "is_valid": is_valid,
            "score": composite_score,
            "evidence": evidence
        }
    
    async def validate_targets_batch(
        self,
        targets: List[Dict[str, str]],
        disease_name: str,
        disease_context: Optional[DiseaseContext] = None,  # ✅ ADDED
        use_gemini_filter: bool = False  # ✅ ADDED
    ) -> List[Dict]:
        """
        Validate a batch of targets with optional Gemini mechanism filtering.
        
        Args:
            targets: List of {"symbol": str, "uniprot_id": str, "ensembl_id": str}
            disease_name: Disease name
            disease_context: Optional DiseaseContext from Gemini
            use_gemini_filter: If True, apply Gemini mechanism validation
        
        Returns:
            List of validated targets with scores
        """
        validated = []
        
        # ✅ Initialize Gemini filter if requested
        gemini_filter = None
        if use_gemini_filter and disease_context and GEMINI_AVAILABLE:
            try:
                gemini_filter = GeminiTargetMechanismFilter()
                logger.info(f"🤖 Gemini mechanism filter enabled for {disease_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize Gemini filter: {e}")
        
        for target in targets[:3]:
            result = await self.validate_target(
                gene_symbol=target["symbol"],
                disease_name=disease_name,
                uniprot_id=target.get("uniprot_id")
            )
            
            if result["is_valid"]:
                # ✅ Apply Gemini mechanism filter if available
                if gemini_filter:
                    gemini_result = await gemini_filter.validate_mechanism_async(
                        target["symbol"],
                        disease_context,
                        result["score"]
                    )
                    
                    if gemini_result["decision"] == "REJECT":
                        logger.warning(
                            f"❌ {result['symbol']} - REJECTED by Gemini filter: {gemini_result['reasoning']}"
                        )
                        continue  # Skip this target
                
                validated.append({
                    **target,
                    "validation_score": result["score"],
                    "evidence": result["evidence"]
                })
                
                logger.info(
                    f"✓ {result['symbol']} - VALID (score: {result['score']:.2f}, "
                    f"sources: DisGeNET:{result['evidence']['disgenet']:.2f}, "
                    f"UniProt:{result['evidence']['uniprot']:.2f}, "
                    f"NCBI:{result['evidence']['ncbi']:.2f})"
                )
            else:
                logger.warning(
                    f"❌ {result['symbol']} - REJECTED (score: {result['score']:.2f}, insufficient evidence)"
                )
        
        return validated


# ============================================================================
# Gemini Mechanism Filter
# ============================================================================

class GeminiTargetMechanismFilter:
    """
    Use Gemini to validate if a target mechanism matches the disease biology.
    Prevents cancer drugs from being recommended for autoimmune diseases.
    """
    
    FILTER_PROMPT = """You are a drug target validation expert.

TASK: Determine if the proposed drug target is biologically relevant for the disease.

EXAMPLES (Few-Shot Learning):

Example 1:
Disease: Alopecia Areata (Autoimmune)
Target: ABL1 (Tyrosine kinase, CML leukemia target)
Decision: REJECT
Reason: ABL1 is a cancer target (Chronic Myeloid Leukemia). Alopecia Areata is autoimmune (T-cell mediated). Mechanism mismatch. Imatinib (ABL1 inhibitor) would be toxic and ineffective.

Example 2:
Disease: Alopecia Areata (Autoimmune)
Target: JAK1 (JAK-STAT pathway)
Decision: KEEP
Reason: JAK1 is a validated target for autoimmune diseases. Baricitinib (JAK1/2 inhibitor) is FDA-approved for Alopecia Areata.

Example 3:
Disease: Chronic Myeloid Leukemia (Cancer)
Target: ABL1
Decision: KEEP
Reason: ABL1 is the STANDARD target for CML. Imatinib (ABL1 inhibitor) is first-line therapy.

Example 4:
Disease: Narcolepsy (Neurological - Sleep disorder)
Target: HCRT (Hypocretin/Orexin receptor)
Decision: KEEP
Reason: HCRT is the PRIMARY pathophysiology of narcolepsy. Hypocretin deficiency causes excessive daytime sleepiness and cataplexy.

Example 5:
Disease: Narcolepsy (Neurological)
Target: ABL1 (Tyrosine kinase for cancer)
Decision: REJECT
Reason: ABL1 is a cancer target (CML). Narcolepsy is a neurological sleep disorder caused by hypocretin deficiency. No mechanism overlap.

NOW EVALUATE:

Disease: {disease_name} ({disease_type})
Disease Description: {disease_description}
Target Gene: {target_symbol}
Evidence Score: {evidence_score:.2f}/1.0

Return JSON:
{{
  "decision": "KEEP" or "REJECT",
  "confidence": 0.95,
  "reasoning": "<2-sentence explanation>"
}}
"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        self.api_key = ""
        genai.configure(api_key="")
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    async def validate_mechanism_async(
        self,
        target_symbol: str,
        disease_context: DiseaseContext,
        evidence_score: float
    ) -> Dict:
        """
        ✅ ASYNC VERSION: Use Gemini to validate if target matches disease mechanism.
        
        Returns:
            {"decision": "KEEP"|"REJECT", "confidence": float, "reasoning": str}
        """
        return await asyncio.to_thread(
            self._validate_mechanism_sync,
            target_symbol,
            disease_context,
            evidence_score
        )
    
    def _validate_mechanism_sync(
        self,
        target_symbol: str,
        disease_context: DiseaseContext,
        evidence_score: float
    ) -> Dict:
        """Synchronous version (runs in thread pool)."""
        try:
            # Determine disease type
            disease_type = disease_context.therapeutic_area
            if disease_context.is_cancer:
                disease_type = "Cancer"
            elif disease_context.is_autoimmune:
                disease_type = "Autoimmune"
            elif disease_context.is_infectious:
                disease_type = "Infectious"
            
            prompt = self.FILTER_PROMPT.format(
                disease_name=disease_context.corrected_name,
                disease_type=disease_type,
                disease_description=disease_context.description,
                target_symbol=target_symbol,
                evidence_score=evidence_score
            )
            
            response = llm.generate_sync(prompt)
            response_text = response.text.strip()
            
            # Clean JSON (remove markdown code blocks if present)
            text = response_text.strip()
            if "```json" in text:
                # Split on ```json and take the part after it
                parts = text.split("```json")
                if len(parts) > 1:
                    # Now split on ending ```
                    text = parts[1].split("```")[0].strip()

            elif "```" in text:
                # Handle generic code blocks ``` ... ```
                parts = text.split("```")
                if len(parts) >= 3:
                    # The JSON is usually in parts[1]
                    text = parts[1].strip()

            # Method 2: Extract JSON object manually (first { to last })
            if not text.startswith("{"):
                start_idx = text.find("{")
                end_idx = text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    text = text[start_idx:end_idx + 1]
            
            result = json.loads(text)
            
            logger.info(f"🤖 Gemini filter: {target_symbol} → {result['decision']} (confidence: {result['confidence']:.2f})")
            logger.debug(f"   Reason: {result['reasoning']}")
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Gemini mechanism filter failed for {target_symbol}: {e}")
            # Fallback: accept the target if Gemini fails
            return {
                "decision": "KEEP",
                "confidence": 0.5,
                "reasoning": "LLM unavailable, defaulting to KEEP"
            }


# ============================================================================
# Public API
# ============================================================================

# Global instance
_validator = None

def get_validator() -> AutomatedTargetValidator:
    """Get singleton validator instance."""
    global _validator
    if _validator is None:
        _validator = AutomatedTargetValidator()
    return _validator


async def validate_targets_for_disease(
    disease_name: str,
    targets: List[Dict[str, str]],
    disease_context: Optional[DiseaseContext] = None,  # ✅ ADDED
    use_gemini_filter: bool = False  # ✅ ADDED
) -> List[Dict[str, str]]:
    """
    ✅ FIXED: Public API now accepts disease_context and use_gemini_filter.
    
    Validate targets for a disease with optional Gemini mechanism filtering.
    
    Args:
        disease_name: Disease name
        targets: List of {"symbol": str, "uniprot_id": str, "ensembl_id": str}
        disease_context: Optional DiseaseContext from Gemini (for mechanism filtering)
        use_gemini_filter: If True and disease_context provided, apply Gemini filtering
    
    Returns:
        Validated targets (filtered list)
    """
    logger.info(f"🧬 Validating {len(targets)} targets for {disease_name}")
    
    if use_gemini_filter and disease_context:
        logger.info(f"   Mode: DisGeNET + UniProt + NCBI + Gemini Mechanism Filter")
    else:
        logger.info(f"   Mode: DisGeNET + UniProt + NCBI only")
    
    validator = get_validator()
    validated = await validator.validate_targets_batch(
        targets,
        disease_name,
        disease_context=disease_context,  # ✅ PASS THROUGH
        use_gemini_filter=use_gemini_filter  # ✅ PASS THROUGH
    )
    
    logger.info(f"✅ Validation complete: {len(validated)}/{len(targets)} targets passed")
    
    return validated
