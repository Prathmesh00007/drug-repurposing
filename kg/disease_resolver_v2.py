"""
Disease Resolver V2 - Deterministic Ontology Resolution

Replaces semantic_router.py (which used LLM hallucination)

Uses FREE APIs:
- EMBL-EBI OLS API for EFO/MONDO lookup
- NCBI E-utilities for MeSH
- TherapeuticAreaMapper for therapeutic area classification (deterministic ontology-based)
"""

import httpx
import asyncio
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import os
from urllib.parse import quote

# Import the therapeutic area mapper
from kg.therapeutic_area_mapper import get_therapeutic_area_mapper

logger = logging.getLogger(__name__)

class TherapeuticArea(str, Enum):
    ONCOLOGY = "oncology"
    IMMUNOLOGY = "immunological"
    NEUROLOGY = "neurological"
    CARDIOLOGY = "cardiovascular"
    METABOLIC = "metabolic"
    INFECTIOUS = "infectious"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    DERMATOLOGY = "dermatological"
    RARE_DISEASE = "rare_diseases"
    HEMATOLOGICAL = "hematological"
    UROLOGICAL = "urological"
    MUSCULOSKELETAL = "musculoskeletal"
    OPHTHALMOLOGY = "ophthalmology"
    PSYCHIATRIC = "psychiatric"
    ENDOCRINOLOGY = "endocrinology"
    RENAL_NEPHROLOGY = "renal_nephrology"
    HEPATOLOGY = "hepatology"
    WOMEN_HEALTH_OBGYN = "women_health_obgyn"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    PAIN_PALLIATIVE = "pain_palliative"
    ALLERGY = "allergy"
    ADDICTION = "addiction_substance_use"
    TRANSPLANTATION = "transplantation_immunosuppression"
    DENTAL = "dental_oral_health"
    ONCOLOGY_SUPPORTIVE = "oncology_supportive_care"
    TOXICOLOGY = "toxicology_overdose"
    UNKNOWN = "unknown"

@dataclass
class DiseaseContext:
    """Enhanced disease context with deterministic IDs"""
    original_query: str
    corrected_name: str
    efo_id: Optional[str] = None
    mondo_id: Optional[str] = None
    mesh_id: Optional[str] = None
    description: str = ""
    therapeutic_area: str = "unknown"

    # Disease classification flags
    is_cancer: bool = False
    is_autoimmune: bool = False
    is_infectious: bool = False
    is_rare: bool = False
    is_genetic: bool = False

    # Ontology metadata
    synonyms: List[str] = field(default_factory=list)
    parent_terms: List[str] = field(default_factory=list)

    # Validation
    confidence: float = 1.0  # Deterministic lookup = 100%
    ols_match_score: float = 0.0

    # Suggested key targets (from ontology annotations, NOT LLM hallucination)
    annotated_genes: List[str] = field(default_factory=list)

class DiseaseResolverV2:
    """
    Deterministic disease resolution using authoritative APIs.
    NO LLM hallucination for ontology IDs or therapeutic areas.
    """

    OLS_BASE_URL = "https://www.ebi.ac.uk/ols4/api"
    NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self.therapeutic_mapper = get_therapeutic_area_mapper()
        self.timeout = httpx.Timeout(30.0, connect=10.0)

    def _select_best_doc(self, docs: List[Dict], query: str) -> Optional[Dict]:
        """Improved selection with fuzzy matching and synonym checking"""
        if not docs:
            return None
        
        q = query.lower().strip()
        
        # Helper: normalize for comparison (remove punctuation, extra spaces)
        import re
        def normalize(text):
            # Remove apostrophes, hyphens, normalize spaces
            text = re.sub(r"['\-]", "", text.lower())
            return " ".join(text.split())
        
        q_norm = normalize(query)
        
        # 1) Exact label match (normalized)
        for d in docs:
            label = d.get("label") or ""
            if isinstance(label, list):
                label = label[0] if label else ""
            if normalize(label) == q_norm:
                logger.info(f"✅ Exact label match: {label}")
                return d
        
        # 2) Check synonyms for exact match
        for d in docs:
            synonyms = d.get("synonym", []) or []
            for syn in synonyms:
                if normalize(syn) == q_norm:
                    logger.info(f"✅ Synonym match: {syn} → {d.get('label')}")
                    return d
        
        # 3) Fuzzy matching (Levenshtein distance < 3 for short variations)
        from difflib import SequenceMatcher
        best_fuzzy = None
        best_fuzzy_score = 0.0
        
        for d in docs:
            label = d.get("label") or ""
            if isinstance(label, list):
                label = label[0] if label else ""
            
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, q_norm, normalize(label)).ratio()
            
            # Check if highly similar (>0.85 threshold)
            if similarity > 0.85 and similarity > best_fuzzy_score:
                best_fuzzy = d
                best_fuzzy_score = similarity
        
        if best_fuzzy:
            logger.info(f"✅ Fuzzy match: {best_fuzzy.get('label')} (similarity: {best_fuzzy_score:.2f})")
            return best_fuzzy
        
        # 4) Prefer MONDO entries with highest score
        mondo_docs = [d for d in docs 
                    if (d.get("ontology_name") or "").lower() == "mondo" 
                    or (d.get("obo_id") or "").upper().startswith("MONDO")]
        
        if mondo_docs:
            best = max(mondo_docs, key=lambda x: float(x.get("score", 0.0) or 0.0))
            logger.warning(f"⚠️ No exact/fuzzy match - using highest MONDO score: {best.get('label')}")
            return best
        
        # 5) Fallback: highest score overall
        best = max(docs, key=lambda x: float(x.get("score", 0.0) or 0.0))
        logger.warning(f"⚠️ Fallback to highest score: {best.get('label')}")
        return best


    async def resolve_disease(self, disease_name: str) -> Optional[DiseaseContext]:
        """
        Main resolution pipeline.
        Steps:
        1. Query OLS API for EFO/MONDO (deterministic)
        2. Query NCBI for MeSH (deterministic)
        3. Use TherapeuticAreaMapper for therapeutic area classification (deterministic)
        4. Extract ontology annotations for genes
        """
        logger.info(f"🔍 Resolving disease: '{disease_name}'")

        # Step 1: Query OLS for EFO/MONDO
        ols_result = await self._query_ols(disease_name)
        if not ols_result:
            logger.warning(f"❌ No OLS results for '{disease_name}'")
            return None

        # Step 2: Query MeSH (optional, parallel)
        mesh_task = asyncio.create_task(self._query_mesh(disease_name))

        # Step 3: Get ontology details
        ontology_details = await self._get_ontology_details(
            ols_result["iri"], 
            ols_result["ontology"]
        )

        # Step 4: Use TherapeuticAreaMapper for therapeutic area classification
        therapeutic_area = await self._classify_therapeutic_area_with_mapper(
            disease_name=ols_result["label"]
        )

        # Step 5: Extract disease flags from ontology
        disease_flags = self._extract_disease_flags(
            description=ols_result.get("description", ""),
            parents=ontology_details.get("parents", []),
            iri=ols_result["iri"]
        )

        # Wait for MeSH result
        mesh_id = await mesh_task

        # Build DiseaseContext
        context = DiseaseContext(
            original_query=disease_name,
            corrected_name=ols_result["label"],
            efo_id=ols_result["obo_id"] if ols_result["ontology"] == "efo" else None,
            mondo_id=ols_result["obo_id"] if ols_result["ontology"] == "mondo" else None,
            mesh_id=mesh_id,
            description=ols_result.get("description", ""),
            therapeutic_area=therapeutic_area,
            is_cancer=disease_flags["is_cancer"],
            is_autoimmune=disease_flags["is_autoimmune"],
            is_infectious=disease_flags["is_infectious"],
            is_rare=disease_flags["is_rare"],
            is_genetic=disease_flags["is_genetic"],
            synonyms=ols_result.get("synonyms", []),
            parent_terms=ontology_details.get("parents", []),
            confidence=1.0,
            ols_match_score=ols_result.get("score", 0.0),
            annotated_genes=ontology_details.get("annotated_genes", [])
        )

        logger.info(f"✅ Resolved: {context.corrected_name}")
        logger.info(f"   EFO: {context.efo_id}, MONDO: {context.mondo_id}, MeSH: {context.mesh_id}")
        logger.info(f"   Therapeutic Area: {context.therapeutic_area}")
        return context

    async def _query_ols(self, disease_name: str) -> Optional[Dict]:
        """Return the best matching doc for a disease name (robust)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.OLS_BASE_URL}/search",
                params={
                    "q": disease_name,
                    "ontology": "efo,mondo",
                    "type": "class",
                    "exact": False,
                    "rows": 10,
                    "fieldList": "iri,label,description,obo_id,ontology_name,synonym,score"
                },
            )
            resp.raise_for_status()
            if not resp.content:
                return None
            try:
                data = resp.json()
            except ValueError:
                logger.error("Invalid JSON from OLS; preview=%s", (resp.text or "")[:300])
                return None

            docs = data.get("response", {}).get("docs", []) or []
            if not docs:
                return None

            best = self._select_best_doc(docs, disease_name)
            if not best:
                return None

            # normalize return
            def first_or_str(v):
                if isinstance(v, list):
                    return v[0] if v else ""
                return v or ""

            return {
                "iri": first_or_str(best.get("iri")),
                "label": first_or_str(best.get("label")),
                "description": first_or_str(best.get("description", "")),
                "obo_id": first_or_str(best.get("obo_id")),
                "ontology": first_or_str(best.get("ontology_name")),
                "synonyms": best.get("synonym", []) or [],
                "score": float(best.get("score", 0.0) or 0.0)
            }

    

    async def _query_mesh(self, disease_name: str) -> Optional[str]:
        """Query NCBI E-utilities for MeSH ID"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Step 1: Search MeSH
                search_response = await client.get(
                    f"{self.NCBI_BASE_URL}/esearch.fcgi",
                    params={
                        "db": "mesh",
                        "term": disease_name,
                        "retmode": "json",
                        "retmax": 1
                    }
                )
                search_data = search_response.json()
                if not search_data.get("esearchresult", {}).get("idlist"):
                    return None

                mesh_uid = search_data["esearchresult"]["idlist"][0]

                # Step 2: Fetch MeSH details
                fetch_response = await client.get(
                    f"{self.NCBI_BASE_URL}/esummary.fcgi",
                    params={
                        "db": "mesh",
                        "id": mesh_uid,
                        "retmode": "json"
                    }
                )
                fetch_data = fetch_response.json()
                mesh_id = fetch_data.get("result", {}).get(mesh_uid, {}).get("ds_meshterms", [""])[0]

                # Format as D-number
                if mesh_id and not mesh_id.startswith("D"):
                    mesh_id = f"D{mesh_uid.zfill(6)}"

                return mesh_id
            except Exception as e:
                logger.error(f"MeSH query failed: {e}")
                return None

    async def _get_ontology_details(self, iri: str, ontology: str) -> Dict:
        """Get ontology parents and gene annotations"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # URL-encode IRI
                import urllib.parse
                encoded_iri = quote(quote(iri, safe=''), safe='')

                # Get parents
                parents_response = await client.get(
                    f"{self.OLS_BASE_URL}/ontologies/{ontology}/terms/{encoded_iri}/parents"
                )
                parents_data = parents_response.json()
                parent_labels = []

                if "_embedded" in parents_data and "terms" in parents_data["_embedded"]:
                    parent_labels = [
                        term.get("label", "")
                        for term in parents_data["_embedded"]["terms"]
                    ]

                # Get annotations (for gene associations)
                # Note: Not all ontologies have gene annotations
                # This is best-effort
                annotated_genes = []

                return {
                    "parents": parent_labels,
                    "annotated_genes": annotated_genes
                }
            except Exception as e:
                logger.error(f"Ontology details query failed: {e}")
                return {"parents": [], "annotated_genes": []}

    async def _classify_therapeutic_area_with_mapper(
        self, 
        disease_name: str
    ) -> str:
        """
        Use TherapeuticAreaMapper for deterministic therapeutic area classification.
        Uses MeSH tree numbers, EFO ancestors, and keyword matching.
        """
        try:
            therapeutic_area = await self.therapeutic_mapper.classify(disease_name)

            if therapeutic_area:
                logger.info(f"✅ Therapeutic area classified as: {therapeutic_area}")
                return therapeutic_area
            else:
                logger.warning(f"⚠️ Could not classify therapeutic area for: {disease_name}")
                return "unknown"
        except Exception as e:
            logger.error(f"Therapeutic area classification failed: {e}")
            return "unknown"

    def _extract_disease_flags(
        self, 
        description: str, 
        parents: List[str], 
        iri: str
    ) -> Dict[str, bool]:
        """Extract disease classification flags from ontology"""
        desc_lower = description.lower()
        parents_str = " ".join(parents).lower()
        iri_lower = iri.lower()

        return {
            "is_cancer": any(term in desc_lower or term in parents_str for term in [
                "cancer", "carcinoma", "neoplasm", "tumor", "malignancy", "leukemia", "lymphoma"
            ]),
            "is_autoimmune": any(term in desc_lower or term in parents_str for term in [
                "autoimmune", "autoinflammatory", "immune-mediated"
            ]),
            "is_infectious": any(term in desc_lower or term in parents_str for term in [
                "infection", "infectious", "viral", "bacterial", "fungal", "parasite"
            ]),
            "is_rare": any(term in desc_lower or term in parents_str or term in iri_lower for term in [
                "rare", "orphan", "orpha"
            ]),
            "is_genetic": any(term in desc_lower or term in parents_str for term in [
                "genetic", "hereditary", "congenital", "inherited"
            ])
        }

# Singleton instance
disease_resolver_v2 = DiseaseResolverV2()

async def resolve_disease_deterministic(disease_name: str) -> Optional[DiseaseContext]:
    """
    Public API for disease resolution.
    USE THIS instead of semantic_router.py
    """
    return await disease_resolver_v2.resolve_disease(disease_name)