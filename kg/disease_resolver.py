"""Disease Resolution using FREE robust APIs - Zero hardcoded data."""
import logging
import httpx
import re
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher
import asyncio

logger = logging.getLogger(__name__)

class DiseaseResolver:
    """
    Resolve disease names using multiple FREE APIs:
    - EBI OLS (Ontology Lookup Service)
    - EBI OxO (Ontology Xref Service) 
    - NCBI E-utilities
    """
    
    def __init__(self):
        # EBI Services (FREE, no API key)
        self.ols_base = "https://www.ebi.ac.uk/ols4/api"
        self.oxo_base = "https://www.ebi.ac.uk/spot/oxo/api"
        
        # NCBI E-utilities (FREE, no API key for low volume)
        self.ncbi_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        self.timeout = 15.0
    
    async def get_disease_synonyms_ncbi(self, disease_name: str) -> List[str]:
        """
        Get disease synonyms from NCBI MedGen API (FREE).

        Returns synonyms, abbreviations, and alternate names.
        Defensive: handles strings/lists/dicts returned by the API.
        """
        logger.info(f"🔍 Fetching synonyms from NCBI for: {disease_name}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Step 1: Search MedGen
                search_response = await client.get(
                    f"{self.ncbi_base}/esearch.fcgi",
                    params={
                        "db": "medgen",
                        "term": disease_name,
                        "retmode": "json",
                        "retmax": 5
                    }
                )

                if search_response.status_code != 200:
                    logger.warning("NCBI esearch returned status %s", search_response.status_code)
                    return []

                try:
                    search_data = search_response.json()
                except Exception as e:
                    logger.warning("Failed to decode esearch JSON: %s", e)
                    return []

                id_list = (
                    search_data.get("esearchresult", {}) if isinstance(search_data, dict) else {}
                ).get("idlist", [])

                if not id_list:
                    return []

                # Step 2: Fetch details for first result
                uid = id_list[0]
                summary_response = await client.get(
                    f"{self.ncbi_base}/esummary.fcgi",
                    params={
                        "db": "medgen",
                        "id": uid,
                        "retmode": "json"
                    }
                )

                if summary_response.status_code != 200:
                    logger.warning("NCBI esummary returned status %s", summary_response.status_code)
                    return []

                try:
                    summary_data = summary_response.json()
                except Exception as e:
                    logger.warning("Failed to decode esummary JSON: %s", e)
                    return []

                # Defensive extraction of the result object:
                result_obj = summary_data.get("result")
                result = {}
                if isinstance(result_obj, dict):
                    # many responses: result is a dict mapping uid->object and has "uids" list
                    if uid in result_obj:
                        # typical case
                        maybe = result_obj.get(uid)
                        if isinstance(maybe, dict):
                            result = maybe
                        else:
                            # unexpected: the uid key exists but is not a dict
                            logger.debug("Unexpected shape for result[%s]: %s", uid, type(maybe))
                            result = {}
                    else:
                        # sometimes summary_data['result'] itself *is* the desired dict
                        # try to pick first dict-like entry
                        found = None
                        for k, v in result_obj.items():
                            if k == "uids":
                                continue
                            if isinstance(v, dict):
                                found = v
                                break
                        if found:
                            result = found
                        else:
                            logger.debug("No dict found inside summary_data['result']")
                            result = {}
                else:
                    # weird case: result is not a dict (string/list/None)
                    logger.debug("summary_data['result'] unexpected type: %s", type(result_obj))
                    result = {}

                # Collect synonyms
                synonyms: List[str] = []

                # 1) concept/title
                concept_name = result.get("title") if isinstance(result, dict) else None
                if concept_name and isinstance(concept_name, str) and concept_name.strip():
                    synonyms.append(concept_name.strip())

                # 2) semantic types (kept for validation / future use)
                semantic_types = []
                semantic_type_raw = result.get("semantictype") if isinstance(result, dict) else None
                if semantic_type_raw:
                    if isinstance(semantic_type_raw, str):
                        semantic_types.append(semantic_type_raw)
                    elif isinstance(semantic_type_raw, list):
                        for item in semantic_type_raw:
                            if isinstance(item, dict):
                                val = item.get("value") or item.get("name") or item.get("semantictype")
                                if val:
                                    semantic_types.append(val)
                            elif isinstance(item, str):
                                semantic_types.append(item)
                    elif isinstance(semantic_type_raw, dict):
                        val = semantic_type_raw.get("value") or semantic_type_raw.get("name")
                        if val:
                            semantic_types.append(val)
                logger.debug("Semantic types: %s", semantic_types)

                # 3) conceptmeta: can be dict, string, or missing
                concept_meta = result.get("conceptmeta", {})
                if isinstance(concept_meta, dict):
                    # 'names' often contains synonyms
                    names = concept_meta.get("names") or concept_meta.get("synonyms") or []
                    if isinstance(names, list):
                        for term in names:
                            if isinstance(term, str) and term.strip():
                                if term.strip() not in synonyms:
                                    synonyms.append(term.strip())
                            elif isinstance(term, dict):
                                # sometimes name objects have 'name' or 'value'
                                nm = term.get("name") or term.get("value")
                                if nm and isinstance(nm, str) and nm.strip() and nm.strip() not in synonyms:
                                    synonyms.append(nm.strip())
                    elif isinstance(names, str):
                        # comma/semicolon separated string
                        for term in re.split(r"[;,]", names):
                            t = term.strip()
                            if t and t not in synonyms:
                                synonyms.append(t)
                    # Some entries also contain 'preferred', 'preferredName' etc
                    pref = concept_meta.get("preferred") or concept_meta.get("preferredName")
                    if isinstance(pref, str) and pref.strip() and pref.strip() not in synonyms:
                        synonyms.append(pref.strip())

                elif isinstance(concept_meta, str):
                    # If it's a plain string, split and use tokens
                    for term in re.split(r"[;,]", concept_meta):
                        t = term.strip()
                        if t and t not in synonyms:
                            synonyms.append(t)
                else:
                    # concept_meta missing or unexpected type
                    logger.debug("conceptmeta unexpected type: %s", type(concept_meta))

                # 4) other possible synonym fields (defensive)
                for candidate_key in ("synonymlist", "synonyms", "otheraliases", "alias"):
                    val = result.get(candidate_key)
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str) and item.strip() and item.strip() not in synonyms:
                                synonyms.append(item.strip())
                    elif isinstance(val, str):
                        for term in re.split(r"[;,]", val):
                            t = term.strip()
                            if t and t not in synonyms:
                                synonyms.append(t)

                # 5) extra: some records include 'clinicalSynopsis' text — extract simple noun phrases (light touch)
                clinical = result.get("clinicalSynopsis") if isinstance(result, dict) else None
                if isinstance(clinical, str) and len(synonyms) < 10:
                    # take short phrases split by semicolon or newline
                    for line in clinical.splitlines():
                        t = line.strip()
                        if t and len(t) < 80 and t not in synonyms:
                            synonyms.append(t)

                # Deduplicate, preserve order, limit
                seen = set()
                cleaned: List[str] = []
                for s in synonyms:
                    s2 = s.strip()
                    if not s2:
                        continue
                    if s2 not in seen:
                        seen.add(s2)
                        cleaned.append(s2)
                    if len(cleaned) >= 10:
                        break

                logger.info("✓ NCBI found %d synonyms (returning %d)", len(synonyms), len(cleaned))
                return cleaned

        except Exception as e:
            logger.warning("NCBI synonym lookup failed: %s", e)
            logger.debug("Exception detail:", exc_info=True)
            return []

    
    async def get_disease_synonyms_ols(self, disease_name: str) -> List[str]:
        """
        Get disease synonyms from EBI OLS API (FREE).
        
        OLS contains MONDO, EFO, DOID with full synonym support.
        """
        logger.info(f"🔍 Fetching synonyms from EBI OLS for: {disease_name}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.ols_base}/search",
                    params={
                        "q": disease_name,
                        "ontology": "mondo,efo,doid",
                        "type": "class",
                        "rows": 5,
                        "exact": "false"
                    }
                )
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                docs = data.get("response", {}).get("docs", [])
                
                if not docs:
                    return []
                
                # Get synonyms from best match
                best_match = docs[0]
                synonyms = []
                
                # Label
                label = best_match.get("label", "")
                if label:
                    synonyms.append(label)
                
                # Synonyms field
                synonym_field = best_match.get("synonym", [])
                if isinstance(synonym_field, list):
                    synonyms.extend(synonym_field)
                
                # Short form (abbreviations)
                short_form = best_match.get("short_form", "")
                if short_form:
                    synonyms.append(short_form)
                
                # Remove duplicates
                synonyms = list(set(synonyms))
                
                logger.info(f"✓ EBI OLS found {len(synonyms)} synonyms")
                return synonyms[:10]
                
        except Exception as e:
            logger.warning(f"EBI OLS synonym lookup failed: {e}")
            return []
    
    async def get_all_synonyms(self, disease_name: str) -> List[str]:
        """
        Aggregate synonyms from multiple FREE APIs.
        """
        # Run both APIs in parallel
        results = await asyncio.gather(
            self.get_disease_synonyms_ols(disease_name),
            self.get_disease_synonyms_ncbi(disease_name),
            return_exceptions=True
        )
        
        # Combine results
        all_synonyms = [disease_name]  # Start with original
        
        for result in results:
            if isinstance(result, list):
                all_synonyms.extend(result)
        
        # Remove duplicates, normalize
        unique_synonyms = []
        seen = set()
        
        for syn in all_synonyms:
            normalized = syn.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_synonyms.append(syn)
        
        logger.info(f"✓ Total unique synonyms: {len(unique_synonyms)}")
        return unique_synonyms
    
    async def map_ontology_id_oxo(self, source_id: str, target_ontology: str = "EFO") -> Optional[str]:
        """
        Map ontology IDs using EBI OxO API (FREE).
        
        Examples:
            MONDO:0005015 → EFO:0000249 (Alzheimer's)
            DOID:1470 → EFO:... (Malaria)
        
        Args:
            source_id: Source ontology ID (e.g., "MONDO:0005015")
            target_ontology: Target ontology prefix (e.g., "EFO")
        
        Returns:
            Mapped ID in target ontology or None
        """
        logger.info(f"🔄 Mapping {source_id} → {target_ontology} via OxO...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.oxo_base}/mappings",
                    params={
                        "fromId": source_id,
                        "mappingTarget": target_ontology,
                        "distance": "2"  # Allow 1-2 hop mappings
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"OxO mapping failed: {response.status_code}")
                    return None
                
                data = response.json()
                embedded = data.get("_embedded", {})
                mappings = embedded.get("mappings", [])
                
                if not mappings:
                    logger.info(f"No OxO mapping found for {source_id}")
                    return None
                
                # Get first mapping
                first_mapping = mappings[0]
                target_id = first_mapping.get("toTerm", {}).get("curie", "")
                
                if target_id:
                    logger.info(f"✓ OxO mapped: {source_id} → {target_id}")
                    return target_id
                
        except Exception as e:
            logger.warning(f"OxO API error: {e}")
        
        return None
    
    async def search_ols_comprehensive(self, disease_name: str, synonyms: List[str]) -> Optional[Dict]:
        """
        Search EBI OLS using disease name + all synonyms.
        """
        search_terms = [disease_name] + synonyms
        
        for term in search_terms[:5]:  # Try first 5 variants
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(
                        f"{self.ols_base}/search",
                        params={
                            "q": term,
                            "ontology": "mondo,efo,doid,hp",
                            "type": "class",
                            "rows": 10,
                            "exact": "false",
                            "fieldList": "iri,label,short_form,obo_id,ontology_name,synonym,description"
                        }
                    )
                    
                    if response.status_code != 200:
                        continue
                    
                    data = response.json()
                    docs = data.get("response", {}).get("docs", [])
                    
                    if not docs:
                        continue
                    
                    # Find best match using similarity
                    best_match = self._find_best_match(term, docs)
                    
                    if best_match and best_match.get("score", 0) > 0.6:
                        logger.info(f"✓ OLS match for '{term}': {best_match['label']} ({best_match['obo_id']})")
                        return best_match
                        
            except Exception as e:
                logger.warning(f"OLS search failed for '{term}': {e}")
                continue
        
        return None
    
    def _find_best_match(self, query: str, docs: List[Dict]) -> Optional[Dict]:
        """Find best matching disease from search results."""
        query_lower = query.lower()
        best_match = None
        best_score = 0.0
        
        for doc in docs:
            label = doc.get("label", "").lower()
            synonyms = doc.get("synonym", [])
            obo_id = doc.get("obo_id", "")
            ontology = doc.get("ontology_name", "")
            
            # Calculate similarity
            label_score = SequenceMatcher(None, query_lower, label).ratio()
            
            # Exact match boost
            if query_lower == label:
                label_score = 1.0
            
            # Check synonyms
            synonym_score = 0.0
            for syn in synonyms:
                syn_lower = syn.lower() if isinstance(syn, str) else ""
                score = SequenceMatcher(None, query_lower, syn_lower).ratio()
                if syn_lower == query_lower:
                    score = 1.0
                synonym_score = max(synonym_score, score)
            
            # Combined score
            score = max(label_score, synonym_score)
            
            # Prefer MONDO and EFO
            if ontology in ["mondo", "efo"]:
                score += 0.1
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = {
                    "label": doc.get("label", ""),
                    "obo_id": obo_id,
                    "iri": doc.get("iri", ""),
                    "ontology_name": ontology,
                    "score": score,
                    "description": doc.get("description", [""])[0] if doc.get("description") else ""
                }
        
        return best_match
    
    async def resolve(self, disease_name: str) -> Optional[Dict]:
        """
        Main resolution pipeline using FREE APIs.
        
        Pipeline:
        1. Get synonyms from NCBI + EBI OLS
        2. Search EBI OLS with all variants
        3. Map to EFO using OxO if needed
        
        Returns:
            {
                "disease_id": "EFO_0000275",
                "disease_name": "Alzheimer's disease",
                "source": "EBI_OLS_MONDO",
                "confidence": 0.95,
                "original_id": "MONDO:0004975"
            }
        """
        logger.info(f"🔍 Resolving disease: '{disease_name}'")
        
        # Step 1: Get all synonyms from APIs
        synonyms = await self.get_all_synonyms(disease_name)
        logger.info(f"📚 Found {len(synonyms)} total search variants")
        
        # Step 2: Search OLS with all variants
        ols_result = await self.search_ols_comprehensive(disease_name, synonyms)
        
        if not ols_result:
            logger.error(f"❌ Could not resolve '{disease_name}' in any ontology")
            return None
        
        obo_id = ols_result.get("obo_id", "")
        ontology = ols_result.get("ontology_name", "")
        
        # Step 3: Convert to EFO if needed using OxO
        final_id = obo_id
        mapping_source = ontology.upper()
        
        if not obo_id.startswith("EFO:"):
            # Try to map to EFO using OxO API
            efo_id = await self.map_ontology_id_oxo(obo_id, "EFO")
            
            if efo_id:
                final_id = efo_id
                mapping_source = f"{ontology.upper()}_to_EFO"
            else:
                # Use original ID (Open Targets accepts MONDO/DOID)
                logger.info(f"⚠️ No EFO mapping found, using {obo_id}")
        
        # Format ID for Open Targets (replace : with _)
        formatted_id = final_id.replace(":", "_")
        
        return {
            "disease_id": formatted_id,
            "disease_name": ols_result["label"],
            "source": mapping_source,
            "confidence": ols_result["score"],
            "original_id": obo_id,
            "description": ols_result.get("description", "")
        }

# Global instance
disease_resolver = DiseaseResolver()
