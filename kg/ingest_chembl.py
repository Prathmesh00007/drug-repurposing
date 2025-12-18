"""Multi-source drug aggregator with OpenTargets primary, ChEMBL fallback."""
import json  # Add this with other imports
from json import JSONDecodeError
import logging
import httpx
import asyncio
import traceback
from typing import List, Dict, Optional, Union, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from agents.base import cache_manager
from kg.therapeutic_area_mapper import classify_disease_therapeutic_area, TherapeuticAreaMapper
from kg.dgidb_graphql import fetch_dgidb_drugs_graphql
from kg.drug_mechanism_explainer import DrugMechanismExplainer


logger = logging.getLogger(__name__)


# ============================================================================
# INDICATION FILTERING (SHARED LOGIC)
# ============================================================================
# ingest_chembl.py - REPLACE _is_relevant_indication function ONLY

def _is_relevant_indication(
    indication: str,
    disease_name: str,
    therapeutic_area: Optional[str],
    disease_keywords: List[str],
    is_rare_disease: bool = False
) -> tuple:  # ✅ CHANGED: Now returns tuple (bool, str)
    """
    ✅ REPURPOSING VERSION: INVERTED LOGIC!
    
    OLD (Discovery): Accept drugs that match disease
    NEW (Repurposing): REJECT drugs that match disease, accept others
    
    Returns:
        tuple: (is_relevant: bool, match_reason: str)
    """
    if not indication:
        # No indication data - be lenient, might be repurposing candidate
        return (True, "no_indication_data")
    
    indication_lower = indication.lower()
    disease_lower = disease_name.lower()
    
    # ✅ REPURPOSING FILTER #1: REJECT direct disease matches
    if disease_lower in indication_lower:
        return (False, "already_treats_disease")
    
    # ✅ REPURPOSING FILTER #2: REJECT disease keyword matches
    if any(keyword in indication_lower for keyword in disease_keywords if len(keyword) > 3):
        return (False, "treats_related_condition")
    
    # ✅ REPURPOSING FILTER #3: ACCEPT different therapeutic areas (cross-domain repurposing!)
    if therapeutic_area:
        area_config = TherapeuticAreaMapper.THERAPEUTIC_AREAS.get(therapeutic_area, {})
        area_keywords = area_config.get("mondo_patterns", [])
        matches = sum(1 for kw in area_keywords if kw in indication_lower and len(kw) > 4)
        
        # If indication matches query disease area TOO closely, reject
        if matches >= 3:
            return (False, "same_therapeutic_area")
        
        # If indication is in DIFFERENT area, accept (repurposing!)
        for other_area, other_config in TherapeuticAreaMapper.THERAPEUTIC_AREAS.items():
            if other_area == therapeutic_area:
                continue
            other_keywords = other_config.get("mondo_patterns", [])
            other_matches = sum(1 for kw in other_keywords if kw in indication_lower and len(kw) > 4)
            if other_matches >= 2:
                return (True, f"cross_domain_from_{other_area}")
    
    # ✅ REPURPOSING FILTER #4: Rare diseases - still lenient
    if is_rare_disease:
        return (True, "rare_disease_lenient")
    
    # Default: Accept (might be repurposing candidate)
    return (True, "potential_repurposing")


# ============================================================================
# RATE LIMITING (Fix Issue #4)
# ============================================================================
_last_chembl_request = 0
_chembl_circuit_breaker_failures = 0
_chembl_circuit_breaker_open = False
CHEMBL_MIN_DELAY = 3.0  # 3 seconds between requests
CIRCUIT_BREAKER_THRESHOLD = 5  # Open after 5 consecutive failures


async def _rate_limit_chembl():
    """Enforce rate limiting for ChEMBL API."""
    global _last_chembl_request
    now = asyncio.get_event_loop().time()
    elapsed = now - _last_chembl_request
    
    if elapsed < CHEMBL_MIN_DELAY:
        delay = CHEMBL_MIN_DELAY - elapsed
        logger.debug(f"⏱️ Rate limiting ChEMBL: waiting {delay:.1f}s")
        await asyncio.sleep(delay)
    
    _last_chembl_request = asyncio.get_event_loop().time()


def _check_circuit_breaker() -> bool:
    """Check if ChEMBL circuit breaker is open."""
    global _chembl_circuit_breaker_open, _chembl_circuit_breaker_failures
    
    if _chembl_circuit_breaker_failures >= CIRCUIT_BREAKER_THRESHOLD:
        if not _chembl_circuit_breaker_open:
            logger.warning(f"🔴 ChEMBL circuit breaker OPEN ({_chembl_circuit_breaker_failures} failures)")
            _chembl_circuit_breaker_open = True
        return False
    return True


def _record_chembl_success():
    """Record successful ChEMBL request."""
    global _chembl_circuit_breaker_failures, _chembl_circuit_breaker_open
    _chembl_circuit_breaker_failures = 0
    _chembl_circuit_breaker_open = False


def _record_chembl_failure():
    """Record failed ChEMBL request."""
    global _chembl_circuit_breaker_failures
    _chembl_circuit_breaker_failures += 1


# Normalize both
def normalize_phase(value):
    if value in (None, "", "null"):
        return 0
    try:
        return int(value)
    except:
        return 0


# ============================================================================
# PRIMARY SOURCE: OpenTargets Drugs (Most Reliable)
# ============================================================================
# ingest_chembl.py - REPLACE _fetch_opentargets_drugs function COMPLETELY

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_opentargets_drugs(
    ensembl_ids: List[str],
    disease_name: str,
    min_phase: int = 1
) -> List[Dict]:
    """
    ✅ FIXED: Fetch drugs from OpenTargets Platform API (PRIMARY SOURCE).
    
    NOW QUERIES THE CORRECT ENDPOINT:
    - OLD (WRONG): disease.associatedTargets (8,619 rows, 2-3 minutes)
    - NEW (CORRECT): targets.knownDrugs (50 targets × 20 drugs = 1,000 rows, 10 seconds)
    
    ALSO INCLUDES REPURPOSING FILTER!
    """
    # ✅ CORRECT QUERY: targets → knownDrugs (NOT disease → associatedTargets!)
    query = """
    query KnownDrugsQuery($ensemblIds: [String!]!) {
      targets(ensemblIds: $ensemblIds) {
        id
        approvedSymbol
        knownDrugs(size: 100) {
          count
          rows {
            drug {
              id
              name
              drugType
              maximumClinicalTrialPhase
              isApproved
            }
            mechanismOfAction
            phase
            status
            targetClass
            disease {
              id
              name
            }
          }
        }
      }
    }
    """
    
    if min_phase is None:
        min_phase = 1
    
    try:
        # ✅ STEP 1: Classify therapeutic area (for repurposing filter)
        logger.info(f"🔍 Classifying therapeutic area for '{disease_name}'...")
        therapeutic_area = await classify_disease_therapeutic_area(disease_name)
        if therapeutic_area:
            logger.info(f"✓ Therapeutic area: {therapeutic_area}")
        else:
            logger.warning(f"⚠️ Could not classify therapeutic area")
        
        disease_keywords = [w.lower() for w in disease_name.split() if len(w) > 3]
        logger.info(f"✓ Disease keywords: {disease_keywords}")
        
        # ✅ STEP 2: Query OpenTargets (CORRECT ENDPOINT!)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.platform.opentargets.org/api/v4/graphql",
                json={"query": query, "variables": {"ensemblIds": ensembl_ids}},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.warning(f"OpenTargets API returned {response.status_code}")
                return []
            
            data = response.json()
            targets = data.get("data", {}).get("targets", [])
            
            all_drugs = []  # Before filtering
            filtered_drugs = []  # After repurposing filter
            
            for target in targets:
                symbol = target.get("approvedSymbol", "UNKNOWN")
                known_drugs = target.get("knownDrugs", {}).get("rows", [])
                
                target_drug_list = []  

                for drug_row in known_drugs:
                    drug = drug_row.get("drug", {})
                    
                    # Normalize phase
                    raw_phase = drug.get("maximumClinicalTrialPhase")
                    row_phase = drug_row.get("phase")
                    phase = normalize_phase(raw_phase)
                    row_phase_norm = normalize_phase(row_phase)
                    phase = max(phase, row_phase_norm)
                    
                    if not isinstance(phase, int):
                        phase = 0
                    
                    # Phase filter
                    if phase < min_phase:
                        continue
                    
                    # Extract indication (what the drug treats)
                    disease_obj = drug_row.get("disease", {})
                    indication = disease_obj.get("name", "")
                    
                    drug_obj = {
                        "source": "opentargets",
                        "target_symbol": symbol,
                        "target_id": target.get("id"),
                        "drug_id": drug.get("id"),
                        "drug_name": drug.get("name", "Unknown"),
                        "mechanism": drug_row.get("mechanismOfAction", "Unknown"),
                        "phase": phase,
                        "approved": drug.get("isApproved", False),
                        "indication": indication,
                        "target_class": drug_row.get("targetClass", [])
                    }
                    
                    all_drugs.append(drug_obj)
                    
                    # ✅ STEP 3: Apply REPURPOSING filter
                    is_relevant, match_reason = _is_relevant_indication(
                        indication=indication,
                        disease_name=disease_name,
                        therapeutic_area=therapeutic_area,
                        disease_keywords=disease_keywords,
                        is_rare_disease=False
                    )

                    # ✅ This now works because function returns tuple!
                    if is_relevant:
                        target_drug_list.append(drug_obj) 
                        logger.debug(f"  ✓ Accept ({match_reason}): {drug.get('drug_name')} - {indication}")
                    else:
                        logger.debug(f"  ❌ Reject ({match_reason}): {drug.get('drug_name')} - {indication}")

                    target_drug_list.sort(key=lambda d: d["phase"], reverse=True)
                    filtered_drugs.extend(target_drug_list[:15])

            
            # ✅ STEP 4: Log results
            logger.info(
                f"✅ OpenTargets: {len(filtered_drugs)}/{len(all_drugs)} drugs passed repurposing filter "
                f"from {len(targets)} targets"
            )
            
            if len(all_drugs) > 0 and len(filtered_drugs) < len(all_drugs) * 0.1:
                logger.warning(
                    f"⚠️ Heavy filtering! {len(filtered_drugs)}/{len(all_drugs)} drugs kept. "
                    f"This is expected for repurposing (we filter OUT drugs already treating the disease)"
                )
            
            return filtered_drugs
    
    except Exception as e:
        logger.error(f"❌ OpenTargets drugs failed: {e}")
        logger.error(traceback.format_exc())
        return []


# ============================================================================
# SECONDARY SOURCE: DGIdb (Batch Gene-Drug Interactions)
# ============================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_dgidb_drugs(
    gene_symbols: List[str],
    min_phase: int = 1
) -> List[Dict]:
    """
    Fetch drug-gene interactions from DGIdb GraphQL API.
    ✅ FIXED: Now uses GraphQL instead of deprecated REST API
    """
    return await fetch_dgidb_drugs_graphql(gene_symbols, min_phase)


# ============================================================================
# TERTIARY SOURCE: ChEMBL (Fallback Only)
# ============================================================================
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=4, max=15))
async def _fetch_chembl_drugs(
    gene_symbol: str,
    disease_name: str,
    therapeutic_area: Optional[str],
    min_phase: int = 1
) -> List[Dict]:
    """
    ChEMBL fallback (TERTIARY SOURCE - use only for gaps).
    
    FIXED ISSUES:
    - Issue #4: Rate limiting added
    - Issue #5: Lenient mode logic fixed
    - Issue #7: Full exception logging
    """
    if not _check_circuit_breaker():
        logger.warning(f"⚠️ ChEMBL circuit breaker open, skipping {gene_symbol}")
        return []
    
    try:
        await _rate_limit_chembl()
        
        # Search for target
        async with httpx.AsyncClient(timeout=15.0) as client:
            search_resp = await client.get(
                "https://www.ebi.ac.uk/chembl/api/data/target/search.json",
                params={"q": gene_symbol, "limit": 3}
            )
            
            if search_resp.status_code != 200:
                _record_chembl_failure()
                return []
            
            targets = search_resp.json().get("targets", [])
            if not targets:
                return []
            
            chembl_id = targets[0].get("target_chembl_id")
            
            # Get mechanisms
            await _rate_limit_chembl()
            mech_resp = await client.get(
                "https://www.ebi.ac.uk/chembl/api/data/mechanism.json",
                params={"target_chembl_id": chembl_id, "limit": 30}
            )
            
            if mech_resp.status_code != 200:
                _record_chembl_failure()
                return []
            
            mechanisms = mech_resp.json().get("mechanisms", [])
            drugs = []
            
            disease_keywords = [w.lower() for w in disease_name.split() if len(w) > 3]
            
            for mech in mechanisms:
                molecule_id = mech.get("molecule_chembl_id")
                if not molecule_id:
                    continue
                
                # Get molecule details
                await _rate_limit_chembl()
                mol_resp = await client.get(
                    f"https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_id}.json"
                )
                
                if mol_resp.status_code != 200:
                    continue
                
                mol_data = mol_resp.json()
                phase = mol_data.get("max_phase", 0)
                
                if phase < min_phase:
                    continue
                
                # Use shared filtering logic
                indications = mol_data.get("drug_indications", [])
                relevant = False
                matched_indication = ""
                
                if indications:
                    # Has indications - check relevance
                    for ind in indications:
                        ind_text = ind.get("indication", "")
                        
                        if _is_relevant_indication(
                            ind_text,
                            disease_name,
                            therapeutic_area,
                            disease_keywords
                        ):
                            relevant = True
                            matched_indication = ind_text
                            break
                else:
                    # NO INDICATIONS - LENIENT MODE
                    # Accept if phase >= 1 (has clinical data) OR approved
                    if phase >= 1 or phase == 4:
                        relevant = True
                        matched_indication = f"Clinical phase {phase} compound"
                
                if relevant:
                    drugs.append({
                        "source": "chembl",
                        "target_symbol": gene_symbol,
                        "drug_id": molecule_id,
                        "drug_name": mol_data.get("pref_name", "Unknown"),
                        "mechanism": mech.get("mechanism_of_action", "Unknown"),
                        "phase": phase,
                        "approved": phase == 4,
                        "indication": matched_indication
                    })
            
            _record_chembl_success()
            logger.info(f"✅ ChEMBL: {len(drugs)} drugs for {gene_symbol}")
            return drugs
            
    except Exception as e:
        _record_chembl_failure()
        logger.error(f"❌ ChEMBL failed for {gene_symbol}: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return []


# ============================================================================
# MAIN AGGREGATOR: Multi-Source with Fallback Chain
# ============================================================================
async def ingest_drugs_multisource(
    targets: List[Union[Dict, str]],
    neo4j_client,
    disease_name: str,
    disease_context=None, 
    include_clinical_candidates: bool = True,
    min_phase: int = 1,
    enable_reasoning: bool = True 
) -> List[Dict[str, Any]]:
    """
    ROBUST multi-source drug ingestion with fallback chain.
    
    Strategy:
    1. PRIMARY: OpenTargets (batch query, most reliable) - NOW WITH FILTERING! ✅
    2. SECONDARY: DGIdb (batch query, fills gaps)
    3. TERTIARY: ChEMBL (per-target, fallback only)
    
    Returns:
        Number of compounds ingested
    """
    logger.info(f"🚀 Multi-source drug ingestion: {len(targets)} targets for {disease_name}")
    
    # Parse targets
    gene_symbols = []
    ensembl_ids = []
    target_map = {}  # symbol -> ensembl_id
    
    for t in targets:
        if isinstance(t, str):
            gene_symbols.append(t)
        else:
            symbol = t.get("symbol", "")
            ensembl = t.get("ensembl_id", "")
            gene_symbols.append(symbol)
            if ensembl:
                ensembl_ids.append(ensembl)
                target_map[symbol] = ensembl
    
    all_drugs = []
    covered_targets = set()
    
    # ===== STAGE 1: OpenTargets (Primary) - NOW WITH FILTERING! =====
    if ensembl_ids:
        logger.info(f"📊 Stage 1: Querying OpenTargets for {len(ensembl_ids)} targets...")
        ot_drugs = await _fetch_opentargets_drugs(ensembl_ids, disease_name, min_phase)
        all_drugs.extend(ot_drugs)
        covered_targets.update({d["target_symbol"] for d in ot_drugs})
        logger.info(f"✓ OpenTargets covered {len(covered_targets)} targets with {len(ot_drugs)} drugs")
    
    # ===== STAGE 2: DGIdb (Secondary) =====
    missing = [s for s in gene_symbols if s not in covered_targets]
    if missing:
        logger.info(f"📊 Stage 2: Querying DGIdb for {len(missing)} remaining targets...")
        dgidb_drugs = await _fetch_dgidb_drugs(missing, min_phase)
        all_drugs.extend(dgidb_drugs)
        covered_targets.update({d["target_symbol"] for d in dgidb_drugs})
        logger.info(f"✓ DGIdb covered {len({d['target_symbol'] for d in dgidb_drugs})} additional targets")
    
    # ===== STAGE 3: ChEMBL (Tertiary Fallback) - OPTIONAL =====
    # Uncomment if you want ChEMBL as ultimate fallback
    # missing = [s for s in gene_symbols if s not in covered_targets]
    # if missing and _check_circuit_breaker():
    #     logger.info(f"📊 Stage 3: Querying ChEMBL for {len(missing)} remaining targets...")
    #     # ... ChEMBL code ...

    if enable_reasoning and disease_context:
        logger.info(f"💊 Generating mechanism explanations for {len(unique_drugs)} drugs...")
        
        explainer = DrugMechanismExplainer()
        
        for drug in unique_drugs[:10]:  # Limit to top 10 to save API calls
            try:
                mechanism_explanation = await explainer.explain_drug_mechanism(
                    drug_name=drug.get("drug_name"),
                    drug_moa=drug.get("mechanism", "Unknown"),
                    drug_phase=str(drug.get("phase", 0)),
                    target_symbol=drug.get("target_symbol"),
                    target_pathways="...",  # Get from pathway integrator
                    disease_name=disease_name,
                    disease_mechanism=disease_context.description
                )
                
                # Store in Neo4j
                neo4j_client.add_drug_mechanism(
                    drug_id=drug.get("drug_id"),
                    mechanism_json=json.dumps(mechanism_explanation)
                )
                
                logger.info(f"   ✓ {drug['drug_name']}: Mechanism explained")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Could not explain {drug['drug_name']}: {e}")
    
    # ===== Deduplicate and Ingest to Neo4j =====
    seen = set()
    unique_drugs = []
    
    for drug in all_drugs:
        key = (drug.get("drug_name", ""), drug.get("target_symbol", ""))
        if key not in seen:
            seen.add(key)
            unique_drugs.append(drug)
    
    logger.info(f"📊 Total unique drugs: {len(unique_drugs)} (from {len(all_drugs)} raw)")
    
    candidates_data = []
    # Create Neo4j nodes
    for drug in unique_drugs:
        phase = drug.get("phase", 0)
        stage = (
            "approved" if phase == 4 else
            "clinical" if phase in (1, 2, 3) else
            "preclinical" if phase == 0 else
            "unknown"
        )

        candidates_data.append({
            "candidate_id": drug.get("drug_id", drug.get("drug_name")),
            "name": drug.get("drug_name", "Unknown"),
            "stage": stage,
            "source": drug.get("source", ""),
            "target_symbol": drug.get("target_symbol", ""),
            "mechanism": drug.get("mechanism", "modulates"),
            "indication": drug.get("indication", "N/A")
        })
        
        # ✅ Use create_candidate_target_modulation
    neo4j_client.batch_create_candidates(candidates_data)
        
        
        
    logger.info(f"✅ Multi-source ingestion complete: {len(unique_drugs)} compounds")
    return unique_drugs


# Backward compatibility alias
ingest_chembl_candidates = ingest_drugs_multisource
