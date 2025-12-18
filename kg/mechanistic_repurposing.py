"""
Mechanistic Drug Repurposing Engine
====================================

TRUE drug repurposing using mechanism-first approach:
1. Drug → Target → Pathway → Disease chain
2. Filters OUT drugs already treating query disease
3. Generates experimental validation plans
4. Assesses safety and contraindications

Example output:
    Drug: Metformin
    Target: AMPK (PRKAA1/2)
    Original indication: Type 2 diabetes
    Proposed indication: Cancer
    Mechanism: AMPK activation → mTOR inhibition → reduced tumor growth
    Evidence: Epidemiology (lower cancer rates in diabetics on metformin)
    Experiments: Cell viability, xenograft models, biomarker analysis
"""

import logging
import httpx
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential

from kg.pathway_integrator import PathwayIntegrator

logger = logging.getLogger(__name__)


@dataclass
class RepurposingCandidate:
    """Structured repurposing candidate with mechanistic rationale."""
    # Drug identification
    drug_name: str
    drug_id: str
    phase: int
    drug_type: str  # small molecule, antibody, etc.
    
    # Target information
    molecular_target: str  # Gene symbol (e.g., "PRKAA1")
    target_protein: str    # Protein name (e.g., "AMPK alpha-1")
    target_ensembl_id: str
    
    # Repurposing rationale
    original_indication: str
    proposed_indication: str
    mechanism_of_action: str
    
    # Mechanistic explanation
    disease_pathway_link: str  # How target relates to disease
    shared_pathways: List[str]  # Specific pathways
    pathway_overlap_score: float  # 0-1
    
    # Evidence
    opentargets_score: float  # Target-disease association score
    clinical_phase_original: int  # Phase for original indication
    preclinical_evidence: List[str]  # From literature
    epidemiological_signal: Optional[str]  # e.g., "Lower cancer rates in diabetics"
    
    # Experimental validation plan
    in_vitro_experiments: List[str]
    in_vivo_experiments: List[str]
    biomarkers_to_measure: List[str]
    
    # Safety assessment
    safety_concerns: List[str]
    contraindications: List[str]
    pharmacokinetic_considerations: List[str]
    
    # Scoring
    mechanistic_confidence: float  # 0-1
    novelty_score: float  # 0-100
    repurposing_feasibility: str  # "HIGH", "MEDIUM", "LOW"


class MechanisticRepurposingEngine:
    """
    Core engine for mechanism-based drug repurposing.
    
    This does TRUE repurposing:
    1. Identify disease-associated targets/pathways
    2. Find drugs modulating those targets
    3. Filter out drugs already treating the disease
    4. Build mechanistic rationale for new use
    5. Design experimental validation plan
    """
    
    def __init__(self):
        self.pathway_integrator = PathwayIntegrator()
        self._gene_cache = {}  # Cache gene symbol → Ensembl ID mapping

    async def _process_single_target(
        self,
        idx: int,
        target: dict,
        disease_name: str,
        disease_pathways: list,
        therapeutic_area: str,
        min_phase: int
    ) -> list:
        """Process a single target and return its drug candidates."""
        candidates = []
        
        target_symbol = target.get("symbol")
        target_ensembl = target.get("ensembl_id") or target.get("ensemblid")
        target_score = target.get("opentargets_score", 0.0)
        
        logger.info(f"   [{idx}/30] Target: {target_symbol} (score: {target_score:.3f})")
        
        # Fetch pathway context for this target
        target_pathways = await self.pathway_integrator.get_target_pathways(
            gene_symbol=target_symbol
        )
        target_pathway_ids = [p["pathway_id"] for p in target_pathways]
        
        # Calculate pathway overlap with disease
        pathway_overlap = await self.pathway_integrator.find_pathway_overlap(
            disease_pathways=disease_pathways,
            target_pathways=target_pathway_ids
        )
        
        jaccard = pathway_overlap.get("jaccard_similarity", 0.0)
        overlap_pathways = pathway_overlap.get("overlap_pathways", [])
        
        logger.info(f"      Pathway overlap: {jaccard:.2%} ({len(overlap_pathways)} shared pathways)")
        
        # Query OpenTargets for drugs targeting this protein
        drugs_for_target = await self._fetch_drugs_for_target(
            ensembl_id=target_ensembl,
            min_phase=min_phase
        )
        
        logger.info(f"      Found {len(drugs_for_target)} drugs targeting {target_symbol}")
        
        # STEP 2: Filter and classify drugs
        for drug in drugs_for_target:
            # ❌ Skip if drug already treats query disease
            if self._drug_treats_disease(drug["indication"], disease_name):
                logger.debug(f"      ❌ {drug['name']}: Already treats {disease_name}")
                continue
            
            # ✅ This is a repurposing candidate!
            #logger.info(f"      ✅ {drug['name']}: REPURPOSING CANDIDATE!")
            #logger.info(f"         Original: {drug['indication']}")
            #logger.info(f"         Phase: {drug['phase']}")
            #logger.info(f"         Mechanism: {drug['mechanism']}")
            
            # STEP 3: Build mechanistic rationale
            try:
                candidate = await self._build_mechanistic_candidate(
                    drug=drug,
                    target_symbol=target_symbol,
                    target_ensembl=target_ensembl,
                    target_score=target_score,
                    original_indication=drug["indication"],
                    proposed_disease=disease_name,
                    disease_pathways=disease_pathways,
                    shared_pathways=overlap_pathways,
                    pathway_overlap_score=jaccard,
                    therapeutic_area=therapeutic_area
                )
                
                if candidate:
                    candidates.append(candidate)
                    logger.info(f"         Confidence: {candidate.mechanistic_confidence:.1%}")
            
            except Exception as e:
                logger.warning(f"      Failed to build candidate for {drug['name']}: {e}")
                continue
        
        return candidates
    
    async def find_repurposing_candidates(
        self,
        disease_name: str,
        disease_id: str,
        disease_targets: List[Dict],
        disease_pathways: List[str],
        therapeutic_area: Optional[str] = None,
        min_phase: int = 1,
        top_n: int = 50
    ) -> List[RepurposingCandidate]:
        """
        Find repurposing candidates using mechanism-first approach.
        
        Args:
            disease_name: Human-readable disease name (e.g., "breast cancer")
            disease_id: EFO/MONDO disease ID
            disease_targets: Validated targets from OpenTargets
            disease_pathways: Pathway IDs implicated in disease
            therapeutic_area: Therapeutic area (for safety assessment)
            min_phase: Minimum clinical phase (default: Phase 1)
            top_n: Maximum candidates to return
            
        Returns:
            List of mechanistic repurposing candidates with full rationale
        """
        logger.info(f"🔬 MECHANISTIC REPURPOSING for {disease_name}")
        logger.info(f"   Disease ID: {disease_id}")
        logger.info(f"   Analyzing {len(disease_targets)} disease targets")
        logger.info(f"   Analyzing {len(disease_pathways)} disease pathways")
        logger.info(f"   Minimum phase: {min_phase}")
           
        
        tasks = [
            self._process_single_target(
                idx=idx,
                target=target,
                disease_name=disease_name,
                disease_pathways=disease_pathways,
                therapeutic_area=therapeutic_area,
                min_phase=min_phase
            )
            for idx, target in enumerate(disease_targets[:5], 1)
        ]
        
        # STEP 1: For each disease target, find ALL drugs (regardless of indication)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        candidates = []
        for idx, result in enumerate(results, 1):
            if isinstance(result, BaseException):
                logger.warning(f"   Target {idx} failed: {repr(result)}")
                continue
            
            # CHECK 2: Ensure it is actually a list before extending
            if isinstance(result, list):
                candidates.extend(result)
            else:
                logger.warning(f"   Target {idx} returned unexpected type: {type(result)}")

        # STEP 4: Rank and filter
        candidates.sort(key=lambda c: (
            c.mechanistic_confidence * 0.35 +
            c.pathway_overlap_score * 0.2 +
            c.opentargets_score * 0.35 +
            (c.phase / 4.0) * 0.1
        ), reverse=True)

        top_candidates = candidates[:top_n]

        logger.info(f"✅ MECHANISTIC REPURPOSING COMPLETE")
        logger.info(f"   Found {len(candidates)} total candidates")
        logger.info(f"   Returning top {len(top_candidates)} candidates")

        return top_candidates
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch_drugs_for_target(
        self,
        ensembl_id: str,
        min_phase: int
    ) -> List[Dict]:
        """
        Fetch ALL drugs targeting a protein (no disease filter).
        
        This is the key difference from discovery: we want drugs for ANY disease,
        then we'll filter OUT the ones already treating our query disease.
        """
        query = """
        query TargetDrugs($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            approvedSymbol
            approvedName
            knownDrugs(size: 10) {
              rows {
                drug {
                  id
                  name
                  drugType
                  maximumClinicalTrialPhase
                  isApproved
                }
                disease {
                  id
                  name
                }
                mechanismOfAction
                phase
              }
            }
          }
        }
        """
        
        if not min_phase:
            min_phase = 1
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.platform.opentargets.org/api/v4/graphql",
                    json={"query": query, "variables": {"ensemblId": ensembl_id}}
                )
                
                if response.status_code != 200:
                    logger.warning(f"OpenTargets API returned {response.status_code}")
                    return []
                
                data = response.json()
                target_data = data.get("data", {}).get("target", {})
                
                if not target_data:
                    return []
                
                rows = target_data.get("knownDrugs", {}).get("rows", [])
                
                drugs = []
                for row in rows:
                    drug_data = row.get("drug", {})
                    disease_data = row.get("disease", {})
                    
                    # Normalize phase
                    raw_phase = drug_data.get("maximumClinicalTrialPhase")
                    row_phase = row.get("phase")
                    
                    phase = self._normalize_phase(raw_phase)
                    row_phase_norm = self._normalize_phase(row_phase)
                    phase = max(phase, row_phase_norm)
                    
                    if not isinstance(phase, int):
                        phase = 0
                    
                    # Phase filter
                    if phase < min_phase:
                        continue
                    
                    drugs.append({
                        "id": drug_data.get("id"),
                        "name": drug_data.get("name", "Unknown"),
                        "drug_type": drug_data.get("drugType", "Unknown"),
                        "phase": phase,
                        "approved": drug_data.get("isApproved", False),
                        "indication": disease_data.get("name", "Unknown indication"),
                        "indication_id": disease_data.get("id", ""),
                        "mechanism": row.get("mechanismOfAction", "Unknown mechanism")
                    })
                
                return drugs
        
        except Exception as e:
            logger.error(f"Failed to fetch drugs for {ensembl_id}: {e}")
            return []
    
    def _drug_treats_disease(self, indication: str, query_disease: str) -> bool:
        """
        Check if drug already treats the query disease (strict matching).
        
        Returns True only if there's strong evidence the drug is already
        used for this disease (not a repurposing candidate).
        """
        if not indication or indication in ["Unknown indication", "Unknown", ""]:
            return False  # No indication = potential repurposing
        
        indication_lower = indication.lower()
        disease_lower = query_disease.lower()
        
        # Exact substring match
        if disease_lower in indication_lower:
            return True
        
        # Check for significant word overlap (at least 2 key words)
        disease_words = set(w for w in disease_lower.split() if len(w) > 3)
        indication_words = set(w for w in indication_lower.split() if len(w) > 3)
        
        overlap = disease_words.intersection(indication_words)
        
        if len(overlap) >= 2:
            return True
        
        return False
    
    async def _build_mechanistic_candidate(
        self,
        drug: Dict,
        target_symbol: str,
        target_ensembl: str,
        target_score: float,
        original_indication: str,
        proposed_disease: str,
        disease_pathways: List[str],
        shared_pathways: List[str],
        pathway_overlap_score: float,
        therapeutic_area: Optional[str]
    ) -> Optional[RepurposingCandidate]:
        """
        Build mechanistic repurposing candidate with full rationale.
        
        This generates the "Metformin → AMPK → Cancer" style explanation.
        """
        drug_name = drug["name"]
        drug_id = drug["id"]
        phase = drug["phase"]
        drug_type = drug["drug_type"]
        mechanism = drug["mechanism"]
        
        # STEP 1: Build mechanistic explanation
        mechanistic_link = self._explain_target_disease_link(
            drug_name=drug_name,
            target_symbol=target_symbol,
            mechanism=mechanism,
            disease=proposed_disease,
            shared_pathways=shared_pathways,
            pathway_overlap=pathway_overlap_score
        )
        
        # STEP 2: Design experimental validation
        experiments = self._design_validation_experiments(
            drug_name=drug_name,
            target_symbol=target_symbol,
            disease=proposed_disease,
            phase=phase
        )
        
        # STEP 3: Assess safety for repurposing
        safety = self._assess_repurposing_safety(
            drug_name=drug_name,
            drug_type=drug_type,
            original_indication=original_indication,
            proposed_indication=proposed_disease,
            therapeutic_area=therapeutic_area,
            current_phase=phase
        )
        
        # STEP 4: Calculate confidence scores
        mechanistic_confidence = self._calculate_mechanistic_confidence(
            pathway_overlap=pathway_overlap_score,
            target_score=target_score,
            phase=phase,
            mechanism_known=(mechanism != "Unknown mechanism")
        )
        
        # STEP 5: Assess feasibility
        feasibility = self._assess_feasibility(
            phase=phase,
            pathway_overlap=pathway_overlap_score,
            safety_concerns=len(safety["concerns"])
        )
        
        return RepurposingCandidate(
            # Drug identification
            drug_name=drug_name,
            drug_id=drug_id,
            phase=phase,
            drug_type=drug_type,
            
            # Target information
            molecular_target=target_symbol,
            target_protein=f"{target_symbol} protein",  # Simplified
            target_ensembl_id=target_ensembl,
            
            # Repurposing rationale
            original_indication=original_indication,
            proposed_indication=proposed_disease,
            mechanism_of_action=mechanism,
            
            # Mechanistic explanation
            disease_pathway_link=mechanistic_link["explanation"],
            shared_pathways=mechanistic_link["pathway_names"],
            pathway_overlap_score=pathway_overlap_score,
            
            # Evidence
            opentargets_score=target_score,
            clinical_phase_original=phase,
            preclinical_evidence=[],  # Filled by literature agent
            epidemiological_signal=None,  # Filled by literature agent
            
            # Experimental validation
            in_vitro_experiments=experiments["in_vitro"],
            in_vivo_experiments=experiments["in_vivo"],
            biomarkers_to_measure=experiments["biomarkers"],
            
            # Safety
            safety_concerns=safety["concerns"],
            contraindications=safety["contraindications"],
            pharmacokinetic_considerations=safety["pk_considerations"],
            
            # Scoring
            mechanistic_confidence=mechanistic_confidence,
            novelty_score=100.0,  # High novelty (new indication)
            repurposing_feasibility=feasibility
        )
    
    def _explain_target_disease_link(
        self,
        drug_name: str,
        target_symbol: str,
        mechanism: str,
        disease: str,
        shared_pathways: List[str],
        pathway_overlap: float
    ) -> Dict:
        """
        Generate mechanistic explanation linking drug → target → disease.
        
        Example output:
        "Metformin activates AMPK, which inhibits mTOR signaling. This pathway
        is dysregulated in cancer, where mTOR drives tumor growth. By activating
        AMPK, metformin may suppress cancer cell proliferation."
        """
        if pathway_overlap >= 0.3 and shared_pathways:
            pathway_names = [p.replace("R-HSA-", "").replace("_", " ") for p in shared_pathways[:3]]
            
            explanation = (
                f"{drug_name} modulates {target_symbol} via {mechanism}. "
                f"This target is implicated in {disease} through {len(shared_pathways)} shared biological pathways, "
                f"including: {', '.join(pathway_names[:2])}. "
                f"The {pathway_overlap:.0%} pathway overlap suggests strong mechanistic relevance. "
                f"Targeting {target_symbol} may disrupt disease-driving processes in {disease}."
            )
        else:
            explanation = (
                f"{drug_name} modulates {target_symbol} via {mechanism}. "
                f"While pathway overlap is limited ({pathway_overlap:.0%}), "
                f"{target_symbol} is associated with {disease} and may represent a novel therapeutic angle."
            )
        
        return {
            "explanation": explanation,
            "pathway_names": [p.replace("R-HSA-", "").replace("_", " ") for p in shared_pathways[:5]],
            "confidence": min(pathway_overlap * 1.5, 1.0)
        }
    
    def _design_validation_experiments(
        self,
        drug_name: str,
        target_symbol: str,
        disease: str,
        phase: int
    ) -> Dict:
        """
        Design experimental validation plan (like Metformin → Cancer example).
        
        Returns structured experimental plan with:
        - In vitro experiments
        - In vivo experiments
        - Biomarkers to measure
        """
        # In vitro experiments
        in_vitro = [
            f"Cell viability assay: Treat {disease}-relevant cell lines with {drug_name} at therapeutic concentrations",
            f"Mechanism validation: Measure {target_symbol} activity (Western blot, ELISA) after {drug_name} treatment",
            f"Functional assays: Assess cell proliferation, apoptosis, migration in disease models",
            f"Dose-response: Determine IC50 and optimal concentration range"
        ]
        
        if phase >= 4:  # Approved drug
            in_vitro.append(
                f"Combination studies: Test {drug_name} synergy with standard-of-care {disease} treatments"
            )
        
        # In vivo experiments
        in_vivo = []
        
        if phase >= 2:  # Has safety data
            in_vivo = [
                f"Animal efficacy: Test {drug_name} in {disease} xenograft or syngeneic models",
                f"Pharmacodynamics: Measure {target_symbol} modulation in tumor/tissue biopsies",
                f"Dosing optimization: Determine optimal dose and schedule for {disease} indication",
                f"Survival benefit: Assess impact on disease progression and survival"
            ]
        else:
            in_vivo = [
                f"Preclinical safety: Assess {drug_name} toxicity in relevant animal models before {disease} studies",
                f"Proof-of-concept: Single-arm efficacy study in {disease} animal model"
            ]
        
        # Biomarkers
        biomarkers = [
            f"Phospho-{target_symbol} (target engagement)",
            f"Downstream pathway markers (e.g., p-S6K, p-4EBP1 if mTOR pathway)",
            f"{disease} progression biomarkers (tumor markers, imaging)",
            "Pharmacokinetic markers (drug levels in plasma/tissue)"
        ]
        
        return {
            "in_vitro": in_vitro,
            "in_vivo": in_vivo,
            "biomarkers": biomarkers
        }
    
    def _assess_repurposing_safety(
        self,
        drug_name: str,
        drug_type: str,
        original_indication: str,
        proposed_indication: str,
        therapeutic_area: Optional[str],
        current_phase: int
    ) -> Dict:
        """
        Assess safety concerns for repurposing to new indication.
        
        Considers:
        - Drug-disease interactions
        - Therapeutic area mismatch
        - Known toxicities
        - PK/PD considerations
        """
        concerns = []
        contraindications = []
        pk_considerations = []
        
        # Phase-based safety assessment
        if current_phase < 2:
            concerns.append("Limited safety data in humans")
            pk_considerations.append("Human PK/PD not well characterized")
        elif current_phase >= 4:
            pk_considerations.append(
                f"Approved drug with known PK profile - dose may need adjustment for {proposed_indication}"
            )
        
        # Therapeutic area considerations
        original_lower = original_indication.lower()
        proposed_lower = proposed_indication.lower()
        
        # Cancer repurposing
        if "cancer" in proposed_lower or "tumor" in proposed_lower:
            if "diabetes" in original_lower or "metabolic" in original_lower:
                concerns.append("Monitor for metabolic disturbances in cancer patients")
            if "cardiovascular" in original_lower or "heart" in original_lower:
                concerns.append("Monitor for cardiotoxicity (may be additive with chemotherapy)")
        
        # Cardiovascular repurposing
        if "cardio" in proposed_lower or "heart" in proposed_lower:
            if "cancer" in original_lower:
                contraindications.append("Many cancer drugs are cardiotoxic - careful monitoring required")
        
        # Immunology considerations
        if "immune" in original_lower or "autoimmune" in original_lower:
            if "infection" in proposed_lower or "sepsis" in proposed_lower:
                contraindications.append("Immunosuppression contraindicated in infectious diseases")
        
        # Biologics require special consideration
        if drug_type in ["Antibody", "Protein"]:
            concerns.append("Biologic drug - immunogenicity and dosing may differ for new indication")
            pk_considerations.append("Antibody PK may vary across indications due to target expression differences")
        
        # Small molecules - PK considerations
        if drug_type == "Small molecule":
            pk_considerations.append(
                f"Small molecule with predictable PK - existing formulation may be suitable for {proposed_indication}"
            )
        
        return {
            "concerns": concerns,
            "contraindications": contraindications,
            "pk_considerations": pk_considerations
        }
    
    def _calculate_mechanistic_confidence(
        self,
        pathway_overlap: float,
        target_score: float,
        phase: int,
        mechanism_known: bool
    ) -> float:
        """
        Calculate mechanistic confidence score (0-1).
        
        Higher confidence = stronger mechanistic rationale
        """
        # Pathway overlap (40% weight)
        pathway_score = pathway_overlap * 0.4
        
        # Target-disease association (30% weight)
        target_score_norm = min(target_score, 1.0) * 0.3
        
        # Clinical validation (20% weight)
        phase_score = (phase / 4.0) * 0.2
        
        # Mechanism understanding (10% weight)
        mechanism_score = 0.1 if mechanism_known else 0.05
        
        confidence = pathway_score + target_score_norm + phase_score + mechanism_score
        
        return min(confidence, 1.0)
    
    def _assess_feasibility(
        self,
        phase: int,
        pathway_overlap: float,
        safety_concerns: int
    ) -> str:
        """
        Assess repurposing feasibility: HIGH, MEDIUM, or LOW.
        
        HIGH: Approved drug, strong mechanism, low safety concerns
        MEDIUM: Phase 2+, moderate mechanism
        LOW: Early phase or weak mechanism
        """
        score = 0
        
        # Phase contribution
        if phase == 4:
            score += 40
        elif phase == 3:
            score += 30
        elif phase == 2:
            score += 20
        else:
            score += 10
        
        # Mechanism contribution
        if pathway_overlap >= 0.4:
            score += 40
        elif pathway_overlap >= 0.2:
            score += 25
        else:
            score += 10
        
        # Safety contribution
        if safety_concerns == 0:
            score += 20
        elif safety_concerns <= 2:
            score += 10
        else:
            score += 0
        
        if score >= 70:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _normalize_phase(self, value) -> int:
        """Normalize clinical phase to integer (0-4)."""
        if value in [None, "", "null"]:
            return 0
        try:
            return int(value)
        except:
            return 0
    
    async def _symbol_to_ensembl(self, symbol: str) -> Optional[str]:
        """
        Convert gene symbol to Ensembl ID.
        
        Uses cache to avoid repeated queries.
        """
        if symbol in self._gene_cache:
            return self._gene_cache[symbol]
        
        # Query OpenTargets for gene info
        query = """
        query GeneSearch($symbol: String!) {
          search(queryString: $symbol, entityNames: ["target"]) {
            hits {
              id
              entity
              name
            }
          }
        }
        """
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.platform.opentargets.org/api/v4/graphql",
                    json={"query": query, "variables": {"symbol": symbol}}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    hits = data.get("data", {}).get("search", {}).get("hits", [])
                    
                    for hit in hits:
                        if hit.get("entity") == "target":
                            ensembl_id = hit.get("id")
                            self._gene_cache[symbol] = ensembl_id
                            return ensembl_id
        
        except Exception as e:
            logger.debug(f"Failed to resolve {symbol} to Ensembl ID: {e}")
        
        return None
