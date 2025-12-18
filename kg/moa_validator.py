"""
Mechanism of Action (MOA) Validator
Uses ChEMBL API + Gemini for complex MOA appropriateness decisions.

Prevents recommending drugs with OPPOSITE mechanism (e.g., agonist instead of antagonist).
"""

import httpx
import logging
from typing import Optional, Dict, List
from enum import Enum
from dataclasses import dataclass
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

genai.configure(api_key="")


class MOAType(str, Enum):
    INHIBITOR = "inhibitor"
    ANTAGONIST = "antagonist"
    AGONIST = "agonist"
    ACTIVATOR = "activator"
    MODULATOR = "modulator"
    BLOCKER = "blocker"
    UNKNOWN = "unknown"


class PathologyType(str, Enum):
    OVERACTIVE = "overactive"      # Need suppression/inhibition
    UNDERACTIVE = "underactive"    # Need activation
    DYSREGULATED = "dysregulated"  # Complex, either direction
    UNKNOWN = "unknown"


@dataclass
class MOAValidationResult:
    is_appropriate: bool
    confidence: float
    reasoning: str
    drug_moa: MOAType
    target_pathology: PathologyType
    recommendation: str


class MOAValidator:
    """
    Validates if drug MOA matches disease pathology.
    
    Strategy:
    1. Extract drug MOA from ChEMBL (deterministic)
    2. Determine target pathology:
       - Simple cases: Use disease flags (cancer/autoimmune)
       - Complex cases: Use Gemini with structured prompting
    3. Validate match
    """
    
    CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
    
    def __init__(self):
        self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        self.timeout = httpx.Timeout(30.0)
    
    async def validate_moa_appropriateness(
        self,
        drug_chembl_id: str,
        target_symbol: str,
        disease_context,  # DiseaseContext
        mechanism_text: Optional[str] = None
    ) -> MOAValidationResult:
        """
        Validate if drug MOA is appropriate for disease.
        
        Args:
            drug_chembl_id: ChEMBL ID (e.g., "CHEMBL123")
            target_symbol: Gene symbol (e.g., "JAK1")
            disease_context: DiseaseContext
            mechanism_text: Optional mechanism description
        
        Returns:
            MOAValidationResult with decision
        """
        logger.info(f"🎯 Validating MOA: {drug_chembl_id} for {disease_context.corrected_name}")
        
        # Step 1: Get drug MOA from ChEMBL
        drug_moa = await self._get_drug_moa_from_chembl(drug_chembl_id, target_symbol)
        
        if drug_moa == MOAType.UNKNOWN and mechanism_text:
            # Fallback: parse from mechanism text
            drug_moa = self._parse_moa_from_text(mechanism_text)
        
        # Step 2: Determine target pathology
        target_pathology = await self._determine_target_pathology(
            target_symbol=target_symbol,
            disease_context=disease_context
        )
        
        # Step 3: Validate match
        return self._validate_moa_match(
            drug_moa=drug_moa,
            target_pathology=target_pathology,
            target_symbol=target_symbol,
            disease_name=disease_context.corrected_name
        )
    
    async def _get_drug_moa_from_chembl(
        self,
        chembl_id: str,
        target_symbol: str
    ) -> MOAType:
        """Query ChEMBL for drug mechanism"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Get drug mechanisms
                response = await client.get(
                    f"{self.CHEMBL_API}/mechanism.json",
                    params={
                        "molecule_chembl_id": chembl_id,
                        "limit": 100
                    },
                    headers={"Accept": "application/json"}
                )
                
                if response.status_code != 200:
                    return MOAType.UNKNOWN
                
                data = response.json()
                mechanisms = data.get("mechanisms", [])
                
                # Find mechanism for our target
                for mech in mechanisms:
                    target_name = mech.get("target_chembl_id", "")
                    action_type = mech.get("action_type", "").lower()
                    mechanism_of_action = mech.get("mechanism_of_action", "").lower()
                    
                    # Check if this mechanism is for our target
                    # (Simple heuristic: target symbol in MOA text)
                    if target_symbol.lower() in mechanism_of_action:
                        return self._parse_moa_from_text(action_type or mechanism_of_action)
                
                # If no specific match, use first mechanism
                if mechanisms:
                    action_type = mechanisms[0].get("action_type", "").lower()
                    return self._parse_moa_from_text(action_type)
                
                return MOAType.UNKNOWN
                
            except Exception as e:
                logger.error(f"ChEMBL MOA query failed: {e}")
                return MOAType.UNKNOWN
    
    def _parse_moa_from_text(self, text: str) -> MOAType:
        """Extract MOA type from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["inhibitor", "inhibits", "inhibition"]):
            return MOAType.INHIBITOR
        elif any(word in text_lower for word in ["antagonist", "antagonizes"]):
            return MOAType.ANTAGONIST
        elif any(word in text_lower for word in ["agonist", "activates"]):
            return MOAType.AGONIST
        elif any(word in text_lower for word in ["blocker", "blocks", "blocking"]):
            return MOAType.BLOCKER
        elif "modulator" in text_lower:
            return MOAType.MODULATOR
        else:
            return MOAType.UNKNOWN
    
    async def _determine_target_pathology(
        self,
        target_symbol: str,
        disease_context
    ) -> PathologyType:
        """
        Determine if target is overactive or underactive in disease.
        
        Strategy:
        1. Simple rules for cancer/autoimmune/infectious
        2. For complex cases, use Gemini with expert reasoning
        """
        # Simple rule-based classification
        if disease_context.is_cancer:
            # Cancer: targets are typically OVERACTIVE (need inhibition)
            return PathologyType.OVERACTIVE
        
        elif disease_context.is_autoimmune:
            # Autoimmune: immune targets are OVERACTIVE (need suppression)
            return PathologyType.OVERACTIVE
        
        elif disease_context.is_infectious:
            # Infection: immune targets are UNDERACTIVE (need activation)
            return PathologyType.UNDERACTIVE
        
        else:
            # Complex case: use Gemini for expert reasoning
            return await self._classify_pathology_with_gemini(
                target_symbol, disease_context
            )
    
    async def _classify_pathology_with_gemini(
        self,
        target_symbol: str,
        disease_context
    ) -> PathologyType:
        """Use Gemini for complex pathology classification"""
        prompt = f"""You are a molecular pathology expert. Determine if this target is OVERACTIVE or UNDERACTIVE in the disease.

Target: {target_symbol}
Disease: {disease_context.corrected_name}
Description: {disease_context.description}
Therapeutic Area: {disease_context.therapeutic_area}

Classification Rules:
- OVERACTIVE: Target is upregulated, hyperactive, or causing pathology through excessive activity
  → Treatment should INHIBIT/BLOCK this target
  → Examples: oncogenes in cancer, inflammatory mediators in autoimmune disease

- UNDERACTIVE: Target is downregulated, deficient, or protective function is lost
  → Treatment should ACTIVATE/ENHANCE this target
  → Examples: tumor suppressors in cancer, insulin in diabetes

- DYSREGULATED: Target activity is imbalanced (not simply high or low)
  → Treatment needs MODULATION (not simple activation/inhibition)

- UNKNOWN: Insufficient information to classify

Respond with ONLY one word: overactive, underactive, dysregulated, or unknown"""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=20
                )
            )
            
            result = response.text.strip().lower()
            
            for pathology in PathologyType:
                if pathology.value in result:
                    logger.info(f"   Gemini classified {target_symbol} as {pathology.value}")
                    return pathology
            
            return PathologyType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Gemini pathology classification failed: {e}")
            return PathologyType.UNKNOWN
    
    def _validate_moa_match(
        self,
        drug_moa: MOAType,
        target_pathology: PathologyType,
        target_symbol: str,
        disease_name: str
    ) -> MOAValidationResult:
        """Validate if MOA matches pathology"""
        
        # OVERACTIVE target → need INHIBITOR/ANTAGONIST/BLOCKER
        if target_pathology == PathologyType.OVERACTIVE:
            if drug_moa in [MOAType.INHIBITOR, MOAType.ANTAGONIST, MOAType.BLOCKER]:
                return MOAValidationResult(
                    is_appropriate=True,
                    confidence=0.9,
                    reasoning=f"{drug_moa.value} is appropriate for overactive {target_symbol} in {disease_name}",
                    drug_moa=drug_moa,
                    target_pathology=target_pathology,
                    recommendation="KEEP - MOA matches disease pathology"
                )
            elif drug_moa in [MOAType.AGONIST, MOAType.ACTIVATOR]:
                return MOAValidationResult(
                    is_appropriate=False,
                    confidence=0.9,
                    reasoning=f"{drug_moa.value} would WORSEN disease (activates already overactive {target_symbol})",
                    drug_moa=drug_moa,
                    target_pathology=target_pathology,
                    recommendation="REJECT - Opposite MOA (would worsen disease)"
                )
        
        # UNDERACTIVE target → need AGONIST/ACTIVATOR
        elif target_pathology == PathologyType.UNDERACTIVE:
            if drug_moa in [MOAType.AGONIST, MOAType.ACTIVATOR]:
                return MOAValidationResult(
                    is_appropriate=True,
                    confidence=0.9,
                    reasoning=f"{drug_moa.value} is appropriate for underactive {target_symbol} in {disease_name}",
                    drug_moa=drug_moa,
                    target_pathology=target_pathology,
                    recommendation="KEEP - MOA matches disease pathology"
                )
            elif drug_moa in [MOAType.INHIBITOR, MOAType.ANTAGONIST, MOAType.BLOCKER]:
                return MOAValidationResult(
                    is_appropriate=False,
                    confidence=0.9,
                    reasoning=f"{drug_moa.value} would WORSEN disease (inhibits already underactive {target_symbol})",
                    drug_moa=drug_moa,
                    target_pathology=target_pathology,
                    recommendation="REJECT - Opposite MOA (would worsen disease)"
                )
        
        # MODULATOR or UNKNOWN - accept with lower confidence
        return MOAValidationResult(
            is_appropriate=True,
            confidence=0.5,
            reasoning=f"MOA ({drug_moa.value}) appropriateness uncertain for {target_pathology.value} {target_symbol}",
            drug_moa=drug_moa,
            target_pathology=target_pathology,
            recommendation="UNCERTAIN - Accept with caution"
        )


# Singleton
moa_validator = MOAValidator()
