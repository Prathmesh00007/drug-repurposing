"""
Safety Validator - Check contraindications and safety signals
Uses FREE APIs: OpenFDA, DrugBank Open Data

This is a GATE (pass/fail), not a score component.
"""

import httpx
import logging
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

genai.configure(api_key="")


class SafetySignal(str, Enum):
    GREEN = "green"       # No known contraindications - safe to recommend
    YELLOW = "yellow"     # Caution - population restrictions or monitoring required
    RED = "red"          # Contraindicated - DO NOT recommend


@dataclass
class SafetyValidationResult:
    signal: SafetySignal
    is_safe_to_recommend: bool
    red_flags: List[str]
    warnings: List[str]
    population_restrictions: List[str]
    reasoning: str
    fda_warnings: List[str]


class SafetyValidator:
    """
    Validates drug safety for disease context.
    
    Strategy:
    1. Query OpenFDA for adverse events and warnings
    2. Check disease-specific contraindications using Gemini
    3. Population restrictions (pregnancy, pediatric, etc.)
    """
    
    OPENFDA_API = "https://api.fda.gov/drug"
    
    def __init__(self):
        self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        self.timeout = httpx.Timeout(30.0)
    
    async def validate_safety(
        self,
        drug_name: str,
        drug_chembl_id: Optional[str],
        disease_context,  # DiseaseContext
        target_symbol: Optional[str] = None
    ) -> SafetyValidationResult:
        """
        Validate drug safety for disease.
        
        Returns:
            SafetyValidationResult with gate decision
        """
        logger.info(f"🛡️ Validating safety: {drug_name} for {disease_context.corrected_name}")
        
        red_flags = []
        warnings = []
        population_restrictions = []
        fda_warnings = []
        
        # Step 1: Query OpenFDA for warnings (if available)
        fda_data = await self._query_openfda(drug_name)
        
        if fda_data:
            fda_warnings = fda_data.get("warnings", [])
            
            # Check for boxed warnings (black box)
            if fda_data.get("boxed_warning"):
                red_flags.append(f"FDA Black Box Warning: {fda_data['boxed_warning'][:200]}")
        
        # Step 2: Disease-specific contraindication check using Gemini
        disease_contraindications = await self._check_disease_contraindications_with_gemini(
            drug_name=drug_name,
            disease_context=disease_context,
            fda_warnings=fda_warnings
        )
        
        red_flags.extend(disease_contraindications.get("red_flags", []))
        warnings.extend(disease_contraindications.get("warnings", []))
        population_restrictions.extend(disease_contraindications.get("population_restrictions", []))
        
        # Step 3: Determine safety signal
        if red_flags:
            signal = SafetySignal.RED
            is_safe = False
            reasoning = f"CONTRAINDICATED: {'; '.join(red_flags[:3])}"
        elif warnings:
            signal = SafetySignal.YELLOW
            is_safe = True  # Can recommend with caution
            reasoning = f"CAUTION: {'; '.join(warnings[:3])}"
        else:
            signal = SafetySignal.GREEN
            is_safe = True
            reasoning = "No known contraindications for this disease"
        
        logger.info(f"   Safety Signal: {signal}, Safe: {is_safe}")
        
        return SafetyValidationResult(
            signal=signal,
            is_safe_to_recommend=is_safe,
            red_flags=red_flags,
            warnings=warnings,
            population_restrictions=population_restrictions,
            reasoning=reasoning,
            fda_warnings=fda_warnings
        )
    
    async def _query_openfda(self, drug_name: str) -> Optional[Dict]:
        """Query OpenFDA for drug warnings"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Try label endpoint first
                response = await client.get(
                    f"{self.OPENFDA_API}/label.json",
                    params={
                        "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                        "limit": 1
                    }
                )
                
                if response.status_code == 404:
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                if not data.get("results"):
                    return None
                
                label = data["results"][0]
                
                return {
                    "warnings": label.get("warnings", []),
                    "boxed_warning": label.get("boxed_warning", [""])[0] if "boxed_warning" in label else None,
                    "contraindications": label.get("contraindications", []),
                    "adverse_reactions": label.get("adverse_reactions", [])
                }
                
            except Exception as e:
                logger.warning(f"OpenFDA query failed for {drug_name}: {e}")
                return None
    
    async def _check_disease_contraindications_with_gemini(
        self,
        drug_name: str,
        disease_context,
        fda_warnings: List[str]
    ) -> Dict:
        """
        Use Gemini to analyze disease-specific contraindications.
        
        This handles complex reasoning like:
        - Immunosuppressive drug for autoimmune disease (appropriate)
        - Immunosuppressive drug for infectious disease (contraindicated)
        """
        prompt = f"""You are a clinical pharmacology expert. Analyze if this drug has contraindications for this disease.

Drug: {drug_name}
Disease: {disease_context.corrected_name}
Disease Type: {disease_context.therapeutic_area}
Disease Flags:
- Cancer: {disease_context.is_cancer}
- Autoimmune: {disease_context.is_autoimmune}
- Infectious: {disease_context.is_infectious}

FDA Warnings (if available):
{chr(10).join(fda_warnings[:3]) if fda_warnings else "None available"}

Analyze for:
1. **RED FLAGS** (absolute contraindications - DO NOT use):
   - Drug mechanism worsens disease
   - Known disease-drug interactions
   - Example: Immunosuppressant for active infection

2. **WARNINGS** (relative contraindications - use with caution):
   - Population restrictions (pregnancy, pediatric, elderly)
   - Monitoring required
   - Drug-disease interactions requiring caution

3. **POPULATION RESTRICTIONS**:
   - Pregnancy category
   - Pediatric use
   - Renal/hepatic impairment

Respond in JSON format:
{{
  "red_flags": ["list of absolute contraindications"],
  "warnings": ["list of caution items"],
  "population_restrictions": ["list of population limits"],
  "reasoning": "brief explanation"
}}

If no contraindications, return empty lists."""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500
                )
            )
            
            import json
            result_text = response.text.strip()
            
            # Extract JSON (remove markdown code blocks if present)
            if "```json" in result_text:
                # Split on ```json and take the part after it
                parts = result_text.split("```json")
                if len(parts) > 1:
                    # Now split on ending ```
                    result_text = parts[1].split("```")[0].strip()

            elif "```" in result_text:
                # Handle generic code blocks ``` ... ```
                parts = result_text.split("```")
                if len(parts) >= 3:
                    # The JSON is usually in parts[1]
                    result_text = parts[1].strip()

            # Method 2: Extract JSON object manually (first { to last })
            if not result_text.startswith("{"):
                start_idx = result_text.find("{")
                end_idx = result_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    result_text = result_text[start_idx:end_idx + 1]

            
            result = json.loads(result_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini safety check failed: {e}")
            return {
                "red_flags": [],
                "warnings": ["Unable to verify safety - manual review recommended"],
                "population_restrictions": [],
                "reasoning": "Safety check service unavailable"
            }


# Singleton
safety_validator = SafetyValidator()
