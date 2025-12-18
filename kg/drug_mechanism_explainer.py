# kg/drug_mechanism_explainer.py
"""
Explain how drugs work at the molecular level.
"""

import logging
from typing import Dict, Optional
import os
import json
import cerebras_llm as llm  # Cerebras-hosted LLM wrapper
import google.generativeai as genai

GEMINI_AVAILABLE = True


logger = logging.getLogger(__name__)


class DrugMechanismExplainer:
    """
    Generate detailed mechanistic explanations for drug-target-disease relationships.
    """
    
    DRUG_MECHANISM_PROMPT = """You are a pharmacology expert explaining drug mechanisms.

DISEASE:
- Name: {disease_name}
- Mechanism: {disease_mechanism}

TARGET:
- Gene: {target_symbol}
- Pathways: {target_pathways}

DRUG:
- Name: {drug_name}
- Mechanism of Action: {drug_moa}
- Clinical Phase: {drug_phase}

Generate a COMPLETE pharmacological explanation in JSON:
{{
  "molecular_mechanism": {{
    "binding": "<How drug binds to target (atomistic level)>",
    "conformational_change": "<What happens to target protein>",
    "downstream_effects": ["<Step 1>", "<Step 2>", "<Step 3>"]
  }},
  "pathway_effects": {{
    "primary_pathway": "<Direct pathway modulation>",
    "secondary_pathways": ["<Indirect effect 1>", "<Indirect effect 2>"],
    "feedback_loops": "<Compensatory mechanisms>"
  }},
  "clinical_translation": {{
    "expected_outcome": "<Phenotypic result>",
    "time_to_effect": "<Hours/days/weeks>",
    "biomarkers": ["<Measurable indicators>"]
  }},
  "pharmacokinetics": {{
    "absorption": "<Route and extent>",
    "distribution": "<Tissue penetration, BBB crossing>",
    "metabolism": "<Metabolic pathways>",
    "elimination": "<Half-life, clearance>"
  }},
  "safety_profile": {{
    "on_target_toxicity": ["<Mechanism-based AEs>"],
    "off_target_effects": ["<Unintended interactions>"],
    "drug_interactions": ["<CYP450, transporter interactions>"]
  }},
  "limitations": [
    "<What disease aspects are NOT addressed>",
    "<Potential resistance mechanisms>"
  ],
  "optimization_opportunities": [
    "<How to improve efficacy or safety>"
  ]
}}

Be highly specific. Cite molecular details (receptors, enzymes, signaling cascades).
"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        self.api_key = ""
        genai.configure(api_key="")
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    async def explain_drug_mechanism(
        self,
        drug_name: str,
        drug_moa: str,
        drug_phase: str,
        target_symbol: str,
        target_pathways: str,
        disease_name: str,
        disease_mechanism: str
    ) -> Dict:
        """
        Generate comprehensive pharmacological explanation.
        """
        logger.info(f"💊 Explaining mechanism: {drug_name} → {target_symbol} for {disease_name}")
        
        prompt = self.DRUG_MECHANISM_PROMPT.format(
            disease_name=disease_name,
            disease_mechanism=disease_mechanism,
            target_symbol=target_symbol,
            target_pathways=target_pathways,
            drug_name=drug_name,
            drug_moa=drug_moa,
            drug_phase=drug_phase
        )
        
        try:
            response = await llm.generate(prompt)
            response_text = response.text.strip()
            
            # Clean JSON
            if "```json" in response_text:
                # Split on ```json and take the part after it
                parts = response_text.split("```json")
                if len(parts) > 1:
                    # Now split on ending ```
                    response_text = parts[1].split("```")[0].strip()

            elif "```" in response_text:
                # Handle generic code blocks ``` ... ```
                parts = response_text.split("```")
                if len(parts) >= 3:
                    # The JSON is usually in parts[1]
                    response_text = parts[1].strip()

            # Method 2: Extract JSON object manually (first { to last })
            if not response_text.startswith("{"):
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx + 1]

            
            result = json.loads(response_text)
            
            logger.info(f"✅ Generated drug mechanism explanation for {drug_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Drug mechanism explanation failed: {e}")
            return {
                "molecular_mechanism": {"error": str(e)},
                "pathway_effects": {},
                "clinical_translation": {},
                "pharmacokinetics": {},
                "safety_profile": {},
                "limitations": ["Could not generate explanation"],
                "optimization_opportunities": []
            }
