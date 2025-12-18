# kg/mechanism_reasoner.py
"""
Mechanistic reasoning using pathways and LLMs.
"""

import logging
import os
import json
from typing import List, Dict, Optional
from kg.semantic_router import DiseaseContext
from kg.pathway_integrator import PathwayIntegrator
from kg.ppi_integrator import PPIIntegrator
import google.generativeai as genai
import cerebras_llm as llm  # Cerebras-hosted LLM wrapper

GEMINI_AVAILABLE = True

logger = logging.getLogger(__name__)


class MechanismReasoner:
    """
    Generate mechanistic explanations for target-disease associations.
    """
    
    MECHANISM_PROMPT = """You are a systems biology expert analyzing drug targets.

TASK: Explain the COMPLETE mechanistic pathway from disease to target to phenotype.

DISEASE CONTEXT:
- Name: {disease_name}
- Type: {disease_type}
- Pathophysiology: {disease_description}
- Disrupted Pathways: {disease_pathways}

TARGET:
- Gene: {target_symbol}
- Pathways: {target_pathways}
- Protein Interactions: {target_interactions}

PATHWAY OVERLAP:
- Common Pathways: {overlap_pathways}
- Jaccard Similarity: {overlap_score:.2f}

TASK: Generate a mechanistic explanation in this JSON format:
{{
  "mechanistic_fit": "HIGH" | "MEDIUM" | "LOW",
  "confidence": 0.95,
  "reasoning": {{
    "disease_mechanism": "<How disease disrupts normal biology>",
    "target_role": "<Where target sits in pathway>",
    "intervention_effect": "<Predicted effect of targeting this protein>",
    "pathway_cascade": ["Step 1", "Step 2", "Step 3"],
    "phenotypic_outcome": "<Expected clinical outcome>"
  }},
  "risks": [
    {{
      "risk": "Off-target effect on pathway X",
      "severity": "MODERATE",
      "mitigation": "Use selective inhibitor"
    }}
  ],
  "synergies": [
    "<Targets or drugs that would synergize>"
  ],
  "limitations": [
    "<What this target CANNOT address>"
  ]
}}

Be specific and cite molecular mechanisms. Focus on CAUSAL relationships, not just associations.
"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        self.api_key = ""
        genai.configure(api_key="")
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.pathway_integrator = PathwayIntegrator()
        self.ppi_integrator = PPIIntegrator()
    
    async def explain_target_mechanism(
        self,
        target_symbol: str,
        disease_context: DiseaseContext
    ) -> Dict:
        """
        Generate comprehensive mechanistic explanation for a target.
        
        Returns:
        {
            "mechanistic_fit": "HIGH",
            "confidence": 0.92,
            "reasoning": {...},
            "risks": [...],
            "synergies": [...],
            "limitations": [...]
        }
        """
        logger.info(f"🧬 Generating mechanistic explanation: {target_symbol} for {disease_context.corrected_name}")
        
        # Gather pathway data
        disease_pathways = await self.pathway_integrator.get_disease_pathways(
            disease_context.corrected_name
        )
        
        target_pathways = await self.pathway_integrator.get_target_pathways(
            target_symbol
        )
        
        # Calculate pathway overlap
        disease_pathway_ids = [p["pathway_id"] for p in disease_pathways]
        target_pathway_ids = [p["pathway_id"] for p in target_pathways]
        
        overlap = await self.pathway_integrator.find_pathway_overlap(
            disease_pathway_ids,
            target_pathway_ids
        )
        
        # Get protein interactions
        interactions = await self.ppi_integrator.get_protein_interactions(target_symbol)
        
        # Format for LLM
        disease_pathways_str = "\n".join([
            f"  - {p['name']} ({p['source']})"
            for p in disease_pathways[:5]
        ]) or "No specific pathways identified"
        
        target_pathways_str = "\n".join([
            f"  - {p['name']} ({p['source']})"
            for p in target_pathways[:5]
        ]) or "No specific pathways identified"
        
        target_interactions_str = "\n".join([
            f"  - {i['partner']} (score: {i['score']:.2f})"
            for i in interactions[:10]
        ]) or "No interactions found"
        
        overlap_pathways_str = "\n".join([
            f"  - {pid}"
            for pid in overlap["overlap_pathways"]
        ]) or "No pathway overlap"
        
        # Determine disease type
        disease_type = disease_context.therapeutic_area
        if disease_context.is_cancer:
            disease_type = "Cancer"
        elif disease_context.is_autoimmune:
            disease_type = "Autoimmune"
        elif disease_context.is_infectious:
            disease_type = "Infectious"
        
        # Generate mechanistic explanation
        prompt = self.MECHANISM_PROMPT.format(
            disease_name=disease_context.corrected_name,
            disease_type=disease_type,
            disease_description=disease_context.description,
            disease_pathways=disease_pathways_str,
            target_symbol=target_symbol,
            target_pathways=target_pathways_str,
            target_interactions=target_interactions_str,
            overlap_pathways=overlap_pathways_str,
            overlap_score=overlap["jaccard_similarity"]
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
            
            # Add metadata
            result["pathway_overlap_score"] = overlap["jaccard_similarity"]
            result["num_disease_pathways"] = len(disease_pathways)
            result["num_target_pathways"] = len(target_pathways)
            result["num_interactions"] = len(interactions)
            
            logger.info(f"✅ Mechanistic fit: {result['mechanistic_fit']} (confidence: {result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Mechanistic reasoning failed: {e}")
            return {
                "mechanistic_fit": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": {"error": str(e)},
                "risks": [],
                "synergies": [],
                "limitations": ["Could not generate explanation"]
            }
