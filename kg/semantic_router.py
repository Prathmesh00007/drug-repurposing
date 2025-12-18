"""
Semantic Router using Local Qwen 2.5 for disease resolution.
Optimized for 3B parameter models via Native JSON Mode.
"""

import logging
import json
from typing import Optional, List
from dataclasses import dataclass
import cerebras_llm as llm  # Cerebras-hosted LLM wrapper

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiseaseContext:
    original_query: str
    corrected_name: str
    mesh_id: Optional[str]
    mondo_id: Optional[str]
    efo_id: Optional[str]
    therapeutic_area: str
    key_pathways: List[str]
    key_targets: List[str]
    known_drugs: List[str]
    is_cancer: bool
    is_autoimmune: bool
    is_infectious: bool
    description: str
    confidence: float
    # We don't store the chain_of_thought, but we parse it to allow the LLM to think

class GeminiSemanticRouter:
    """
    Disease Resolution for Local LLMs.
    Uses 'Chain of Thought embedded in JSON' pattern.
    """
    
    SYSTEM_PROMPT = """You are a World-Class Computational Biologist and Medical Ontologist.
Your task is to resolve user disease queries into structured, biologically accurate contexts.

EXECUTE THE FOLLOWING STEP-BY-STEP REASONING:

STEP 1: TYPO CORRECTION & NORMALIZATION
- Detect and correct typos (e.g., 'Alopecia Areta' → 'Alopecia Areata')
- Expand abbreviations (e.g., 'RA' → 'Rheumatoid Arthritis')
- Normalize slang (e.g., 'Sugar disease' → 'Diabetes Mellitus')

STEP 2: ONTOLOGY MAPPING (CRITICAL)
- Map to the MOST SPECIFIC ontology IDs:
  * MeSH ID (e.g., 'D000506' for Alopecia Areata)
  * MONDO ID (e.g., 'MONDO:0004907')
  * EFO ID (e.g., 'EFO_0000278')
- Avoid mapping to parent terms (e.g., reject 'D000505 Alopecia' when 'D000506 Alopecia Areata' exists)

STEP 3: DISEASE CLASSIFICATION (FOR TARGET FILTERING)
- Is this Oncology (cancer)? → ABL1 targets are valid
- Is this Autoimmune? → JAK/STAT targets are valid
- Is this Infectious? → Different target class
- Is this Genetic/Metabolic? → Another target class

This classification is CRITICAL. Cancer drugs should NOT be recommended for autoimmune diseases.

STEP 4: BIOLOGICAL MECHANISM RETRIEVAL
- List 3 verified pathways (e.g., 'JAK-STAT signaling', 'T-cell activation')
- List 3-5 key drug targets (genes) known for this disease
- List 3-5 Standard of Care (SoC) drugs (FDA-approved or Phase 3)

STEP 5: SELF-VERIFICATION (HALLUCINATION CHECK)
- Double-check: Are the SoC drugs you listed actually approved/in trials?
- Double-check: Is the MeSH ID the most specific available?
- Double-check: Does the disease classification match the mechanism?

If uncertain about ANY field, return null rather than guessing.

OUTPUT FORMAT (JSON only):
{
  "original_query": "<user input>",
  "corrected_name": "<typo-corrected, normalized name>",
  "mesh_id": "D123456",
  "mondo_id": "MONDO:1234567",
  "efo_id": "EFO_1234567",
  "therapeutic_area": "<Autoimmune|Oncology|Infectious|Metabolic|Neurological|Cardiovascular>",
  "key_pathways": ["pathway1", "pathway2", "pathway3"],
  "key_targets": ["GENE1", "GENE2", "GENE3"],
  "known_drugs": ["Drug A", "Drug B", "Drug C"],
  "is_cancer": false,
  "is_autoimmune": true,
  "is_infectious": false,
  "description": "<2-sentence description>",
  "confidence": 0.95
}
"""

    def resolve(self, disease_name: str) -> Optional[DiseaseContext]:
        """
        Resolve disease name using Local LLM.
        """
        logger.info(f"🧠 Local Router analyzing: '{disease_name}'")
        
        try:
            # We don't need complex prompt engineering here because
            # LLMClient enforces JSON format natively.
            user_prompt = f"Analyze this disease query: '{disease_name}'"
            
            # Synchronous Call
            response = llm.generate_sync([
                {"role": "system", "parts": [{"text": self.SYSTEM_PROMPT}]},
                {"role": "user", "parts": [{"text": user_prompt}]}
            ])

            response_text = response.text.strip()
            
            # Simple JSON parse - Ollama guarantees valid JSON syntax with format='json'
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: sometimes models double escape or add preamble despite enforcement
                # But with format='json' this is extremely rare.
                logger.warning("JSON Decode failed, attempting cleanup...")
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end != -1:
                    data = json.loads(response_text[start:end])
                else:
                    raise

            # Convert to DiseaseContext
            context = DiseaseContext(
                original_query=data.get("original_query", disease_name),
                corrected_name=data.get("corrected_name", disease_name),
                mesh_id=data.get("mesh_id"),
                mondo_id=data.get("mondo_id"),
                efo_id=data.get("efo_id"),
                therapeutic_area=data.get("therapeutic_area", "Unknown"),
                key_pathways=data.get("key_pathways", []),
                key_targets=data.get("key_targets", []),
                known_drugs=data.get("known_drugs", []),
                is_cancer=data.get("is_cancer", False),
                is_autoimmune=data.get("is_autoimmune", False),
                is_infectious=data.get("is_infectious", False),
                description=data.get("description", ""),
                confidence=data.get("confidence", 0.0)
            )
            
            logger.info(f"✅ Resolved: '{context.corrected_name}' ({context.therapeutic_area})")
            logger.debug(f"   Reasoning: {data.get('chain_of_thought', 'N/A')}")
            
            return context
            
        except Exception as e:
            logger.error(f"Local routing failed: {e}")
            logger.error(f"Raw output: {response_text if 'response_text' in locals() else 'No output'}")
            return None
    
    async def resolve_async(self, disease_name: str) -> Optional[DiseaseContext]:
        """Async wrapper."""
        # For actual async performance, we should call llm.generate directly
        # But to match your existing structure:
        import asyncio
        return await asyncio.to_thread(self.resolve, disease_name)

# --- Singleton & Helper Functions ---

_semantic_router = None

def get_semantic_router() -> GeminiSemanticRouter:
    global _semantic_router
    if _semantic_router is None:
        _semantic_router = GeminiSemanticRouter()
    return _semantic_router

async def resolve_disease_with_gemini(disease_name: str) -> Optional[DiseaseContext]:
    router = get_semantic_router()
    # Using thread wrapper to keep sync logic simple
    return await router.resolve_async(disease_name)

# --- Quick Test ---
if __name__ == "__main__":
    # Simple test to verify migration
    router = get_semantic_router()
    result = router.resolve("Bone cancer") # Intentional typo
    if result:
        print("\n--- RESULT ---")
        print(f"Name: {result.corrected_name}")
        print(f"MeSH: {result.mesh_id}")
        print(f"Drugs: {result.mondo_id}")
        print(f"Drugs: {result.efo_id}")
        print(f"Drugs: {result.known_drugs}")
        print(f"Drugs: {result.is_cancer}")
