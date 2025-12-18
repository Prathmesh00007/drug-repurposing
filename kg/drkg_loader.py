"""DRKG Loader: Download and manage DRKG knowledge graph embeddings."""
import logging
import os
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# DRKG Data URLs
DRKG_DATA_URL = "https://dgl-data.s3.us-west-2.amazonaws.com/dataset/DRKG/drkg.tsv"
DRKG_EMBEDDINGS_URL = "https://dgl-data.s3.us-west-2.amazonaws.com/dataset/DRKG/embed/TransE_l2_entity.npy"
DRKG_ENTITIES_URL = "https://dgl-data.s3.us-west-2.amazonaws.com/dataset/DRKG/embed/entities.tsv"

class DRKGLoader:
    """Load and query DRKG embeddings for drug repurposing."""
    
    def __init__(self, cache_dir: str = "./data/drkg"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.entity_embeddings = None
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.triples_df = None
        self.loaded = False
    
    def _download_file(self, url: str, filename: str) -> Path:
        """Download DRKG file if not cached."""
        filepath = self.cache_dir / filename
        
        if filepath.exists():
            logger.info(f"✓ Using cached {filename}")
            return filepath
        
        logger.info(f"⬇️ Downloading {filename}...")
        response = requests.get(
            url,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✓ Downloaded {filename}")
        return filepath
    
    def load(self):
        """Load DRKG data and embeddings."""
        if self.loaded:
            return
        
        try:
            # Download files
            entities_path = self._download_file(DRKG_ENTITIES_URL, "entities.tsv")
            embeddings_path = self._download_file(DRKG_EMBEDDINGS_URL, "DRKG_TransE_l2_entity.npy")
            triples_path = self._download_file(DRKG_DATA_URL, "drkg.tsv")
            
            # Load entity mapping
            # --- replace the "Load entity mapping" block in load() with this ---
            logger.info("📚 Loading entity mappings...")
            # read without assuming column order; read as strings to avoid dtype surprises
            entities_df = pd.read_csv(entities_path, sep='\t', header=None, dtype=str)

            # handle two common formats: (id, entity) or (entity, id)
            if entities_df.shape[1] >= 2:
                col0 = entities_df.iloc[:, 0].str.strip()
                col1 = entities_df.iloc[:, 1].str.strip()
                # Heuristic: if col0 is numeric, assume (id, entity)
                if col0.str.match(r'^\d+$').all():
                    entities_df = pd.DataFrame({
                        'id': col0.astype(int),
                        'entity': col1
                    })
                else:
                    # assume (entity, id)
                    entities_df = pd.DataFrame({
                        'entity': col0,
                        'id': col1.astype(int)
                    })
            else:
                raise ValueError("Unexpected entities.tsv format: need at least 2 columns")

            # build mappings with normalized types
            self.entity_to_id = {str(ent): int(i) for ent, i in zip(entities_df['entity'], entities_df['id'])}
            self.id_to_entity = {int(i): str(ent) for ent, i in zip(entities_df['entity'], entities_df['id'])}

            # --- load embeddings as before ---
            logger.info("🧬 Loading embeddings...")
            self.entity_embeddings = np.load(embeddings_path)

            # validate embedding shape vs ids
            max_id = max(self.id_to_entity.keys())
            if max_id >= self.entity_embeddings.shape[0]:
                logger.warning(
                    f"Embedding array has shape {self.entity_embeddings.shape}, "
                    f"but max entity id is {max_id}. This may cause indexing errors."
                )

            # --- ensure triples are strings so comparisons work ---
            logger.info("🔗 Loading knowledge graph triples...")
            self.triples_df = pd.read_csv(
                triples_path,
                sep='\t',
                header=None,
                names=['head', 'relation', 'tail'],
                nrows=500000,
                dtype=str  # force string dtype
            )
            
            self.loaded = True
            logger.info(f"✅ DRKG loaded: {len(self.entity_to_id)} entities, {len(self.triples_df)} triples")
            
        except Exception as e:
            logger.error(f"❌ Failed to load DRKG: {e}")
            self.loaded = False
    
    def find_disease_entity(self, disease_name: str) -> Optional[str]:
        """Find best matching disease entity in DRKG."""
        disease_name_lower = disease_name.lower()
        
        # Search for disease entities
        candidates = []
        for entity in self.entity_to_id.keys():
            if entity.startswith("Disease::"):
                entity_name = entity.replace("Disease::", "").lower()
                if disease_name_lower in entity_name or entity_name in disease_name_lower:
                    candidates.append(entity)
        
        if candidates:
            return candidates[0]
        
        logger.warning(f"⚠️ Disease '{disease_name}' not found in DRKG")
        return None
    
    def predict_drug_candidates(
        self, 
        disease_entity: str, 
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Predict drug candidates using embedding similarity.
        Returns list of (drug_entity, similarity_score) tuples.
        """
        if not self.loaded:
            return []
        
        if disease_entity not in self.entity_to_id:
            logger.warning(f"⚠️ Entity {disease_entity} not in DRKG")
            return []
        
        disease_id = self.entity_to_id[disease_entity]
        disease_emb = self.entity_embeddings[disease_id]
        
        # Find all drug entities
        drug_entities = [e for e in self.entity_to_id.keys() if e.startswith("Compound::")]
        
        # Calculate similarities
        similarities = []
        for drug_entity in drug_entities:
            drug_id = self.entity_to_id[drug_entity]
            drug_emb = self.entity_embeddings[drug_id]
            
            # Cosine similarity
            similarity = np.dot(disease_emb, drug_emb) / (
                np.linalg.norm(disease_emb) * np.linalg.norm(drug_emb)
            )
            similarities.append((drug_entity, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_drug_disease_paths(self, drug_entity: str, disease_entity: str, max_paths: int = 5) -> List[str]:
        """Find connecting paths between drug and disease in the graph."""
        if not self.loaded:
            return []
        
        paths = []
        
        # Direct connections
        direct = self.triples_df[
            ((self.triples_df['head'] == drug_entity) & (self.triples_df['tail'] == disease_entity)) |
            ((self.triples_df['head'] == disease_entity) & (self.triples_df['tail'] == drug_entity))
        ]
        
        for _, row in direct.iterrows():
            paths.append(f"{row['head']} --[{row['relation']}]--> {row['tail']}")
        
        # One-hop connections (drug -> gene -> disease)
        if len(paths) < max_paths:
            drug_connections = self.triples_df[self.triples_df['head'] == drug_entity]
            for _, row1 in drug_connections.head(50).iterrows():
                intermediate = row1['tail']
                disease_connections = self.triples_df[
                    (self.triples_df['head'] == intermediate) & (self.triples_df['tail'] == disease_entity)
                ]
                for _, row2 in disease_connections.iterrows():
                    path = f"{drug_entity} --[{row1['relation']}]--> {intermediate} --[{row2['relation']}]--> {disease_entity}"
                    paths.append(path)
                    if len(paths) >= max_paths:
                        break
        
        return paths[:max_paths]

# Global instance
drkg_loader = DRKGLoader()