"""Neo4j client for Route A knowledge graph."""
import logging
from neo4j import GraphDatabase
from backend.app.config import get_settings
from typing import List, Dict, Any, Optional
from kg.ingest_opentargets import normalize_disease_id

logger = logging.getLogger(__name__)

class Neo4jClient:
    """Manage Neo4j connections and queries."""
    
    def __init__(self):
        self.settings = get_settings()
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )
        self.session_id = "session"
        logger.info(f"Connected to Neo4j: {self.settings.neo4j_uri}")
    
    def close(self):
        """Close driver connection."""
        self.driver.close()
    
    def create_disease_node(self, disease_id: str, disease_name: str):
        """Create a disease node."""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MERGE (d:Disease {id: $id})
                SET d.name = $name, d.updated_at = timestamp()
                """,
                id=disease_id,
                name=disease_name
            )
        logger.warning(f"Created disease node: {disease_name}")
    
    def create_target_node(self, target_id: str, target_symbol: str, target_name: str = ""):
        """Create a target node."""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MERGE (t:Target {id: $id})
                SET t.symbol = $symbol, t.name = $name, t.updated_at = timestamp()
                """,
                id=target_id,
                symbol=target_symbol,
                name=target_name
            )
        logger.warning(f"Created target node: {target_symbol}")
    
    def create_candidate_node(self, candidate_id: str, name: str, stage: str, source: str = ""):
        """Create a candidate drug node."""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MERGE (c:Candidate {id: $id})
                SET c.name = $name, c.stage = $stage, c.source = $source, c.updated_at = timestamp()
                """,
                id=candidate_id,
                name=name,
                stage=stage,
                source=source
            )
        logger.warning(f"Created candidate node: {name}")
    
    def create_target_disease_association(
        self, 
        target_id: str,
        disease_id: str,
        score: float = 0.5,
        evidence: str = "",
        mechanism_score: float = 0.0
    ):
        """Link target to disease."""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MATCH (t:Target {id: $target_id})
                MATCH (d:Disease {id: $disease_id})
                MERGE (t)-[r:ASSOCIATED_WITH]->(d)
                SET r.score = $score + $mechanism_score, r.evidence = $evidence, r.updated_at = timestamp()
                """,
                target_id=target_id,
                disease_id=disease_id,
                score=score,
                evidence=evidence,
                mechanism_score=mechanism_score
            )
        logger.warning(f"Created association: Target->Disease")
    
    def create_candidate_target_modulation(
        self,
        candidate_id: str,
        target_symbol: str, # Changed from target_id
        interaction_type: str = "modulates",
        score: float = 0.5
    ):
        """Link candidate to target by SYMBOL."""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MATCH (c:Candidate {id: $candidate_id})
                MATCH (t:Target) WHERE t.symbol = $target_symbol  // Match by symbol!
                MERGE (c)-[r:MODULATES]->(t)
                SET r.type = $type, r.score = $score, r.updated_at = timestamp()
                """,
                candidate_id=candidate_id,
                target_symbol=target_symbol, # Pass symbol
                type=interaction_type,
                score=score
            )
        logger.warning(f"Created modulation: Candidate->Target({target_symbol})")
    
    def query_candidates_for_disease(
        self,
        disease_id: str,
        limit: int = 20,
        min_phase: Optional[int] = None,
        oral_only: bool = False,
        exclude_biologics: bool = False
    ) -> List[Dict[str, Any]]:
        """Query candidates for a disease."""

        # Build filter clauses
        filters = ["ta.score > 0.0"]

        # FIX: Use c.stage not c.phase, with proper string matching
        if min_phase:
            phase_mapping = {
                1: "['Phase 1', 'Phase 2', 'Phase 3', 'approved']",
                2: "['Phase 2', 'Phase 3', 'approved']",
                3: "['Phase 3', 'approved']",
                4: "['approved']"
            }
            allowed_stages = phase_mapping.get(min_phase, "['approved']")
            filters.append(f"c.stage IN {allowed_stages}")

        if oral_only:
            filters.append("c.formulation = 'oral'")

        if exclude_biologics:
            filters.append("c.type <> 'biologic'")

        where_clause = " AND ".join(filters)

        query = f"""
        MATCH (d:Disease {{id: $disease_id}})
        MATCH (t:Target)-[ta:ASSOCIATED_WITH]->(d)
        MATCH (c:Candidate)-[ct:MODULATES]->(t)
        WHERE {where_clause}
        RETURN DISTINCT
            c.id as id,
            c.name as name,
            c.stage as stage,
            collect(DISTINCT t.symbol) as targets,
            avg(ta.score) as score,
            collect(DISTINCT c.source) as urls
        ORDER BY score DESC
        LIMIT {limit}
        """

        with self.driver.session(database=self.settings.neo4j_database) as session:
            normalized_disease_id = normalize_disease_id(disease_id)
            result = session.run(query, disease_id=normalized_disease_id)
            candidates = [dict(record) for record in result]
            logger.info(f"Queried {len(candidates)} candidates for {disease_id}")
            return candidates
            
    def add_target_mechanism(self, target_id: str, mechanism_json: str):
        """Add mechanistic reasoning to target node."""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MATCH (t:Target {id: $target_id})
                SET t.mechanism_reasoning = $mechanism_json, 
                    t.updated_at = timestamp()
                """,
                target_id=target_id,
                mechanism_json=mechanism_json
            )
            logger.info(f"✓ Added mechanism to target {target_id}")

    def add_drug_mechanism(self, drug_id: str, mechanism_json: str):
        """Add mechanistic explanation to drug/candidate node."""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MATCH (c:Candidate {id: $drug_id})
                SET c.mechanism_explanation = $mechanism_json,
                    c.updated_at = timestamp()
                """,
                drug_id=drug_id,
                mechanism_json=mechanism_json
            )
            logger.info(f"✓ Added mechanism to drug {drug_id}")

    def batch_create_candidates(self, candidates: List[Dict]):
        """Batch create candidates and modulations in one transaction"""
        with self.driver.session() as session:
            session.execute_write(self._batch_create_tx, candidates)
            logger.debug(f"Created {len(candidates)} candidates in Neo4j")

    @staticmethod
    def _batch_create_tx(tx, candidates):
        query = """
        UNWIND $candidates AS cand
        MERGE (d:Drug {id: cand.candidate_id})
        SET d.name = cand.name,
            d.stage = cand.stage,
            d.source = cand.source
        
        WITH d, cand
        MATCH (t:Target {symbol: cand.target_symbol})
        MERGE (d)-[r:TARGETS]->(t)
        SET r.mechanism = cand.mechanism,
            r.score = 0.5
        """
        tx.run(query, candidates=candidates)
