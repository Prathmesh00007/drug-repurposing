"""
Hybrid Drug Discovery Strategy.

Orchestrates BOTH direct and indirect drug discovery approaches:
1. Direct: Query disease → knownDrugs (reference system approach)
2. Indirect: Query disease → targets → drugs (your system's approach)

Then merges, deduplicates, and enriches the results.
"""

import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DrugCandidate:
    """Standardized drug candidate representation."""
    drug_id: str
    drug_name: str
    target_symbol: Optional[str] = None
    target_name: Optional[str] = None
    drug_type: str = "Unknown"
    phase: int = 0
    source: str = ""
    has_clinical_evidence: bool = False
    mechanism: str = ""
    indication: str = ""
    composite_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.drug_id, self.target_symbol))
    
    def __eq__(self, other):
        if not isinstance(other, DrugCandidate):
            return False
        return self.drug_id == other.drug_id and self.target_symbol == other.target_symbol


class HybridDrugDiscovery:
    """
    Orchestrator for hybrid drug discovery strategy.
    
    Combines direct disease-drug queries with indirect target-based discovery.
    """
    
    def __init__(self):
        self.direct_drugs: List[DrugCandidate] = []
        self.indirect_drugs: List[DrugCandidate] = []
        self.merged_drugs: List[DrugCandidate] = []
    
    def add_direct_drugs(self, drugs: List[Dict]) -> None:
        """
        Add drugs from direct disease-drug query.
        
        Args:
            drugs: List of drug dicts from direct_disease_drugs.py
        """
        for drug in drugs:
            candidate = DrugCandidate(
                drug_id=drug.get("drug_id", ""),
                drug_name=drug.get("drug_name", ""),
                target_symbol=drug.get("target_symbol"),
                target_name=drug.get("target_name"),
                drug_type=drug.get("drug_type", "Unknown"),
                phase=drug.get("phase", 0),
                source="direct",
                has_clinical_evidence=True,
                sources=["opentargets_known_drugs"]
            )
            self.direct_drugs.append(candidate)
        
        logger.info(f"Added {len(drugs)} drugs from direct query")
    
    def add_indirect_drugs(self, drugs: List[Dict]) -> None:
        """
        Add drugs from indirect target-based discovery.
        
        Args:
            drugs: List of drug dicts from ingest_chembl.py
        """
        for drug in drugs:
            candidate = DrugCandidate(
                drug_id=drug.get("drug_id", ""),
                drug_name=drug.get("drug_name", ""),
                target_symbol=drug.get("target_symbol"),
                target_name=drug.get("target_name"),
                drug_type=drug.get("drug_type", "Unknown"),
                phase=drug.get("phase", 0),
                source="indirect",
                has_clinical_evidence=drug.get("has_clinical_evidence", False),
                mechanism=drug.get("mechanism", ""),
                indication=drug.get("indication", ""),
                sources=[drug.get("source", "")]
            )
            self.indirect_drugs.append(candidate)
        
        logger.info(f"Added {len(drugs)} drugs from indirect query")
    
    def merge_and_deduplicate(self) -> List[DrugCandidate]:
        """
        Merge direct and indirect drugs, handling duplicates intelligently.
        
        Strategy:
        1. Direct drugs have priority (higher confidence)
        2. For duplicates, merge information from both sources
        3. Boost scores for drugs found in both sources
        
        Returns:
            Merged and deduplicated list of drug candidates
        """
        logger.info("Merging and deduplicating drug candidates...")
        
        # Create lookup by drug_id
        drug_map: Dict[str, DrugCandidate] = {}
        
        # Process direct drugs first (priority)
        for drug in self.direct_drugs:
            key = drug.drug_id
            if key not in drug_map:
                drug_map[key] = drug
            else:
                # Merge sources
                existing = drug_map[key]
                existing.sources.extend(drug.sources)
                existing.sources = list(set(existing.sources))
        
        # Process indirect drugs
        overlap_count = 0
        new_count = 0
        
        for drug in self.indirect_drugs:
            key = drug.drug_id
            if key in drug_map:
                # Drug found in both sources - merge and boost
                overlap_count += 1
                existing = drug_map[key]
                
                # Merge sources
                existing.sources.extend(drug.sources)
                existing.sources = list(set(existing.sources))
                
                # Take mechanism from indirect if not present
                if not existing.mechanism and drug.mechanism:
                    existing.mechanism = drug.mechanism
                
                # Boost score for multi-source validation
                existing.composite_score = 0.8  # High confidence
                
            else:
                # New drug from indirect source
                new_count += 1
                drug_map[key] = drug
        
        self.merged_drugs = list(drug_map.values())
        
        # Log statistics
        logger.info(
            f"✅ Merge complete: {len(self.merged_drugs)} total candidates"
        )
        logger.info(
            f"   - {len(self.direct_drugs)} from direct query"
        )
        logger.info(
            f"   - {new_count} additional from indirect query"
        )
        logger.info(
            f"   - {overlap_count} found in both sources (high confidence)"
        )
        
        return self.merged_drugs
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about discovered drugs."""
        if not self.merged_drugs:
            return {}
        
        stats = {
            "total_candidates": len(self.merged_drugs),
            "direct_only": sum(1 for d in self.merged_drugs if d.source == "direct"),
            "indirect_only": sum(1 for d in self.merged_drugs if d.source == "indirect"),
            "multi_source": sum(1 for d in self.merged_drugs if len(d.sources) > 1),
            "phase_distribution": {},
            "drug_types": {},
            "with_clinical_evidence": sum(1 for d in self.merged_drugs if d.has_clinical_evidence),
            "unique_targets": len(set(
                d.target_symbol for d in self.merged_drugs 
                if d.target_symbol
            ))
        }
        
        # Phase distribution
        for drug in self.merged_drugs:
            phase = drug.phase
            stats["phase_distribution"][phase] = stats["phase_distribution"].get(phase, 0) + 1
        
        # Drug type distribution
        for drug in self.merged_drugs:
            dtype = drug.drug_type
            stats["drug_types"][dtype] = stats["drug_types"].get(dtype, 0) + 1
        
        return stats
    
    def filter_by_criteria(
        self,
        min_phase: int = 1,
        exclude_biologics: bool = False,
        require_clinical_evidence: bool = False
    ) -> List[DrugCandidate]:
        """
        Filter merged drugs by various criteria.
        
        Args:
            min_phase: Minimum clinical trial phase
            exclude_biologics: If True, exclude antibodies/biologics
            require_clinical_evidence: If True, only keep drugs with trial data
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        for drug in self.merged_drugs:
            # Phase filter
            if drug.phase < min_phase:
                continue
            
            # Biologic filter
            if exclude_biologics and "antibody" in drug.drug_type.lower():
                continue
            
            # Clinical evidence filter
            if require_clinical_evidence and not drug.has_clinical_evidence:
                continue
            
            filtered.append(drug)
        
        logger.info(
            f"Filtered {len(self.merged_drugs)} → {len(filtered)} candidates "
            f"(min_phase={min_phase}, exclude_biologics={exclude_biologics}, "
            f"require_clinical_evidence={require_clinical_evidence})"
        )
        
        return filtered
    
    def to_dict_list(self, candidates: Optional[List[DrugCandidate]] = None) -> List[Dict]:
        """Convert candidates to list of dictionaries."""
        if candidates is None:
            candidates = self.merged_drugs
        
        return [
            {
                "drug_id": c.drug_id,
                "drug_name": c.drug_name,
                "target_symbol": c.target_symbol,
                "target_name": c.target_name,
                "drug_type": c.drug_type,
                "phase": c.phase,
                "source": c.source,
                "sources": c.sources,
                "has_clinical_evidence": c.has_clinical_evidence,
                "mechanism": c.mechanism,
                "indication": c.indication,
                "composite_score": c.composite_score
            }
            for c in candidates
        ]


# Example usage
if __name__ == "__main__":
    # Mock data for testing
    direct_drugs = [
        {"drug_id": "CHEMBL1", "drug_name": "Drug A", "target_symbol": "TP53", "phase": 4},
        {"drug_id": "CHEMBL2", "drug_name": "Drug B", "target_symbol": "EGFR", "phase": 3},
        {"drug_id": "CHEMBL3", "drug_name": "Drug C", "target_symbol": "BRCA1", "phase": 2},
    ]
    
    indirect_drugs = [
        {"drug_id": "CHEMBL2", "drug_name": "Drug B", "target_symbol": "EGFR", "phase": 3, "mechanism": "Inhibitor"},
        {"drug_id": "CHEMBL4", "drug_name": "Drug D", "target_symbol": "ALK", "phase": 1, "mechanism": "Modulator"},
    ]
    
    # Test hybrid discovery
    hybrid = HybridDrugDiscovery()
    hybrid.add_direct_drugs(direct_drugs)
    hybrid.add_indirect_drugs(indirect_drugs)
    merged = hybrid.merge_and_deduplicate()
    
    print(f"\n{'='*60}")
    print("HYBRID DRUG DISCOVERY TEST")
    print(f"{'='*60}")
    
    stats = hybrid.get_statistics()
    print(f"Total candidates: {stats['total_candidates']}")
    print(f"Direct only: {stats['direct_only']}")
    print(f"Indirect only: {stats['indirect_only']}")
    print(f"Multi-source: {stats['multi_source']}")
    
    print("\nMerged candidates:")
    for candidate in merged:
        print(f"  - {candidate.drug_name} ({candidate.drug_id})")
        print(f"    Sources: {', '.join(candidate.sources)}")
        print(f"    Phase: {candidate.phase}, Score: {candidate.composite_score}")
