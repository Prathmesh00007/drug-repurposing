# kg/pathway_integrator.py
"""
Integrate pathway-level knowledge for mechanistic reasoning.
"""

import httpx
import asyncio
from typing import List, Dict, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class PathwayIntegrator:
    """
    Aggregate pathway data from multiple sources:
    - Reactome: Curated pathways
    - KEGG: Metabolic/signaling pathways
    - WikiPathways: Community-curated
    - Pathway Commons: Aggregated interactions
    """

    UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
    # Minimal, polite headers to use for all external calls to avoid filtering/rate-limit surprises
    _DEFAULT_HEADERS = {
        "Accept": "application/json",
        "User-Agent": "pathway-integrator/1.0 (+https://example.org)"
    }

    def __init__(self):
        self.reactome_url = "https://reactome.org/ContentService"
        self.kegg_url = "https://rest.kegg.jp"
        self.wikipathways_url = "https://webservice.wikipathways.org"

    # -----------------------
    # Public methods (unchanged semantics)
    # -----------------------
    async def get_disease_pathways(
        self, 
        disease_targets: List[Dict]  # The 50 validated targets
    ) -> List[str]:
        """Get disease pathways by aggregating pathways from disease targets."""
        all_pathway_ids = set()
        
        for target in disease_targets[:20]:  # Top 20 targets
            gene_symbol = target.get("symbol")
            target_pathways = await self.get_target_pathways(gene_symbol)
            
            for pathway in target_pathways:
                all_pathway_ids.add(pathway["pathway_id"])
        
        logger.info(f"Inferred {len(all_pathway_ids)} disease pathways from targets")
        return list(all_pathway_ids)


    # -----------------------
    # Reactome disease query (kept mostly as-is)
    # -----------------------
    async def _query_reactome(self, disease_name: str) -> List[Dict]:
        """Query Reactome for disease pathways (search endpoint)."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.reactome_url}/search/query",
                    params={"query": disease_name, "species": "Homo sapiens"},
                    headers=self._DEFAULT_HEADERS
                )

                if response.status_code != 200:
                    logger.debug("Reactome search returned %s", response.status_code)
                    return []

                results = response.json()
                pathways = []

                for entry in results.get("results", []):
                    if entry.get("type") == "Pathway":
                        pathway_id = entry.get("stId")
                        pathway_details = await self._get_reactome_pathway_details(pathway_id)

                        if pathway_details:
                            pathways.append({
                                "pathway_id": pathway_id,
                                "name": entry.get("name"),
                                "source": "reactome",
                                "entities": pathway_details.get("entities", []),
                                "description": entry.get("summation", "")
                            })

                logger.info(f"✓ Reactome: Found {len(pathways)} pathways for {disease_name}")
                return pathways

        except Exception as e:
            logger.warning(f"Reactome query failed: {e}")
            return []

    async def _get_reactome_pathway_details(self, pathway_id: str) -> Optional[Dict]:
        """Get detailed information about a Reactome pathway (contained events -> participants)."""
        if not pathway_id:
            return None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.reactome_url}/data/pathway/{pathway_id}/containedEvents",
                    headers=self._DEFAULT_HEADERS
                )

                if response.status_code != 200:
                    logger.debug("Failed to fetch pathway details for %s : %s", pathway_id, response.status_code)
                    return None

                data = response.json()

                entities = []
                for event in data:
                    if event.get("schemaClass") in ["Reaction", "BlackBoxEvent"]:
                        for participant in event.get("input", []) + event.get("output", []):
                            gene = participant.get("geneName") or participant.get("displayName") or participant.get("identifier")
                            if gene:
                                entities.append(gene)

                return {
                    "entities": list(set(entities)),
                    "num_reactions": len(data)
                }

        except Exception as e:
            logger.debug(f"Failed to get Reactome pathway details: {e}")
            return None

    # -----------------------
    # KEGG query (unchanged)
    # -----------------------
    async def _query_kegg(self, disease_name: str) -> List[Dict]:
        """Query KEGG for disease pathways."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.kegg_url}/find/disease/{disease_name}"
                )

                if response.status_code != 200:
                    return []

                lines = response.text.strip().split("\n")
                pathways = []

                for line in lines[:5]:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        disease_id = parts[0]
                        disease_desc = parts[1]

                        pathway_response = await client.get(
                            f"{self.kegg_url}/link/pathway/{disease_id}"
                        )

                        if pathway_response.status_code == 200:
                            for pathway_line in pathway_response.text.strip().split("\n"):
                                pathway_parts = pathway_line.split("\t")
                                if len(pathway_parts) >= 2:
                                    pathway_id = pathway_parts[1].replace("path:", "")
                                    pathways.append({
                                        "pathway_id": pathway_id,
                                        "name": f"KEGG pathway {pathway_id}",
                                        "source": "kegg",
                                        "disease_code": disease_id
                                    })

                logger.info(f"✓ KEGG: Found {len(pathways)} pathways for {disease_name}")
                return pathways

        except Exception as e:
            logger.warning(f"KEGG query failed: {e}")
            return []

    # -----------------------
    # Reactome mapping helpers
    # -----------------------
    async def _map_uniprot_to_pathways(self, uniprot_acc: str) -> List[Dict]:
        if not uniprot_acc:
            return []

        uniprot_acc = uniprot_acc.strip()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{self.reactome_url}/data/mapping/UniProt/{uniprot_acc}/pathways"
                resp = await client.get(url, headers=self._DEFAULT_HEADERS)
                if resp.status_code != 200:
                    logger.debug("Reactome mapping returned %s for %s", resp.status_code, uniprot_acc)
                    return []
                mapped = resp.json()
                out = []
                for m in mapped:
                    out.append({
                        "stId": m.get("stId"),
                        "dbId": m.get("dbId"),
                        "displayName": m.get("displayName"),
                        "speciesName": m.get("speciesName")
                    })
                # debug: log count + small snippet
                logger.debug("Reactome mapped count=%d for %s", len(out), uniprot_acc)
                if len(out) > 0:
                    logger.debug("Reactome mapped sample: %s", out[:3])
                return out
        except Exception as e:
            logger.debug("Reactome map error for %s : %s", uniprot_acc, e)
            return []
    async def _resolve_uniprot(self, gene_symbol: str) -> Optional[str]:
        if not gene_symbol:
            return None

        # slightly widened but same intent
        if re.match(r'^[A-NR-ZOPQ][0-9][A-Z0-9]{3}[0-9]$', gene_symbol):
            return gene_symbol

        params = {
            # request a few hits so we can prefer reviewed / primary accessions
            "query": f"(gene:{gene_symbol}) AND organism_id:9606",
            "fields": "accession,reviewed,id",
            "format": "json",
            "size": 5
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self.UNIPROT_SEARCH_URL, params=params, headers=self._DEFAULT_HEADERS)
                if resp.status_code != 200:
                    logger.debug("UniProt search failed %s for %s", resp.status_code, gene_symbol)
                    return None
                body = resp.json()
                results = body.get("results") or body.get("entries") or []
                if not results:
                    return None

                # 1) Prefer reviewed (Swiss-Prot) entries
                for r in results:
                    entry_type = (r.get("entryType") or "").lower()
                    acc = r.get("primaryAccession") or (r.get("accession") and r.get("accession")[0])
                    if acc and "reviewed" in entry_type:
                        logger.debug("UniProt: choosing reviewed accession %s for %s", acc, gene_symbol)
                        return acc

                # 2) Prefer accessions starting with 'P' (common for reviewed human records)
                for r in results:
                    acc = r.get("primaryAccession") or (r.get("accession") and r.get("accession")[0])
                    if isinstance(acc, str) and acc.startswith("P"):
                        logger.debug("UniProt: choosing P-* accession %s for %s", acc, gene_symbol)
                        return acc

                # 3) Fallback to first primaryAccession if nothing matched
                first = results[0]
                acc = first.get("primaryAccession") or (first.get("accession") and first.get("accession")[0])
                logger.debug("UniProt: falling back to accession %s for %s", acc, gene_symbol)
                return acc

        except Exception as e:
            logger.debug("Failed to resolve UniProt for %s : %s", gene_symbol, e)
            return None

    # -----------------------
    # get_target_pathways (unchanged)
    # -----------------------
    async def get_target_pathways(self, gene_symbol: str) -> List[Dict]:
        try:
            uniprot_acc = None
            if gene_symbol and gene_symbol.upper().startswith(("P", "Q", "O")):
                uniprot_acc = gene_symbol.strip()
            else:
                uniprot_acc = await self._resolve_uniprot(gene_symbol)

            # debug: show what was resolved (or not)
            logger.debug("get_target_pathways: input=%s resolved_uniprot=%s", gene_symbol, uniprot_acc)

            if not uniprot_acc:
                uniprot_acc = gene_symbol

            mapped = await self._map_uniprot_to_pathways(uniprot_acc)
            # If no mapping found for the chosen accession, try other UniProt hits (robust fallback)
            if not mapped:
                logger.debug("No mapping for initial accession %s; trying other UniProt hits as fallback", uniprot_acc)
                # re-query UniProt to get multiple candidate accessions
                try:
                    params = {
                        "query": f"(gene:{gene_symbol}) AND organism_id:9606",
                        "fields": "accession,reviewed,id",
                        "format": "json",
                        "size": 5
                    }
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.get(self.UNIPROT_SEARCH_URL, params=params, headers=self._DEFAULT_HEADERS)
                        if resp.status_code == 200:
                            body = resp.json()
                            results = body.get("results") or body.get("entries") or []
                            # iterate through candidate accessions and try mapping until one works
                            for r in results:
                                candidate = r.get("primaryAccession") or (r.get("accession") and r.get("accession")[0])
                                if not candidate or candidate == uniprot_acc:
                                    continue
                                logger.debug("Trying alternate accession %s for mapping", candidate)
                                mapped_alt = await self._map_uniprot_to_pathways(candidate)
                                if mapped_alt:
                                    mapped = mapped_alt
                                    uniprot_acc = candidate
                                    logger.debug("Found mapping for alternate accession %s", candidate)
                                    break
                except Exception as e:
                    logger.debug("Fallback UniProt re-query failed: %s", e)

            # debug: log mapped result
            logger.debug("get_target_pathways: mapped=%s", mapped[:3] if isinstance(mapped, list) else mapped)

            if not mapped:
                logger.info("No Reactome mapping found for %s (tried accession: %s)", gene_symbol, uniprot_acc)
                return []

            result = []
            for m in mapped:
                result.append({
                    "pathway_id": m.get("stId"),
                    "name": m.get("displayName"),
                    "dbId": m.get("dbId"),
                    "species": m.get("speciesName"),
                    "source": "reactome"
                })

            logger.info(f"✓ Found {len(result)} pathways for target {gene_symbol} (resolved:{uniprot_acc})")
            return result

        except Exception as e:
            logger.warning(f"Failed to get pathways for {gene_symbol}: {e}")
            return []


    # -----------------------
    # Overlap function (unchanged)
    # -----------------------
    async def find_pathway_overlap(
        self,
        disease_pathways: List[str],
        target_pathways: List[str]
    ) -> Dict:
        disease_set = set(disease_pathways)
        target_set = set(target_pathways)
        overlap = disease_set & target_set

        return {
            "overlap_pathways": list(overlap),
            "overlap_count": len(overlap),
            "jaccard_similarity": len(overlap) / len(disease_set | target_set) if (disease_set | target_set) else 0,
            "is_mechanistically_relevant": len(overlap) > 0
        }
