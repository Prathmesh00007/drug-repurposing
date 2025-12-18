"""
Clinical Report Generator.

Generates comprehensive, clinician-friendly reports for drug repurposing candidates.

Output formats:
- Markdown (for documentation)
- JSON (for API responses)
- HTML (for web display)
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """Report metadata."""
    disease_name: str
    disease_id: str
    generation_date: str
    total_candidates: int
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int
    ranking_strategy: str


class ReportGenerator:
    """
    Generates clinical reports for drug repurposing candidates.
    """
    
    def __init__(self):
        self.report_version = "1.0"
    
    def generate_markdown_report(
        self,
        disease_name: str,
        disease_id: str,
        ranked_candidates: List,  # List[RankedCandidate]
        top_n: int = 10,
        include_details: bool = True
    ) -> str:
        """
        Generate Markdown report.
        
        Args:
            disease_name: Disease name
            disease_id: Disease identifier
            ranked_candidates: List of ranked candidates
            top_n: Number of top candidates to include
            include_details: Include detailed breakdown
            
        Returns:
            Markdown-formatted report
        """
        # Get top N
        top_candidates = ranked_candidates[:top_n]
        
        # Count by tier
    
        
        # Generate report
        lines = []
        lines.append(f"# Drug Repurposing Report: {disease_name}")
        lines.append(f"")
        lines.append(f"**Disease ID:** {disease_id}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Report Version:** {self.report_version}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # Executive Summary
        lines.append(f"## Executive Summary")
        lines.append(f"")
        lines.append(f"Analyzed **{len(ranked_candidates)}** drug repurposing candidates for {disease_name}.")
        lines.append(f"")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # Top Candidates
        lines.append(f"## Top {top_n} Candidates")
        lines.append(f"")
        
        for candidate in top_candidates:
            
            lines.append(f"### \{candidate.rank}. {candidate.drug_name}")
            lines.append(f"")
            lines.append(f"**Drug ID:** {candidate.drug_id}")
        
            lines.append(f"**Final Score:** {candidate.final_score:.1f}/100")
            lines.append(f"")
            
            if include_details:
                lines.append(f"**Score Breakdown:**")
                lines.append(f"- Composite Score: {candidate.composite_score:.1f}/100")
                lines.append(f"- Novelty Score: {candidate.novelty_score:.1f}/100")
                lines.append(f"- Feasibility Score: {candidate.feasibility_score:.1f}/100")
                lines.append(f"")
            
            lines.append(f"**Recommendation:**")
            lines.append(f"{candidate.recommendation}")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")
        
        # Methodology
        lines.append(f"## Methodology")
        lines.append(f"")
        lines.append(f"This report was generated using a hybrid drug discovery approach:")
        lines.append(f"")
        lines.append(f"1. **Direct Disease-Drug Query:** Identified drugs with existing clinical evidence")
        lines.append(f"2. **Target-Based Discovery:** Found additional candidates through disease-associated targets")
        lines.append(f"3. **Multi-Factor Scoring:** Evaluated candidates based on:")
        lines.append(f"   - Clinical phase (40% weight)")
        lines.append(f"   - Evidence strength (30% weight)")
        lines.append(f"   - Mechanism overlap (20% weight)")
        lines.append(f"   - Safety profile (10% weight)")
        lines.append(f"4. **Ranking:** Prioritized candidates considering novelty and feasibility")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # Next Steps
        lines.append(f"## Recommended Next Steps")
        lines.append(f"")
        lines.append(f"**For High Priority Candidates:**")
        lines.append(f"- Conduct comprehensive literature review")
        lines.append(f"- Analyze existing clinical trial data")
        lines.append(f"- Assess intellectual property landscape")
        lines.append(f"- Design pilot study or case series")
        lines.append(f"")
        lines.append(f"**For Medium Priority Candidates:**")
        lines.append(f"- Perform detailed mechanism analysis")
        lines.append(f"- Validate pathway overlap computationally")
        lines.append(f"- Review safety and pharmacokinetic data")
        lines.append(f"")
        lines.append(f"**For Low Priority Candidates:**")
        lines.append(f"- Monitor for emerging evidence")
        lines.append(f"- Consider for basic research investigations")
        lines.append(f"")
        
        return "\n".join(lines)
    
    def generate_json_report(
        self,
        disease_name: str,
        disease_id: str,
        ranked_candidates: List,
        include_metadata: bool = True
    ) -> Dict:
        """
        Generate JSON report for API responses.
        
        Args:
            disease_name: Disease name
            disease_id: Disease identifier
            ranked_candidates: List of ranked candidates
            include_metadata: Include report metadata
            
        Returns:
            Dictionary suitable for JSON serialization
        """
        # Count by tier
   
        
        report = {
            "disease": {
                "name": disease_name,
                "id": disease_id
            },
            "summary": {
                "total_candidates": len(ranked_candidates),
                
            },
            "candidates": [c.to_dict() for c in ranked_candidates]
        }
        
        if include_metadata:
            report["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "report_version": self.report_version,
                "methodology": "hybrid_discovery"
            }
        
        return report
    
    def generate_html_report(
        self,
        disease_name: str,
        disease_id: str,
        ranked_candidates: List,
        top_n: int = 10
    ) -> str:
        """
        Generate HTML report for web display.
        
        Args:
            disease_name: Disease name
            disease_id: Disease identifier
            ranked_candidates: List of ranked candidates
            top_n: Number of top candidates to include
            
        Returns:
            HTML string
        """
        # Get top N
        top_candidates = ranked_candidates[:top_n]
        
       
                
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drug Repurposing Report: {disease_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .candidate {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 5px solid #ddd;
        }}
        .candidate.high {{
            border-left-color: #e74c3c;
        }}
        .candidate.medium {{
            border-left-color: #f39c12;
        }}
        .candidate.low {{
            border-left-color: #95a5a6;
        }}
        .score {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .tier {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .tier.high {{
            background: #e74c3c;
            color: white;
        }}
        .tier.medium {{
            background: #f39c12;
            color: white;
        }}
        .tier.low {{
            background: #95a5a6;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Drug Repurposing Report</h1>
        <h2>{disease_name}</h2>
        <p>Disease ID: {disease_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h3>Summary</h3>
        <p>Analyzed <strong>{len(ranked_candidates)}</strong> drug repurposing candidates.</p>
        <ul>
      
        </ul>
    </div>
    
    <h3>Top {top_n} Candidates</h3>
"""
        
        for candidate in top_candidates:
            
            html += f"""
    <div class="candidate">
        <h4>#{candidate.rank} {candidate.drug_name}</h4>
        <p>
            <span class="score">{candidate.final_score:.1f}/100</span>
            <span class="tier "></span>
        </p>
        <p><strong>Drug ID:</strong> {candidate.drug_id}</p>
        <p><strong>Recommendation:</strong> {candidate.recommendation}</p>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        return html


# Example usage
if __name__ == "__main__":
    from kg.candidate_ranker import RankedCandidate
    
    # Mock candidates
    candidates = [
        RankedCandidate(
            drug_id="CHEMBL1",
            drug_name="Drug A",
            rank=1,
            composite_score=85.0,
            novelty_score=60.0,
            feasibility_score=90.0,
            final_score=82.0,
            tier="High Priority",
            recommendation="Strong repurposing candidate"
        )
    ]
    
    # Generate reports
    generator = ReportGenerator()
    
    # Markdown
    md_report = generator.generate_markdown_report(
        "Alzheimer's Disease",
        "EFO_0000249",
        candidates,
        top_n=10
    )
    
    print(f"\n{'='*60}")
    print("REPORT GENERATOR TEST (Markdown)")
    print(f"{'='*60}")
    print(md_report[:500] + "...")
