"""
RegRAG-Xref Three-Stage Pipeline
Implements the RegRAG-Xref methodology for policy document processing and mapping

Stages:
1. Stage 1: Raw PDF/DOCX (source of truth)
2. Stage 2: Standardized TXT (hierarchical segmentation)
3. Stage 3: Structured JSON (machine-readable corpus)
4. Final Mapping: Map to UAE IA Regulation and ISO 27001
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class StandardizedSection:
    """Represents a section in Stage 2 (Standardized TXT)"""
    section_id: str  # e.g., "5.1", "5.2"
    section_header: str  # e.g., "Asset Inventory"
    content: str  # Full text content
    subsections: List['StandardizedSection'] = None  # Nested sections


@dataclass
class StructuredSection:
    """Represents a section in Stage 3 (Structured JSON)"""
    section_id: str
    section_header: str
    content: str
    metadata: Dict
    parent_section_id: Optional[str] = None


@dataclass
class RegulationMapping:
    """Final mapping output linking internal policy to regulations"""
    internal_policy_section: str  # e.g., "Section 5.1 – Asset Inventory"
    uae_ia_control: str  # e.g., "T1.2.1 – Inventory of Assets"
    iso27001_mapping: str  # e.g., "A.5.9 - Inventory of information assets"
    compliance_status: str  # "Fully Addressed", "Partially Addressed", "Not Addressed"
    evidence_text: Optional[str] = None
    mapping_date: str = None


class RegRAGXrefPipeline:
    """
    Implements the RegRAG-Xref three-stage pipeline
    """
    
    def __init__(self, output_dir: str = "data/06_regrag_xref"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each stage
        (self.output_dir / "stage1_raw").mkdir(exist_ok=True)
        (self.output_dir / "stage2_standardized_txt").mkdir(exist_ok=True)
        (self.output_dir / "stage3_structured_json").mkdir(exist_ok=True)
        (self.output_dir / "stage4_mappings").mkdir(exist_ok=True)
    
    def process_policy_document(
        self,
        policy_json_path: str,
        policy_name: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict:
        """
        Process a policy document through all three stages
        
        Args:
            policy_json_path: Path to policy JSON (from flexible_policy_extractor)
            policy_name: Optional policy name override
            document_id: Optional document ID (e.g., "<CLIENT>-ORG-IMS-POL-009")
        
        Returns:
            Dict with paths to all stage outputs
        """
        print(f"\n{'='*60}")
        print(f"Processing Policy Document: {policy_json_path}")
        print(f"{'='*60}\n")
        
        # Load policy JSON
        with open(policy_json_path, 'r', encoding='utf-8') as f:
            policy_data = json.load(f)
        
        if not policy_data:
            raise ValueError(f"No data found in {policy_json_path}")
        
        # Extract metadata
        first_passage = policy_data[0]
        if not policy_name:
            policy_name = first_passage.get('metadata', {}).get('policy_name', 'Unknown Policy')
        if not document_id:
            document_id = first_passage.get('metadata', {}).get('policy_id', 'UNKNOWN-POL')
        
        # Stage 2: Convert to Standardized TXT
        print("Stage 2: Creating Standardized TXT (Hierarchical Segmentation)...")
        txt_path = self._create_standardized_txt(policy_data, policy_name, document_id)
        print(f"✓ Created: {txt_path}\n")
        
        # Stage 3: Convert to Structured JSON
        print("Stage 3: Creating Structured JSON (Machine-Readable Corpus)...")
        json_path = self._create_structured_json(policy_data, policy_name, document_id)
        print(f"✓ Created: {json_path}\n")
        
        return {
            "stage1_raw": policy_json_path,  # Original is already raw
            "stage2_standardized_txt": str(txt_path),
            "stage3_structured_json": str(json_path),
            "document_id": document_id,
            "policy_name": policy_name
        }
    
    def _create_standardized_txt(
        self,
        policy_data: List[Dict],
        policy_name: str,
        document_id: str
    ) -> Path:
        """
        Stage 2: Create standardized TXT with hierarchical segmentation
        Format: SECTION X: SECTION_NAME\nContent...
        """
        output_path = self.output_dir / "stage2_standardized_txt" / f"{document_id}_standardized.txt"
        
        lines = []
        lines.append(f"File Name: {policy_name}_v1.txt")
        lines.append("=" * 80)
        lines.append("")
        
        # Group passages by section/heading
        sections = self._group_by_section(policy_data)
        
        section_num = 1
        for section_header, passages in sections.items():
            # Determine section ID (try to extract from heading, or use sequential)
            section_id = self._extract_section_id(section_header, section_num)
            
            lines.append(f"SECTION {section_id}: {section_header}")
            lines.append("")
            
            # Add content from all passages in this section
            for passage in passages:
                text = passage.get('text', '').strip()
                if text:
                    # Clean up text
                    text = self._clean_text_for_txt(text)
                    lines.append(text)
                    lines.append("")
            
            section_num += 1
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def _create_structured_json(
        self,
        policy_data: List[Dict],
        policy_name: str,
        document_id: str
    ) -> Path:
        """
        Stage 3: Create structured JSON (machine-readable corpus)
        Format matches RegRAG-Xref specification
        """
        output_path = self.output_dir / "stage3_structured_json" / f"{document_id}_structured.json"
        
        # Extract document metadata
        first_passage = policy_data[0]
        metadata = first_passage.get('metadata', {})
        
        # Get last updated date (try to extract from policy or use current date)
        last_updated = self._extract_last_updated(metadata) or datetime.now().strftime("%d-%b-%Y")
        
        # Group passages by section
        sections = self._group_by_section(policy_data)
        
        structured_sections = []
        section_num = 1
        
        for section_header, passages in sections.items():
            section_id = self._extract_section_id(section_header, section_num)
            
            # Combine content from all passages in this section
            content_parts = []
            for passage in passages:
                text = passage.get('text', '').strip()
                if text:
                    content_parts.append(text)
            
            content = ' '.join(content_parts)
            
            # Create structured section
            structured_section = {
                "section_id": section_id,
                "section_header": section_header,
                "content": content,
                "metadata": {
                    "owner": metadata.get('owner', 'CISO'),  # Default to CISO
                    "classification": metadata.get('classification', 'Internal'),
                    "status": metadata.get('status', 'Final'),
                    "policy_id": document_id,
                    "policy_name": policy_name
                }
            }
            
            structured_sections.append(structured_section)
            section_num += 1
        
        # Create final structured JSON
        structured_doc = {
            "document_id": document_id,
            "document_title": policy_name,
            "last_updated": last_updated,
            "sections": structured_sections
        }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_doc, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _group_by_section(self, policy_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group passages by their section/heading"""
        sections = {}
        
        for passage in policy_data:
            # Get section/heading
            section = passage.get('section') or passage.get('name', 'UNKNOWN')
            
            # Clean section name
            section = self._clean_section_name(section)
            
            if section not in sections:
                sections[section] = []
            
            sections[section].append(passage)
        
        return sections
    
    def _extract_section_id(self, section_header: str, default_num: int) -> str:
        """
        Extract section ID from header (e.g., "5.1" from "5.1 Asset Inventory")
        Falls back to sequential numbering
        """
        # Try to find pattern like "5.1", "5.2.1", etc.
        match = re.search(r'(\d+(?:\.\d+)*)', section_header)
        if match:
            return match.group(1)
        
        # Try to find single number
        match = re.search(r'^(\d+)', section_header)
        if match:
            return match.group(1)
        
        # Fallback to sequential
        return str(default_num)
    
    def _clean_section_name(self, section: str) -> str:
        """Clean section name for grouping"""
        # Remove common prefixes
        section = re.sub(r'^(POLICY|SECTION|CHAPTER)\s+', '', section, flags=re.IGNORECASE)
        section = re.sub(r'^[A-Z\s]+:\s*', '', section)  # Remove "POLICY STATEMENT:"
        section = section.strip()
        
        # If empty, use default
        if not section or len(section) < 3:
            return "General"
        
        return section
    
    def _clean_text_for_txt(self, text: str) -> str:
        """Clean text for TXT output"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special markers if any
        text = text.replace('<CLIENT>', '<CLIENT>')
        return text.strip()
    
    def _extract_last_updated(self, metadata: Dict) -> Optional[str]:
        """Extract last updated date from metadata"""
        # Try various date fields
        for field in ['last_updated', 'last_updated_date', 'review_date', 'date']:
            if field in metadata:
                date_val = metadata[field]
                if date_val:
                    return date_val
        
        return None


class RegulationMapper:
    """
    Maps internal policy sections to UAE IA Regulation and ISO 27001
    Uses semantic matching and keyword-based mapping
    """
    
    # UAE IA Control mappings (based on control families)
    UAE_IA_MAPPINGS = {
        # Asset Management (T1 family)
        "asset inventory": ["T1.2.1", "T1.2.2"],
        "asset ownership": ["T1.2.2"],
        "asset labelling": ["T1.3.2"],
        "asset handling": ["T1.3.1", "T1.3.2"],
        "removable media": ["T1.4.1", "T1.4.2"],
        "media disposal": ["T1.4.3"],
        
        # Access Control (T5 family)
        "access control": ["T5.1.1", "T5.2.1"],
        "user access": ["T5.2.1", "T5.2.2"],
        "privilege management": ["T5.2.3"],
        
        # Risk Management (M2 family)
        "risk assessment": ["M2.1.1", "M2.1.2"],
        "risk treatment": ["M2.2.1"],
        
        # Incident Management (M6 family)
        "incident management": ["M6.1.1", "M6.2.1"],
        "incident response": ["M6.1.2"],
    }
    
    # ISO 27001:2022 mappings (comprehensive)
    ISO27001_MAPPINGS = {
        # Asset Management (A.5.9, A.5.13)
        "asset inventory": ["A.5.9"],
        "asset ownership": ["A.5.9"],
        "asset register": ["A.5.9"],
        "asset labelling": ["A.5.13"],
        "asset classification": ["A.5.13"],
        "asset handling": ["A.5.13"],
        "information labelling": ["A.5.13"],
        "media handling": ["A.5.10"],
        "removable media": ["A.5.10"],
        "media disposal": ["A.5.10"],
        
        # Access Control (A.5.15, A.5.16, A.5.17, A.5.18)
        "access control": ["A.5.15", "A.5.16", "A.5.17"],
        "user access": ["A.5.16", "A.5.17"],
        "privilege management": ["A.5.17"],
        "authentication": ["A.5.16"],
        "authorization": ["A.5.17"],
        "access rights": ["A.5.16", "A.5.17"],
        
        # Risk Management (A.5.1, A.5.2, A.5.3)
        "risk assessment": ["A.5.1", "A.5.2"],
        "risk treatment": ["A.5.2"],
        "risk management": ["A.5.1", "A.5.2", "A.5.3"],
        "information security risk": ["A.5.1"],
        
        # Incident Management (A.5.26, A.5.27)
        "incident management": ["A.5.26", "A.5.27"],
        "incident response": ["A.5.26"],
        "security incident": ["A.5.26"],
        "incident reporting": ["A.5.26"],
        
        # Physical Security (A.5.11, A.5.12)
        "physical security": ["A.5.11", "A.5.12"],
        "physical access": ["A.5.11"],
        "secure areas": ["A.5.11"],
        
        # Operations Security (A.5.14, A.5.23)
        "operations management": ["A.5.14"],
        "backup": ["A.5.23"],
        "data backup": ["A.5.23"],
        "backup procedures": ["A.5.23"],
        
        # Network Security (A.5.19, A.5.20)
        "network security": ["A.5.19", "A.5.20"],
        "network controls": ["A.5.19"],
        "network segmentation": ["A.5.19"],
        
        # Cryptography (A.5.24)
        "encryption": ["A.5.24"],
        "cryptographic": ["A.5.24"],
        "cryptography": ["A.5.24"],
        
        # Supplier Relationships (A.5.19)
        "third party": ["A.5.19"],
        "supplier": ["A.5.19"],
        "vendor": ["A.5.19"],
        
        # Compliance (A.5.35, A.5.36)
        "compliance": ["A.5.35", "A.5.36"],
        "legal compliance": ["A.5.35"],
        "regulatory compliance": ["A.5.35"],
    }
    
    def __init__(self, uae_ia_controls_path: Optional[str] = None):
        """
        Initialize mapper with UAE IA controls
        
        Args:
            uae_ia_controls_path: Path to UAE IA controls JSON (optional)
        """
        self.uae_ia_controls = {}
        if uae_ia_controls_path and Path(uae_ia_controls_path).exists():
            self._load_uae_ia_controls(uae_ia_controls_path)
    
    def _load_uae_ia_controls(self, controls_path: str):
        """Load UAE IA controls for detailed mapping"""
        try:
            with open(controls_path, 'r', encoding='utf-8') as f:
                controls_data = json.load(f)
            
            # Index by control ID
            if isinstance(controls_data, list):
                for control in controls_data:
                    control_id = control.get('control', {}).get('id') or control.get('control_id')
                    if control_id:
                        self.uae_ia_controls[control_id] = control
        except Exception as e:
            print(f"Warning: Could not load UAE IA controls: {e}")
    
    def map_policy_section(
        self,
        section_header: str,
        section_content: str,
        use_semantic_matching: bool = False
    ) -> List[RegulationMapping]:
        """
        Map a policy section to regulations
        
        Args:
            section_header: Section header (e.g., "Asset Inventory")
            section_content: Section content text
            use_semantic_matching: Use semantic matching (requires NLI model)
        
        Returns:
            List of RegulationMapping objects
        """
        mappings = []
        
        # Normalize section header for keyword matching
        header_lower = section_header.lower()
        content_lower = section_content.lower()
        
        # Find UAE IA mappings
        uae_ia_controls = self._find_uae_ia_mappings(header_lower, content_lower)
        
        # Find ISO 27001 mappings
        iso27001_controls = self._find_iso27001_mappings(header_lower, content_lower)
        
        # Create mappings
        for uae_control in uae_ia_controls:
            for iso_control in iso27001_controls:
                mapping = RegulationMapping(
                    internal_policy_section=f"Section {section_header}",
                    uae_ia_control=uae_control,
                    iso27001_mapping=iso_control,
                    compliance_status="Fully Addressed",  # Default, can be refined with NLI
                    evidence_text=section_content[:500],  # First 500 chars as evidence
                    mapping_date=datetime.now().strftime("%Y-%m-%d")
                )
                mappings.append(mapping)
        
        # If no mappings found, create placeholder
        if not mappings:
            mapping = RegulationMapping(
                internal_policy_section=f"Section {section_header}",
                uae_ia_control="Not Mapped",
                iso27001_mapping="Not Mapped",
                compliance_status="Not Addressed",
                mapping_date=datetime.now().strftime("%Y-%m-%d")
            )
            mappings.append(mapping)
        
        return mappings
    
    def _find_uae_ia_mappings(self, header_lower: str, content_lower: str) -> List[str]:
        """Find UAE IA control mappings based on keywords"""
        found_controls = []
        
        # Check keyword mappings
        for keyword, control_ids in self.UAE_IA_MAPPINGS.items():
            if keyword in header_lower or keyword in content_lower:
                found_controls.extend(control_ids)
        
        # Remove duplicates and sort
        found_controls = sorted(list(set(found_controls)))
        
        # If we have loaded controls, try to find more specific matches
        if self.uae_ia_controls:
            # Search in control names and descriptions
            for control_id, control_data in self.uae_ia_controls.items():
                control_text = ""
                if isinstance(control_data, dict):
                    control_obj = control_data.get('control', {})
                    control_text = (
                        control_obj.get('name', '') + ' ' +
                        control_obj.get('description', '')
                    ).lower()
                
                # Check if section content matches control
                if control_text and any(word in content_lower for word in control_text.split()[:10]):
                    if control_id not in found_controls:
                        found_controls.append(control_id)
        
        return found_controls[:5]  # Limit to top 5
    
    def _find_iso27001_mappings(self, header_lower: str, content_lower: str) -> List[str]:
        """Find ISO 27001 mappings based on keywords"""
        found_controls = []
        
        # Prioritize header matches over content matches
        header_matches = []
        content_matches = []
        
        for keyword, control_ids in self.ISO27001_MAPPINGS.items():
            if keyword in header_lower:
                header_matches.extend(control_ids)
            elif keyword in content_lower:
                content_matches.extend(control_ids)
        
        # Header matches take priority
        if header_matches:
            found_controls = header_matches
        else:
            found_controls = content_matches
        
        # Remove duplicates while preserving order
        seen = set()
        unique_controls = []
        for control in found_controls:
            if control not in seen:
                seen.add(control)
                unique_controls.append(control)
        
        return unique_controls[:5]  # Limit to top 5


def create_mapping_output(
    structured_json_path: str,
    output_path: str,
    uae_ia_controls_path: Optional[str] = None
):
    """
    Create final mapping output (Stage 4) from structured JSON
    
    Args:
        structured_json_path: Path to Stage 3 structured JSON
        output_path: Path to save mapping CSV/JSON
        uae_ia_controls_path: Optional path to UAE IA controls for detailed mapping
    """
    print(f"\n{'='*60}")
    print(f"Creating Final Mapping Output")
    print(f"{'='*60}\n")
    
    # Load structured JSON
    with open(structured_json_path, 'r', encoding='utf-8') as f:
        structured_doc = json.load(f)
    
    # Initialize mapper
    mapper = RegulationMapper(uae_ia_controls_path)
    
    # Map each section
    all_mappings = []
    for section in structured_doc.get('sections', []):
        section_id = section.get('section_id')
        section_header = section.get('section_header')
        content = section.get('content', '')
        
        print(f"Mapping section: {section_header}...")
        mappings = mapper.map_policy_section(section_header, content)
        all_mappings.extend(mappings)
    
    # Save as CSV
    csv_path = Path(output_path).with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            'Internal Policy Section',
            'UAE IA - Control (Reference)',
            'ISO 27001:2022 Mapping',
            'Compliance Status',
            'Evidence Text',
            'Mapping Date'
        ])
        writer.writeheader()
        
        for mapping in all_mappings:
            writer.writerow({
                'Internal Policy Section': mapping.internal_policy_section,
                'UAE IA - Control (Reference)': mapping.uae_ia_control,
                'ISO 27001:2022 Mapping': mapping.iso27001_mapping,
                'Compliance Status': mapping.compliance_status,
                'Evidence Text': (mapping.evidence_text or '')[:200],  # Truncate for CSV
                'Mapping Date': mapping.mapping_date
            })
    
    print(f"✓ Created mapping CSV: {csv_path}")
    
    # Also save as JSON
    json_path = Path(output_path).with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(m) for m in all_mappings], f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created mapping JSON: {json_path}\n")
    
    return csv_path, json_path


def main():
    """Example usage of RegRAG-Xref pipeline"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python regrag_xref_pipeline.py <policy_json_path> [uae_ia_controls_path]")
        print("\nExample:")
        print("  python regrag_xref_pipeline.py data/02_processed/policies/clientname-IS-POL-00-Asset_Management_Policy_6_for_mapping.json")
        sys.exit(1)
    
    policy_json_path = sys.argv[1]
    uae_ia_controls_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Initialize pipeline
    pipeline = RegRAGXrefPipeline()
    
    # Process policy document
    results = pipeline.process_policy_document(policy_json_path)
    
    # Create final mapping
    structured_json_path = results['stage3_structured_json']
    mapping_output_path = pipeline.output_dir / "stage4_mappings" / f"{results['document_id']}_mapping"
    
    create_mapping_output(
        structured_json_path,
        str(mapping_output_path),
        uae_ia_controls_path
    )
    
    print("\n" + "="*60)
    print("RegRAG-Xref Pipeline Complete!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Stage 2 (TXT): {results['stage2_standardized_txt']}")
    print(f"  Stage 3 (JSON): {results['stage3_structured_json']}")
    print(f"  Stage 4 (Mapping): {mapping_output_path}.csv/json")


if __name__ == "__main__":
    main()
