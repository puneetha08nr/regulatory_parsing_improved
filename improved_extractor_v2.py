"""
Improved Control Extractor for UAE IA Regulation PDF - V2
Handles multiple controls per page in table format

Key improvements:
1. Better table handling using pdfplumber's table extraction
2. Proper detection of multiple controls per page
3. More robust control boundary detection
4. Fallback to text extraction when tables aren't detected
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pdfplumber
from dataclasses import dataclass, asdict


@dataclass
class ExtractedControl:
    """Complete control structure matching reference format"""
    control_family: Dict[str, str]
    control_subfamily: Dict[str, str]
    control: Dict
    applicablility: List[str]  # Note: typo in reference file
    breadcrumb: str


class ImprovedControlExtractorV2:
    """Extract controls from UAE IA PDF with complete structure - handles tables"""
    
    def __init__(self):
        # Family and subfamily mappings
        self.family_map = {
            "M1": "Strategy and Planning",
            "M2": "Information Security Risk Management",
            "M3": "Human Resources Security",
            "M4": "Asset Management",
            "M5": "External Party Access",
            "M6": "Information Security Incident Management",
            "T1": "Physical and Environmental Security",
            "T2": "Operations Management",
            "T3": "Communications Management",
            "T4": "Access Control",
            "T5": "Information Systems Acquisition & Dev",
            "T6": "Business Continuity Management",
            "T7": "Compliance",
            "T8": "Cloud Security",
            "T9": "Emerging Technologies"
        }
        
        self.subfamily_map = {
            "M1.1": "Entity Context and Leadership",
            "M1.2": "Information Security Strategy",
            "M1.3": "Resources and Competence",
            "M1.4": "Information Security Performance",
            "M2.1": "Risk Management Framework",
            "M2.2": "Information Security Risk Assessment",
            "M2.3": "Information Security Risk Treatment",
            "M3.1": "Prior to Employment",
            "M3.2": "During Employment",
            "M3.3": "Termination and Change of Employment",
            "M4.1": "Responsibility for Assets",
            "M4.2": "Information Classification",
            "M4.3": "Media Handling",
            "M5.1": "Information Security in Relationships",
            "M5.2": "Service Delivery Management",
            "M6.1": "Management of Incidents and Improvements",
            "T1.1": "Secure Areas",
            "T1.2": "Equipment Security",
            "T2.1": "Operational Procedures and Responsibilities",
            "T2.2": "Protection from Malware",
            "T2.3": "Backup",
            "T2.4": "Logging and Monitoring",
            "T2.5": "Control of Operational Software",
            "T3.1": "Network Security Management",
            "T3.2": "Information Transfer",
            "T4.1": "Business Requirements of Access Control",
            "T4.2": "User Access Management",
            "T4.3": "User Responsibilities",
            "T4.4": "System and Application Access Control",
            "T5.1": "Security Requirements of Information Systems",
            "T5.2": "Security in Development and Support Processes",
            "T5.3": "Test Data",
            "T6.1": "Information Security Continuity",
            "T6.2": "Redundancies",
            "T7.1": "Compliance with Legal and Contractual Requirements",
            "T7.2": "Information Security Reviews",
            "T8.1": "Cloud Governance and Strategy",
            "T8.2": "Cloud Operational Security",
            "T9.1": "Artificial Intelligence and Machine Learning Security",
            "T9.2": "Internet of Things (IoT) Security",
            "T9.3": "Blockchain and Distributed Ledger Technology"
        }
        
        self.control_pattern = re.compile(r'([MT]\d+\.\d+\.\d+)')
    
    def extract_from_pdf(self, pdf_path: str) -> List[ExtractedControl]:
        """
        Extract all controls from PDF with complete structure.
        Uses table extraction for better handling of tabular control format.
        
        Args:
            pdf_path: Path to UAE IA Regulation PDF
            
        Returns:
            List of ExtractedControl objects
        """
        print(f"Extracting controls from: {pdf_path}")
        
        all_controls = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                if page_num % 10 == 0:
                    print(f"Processing page {page_num}/{len(pdf.pages)}...", end='\r')
                
                # Try to extract controls from this page
                page_controls = self._extract_controls_from_page(page, page_num)
                all_controls.extend(page_controls)
        
        print(f"\n✓ Extracted {len(all_controls)} controls")
        return all_controls
    
    def _extract_controls_from_page(self, page, page_num: int) -> List[ExtractedControl]:
        """
        Extract all controls from a single page.
        Handles multiple controls per page.
        """
        controls = []
        
        # First, try table extraction
        tables = page.extract_tables()
        
        if tables:
            # Process tables - each table might contain one or more controls
            for table in tables:
                table_controls = self._extract_controls_from_table(table)
                controls.extend(table_controls)
        
        # Also extract text to catch any controls not in tables
        # or to supplement table data
        text = page.extract_text()
        if text:
            text = self._clean_text(text, page_num)
            text_controls = self._extract_controls_from_text(text)
            
            # Merge with table controls - avoid duplicates
            existing_ids = {c.control['id'] for c in controls}
            for ctrl in text_controls:
                if ctrl.control['id'] not in existing_ids:
                    controls.append(ctrl)
        
        return controls
    
    def _extract_controls_from_table(self, table: List[List[str]]) -> List[ExtractedControl]:
        """
        Extract controls from a table structure.
        Each table row might represent a control.
        """
        controls = []
        
        if not table:
            return controls
        
        # Convert table to text blocks per control
        # Look for control IDs in the first column
        for row_idx, row in enumerate(table):
            if not row:
                continue
            
            # Join all cells in the row
            row_text = ' '.join([cell or '' for cell in row])
            
            # Check if this row contains a control ID
            control_ids = self.control_pattern.findall(row_text)
            
            if control_ids:
                # Found control ID(s) in this row
                for control_id in control_ids:
                    # Build control text from this row and potentially following rows
                    control_text = self._build_control_text_from_table(
                        table, row_idx, control_id
                    )
                    
                    if control_text:
                        control = self._parse_control_block(control_id, control_text)
                        if control and control not in controls:
                            controls.append(control)
        
        return controls
    
    def _build_control_text_from_table(
        self, 
        table: List[List[str]], 
        start_row: int, 
        control_id: str
    ) -> str:
        """
        Build complete control text from table starting at a specific row.
        Continues until the next control ID is found or table ends.
        """
        control_text_parts = []
        
        for row_idx in range(start_row, len(table)):
            row = table[row_idx]
            row_text = ' '.join([cell or '' for cell in row])
            
            # If we encounter another control ID (different from current), stop
            if row_idx > start_row:
                other_ids = self.control_pattern.findall(row_text)
                if other_ids and other_ids[0] != control_id:
                    break
            
            control_text_parts.append(row_text)
        
        return '\n'.join(control_text_parts)
    
    def _extract_controls_from_text(self, text: str) -> List[ExtractedControl]:
        """
        Extract controls from plain text (fallback method).
        This handles cases where table extraction doesn't work well.
        """
        controls = []
        
        # Find all control IDs in the text
        control_positions = []
        for match in self.control_pattern.finditer(text):
            control_positions.append((match.group(1), match.start()))
        
        # Extract text for each control
        for i, (control_id, start_pos) in enumerate(control_positions):
            # Determine end position (start of next control or end of text)
            if i + 1 < len(control_positions):
                end_pos = control_positions[i + 1][1]
            else:
                end_pos = len(text)
            
            control_text = text[start_pos:end_pos]
            
            # Parse this control
            control = self._parse_control_block(control_id, control_text)
            if control:
                controls.append(control)
        
        return controls
    
    def _parse_control_block(self, control_id: str, text: str) -> Optional[ExtractedControl]:
        """Parse a complete control block into structured format"""
        try:
            # Extract control name (text after ID, before "Control" or "Priority")
            name_match = re.search(
                rf'{re.escape(control_id)}\s+(.+?)(?=\s+Control\s|Priority|Applicability|$)',
                text,
                re.DOTALL | re.IGNORECASE
            )
            control_name = name_match.group(1).strip() if name_match else "Unknown"
            control_name = re.sub(r'\s+', ' ', control_name)
            # Clean up control name - remove common artifacts
            control_name = re.sub(r'Priority.*$', '', control_name, flags=re.IGNORECASE).strip()
            control_name = re.sub(r'Applicability.*$', '', control_name, flags=re.IGNORECASE).strip()
            
            # Get family and subfamily
            family_id = control_id.split('.')[0]
            subfamily_id = '.'.join(control_id.split('.')[:2])
            
            family_name = self.family_map.get(family_id, "Unknown")
            subfamily_name = self.subfamily_map.get(subfamily_id, "")
            
            # Extract Control description
            description = self._extract_section(
                text,
                r'Control\s*(?:Statement)?',
                [r'Sub-Control', r'Implementation\s+Guidance', r'Priority', r'Applicability']
            )
            
            # Extract Sub-controls
            sub_controls = self._extract_sub_controls(text, control_id)
            
            # Extract Implementation Guidelines
            implementation_guidelines = self._extract_section(
                text,
                r'Implementation\s+Guidance',
                [r'External\s+Factors', r'Internal\s+Factors', r'Guidance\s+Points', 
                 r'Communication', r'Priority', r'Applicability', r'Based on risk assessment']
            )
            
            # Extract External Factors
            external_factors = self._extract_list_items(
                text,
                r'External\s+Factors',
                [r'Internal\s+Factors', r'Guidance\s+Points', r'Communication', r'Priority']
            )
            
            # Extract Internal Factors
            internal_factors = self._extract_list_items(
                text,
                r'Internal\s+Factors',
                [r'Guidance\s+Points', r'Communication', r'Priority', r'Applicability']
            )
            
            # Extract Guidance Points
            guidance_points = self._extract_list_items(
                text,
                r'Guidance\s+Points',
                [r'Communication', r'Priority', r'Applicability']
            )
            
            # Extract Communication Requirements
            communication_requirements = self._extract_list_items(
                text,
                r'Communication\s*(?:Requirements)?',
                [r'Priority', r'Applicability', r'Based on risk assessment']
            )
            
            # Extract Applicability
            applicability = self._extract_applicability(text)
            
            # Build control dictionary
            control_dict = {
                "id": control_id,
                "name": control_name,
                "description": description,
                "sub_controls": sub_controls
            }
            
            # Add optional fields only if they exist
            if implementation_guidelines:
                control_dict["implementation_guidelines"] = implementation_guidelines
            if external_factors:
                control_dict["external_factors"] = external_factors
            if internal_factors:
                control_dict["internal_factors"] = internal_factors
            if guidance_points:
                control_dict["guidance_points"] = guidance_points
            if communication_requirements:
                control_dict["communication_requirements"] = communication_requirements
            
            # Build breadcrumb
            breadcrumb = f"Management > {family_name} > {subfamily_name} > {control_name}"
            
            return ExtractedControl(
                control_family={
                    "number": family_id,
                    "name": family_name
                },
                control_subfamily={
                    "number": subfamily_id,
                    "name": subfamily_name
                },
                control=control_dict,
                applicablility=applicability,
                breadcrumb=breadcrumb
            )
        
        except Exception as e:
            print(f"\nError parsing control {control_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_section(self, text: str, start_pattern: str, end_patterns: List[str]) -> str:
        """Extract text between a start pattern and any end pattern"""
        # Create pattern that matches start and stops at any end pattern
        pattern = rf'{start_pattern}\s*(?:\(for information purpose only\))?\s*(.*?)(?={"|".join(end_patterns)}|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content)
            return content
        return ""
    
    def _extract_sub_controls(self, text: str, control_id: str) -> List[str]:
        """Extract sub-controls and format them with control_id.letter prefix"""
        section = self._extract_section(
            text,
            r'Sub-Control',
            [r'Implementation\s+Guidance', r'External\s+Factors', r'Internal\s+Factors', 
             r'Guidance\s+Points', r'Priority', r'Applicability']
        )
        
        if not section:
            return []
        
        # Clean up "The entity shall:" prefix if present
        section = re.sub(r'^The entity shall:\s*', '', section, flags=re.IGNORECASE)
        
        # Split by numbered items: 1), 2), 3), etc.
        items = re.split(r'(?:^|\s)(\d+)\)\s+', section)
        
        formatted_sub_controls = []
        # Items list will be: ['', '1', 'first item', '2', 'second item', ...]
        for i in range(1, len(items), 2):
            if i + 1 < len(items):
                item = items[i + 1].strip()
                if len(item) > 5:  # Filter out noise
                    # Use the number from the original text
                    num = int(items[i])
                    letter = chr(96 + num)  # a, b, c... (96 + 1 = 97 = 'a')
                    formatted_sub_controls.append(f"{control_id}.{letter}: {item}")
        
        return formatted_sub_controls
    
    def _extract_list_items(self, text: str, start_pattern: str, end_patterns: List[str]) -> List[str]:
        """Extract list items (for external_factors, internal_factors, guidance_points, etc.)"""
        section = self._extract_section(text, start_pattern, end_patterns)
        
        if not section:
            return []
        
        # Split by numbered items: 1), 2), 3), etc.
        items = re.split(r'(?:^|\s)(\d+)\)\s+', section)
        
        formatted_items = []
        # Items list will be: ['', '1', 'first item', '2', 'second item', ...]
        for i in range(1, len(items), 2):
            if i + 1 < len(items):
                num = items[i]
                content = items[i + 1].strip()
                if content:
                    formatted_items.append(f"{num}) {content}")
        
        # If no numbered items found, try lettered items: a., b., c., etc.
        if not formatted_items:
            items = re.split(r'([a-z]\.)\s+', section)
            for i in range(1, len(items), 2):
                if i + 1 < len(items):
                    label = items[i]
                    content = items[i + 1].strip()
                    if content:
                        formatted_items.append(f"{label} {content}")
        
        return formatted_items
    
    def _extract_applicability(self, text: str) -> List[str]:
        """Extract applicability (Priority levels and 'always')"""
        applicability = []
        
        # Look for Priority levels: P1, P2, P3, P4
        priority_match = re.search(r'Priority\s+(P\d+)', text, re.IGNORECASE)
        if priority_match:
            applicability.append(priority_match.group(1))
        
        # Check for "Based on risk assessment"
        if re.search(r'Based on risk assessment', text, re.IGNORECASE):
            applicability.append("Based on risk assessment")
        
        # Check for "always" or "always applicable"
        if re.search(r'\balways\s+(?:applicable)?', text, re.IGNORECASE):
            if "Based on risk assessment" not in applicability:
                applicability.append("always")
        
        # If no explicit priority found, default to P1
        if not applicability:
            applicability.append("P1")
        
        return applicability
    
    def _clean_text(self, text: str, page_num: int = None) -> str:
        """Clean PDF artifacts from text, including page numbers"""
        # Remove page numbers in various formats
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*\n\s*\n', '', text)
        
        # Remove page numbers at end of lines
        text = re.sub(r'\n\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove header/footer artifacts
        text = re.sub(r'UAE\s+Information\s+Assurance\s+Regulation', '', text, flags=re.IGNORECASE)
        text = re.sub(r'UAE\s+IA\s+Regulation', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n', text)
        
        return text.strip()
    
    def save_controls(self, controls: List[ExtractedControl], output_path: str):
        """Save extracted controls to JSON file"""
        output_data = [asdict(control) for control in controls]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(controls)} controls to {output_path}")


def main():
    """Main extraction function"""
    pdf_path = "data/01_raw/regulation/UAE_IA_Regulation.pdf"
    output_path = "data/02_processed/uae_ia_controls_structured_v2.json"
    
    extractor = ImprovedControlExtractorV2()
    
    # Extract controls
    controls = extractor.extract_from_pdf(pdf_path)
    
    # Save to JSON
    extractor.save_controls(controls, output_path)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Extraction Statistics")
    print("=" * 60)
    print(f"Total controls: {len(controls)}")
    
    # Count by family
    family_counts = {}
    for control in controls:
        family = control.control_family['number']
        family_counts[family] = family_counts.get(family, 0) + 1
    
    print(f"\nControls by family:")
    for family in sorted(family_counts.keys()):
        family_name = extractor.family_map.get(family, "Unknown")
        print(f"  {family} ({family_name}): {family_counts[family]} controls")
    
    # Count controls with sub-controls
    with_subcontrols = sum(1 for c in controls if c.control.get('sub_controls'))
    print(f"\nControls with sub-controls: {with_subcontrols}")
    
    # Count controls with guidance points
    with_guidance = sum(1 for c in controls if c.control.get('guidance_points'))
    print(f"Controls with guidance points: {with_guidance}")
    
    # Show first few control IDs as sample
    print(f"\nSample control IDs:")
    for control in controls[:10]:
        print(f"  - {control.control['id']}: {control.control['name']}")


if __name__ == "__main__":
    main()