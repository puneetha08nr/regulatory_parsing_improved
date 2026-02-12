"""
Improved Control Extractor for UAE IA Regulation PDF
Extracts controls with full structure matching the reference format:
- Control description
- Sub-controls
- Implementation guidelines
- External/internal factors
- Guidance points
- Communication requirements

Based on RegNLP methodology: manual structuring and tagging of document sections.
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


class ImprovedControlExtractor:
    """Extract controls from UAE IA PDF with complete structure"""
    
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
    
    def extract_from_pdf(self, pdf_path: str) -> List[ExtractedControl]:
        """
        Extract all controls from PDF with complete structure.
        
        Args:
            pdf_path: Path to UAE IA Regulation PDF
            
        Returns:
            List of ExtractedControl objects
        """
        print(f"Extracting controls from: {pdf_path}")
        
        controls = []
        current_control_text = []
        current_control_id = None
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                if page_num % 10 == 0:
                    print(f"Processing page {page_num}/{len(pdf.pages)}...", end='\r')
                
                # Prefer text from ALL tables when present (so multiple controls per page are captured)
                tables = page.extract_tables()
                if tables:
                    table_texts = [self._table_to_text(t) for t in tables if t]
                    text = "\n\n".join(t for t in table_texts if t.strip())
                else:
                    text = None
                if not text:
                    text = page.extract_text()
                if not text:
                    continue
                
                # Clean text (including page numbers)
                text = self._clean_text(text, page_num)
                
                # Split into lines for processing
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line contains a control ID pattern
                    # Look for control ID at start of line (most common) or after whitespace
                    control_id_match = re.match(r'([MT]\d+\.\d+\.\d+)', line)
                    
                    if control_id_match:
                        detected_id = control_id_match.group(1)
                        
                        # Check if this is a NEW control (different from current)
                        if current_control_id and detected_id != current_control_id:
                            # Save previous control
                            accumulated_text = '\n'.join(current_control_text)
                            # Remove any content from next control that may have leaked in
                            accumulated_text = self._remove_trailing_next_control(
                                accumulated_text, 
                                current_control_id
                            )
                            
                            if accumulated_text:
                                control = self._parse_control_block(
                                    current_control_id,
                                    accumulated_text
                                )
                                if control:
                                    controls.append(control)
                            
                            # Start new control
                            current_control_id = detected_id
                            current_control_text = [line]
                        elif not current_control_id:
                            # First control
                            current_control_id = detected_id
                            current_control_text = [line]
                        else:
                            # Same control ID mentioned again (shouldn't happen, but handle gracefully)
                            current_control_text.append(line)
                    else:
                        # No control ID at start of line - check if line contains a different control ID
                        # (this handles cases where next control appears mid-text)
                        if current_control_id:
                            # Check for any control ID pattern in this line
                            other_control_match = re.search(r'([MT]\d+\.\d+\.\d+)', line)
                            if other_control_match:
                                detected_id = other_control_match.group(1)
                                if detected_id != current_control_id:
                                    # Found next control ID in this line
                                    # Split the line: content before ID goes to current control,
                                    # content from ID onwards starts new control
                                    split_pos = other_control_match.start()
                                    before_id = line[:split_pos].strip()
                                    from_id = line[split_pos:].strip()
                                    
                                    # Add content before ID to current control
                                    if before_id:
                                        current_control_text.append(before_id)
                                    
                                    # Save current control
                                    accumulated_text = '\n'.join(current_control_text)
                                    accumulated_text = self._remove_trailing_next_control(
                                        accumulated_text,
                                        current_control_id
                                    )
                                    
                                    if accumulated_text:
                                        control = self._parse_control_block(
                                            current_control_id,
                                            accumulated_text
                                        )
                                        if control:
                                            controls.append(control)
                                    
                                    # Start new control with content from ID onwards
                                    current_control_id = detected_id
                                    current_control_text = [from_id] if from_id else []
                                else:
                                    # Same control ID, just accumulate
                                    current_control_text.append(line)
                            else:
                                # No control ID in line, accumulate normally
                                current_control_text.append(line)
            
            # Don't forget the last control
            if current_control_id and current_control_text:
                accumulated_text = '\n'.join(current_control_text)
                # Remove any trailing next control IDs
                accumulated_text = self._remove_trailing_next_control(accumulated_text, current_control_id)
                
                control = self._parse_control_block(
                    current_control_id,
                    accumulated_text
                )
                if control:
                    controls.append(control)
        
        print(f"\n✓ Extracted {len(controls)} controls")
        return controls
    
    def _parse_control_block(self, control_id: str, text: str) -> Optional[ExtractedControl]:
        """Parse a complete control block into structured format"""
        try:
            # Extract control name (text after ID, before "Control" or "Priority")
            name_match = re.search(
                rf'{re.escape(control_id)}\s+(.+?)(?=\s+Control\s|Priority|$)',
                text,
                re.DOTALL
            )
            control_name = name_match.group(1).strip() if name_match else "Unknown"
            control_name = re.sub(r'\s+', ' ', control_name)
            
            # Get family and subfamily
            family_id = control_id.split('.')[0]
            subfamily_id = '.'.join(control_id.split('.')[:2])
            
            family_name = self.family_map.get(family_id, "Unknown")
            subfamily_name = self.subfamily_map.get(subfamily_id, "")
            
            # Extract Control description
            description = self._extract_section(
                text,
                r'Control\s+(?:Statement)?',
                [r'Sub-Control', r'Implementation\s+Guidance', r'Priority', r'Applicability']
            )
            
            # Extract Sub-controls
            sub_controls = self._extract_sub_controls(text, control_id)
            
            # Extract Implementation Guidelines
            implementation_guidelines = self._extract_section(
                text,
                r'Implementation\s+Guidance',
                [r'External\s+Factors', r'Internal\s+Factors', r'Guidance\s+Points', r'Communication', r'Priority', r'Applicability']
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
                r'Communication\s+(?:Requirements)?',
                [r'Priority', r'Applicability']
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
                applicablility=applicability,  # Note: keeping typo to match reference
                breadcrumb=breadcrumb
            )
        
        except Exception as e:
            print(f"Error parsing control {control_id}: {e}")
            return None
    
    def _extract_section(self, text: str, start_pattern: str, end_patterns: List[str]) -> str:
        """Extract text between a start pattern and any end pattern"""
        # Create pattern that matches start and stops at any end pattern
        pattern = rf'{start_pattern}\s*(.*?)(?={"|".join(end_patterns)}|$)'
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
            [r'Implementation\s+Guidance', r'External\s+Factors', r'Internal\s+Factors', r'Guidance\s+Points', r'Priority']
        )
        
        if not section:
            return []
        
        # Split by numbered or lettered items: 1), 2), a), b), etc.
        # Pattern matches: number/letter followed by ) or .
        items = re.split(r'(?:\d+[\)\.]|[a-z][\)\.])\s+', section)
        
        formatted_sub_controls = []
        for idx, item in enumerate(items):
            item = item.strip()
            if len(item) > 5:  # Filter out noise
                letter = chr(97 + idx)  # a, b, c...
                formatted_sub_controls.append(f"{control_id}.{letter}: {item}")
        
        return formatted_sub_controls
    
    def _extract_list_items(self, text: str, start_pattern: str, end_patterns: List[str]) -> List[str]:
        """Extract list items (for external_factors, internal_factors, guidance_points, etc.)"""
        section = self._extract_section(text, start_pattern, end_patterns)
        
        if not section:
            return []
        
        # Split by lettered items: a., b., c., etc.
        items = re.split(r'([a-z]\.)\s+', section)
        
        # Recombine items with their labels
        formatted_items = []
        for i in range(1, len(items), 2):  # Start from index 1 (first label)
            if i + 1 < len(items):
                label = items[i]
                content = items[i + 1].strip()
                if content:
                    formatted_items.append(f"{label} {content}")
        
        # If no lettered items, try numbered items
        if not formatted_items:
            items = re.split(r'(\d+\.)\s+', section)
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
        
        # Check for "always" or "always applicable"
        if re.search(r'\balways\s+(?:applicable)?', text, re.IGNORECASE):
            applicability.append("always")
        
        # If no explicit priority found, default to P1
        if not applicability:
            applicability.append("P1")
        
        return applicability
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert a table (list of rows, each row list of cells) to plain text.
        Each row becomes a line; cells in a row are joined with spaces.
        Ensures multiple controls per page (e.g. M2.3.2 and M2.3.3 in separate tables) are all included.
        """
        if not table:
            return ""
        lines = []
        for row in table:
            cells = [str(c).strip() if c else "" for c in row]
            line = " ".join(cells).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)
    
    def _clean_text(self, text: str, page_num: int = None) -> str:
        """Clean PDF artifacts from text, including page numbers"""
        # Remove page numbers in various formats
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*\n\s*\n', '', text)  # Standalone page numbers with blank lines
        
        # Remove page numbers at end of lines (common footer pattern)
        # Pattern: number at end of line, possibly with whitespace
        text = re.sub(r'\n\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone number lines
        
        # Remove page numbers that appear at end of text block
        # (often isolated on their own line at bottom of page)
        lines = text.split('\n')
        cleaned_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip lines that are just numbers (likely page numbers)
            # But keep numbers that are part of control IDs or content
            if stripped.isdigit() and len(stripped) <= 3:
                # Check if it's likely a page number (not part of a sentence)
                # Skip if it's isolated or at end
                if i == len(lines) - 1 or (i > 0 and not lines[i-1].strip()):
                    continue  # Skip this line (page number)
            cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        # Remove header/footer artifacts
        text = re.sub(r'UAE\s+Information\s+Assurance\s+Regulation', '', text, flags=re.IGNORECASE)
        
        # Remove common footer patterns
        text = re.sub(r'UAE\s+IA\s+Regulation', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n', text)  # Multiple blank lines to single
        
        return text.strip()
    
    def _remove_trailing_next_control(self, text: str, current_control_id: str) -> str:
        """Remove content from next control that may have been included"""
        # Find the next control ID pattern in the text
        # Pattern: M or T followed by digits.digits.digits
        control_pattern = r'([MT]\d+\.\d+\.\d+)'
        matches = list(re.finditer(control_pattern, text))
        
        for match in matches:
            detected_id = match.group(1)
            # If we find a different control ID, truncate before it
            if detected_id != current_control_id:
                # Check if this is likely the start of next control
                # (appears after significant content, not mid-sentence)
                pos = match.start()
                before_text = text[:pos].strip()
                
                # If we have substantial content before this ID, it's likely next control
                if len(before_text) > 50:  # Reasonable control has content
                    return before_text
        
        return text
    
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
    output_path = "data/02_processed/uae_ia_controls_structured.json"
    
    extractor = ImprovedControlExtractor()
    
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
        print(f"  {family}: {family_counts[family]} controls")
    
    # Count controls with sub-controls
    with_subcontrols = sum(1 for c in controls if c.control.get('sub_controls'))
    print(f"\nControls with sub-controls: {with_subcontrols}")
    
    # Count controls with guidance points
    with_guidance = sum(1 for c in controls if c.control.get('guidance_points'))
    print(f"Controls with guidance points: {with_guidance}")


if __name__ == "__main__":
    main()
