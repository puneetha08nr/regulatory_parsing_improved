# src/pipeline/control_extractor.py

import re
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import pdfplumber

@dataclass
class Control:
    """Represents a single control from UAE IA Regulation"""
    control_id: str  # e.g., "5.1.1" or "ACCESS-001"
    control_number: str  # Official numbering
    control_name: str
    control_statement: str
    sub_controls: List[str]
    section: str
    section_title: str
    control_family: Optional[str]
    control_subfamily: Optional[str]
    applicability: List[str]  # P1, P2, P3, P4
    page_number: int
    is_obligation: bool
    metadata: Dict
    
    def to_label_studio_task(self) -> Dict:
        """Convert to Label Studio task format"""
        return {
            'data': {
                'text': self.control_statement,
                'section_id': self.section,
                'article_id': self.control_number,
                'control_id': self.control_id,
                'parent_section': f"Section {self.section}: {self.section_title}",
                'regulation_source': 'UAE IA Regulation v1.1',
                'control_name': self.control_name,
                'control_family': self.control_family or 'N/A',
                'control_subfamily': self.control_subfamily or 'N/A',
                'applicability_levels': ', '.join(self.applicability),
                'has_subcontrols': len(self.sub_controls) > 0,
                'subcontrol_count': len(self.sub_controls)
            },
            'meta': {
                'original_id': self.control_id,
                'page': self.page_number,
                'type': 'control',
                'pre_identified_obligation': self.is_obligation
            }
        }


class UAEIAControlExtractor:
    """Extract controls from UAE IA Regulation"""
    
    # Obligation keywords
    OBLIGATION_KEYWORDS = [
        'shall', 'must', 'required', 'is required',
        'should', 'may not', 'shall not', 'must not',
        'prohibited', 'forbidden', 'mandatory'
    ]
    
    def __init__(self):
        self.controls = []
        self.current_section = None
        self.current_family = None
        self.current_subfamily = None
    
    def extract_controls(self, pdf_path: str) -> List[Control]:
        """
        Extract all controls from the PDF
        
        Args:
            pdf_path: Path to UAE IA Regulation PDF
            
        Returns:
            List of Control objects
        """
        print(f"Extracting controls from: {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                print(f"Processing page {page_num}/{len(pdf.pages)}", end='\r')
                
                # Extract text
                text = page.extract_text()
                
                if not text:
                    continue
                
                # Update current section context
                self._update_context(text)
                
                # Extract tables (controls are often in tables)
                tables = page.extract_tables()
                if tables:
                    self._process_tables(tables, page_num)
                
                # Also process regular text for controls
                self._process_text(text, page_num)
        
        print(f"\nExtracted {len(self.controls)} controls")
        return self.controls
    
    def _update_context(self, text: str):
        """Update current section/family context from text"""
        
        # Look for section headers like "5.1 Control Structure"
        section_match = re.search(r'(\d+(?:\.\d+)?)\s+([A-Z][^\n]+)', text)
        if section_match:
            section_num = section_match.group(1)
            section_title = section_match.group(2).strip()
            
            # Only update if it looks like a real section
            if len(section_title) > 5 and len(section_title) < 100:
                self.current_section = {
                    'number': section_num,
                    'title': section_title
                }
        
        # Look for control family indicators
        if 'Control Family' in text:
            # Extract from nearby text
            family_match = re.search(r'Control Family[:\s]+([^\n]+)', text)
            if family_match:
                self.current_family = family_match.group(1).strip()
        
        if 'Sub-Family' in text or 'Control Sub-Family' in text:
            subfamily_match = re.search(r'(?:Sub-Family|Control Sub-Family)[:\s]+([^\n]+)', text)
            if subfamily_match:
                self.current_subfamily = subfamily_match.group(1).strip()
    
    def _process_tables(self, tables: List, page_num: int):
        """Extract controls from table structures"""
        
        for table in tables:
            if not table or len(table) < 2:
                continue
            
            # Check if this is a control table
            header_row = [str(cell).lower() if cell else '' for cell in table[0]]
            
            # Look for control-related headers
            is_control_table = any(
                keyword in ' '.join(header_row)
                for keyword in ['control number', 'control name', 'control statement']
            )
            
            if not is_control_table:
                continue
            
            # Find column indices
            control_num_col = self._find_column_index(header_row, ['control number', 'number'])
            control_name_col = self._find_column_index(header_row, ['control name', 'name'])
            control_stmt_col = self._find_column_index(header_row, ['control statement', 'statement', 'description'])
            applicability_cols = self._find_applicability_columns(header_row)
            
            # Process data rows
            for row in table[1:]:
                if not row or len(row) < 2:
                    continue
                
                # Skip empty rows
                if all(not cell or str(cell).strip() == '' for cell in row):
                    continue
                
                # Extract control information
                control_number = self._get_cell_value(row, control_num_col)
                control_name = self._get_cell_value(row, control_name_col)
                control_statement = self._get_cell_value(row, control_stmt_col)
                
                # Skip if no meaningful content
                if not control_statement or len(control_statement) < 10:
                    continue
                
                # Extract applicability
                applicability = []
                for level, col_idx in applicability_cols.items():
                    cell_value = self._get_cell_value(row, col_idx)
                    if cell_value and cell_value.strip():
                        applicability.append(level)
                
                # Check if this is an obligation
                is_obligation = self._contains_obligation_keyword(control_statement)
                
                # Create control object
                control = Control(
                    control_id=self._generate_control_id(control_number),
                    control_number=control_number or 'N/A',
                    control_name=control_name or 'Unnamed Control',
                    control_statement=control_statement,
                    sub_controls=[],  # Will be populated separately
                    section=self.current_section['number'] if self.current_section else 'N/A',
                    section_title=self.current_section['title'] if self.current_section else 'N/A',
                    control_family=self.current_family,
                    control_subfamily=self.current_subfamily,
                    applicability=applicability if applicability else ['P1', 'P2', 'P3', 'P4'],
                    page_number=page_num,
                    is_obligation=is_obligation,
                    metadata={
                        'extracted_from': 'table',
                        'has_applicability_matrix': len(applicability) > 0
                    }
                )
                
                self.controls.append(control)
    
    def _process_text(self, text: str, page_num: int):
        """Extract controls from regular text (non-table format)"""
        
        # Look for control patterns in text
        # Pattern: Control number followed by description
        control_pattern = r'(?:^|\n)([A-Z\-]+\d+(?:\.\d+)*)\s+([^\n]+)'
        
        matches = re.finditer(control_pattern, text)
        
        for match in matches:
            control_number = match.group(1)
            control_text = match.group(2).strip()
            
            # Only process if it looks like a control
            if len(control_text) < 20:
                continue
            
            is_obligation = self._contains_obligation_keyword(control_text)
            
            control = Control(
                control_id=self._generate_control_id(control_number),
                control_number=control_number,
                control_name='Extracted Control',
                control_statement=control_text,
                sub_controls=[],
                section=self.current_section['number'] if self.current_section else 'N/A',
                section_title=self.current_section['title'] if self.current_section else 'N/A',
                control_family=self.current_family,
                control_subfamily=self.current_subfamily,
                applicability=['P1', 'P2', 'P3', 'P4'],
                page_number=page_num,
                is_obligation=is_obligation,
                metadata={
                    'extracted_from': 'text'
                }
            )
            
            self.controls.append(control)
    
    def _find_column_index(self, header_row: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by matching keywords"""
        for idx, cell in enumerate(header_row):
            if any(keyword in cell.lower() for keyword in keywords):
                return idx
        return None
    
    def _find_applicability_columns(self, header_row: List[str]) -> Dict[str, int]:
        """Find P1, P2, P3, P4 applicability columns"""
        applicability = {}
        for idx, cell in enumerate(header_row):
            cell_lower = cell.lower()
            for level in ['p1', 'p2', 'p3', 'p4']:
                if level in cell_lower:
                    applicability[level.upper()] = idx
        return applicability
    
    def _get_cell_value(self, row: List, col_idx: Optional[int]) -> str:
        """Safely get cell value from row"""
        if col_idx is None or col_idx >= len(row):
            return ''
        
        cell = row[col_idx]
        return str(cell).strip() if cell else ''
    
    def _contains_obligation_keyword(self, text: str) -> bool:
        """Check if text contains obligation keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.OBLIGATION_KEYWORDS)
    
    def _generate_control_id(self, control_number: str) -> str:
        """Generate unique control ID"""
        if control_number and control_number != 'N/A':
            # Use control number as base
            clean_number = re.sub(r'[^A-Za-z0-9\-\.]', '_', control_number)
            return f"UAE_IA_CTRL_{clean_number}"
        else:
            # Generate UUID
            return f"UAE_IA_CTRL_{str(uuid.uuid4())[:8]}"


def extract_controls_for_label_studio():
    """
    Main function to extract controls and prepare for Label Studio
    
    Args:
        pdf_path: Path to UAE IA Regulation PDF
        output_path: Where to save Label Studio task JSON
    """
    pdf_path = "data/01_raw/regulation/UAE_IA_Regulation.pdf"
    output_path = "data/04_label_studio/imports/uae_ia_controls.json"
    # Extract controls
    extractor = UAEIAControlExtractor()
    controls = extractor.extract_controls(pdf_path)
    
    # Convert to Label Studio format
    label_studio_tasks = [control.to_label_studio_task() for control in controls]
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_studio_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(label_studio_tasks)} controls to: {output_path}")
    
    # Also save raw control data
    raw_output = output_path.replace('.json', '_raw.json')
    with open(raw_output, 'w', encoding='utf-8') as f:
        json.dump(
            [asdict(c) for c in controls],
            f,
            indent=2,
            ensure_ascii=False
        )
    
    print(f"✓ Saved raw control data to: {raw_output}")
    
    # Print statistics
    print_control_statistics(controls)
    
    return label_studio_tasks


def print_control_statistics(controls: List[Control]):
    """Print extraction statistics"""
    
    sections = {}
    families = {}
    obligations = sum(1 for c in controls if c.is_obligation)
    applicability_dist = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0}
    
    for control in controls:
        # Count by section
        sections[control.section] = sections.get(control.section, 0) + 1
        
        # Count by family
        if control.control_family:
            families[control.control_family] = families.get(control.control_family, 0) + 1
        
        # Count applicability
        for level in control.applicability:
            if level in applicability_dist:
                applicability_dist[level] += 1
    
    print("\n" + "="*60)
    print("CONTROL EXTRACTION STATISTICS")
    print("="*60)
    print(f"Total controls extracted: {len(controls)}")
    print(f"Controls with obligation keywords: {obligations} ({obligations/len(controls)*100:.1f}%)")
    
    print(f"\nControls by section:")
    for section, count in sorted(sections.items()):
        print(f"  Section {section}: {count}")
    
    if families:
        print(f"\nControls by family:")
        for family, count in sorted(families.items())[:10]:  # Top 10
            print(f"  {family[:40]:40s}: {count}")
    
    print(f"\nApplicability distribution:")
    for level in ['P1', 'P2', 'P3', 'P4']:
        print(f"  {level}: {applicability_dist[level]} controls")
    
    print("="*60)


if __name__ == "__main__":
   
    
    extract_controls_for_label_studio()