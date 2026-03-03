"""
Improved ADHICS control extractor using PyMuPDF (fitz) with position-based
hierarchy detection. This version properly captures nested structures by
analyzing text positioning and indentation levels.

Key improvements:
- Uses text block coordinates to determine hierarchy
- Identifies indentation levels for nested items
- Captures numbered items (1., 2., 3.) and lettered sub-items (a., b., c.)
- Preserves the relationship between parent and child items

Usage:
    python3 extract_adhic_controls_improved.py --pdf ADHIC.pdf --output controls.json
"""

import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# Domain definitions
DOMAIN_NAMES_CLEAN = {
    1: "Human Resources Security",
    2: "Asset Management",
    3: "Physical and Environmental Security",
    4: "Access Control",
    5: "Operations Management",
    6: "Communications",
    7: "Health Information and Security",
}

PREFIX_TO_DOMAIN = {"HR": 1, "AM": 2, "PE": 3, "AC": 4, "OM": 5, "CO": 6, "HI": 7}

# Patterns
CONTROL_ID_PATTERN = re.compile(
    r"^(HR|AM|PE|AC|OM|CO|HI)\s+(\d+\.\d+)(?:\s|$)",
    re.IGNORECASE,
)

SUB_DOMAIN_PATTERN = re.compile(
    r"^(HR|AM|PE|AC|OM|CO|HI)\s+(\d+)\s+(.+)$",
    re.IGNORECASE,
)

# List item patterns
NUMBERED_ITEM = re.compile(r"^(\d+)\.\s+(.*)$")
LETTERED_ITEM = re.compile(r"^([a-z])\.\s+(.*)$", re.IGNORECASE)

# UAE Reference pattern
UAE_REF_PATTERN = re.compile(
    r"UAE\s+IA\s+References?\s*:?\s*([^\n]+)",
    re.IGNORECASE,
)

# Applicability pattern
APPLICABILITY_PATTERN = re.compile(r"\[([BTA])\]", re.IGNORECASE)
BRACKET_TO_APPLICABILITY = {"B": "Basic", "T": "Transitional", "A": "Advanced"}


class TextBlock:
    """Represents a text block with position information."""
    
    def __init__(self, text: str, x0: float, y0: float, x1: float, y1: float):
        self.text = text.strip()
        self.x0 = x0  # Left edge
        self.y0 = y0  # Top edge
        self.x1 = x1  # Right edge
        self.y1 = y1  # Bottom edge
        self.indent_level = 0
        
    def __repr__(self):
        return f"TextBlock('{self.text[:30]}...', x0={self.x0:.1f}, indent={self.indent_level})"


class PositionBasedExtractor:
    """Extract ADHICS controls using position-based hierarchy detection."""
    
    def __init__(self, indent_threshold: float = 20.0):
        """
        Args:
            indent_threshold: Minimum x-coordinate difference to detect new indent level (points)
        """
        self.indent_threshold = indent_threshold
        self.standard_name = "Abu Dhabi Healthcare Information and Cyber Security Standard (ADHICS)"
        self.version = "0.9"
        
    def extract_text_blocks(self, page) -> List[TextBlock]:
        """Extract text blocks with position information from a page."""
        blocks = []
        
        # Get text with position information
        # dict format gives us detailed block information
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    line_text_parts = []
                    x0_min = float('inf')
                    y0_min = float('inf')
                    x1_max = 0
                    y1_max = 0
                    
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text_parts.append(text)
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            x0_min = min(x0_min, bbox[0])
                            y0_min = min(y0_min, bbox[1])
                            x1_max = max(x1_max, bbox[2])
                            y1_max = max(y1_max, bbox[3])
                    
                    if line_text_parts:
                        full_text = " ".join(line_text_parts)
                        if x0_min != float('inf'):
                            blocks.append(TextBlock(full_text, x0_min, y0_min, x1_max, y1_max))
        
        return blocks
    
    def calculate_indent_levels(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Calculate indent levels based on x-coordinates."""
        if not blocks:
            return blocks
        
        # Group blocks by similar x-coordinates (within threshold)
        x_positions = sorted(set(round(b.x0 / self.indent_threshold) * self.indent_threshold 
                                for b in blocks))
        
        # Create indent level mapping
        indent_map = {x_pos: level for level, x_pos in enumerate(x_positions)}
        
        # Assign indent levels
        for block in blocks:
            rounded_x = round(block.x0 / self.indent_threshold) * self.indent_threshold
            block.indent_level = indent_map.get(rounded_x, 0)
        
        return blocks
    
    def is_control_heading(self, text: str) -> Optional[Tuple[str, str]]:
        """Check if text is a control heading. Returns (control_id, title) or None."""
        match = CONTROL_ID_PATTERN.match(text)
        if match:
            prefix = match.group(1).upper()
            number = match.group(2)
            control_id = f"{prefix} {number}"
            # Extract title (everything after control ID)
            title = text[match.end():].strip()
            # Remove applicability brackets like [B], [T], [A]
            title = APPLICABILITY_PATTERN.sub('', title).strip()
            return (control_id, title)
        return None
    
    def extract_applicability(self, text: str) -> str:
        """Extract applicability level from text with [B], [T], or [A]."""
        match = APPLICABILITY_PATTERN.search(text)
        if match:
            letter = match.group(1).upper()
            return BRACKET_TO_APPLICABILITY.get(letter, "Basic")
        return "Basic"
    
    def parse_control_criteria(self, blocks: List[TextBlock], start_idx: int) -> Tuple[str, List[Dict], int]:
        """
        Parse control criteria starting from start_idx.
        Returns (description, criteria_list, end_idx)
        
        criteria_list format: [{"text": "requirement", "sub": ["a. item", "b. item"]}, ...]
        """
        description = ""
        criteria = []
        current_item = None
        base_indent = blocks[start_idx].indent_level if start_idx < len(blocks) else 0
        
        i = start_idx
        while i < len(blocks):
            block = blocks[i]
            text = block.text
            
            # Stop if we hit another control heading
            if self.is_control_heading(text):
                break
            
            # Stop if we hit UAE IA Reference
            if UAE_REF_PATTERN.match(text):
                break
            
            # Stop if we go back to lower or same indent level and have already captured content
            if criteria and block.indent_level <= base_indent:
                # Check if this might be continuation or new section
                if not (NUMBERED_ITEM.match(text) or LETTERED_ITEM.match(text) or 
                       text.lower().startswith(('the policy shall', 'the healthcare entity'))):
                    break
            
            # Check for main description (usually starts with "The healthcare entity shall...")
            if not description and i == start_idx:
                if "shall" in text.lower():
                    description = text
                    i += 1
                    continue
            
            # Check for "The policy shall:" header
            if text.lower().strip() in ["the policy shall:", "the healthcare entity shall:"]:
                i += 1
                continue
            
            # Check for numbered item (1., 2., 3.)
            num_match = NUMBERED_ITEM.match(text)
            if num_match:
                # Save previous item
                if current_item and current_item.get("text"):
                    criteria.append(current_item)
                
                # Start new item
                current_item = {
                    "text": num_match.group(2).strip(),
                    "sub": []
                }
                i += 1
                continue
            
            # Check for lettered sub-item (a., b., c.)
            letter_match = LETTERED_ITEM.match(text)
            if letter_match and current_item is not None:
                sub_text = letter_match.group(2).strip()
                current_item["sub"].append(sub_text)
                i += 1
                continue
            
            # Check if this is a continuation of current item or sub-item
            if current_item is not None:
                # Higher indent likely means continuation of sub-item
                if len(current_item.get("sub", [])) > 0 and block.indent_level > base_indent:
                    # Append to last sub-item
                    current_item["sub"][-1] = current_item["sub"][-1] + " " + text
                else:
                    # Append to main item text
                    current_item["text"] = current_item["text"] + " " + text
            elif description:
                # Continuation of description
                description = description + " " + text
            
            i += 1
        
        # Don't forget the last item
        if current_item and current_item.get("text"):
            criteria.append(current_item)
        
        return description, criteria, i
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract controls from PDF using position-based hierarchy."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"Extracting ADHICS controls from: {pdf_path}")
        print("Using PyMuPDF with position-based hierarchy detection...")
        
        doc = fitz.open(pdf_path)
        controls = []
        current_domain = ""
        current_sub_domain = ""
        
        for page_num in range(len(doc)):
            if (page_num + 1) % 10 == 0:
                print(f"  Processing page {page_num + 1}/{len(doc)}...", end="\r")
            
            page = doc[page_num]
            
            # Extract text blocks with positions
            blocks = self.extract_text_blocks(page)
            blocks = self.calculate_indent_levels(blocks)
            
            # Track domain and sub-domain context
            for block in blocks:
                text = block.text
                
                # Check for domain heading
                if re.match(r"^(\d+)\.\s+(HUMAN RESOURCES|ASSET MANAGEMENT|PHYSICAL|ACCESS CONTROL|OPERATIONS|COMMUNICATIONS|HEALTH INFORMATION)", text.upper()):
                    current_domain = text
                
                # Check for sub-domain
                sub_match = SUB_DOMAIN_PATTERN.match(text)
                if sub_match:
                    current_sub_domain = sub_match.group(3).strip()
            
            # Extract controls
            i = 0
            while i < len(blocks):
                block = blocks[i]
                control_info = self.is_control_heading(block.text)
                
                if control_info:
                    control_id, title = control_info
                    applicability = self.extract_applicability(block.text)
                    
                    # Parse criteria from subsequent blocks
                    description, criteria, end_idx = self.parse_control_criteria(blocks, i + 1)
                    
                    # Extract UAE IA references (look ahead a bit)
                    references = []
                    for j in range(end_idx, min(end_idx + 5, len(blocks))):
                        ref_match = UAE_REF_PATTERN.search(blocks[j].text)
                        if ref_match:
                            ref_text = ref_match.group(1).strip()
                            references = self._parse_references(ref_text)
                            break
                    
                    # Determine domain from control ID prefix
                    prefix = control_id.split()[0]
                    domain_num = PREFIX_TO_DOMAIN.get(prefix.upper())
                    domain = DOMAIN_NAMES_CLEAN.get(domain_num, current_domain)
                    
                    # Flatten sub-controls for easy access
                    subcontrols = []
                    for item in criteria:
                        subcontrols.extend(item.get("sub", []))
                    
                    control = {
                        "domain": domain,
                        "sub_domain": current_sub_domain,
                        "control_id": control_id,
                        "control_title": title,
                        "control_demand": description or title,
                        "applicability": applicability,
                        "criteria": criteria,
                        "subcontrols": subcontrols,
                        "references": references,
                        "source_page": page_num + 1
                    }
                    
                    controls.append(control)
                    print(f"\n  ✓ Extracted: {control_id} - {title[:50]}...")
                    print(f"    - Criteria items: {len(criteria)}")
                    print(f"    - Sub-controls: {len(subcontrols)}")
                    
                    i = end_idx
                else:
                    i += 1
        
        doc.close()
        print(f"\n\nExtraction complete! Total controls: {len(controls)}")
        
        return {
            "standard_name": self.standard_name,
            "version": self.version,
            "controls": controls
        }
    
    def _parse_references(self, text: str) -> List[str]:
        """Parse UAE IA references from text like 'M3.1.1, M4.1.1'."""
        text = re.sub(r"\s+&\s+", ", ", text)
        refs = []
        for part in re.split(r"[,;]", text):
            part = part.strip()
            if re.match(r"^[MT]\d+\.\d+", part):
                refs.append(part)
        return refs
    
    def save(self, data: Dict[str, Any], output_path: str):
        """Save extracted data to JSON."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with output.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved to: {output_path}")
        print(f"  Controls: {len(data.get('controls', []))}")
        
        # Print sample of first control
        if data.get('controls'):
            first = data['controls'][0]
            print(f"\n  Sample control:")
            print(f"    ID: {first.get('control_id')}")
            print(f"    Title: {first.get('control_title')}")
            print(f"    Criteria: {len(first.get('criteria', []))} items")
            print(f"    Sub-controls: {len(first.get('subcontrols', []))} items")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract ADHICS controls using position-based hierarchy detection"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to ADHIC PDF file"
    )
    parser.add_argument(
        "--output",
        default="adhics_controls_structured.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--indent-threshold",
        type=float,
        default=20.0,
        help="Indent detection threshold in points (default: 20.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    try:
        extractor = PositionBasedExtractor(indent_threshold=args.indent_threshold)
        result = extractor.extract_from_pdf(args.pdf)
        extractor.save(result, args.output)
        
        print("\n" + "="*60)
        print("SUCCESS! Extraction completed.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())