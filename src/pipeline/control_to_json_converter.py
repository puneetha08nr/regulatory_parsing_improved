import re
import json
from typing import Dict, List, Tuple, Optional

class ControlToJSONConverter:
    def __init__(self):
        # Precise mapping based on UAE IA Regulation chapters
        self.family_map = {
            "M1": "Strategy and Planning",
            "M2": "Information Security Risk Management",
            "M3": "Human Resources Security",
            "M4": "Asset Management",
            "M5": "External Party Access",
            "M6": "Information Security Incident Management",
            "T1": "Physical and Environmental Security",
            "T2": "Operations Management",
            "T3": "Communications Management"
        }

    def parse_control_text(self, text: str) -> Dict:
        """Fully parses the raw PDF text into the hierarchical JSON format."""
        
        # 1. CLEANING & PRE-PROCESSING
        text = self._pre_process_text(text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 2. HEADER PARSING (ID, Name, Applicability)
        control_id, control_name, priority, is_always = self._parse_header(text)
        family_num, subfamily_num = self._infer_family(control_id)
        
        # 3. EXTRACTION OF SECTIONS
        description = self._extract_section_text(text, "Control", ["Sub-Control", "Implementation"])
        
        # Extract Sub-controls and add the M2.1.1.a prefix logic
        raw_subs = self._extract_list_items(text, "Sub-Control", ["Implementation"])
        formatted_subs = []
        for idx, sub in enumerate(raw_subs):
            letter = chr(97 + idx) # Generates a, b, c...
            formatted_subs.append(f"{control_id}.{letter}: {sub}")

        # Extract Guidelines (Handling the multi-paragraph logic)
        guidelines = self._extract_section_text(text, "Implementation Guidance", ["Priority", "Applicability"])

        # 4. CROSS-REFERENCES (Regex for (see M#.#.#) or (refer to T#.#.#))
        cross_refs = sorted(list(set(re.findall(r'[A-Z]\d+\.\d+(?:\.\d+)*', text))))
        # Remove current ID from cross-refs if it found itself
        if control_id in cross_refs: cross_refs.remove(control_id)

        # 5. BREADCRUMB CONSTRUCTION
        family_name = self.family_map.get(family_num, "Management")
        # Structure: Management > Family > Sub-Family (Optional) > Control Name
        breadcrumb = f"Management > {family_name} > {control_name} > {control_name}"

        # 6. FINAL ASSEMBLY
        return {
            "control_family": {
                "number": family_num,
                "name": family_name
            },
            "control_subfamily": {
                "number": subfamily_num + ".", # Matching your M2. pattern
                "name": "" 
            },
            "control": {
                "id": control_id,
                "name": control_name,
                "description": description,
                "sub_controls": formatted_subs,
                "implementation_guidelines": guidelines
            },
            "cross_references": cross_refs,
            "applicability": [priority, "always"] if is_always else [priority],
            "breadcrumb": breadcrumb
        }

    def _pre_process_text(self, text: str) -> str:
        """Removes PDF artifacts like 'UAE Information Assurance Regulation' and page numbers."""
        text = re.sub(r'UAE Information Assurance Regulation', '', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text) # Remove standalone page numbers
        return text

    def _parse_header(self, text: str) -> Tuple[str, str, str, bool]:
        """Extracts Metadata from the top table-like structure of the PDF."""
        # Finds ID like M2.1.1
        id_match = re.search(r'^([A-Z]\d+(?:\.\d+)+)', text, re.MULTILINE)
        control_id = id_match.group(1) if id_match else "Unknown"
        
        # Heuristic for Name: Content after ID but before 'Priority'
        name_pattern = rf"{control_id}\s+(.*?)(?=Priority)"
        name_match = re.search(name_pattern, text, re.DOTALL)
        name = re.sub(r'\s+', ' ', name_match.group(1)).strip() if name_match else "Unknown"
        
        # Priority (P1, P2, etc)
        priority_match = re.search(r'Priority\s+(P\d+)', text)
        priority = priority_match.group(1) if priority_match else "P1"
        
        # Always applicable check
        is_always = "always" in text.lower() or "always applicable" in text.lower()
        
        return control_id, name, priority, is_always

    def _extract_section_text(self, text: str, start_key: str, end_keys: List[str]) -> str:
        """Grabs blocks of text between headers like 'Control' and 'Sub-Control'."""
        pattern = rf"{start_key}(?:\s+Guidance)?\s+(.*?)(?={'|'.join(end_keys)}|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            clean = re.sub(r'\s+', ' ', match.group(1)).strip()
            # Remove leading boilerplate like "The entity shall ensure..." if it's repeated
            return clean
        return ""

    def _extract_list_items(self, text: str, start_key: str, end_keys: List[str]) -> List[str]:
        """Splits sub-controls based on 1) 2) or a) b) patterns."""
        section = self._extract_section_text(text, start_key, end_keys)
        # Regex splits on "1) " or "2) " while keeping the content
        parts = re.split(r'\d+\)\s+', section)
        return [p.strip() for p in parts if len(p.strip()) > 5]

    def _infer_family(self, cid: str) -> Tuple[str, str]:
        parts = cid.split('.')
        family = parts[0] if len(parts) > 0 else ""
        subfamily = ".".join(parts[:2]) if len(parts) > 1 else family
        return family, subfamily