import re
import json
from typing import Dict, List, Tuple, Optional

class ControlToJSONConverter:
    def __init__(self):
        # 1. Full UAE IA v2.0 Family Mapping
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

        # 2. Full UAE IA v2.0 Sub-Family Mapping
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

    def convert_file(self, input_path: str, output_path: str):
        """Reads a text file, splits by control, and converts to JSON list."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # FIX: More robust splitter. 
            # It looks for a Control ID (e.g., M2.1.1) that is followed strictly by the control name logic
            # or simply splits by the ID pattern appearing with some whitespace before it.
            
            # 1. Normalize newlines to ensure consistent splitting
            content = content.replace('\r\n', '\n')

            # 2. Split by looking ahead for the Control ID Pattern (M#.#.# or T#.#.#)
            # The regex finds the boundary where a new Control ID appears.
            controls_raw = re.split(r'(?=\n\s*[A-Z]\d+\.\d+\.\d+)', content)
            
            json_output = []
            
            print(f"Found {len(controls_raw)} potential blocks. Processing...")

            for i, chunk in enumerate(controls_raw):
                chunk = chunk.strip()
                if not chunk: continue
                
                # Check if this chunk actually contains a valid Control header
                # We check the first few lines to see if it starts with an ID
                if re.match(r'^[A-Z]\d+\.\d+\.\d+', chunk):
                    print(f"  -> Converting chunk {i+1} starting with: {chunk[:10]}...") 
                    try:
                        control_json = self.parse_single_control(chunk)
                        json_output.append(control_json)
                    except Exception as e:
                        print(f"     [!] Failed to parse chunk {i+1}: {str(e)}")
                else:
                    # Debugging: Print chunks that were skipped to help you see what happened
                    print(f"  -> Skipping chunk {i+1} (No valid Control ID found at start)")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=4, ensure_ascii=False)
            
            print(f"\nSuccessfully converted {len(json_output)} controls to {output_path}")

        except Exception as e:
            print(f"Error processing file: {str(e)}")

    def parse_single_control(self, text: str) -> Dict:
        """Parses a single control text block into the strict JSON schema."""
        
        # 1. Clean Text
        text = self._clean_pdf_artifacts(text)
        
        # 2. Extract Metadata (ID, Name, Priority)
        control_id, control_name, applicability = self._parse_header(text)
        
        # 3. Hierarchy Lookup
        family_id = control_id.split('.')[0] # M2
        subfamily_id = '.'.join(control_id.split('.')[:2]) # M2.1
        
        family_name = self.family_map.get(family_id, "Management")
        subfamily_name = self.subfamily_map.get(subfamily_id, "")
        
        # 4. Extract Description (The "Control" section)
        description = self._extract_section(text, "Control", ["Sub-Control", "Implementation"])
        
        # 5. Extract and Format Sub-Controls
        sub_controls = self._parse_sub_controls(text, control_id)
        
        # 6. Extract Implementation Guidelines
        guidelines = self._extract_section(text, "Implementation Guidance", ["Priority", "Applicability"])
        
        # 7. Build Breadcrumb
        breadcrumb = f"Management > {family_name} > {subfamily_name} > {control_name}"

        # 8. Construct Final JSON
        return {
            "control_family": {
                "number": family_id,
                "name": family_name
            },
            "control_subfamily": {
                "number": subfamily_id,
                "name": subfamily_name
            },
            "control": {
                "id": control_id,
                "name": control_name,
                "description": description,
                "sub_controls": sub_controls,
                "implementation_guidelines": guidelines
            },
            "applicability": applicability,
            "breadcrumb": breadcrumb
        }

    def _clean_pdf_artifacts(self, text: str) -> str:
        text = re.sub(r'UAE Information Assurance Regulation', '', text)
        text = re.sub(r'Page \d+', '', text)
        return text.strip()

    def _parse_header(self, text: str) -> Tuple[str, str, List[str]]:
        # Find ID (e.g., M2.1.1)
        id_match = re.search(r'([A-Z]\d+\.\d+\.\d+)', text)
        control_id = id_match.group(1) if id_match else "UNKNOWN"
        
        # Find Name (Text between ID and "Priority")
        name_match = re.search(rf"{re.escape(control_id)}\s+(.*?)(?=\s+Priority|\n)", text, re.DOTALL)
        name = " ".join(name_match.group(1).split()) if name_match else "Unknown Name"
        
        # Find Priority/Applicability
        applicability = []
        if "P1" in text: applicability.append("P1")
        if "always" in text.lower(): applicability.append("always")
        
        return control_id, name, applicability

    def _extract_section(self, text: str, header: str, stops: List[str]) -> str:
        # Regex to find content between "Header" and any "Stop Word"
        pattern = rf"{header}\s*(.*?)(?={'|'.join(stops)}|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # Clean up newlines and extra spaces
            clean_text = re.sub(r'\s+', ' ', match.group(1))
            return clean_text.strip()
        return ""

    def _parse_sub_controls(self, text: str, control_id: str) -> List[str]:
        # Extract the raw block first
        raw_block = self._extract_section(text, "Sub-Control", ["Implementation Guidance"])
        if not raw_block: return []

        # Logic to split by 1), 2), 3) OR a), b), c)
        # We look for patterns like "1) " or "a. "
        items = re.split(r'(?:\d+\)|\d+\.|[a-z]\)|\n[a-z]\.)\s+', raw_block)
        
        formatted_list = []
        counter = 0
        for item in items:
            clean_item = item.strip()
            if len(clean_item) > 5: # Filter out empty splits or noise
                letter = chr(97 + counter) # a, b, c...
                # Format: M2.1.1.a: The policy shall...
                formatted_list.append(f"{control_id}.{letter}: {clean_item}")
                counter += 1
                
        return formatted_list

# --- Execution ---
if __name__ == "__main__":
    converter = ControlToJSONConverter()
    # Create a dummy input file for testing if it doesn't exist
    print("Looking for 'input.txt'...")
    converter.convert_file("input.txt", "output.json")