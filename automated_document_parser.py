r"""
Automated Document Parser for ADGM Documents
Replaces manual \Table Start/End tagging with modern IDP tools

Supports:
1. LlamaParse (Recommended)
2. Unstructured.io (Open-source)
3. Vision-Language Models (Gemini/GPT-4o)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import re


# Check for LlamaParse
try:
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False

# Check for Unstructured
try:
    from unstructured.partition.docx import partition_docx
    from unstructured.staging.base import elements_to_json
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


class AutomatedDocumentParser:
    """
    Automated parser to replace manual structuring
    Supports multiple backends: LlamaParse, Unstructured, Vision Models
    """
    
    def __init__(self, method: str = "llamaparse", api_key: Optional[str] = None):
        """
        Initialize parser
        
        Args:
            method: "llamaparse", "unstructured", or "vision"
            api_key: API key (or set env vars)
        """
        self.method = method.lower()
        
        if method == "llamaparse":
            if not LLAMAPARSE_AVAILABLE:
                raise ImportError("llama-parse required. Install: pip install llama-parse")
            self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
            if not self.api_key:
                raise ValueError("LlamaParse API key required. Set LLAMA_CLOUD_API_KEY")
            self.parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                verbose=True,
                num_workers=4
            )
        
        elif method == "unstructured":
            if not UNSTRUCTURED_AVAILABLE:
                raise ImportError("unstructured required. Install: pip install unstructured")
            # No API key needed for unstructured
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'llamaparse' or 'unstructured'")
    
    def parse_docx(self, docx_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Parse .docx file with automatic table/figure detection
        
        Args:
            docx_path: Path to .docx file
            output_path: Optional path to save parsed output
        
        Returns:
            Dict with parsed content and metadata
        """
        print(f"Parsing {docx_path} with {self.method}...")
        
        if self.method == "llamaparse":
            return self._parse_with_llamaparse(docx_path, output_path)
        elif self.method == "unstructured":
            return self._parse_with_unstructured(docx_path, output_path)
    
    def _parse_with_llamaparse(self, docx_path: str, output_path: Optional[str] = None) -> Dict:
        """Parse using LlamaParse"""
        documents = self.parser.load_data(docx_path)
        
        parsed_content = {
            "source_file": docx_path,
            "method": "llamaparse",
            "pages": [],
            "tables": [],
            "figures": [],
            "full_text": ""
        }
        
        for doc in documents:
            markdown_content = doc.text
            
            # Extract tables from Markdown
            tables = self._extract_tables_from_markdown(markdown_content)
            parsed_content["tables"].extend(tables)
            
            # Store page content
            parsed_content["pages"].append({
                "page_num": getattr(doc, 'page_label', 'unknown'),
                "content": markdown_content
            })
            
            parsed_content["full_text"] += markdown_content + "\n\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_content, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved to {output_path}")
        
        return parsed_content
    
    def _parse_with_unstructured(self, docx_path: str, output_path: Optional[str] = None) -> Dict:
        """Parse using Unstructured.io"""
        elements = partition_docx(docx_path)
        
        parsed_content = {
            "source_file": docx_path,
            "method": "unstructured",
            "elements": [],
            "tables": [],
            "figures": [],
            "text": ""
        }
        
        for elem in elements:
            elem_dict = {
                "type": elem.category,
                "text": elem.text,
                "metadata": elem.metadata.to_dict() if hasattr(elem, 'metadata') else {}
            }
            parsed_content["elements"].append(elem_dict)
            
            if elem.category == "Table":
                parsed_content["tables"].append({
                    "table_id": f"table_{len(parsed_content['tables']) + 1}",
                    "text": elem.text,
                    "metadata": elem_dict["metadata"]
                })
            elif elem.category == "Figure":
                parsed_content["figures"].append({
                    "figure_id": f"figure_{len(parsed_content['figures']) + 1}",
                    "text": elem.text,
                    "metadata": elem_dict["metadata"]
                })
            else:
                parsed_content["text"] += elem.text + "\n\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_content, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved to {output_path}")
        
        return parsed_content
    
    def _extract_tables_from_markdown(self, markdown: str) -> List[Dict]:
        """Extract tables from Markdown content"""
        tables = []
        # Markdown tables: | col1 | col2 |
        table_pattern = r'(\|.+\|\n\|[-:|\s]+\|\n(?:\|.+\|\n?)+)'
        
        matches = re.finditer(table_pattern, markdown)
        for i, match in enumerate(matches, 1):
            table_md = match.group(1)
            tables.append({
                "table_id": f"table_{i}",
                "markdown": table_md,
                "start_pos": match.start(),
                "end_pos": match.end()
            })
        
        return tables
    
    def convert_to_legacy_format(self, parsed_content: Dict) -> str:
        r"""
        Convert automated output to legacy format with \Table Start/End tags
        For backward compatibility with existing pipelines
        """
        text = parsed_content.get("full_text") or parsed_content.get("text", "")
        
        # Insert table markers
        for table in parsed_content.get("tables", []):
            table_md = table.get("markdown") or table.get("text", "")
            table_marker = f"\n\\Table Start\n{table_md}\n\\Table End\n"
            text = text.replace(table_md, table_marker, 1)  # Replace first occurrence
        
        # Insert figure markers
        for figure in parsed_content.get("figures", []):
            fig_text = figure.get("text") or figure.get("description", "Figure")
            figure_marker = f"\n\\Figure Start\n{fig_text}\n\\Figure End\n"
            text += figure_marker
        
        return text


def batch_parse_documents(
    input_dir: str,
    output_dir: str,
    method: str = "llamaparse",
    api_key: Optional[str] = None
):
    """
    Batch process all .docx files in directory
    
    Args:
        input_dir: Directory containing .docx files
        output_dir: Directory to save parsed outputs
        method: "llamaparse" or "unstructured"
        api_key: API key (for LlamaParse)
    """
    parser = AutomatedDocumentParser(method=method, api_key=api_key)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        print(f"   Please create the directory or check the path.")
        return
    
    docx_files = list(input_path.glob("*.docx"))
    
    if len(docx_files) == 0:
        print(f"⚠️  No .docx files found in: {input_dir}")
        print(f"   Looking for files matching: {input_path}/*.docx")
        print(f"   Directory exists: {input_path.exists()}")
        if input_path.exists():
            all_files = list(input_path.glob("*"))
            if all_files:
                print(f"   Found {len(all_files)} other files:")
                for f in all_files[:5]:
                    print(f"     - {f.name}")
                if len(all_files) > 5:
                    print(f"     ... and {len(all_files) - 5} more")
        return
    
    print(f"Found {len(docx_files)} .docx files to process\n")
    
    for i, docx_file in enumerate(docx_files, 1):
        print(f"[{i}/{len(docx_files)}] Processing: {docx_file.name}")
        
        try:
            output_file = output_path / f"{docx_file.stem}_parsed.json"
            parsed = parser.parse_docx(str(docx_file), str(output_file))
            
            # Also save in legacy format if needed
            legacy_text = parser.convert_to_legacy_format(parsed)
            legacy_file = output_path / f"{docx_file.stem}_legacy.txt"
            with open(legacy_file, 'w', encoding='utf-8') as f:
                f.write(legacy_text)
            
            print(f"  ✓ Pages: {len(parsed.get('pages', []))}")
            print(f"  ✓ Tables: {len(parsed.get('tables', []))}")
            print(f"  ✓ Figures: {len(parsed.get('figures', []))}\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python automated_document_parser.py <input_dir> [output_dir] [method]")
        print("\nMethods:")
        print("  llamaparse    - LlamaParse API (recommended, requires API key)")
        print("  unstructured - Unstructured.io (open-source, no API key)")
        print("\nExample:")
        print("  python automated_document_parser.py data/01_raw/adgm data/02_processed/parsed llamaparse")
        print("\nSet API key:")
        print("  export LLAMA_CLOUD_API_KEY='your-key-here'")
        print("\nNote: Input directory should contain .docx files")
        print("      If directory doesn't exist, create it and add your .docx files")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/02_processed/parsed"
    method = sys.argv[3] if len(sys.argv) > 3 else "llamaparse"
    
    # Get API key if needed
    api_key = None
    if method == "llamaparse":
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            print("⚠️  LLAMA_CLOUD_API_KEY not set.")
            print("   Get API key from: https://cloud.llamaindex.ai/")
            print("   Set it with: export LLAMA_CLOUD_API_KEY='your-key-here'")
            print("\n   Or use unstructured method (no API key needed):")
            print("   python automated_document_parser.py <input_dir> <output_dir> unstructured")
            sys.exit(1)
    
    batch_parse_documents(input_dir, output_dir, method, api_key)
