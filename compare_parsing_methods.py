"""
Compare Different Document Parsing Methods
Tests LlamaParse, Unstructured, and Manual methods on the same document
Helps you choose the best approach for your ADGM documents
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime

try:
    from automated_document_parser import AutomatedDocumentParser
    AUTOMATED_PARSER_AVAILABLE = True
except ImportError:
    AUTOMATED_PARSER_AVAILABLE = False
    print("⚠️  automated_document_parser.py not found")


class ParsingMethodComparator:
    """
    Compare different parsing methods on the same document
    """
    
    def __init__(self):
        self.results = {}
    
    def compare_methods(
        self,
        docx_path: str,
        manual_reference_path: Optional[str] = None,
        methods: List[str] = ["llamaparse", "unstructured"]
    ) -> Dict:
        """
        Compare parsing methods on the same document
        
        Args:
            docx_path: Path to .docx file to test
            manual_reference_path: Optional path to manually structured reference
            methods: List of methods to test
        
        Returns:
            Comparison results
        """
        print("="*70)
        print("Document Parsing Method Comparison")
        print("="*70)
        print(f"\nTest Document: {docx_path}\n")
        
        comparison = {
            "test_file": docx_path,
            "test_date": datetime.now().isoformat(),
            "methods": {},
            "summary": {}
        }
        
        # Test each method
        for method in methods:
            print(f"\n{'='*70}")
            print(f"Testing: {method.upper()}")
            print(f"{'='*70}\n")
            
            try:
                if method == "manual" and manual_reference_path:
                    result = self._load_manual_reference(manual_reference_path)
                elif method in ["llamaparse", "unstructured"]:
                    parser = AutomatedDocumentParser(method=method)
                    result = parser.parse_docx(docx_path)
                else:
                    print(f"⚠️  Unknown method: {method}")
                    continue
                
                # Analyze results
                analysis = self._analyze_result(result, method)
                comparison["methods"][method] = {
                    "result": result,
                    "analysis": analysis
                }
                
                print(f"✓ {method.upper()} completed")
                print(f"  - Tables found: {analysis['table_count']}")
                print(f"  - Figures found: {analysis['figure_count']}")
                print(f"  - Text length: {analysis['text_length']} chars")
                print(f"  - Processing time: {analysis.get('time', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Error with {method}: {e}")
                comparison["methods"][method] = {
                    "error": str(e)
                }
        
        # Compare against manual reference if available
        if manual_reference_path and "manual" in comparison["methods"]:
            manual_result = comparison["methods"]["manual"]["result"]
            for method in methods:
                if method != "manual" and method in comparison["methods"]:
                    similarity = self._compare_with_manual(
                        comparison["methods"][method]["result"],
                        manual_result
                    )
                    comparison["methods"][method]["similarity_to_manual"] = similarity
                    print(f"\n{method.upper()} vs Manual:")
                    print(f"  - Table match: {similarity['table_match']:.1%}")
                    print(f"  - Text similarity: {similarity['text_similarity']:.1%}")
        
        # Generate summary
        comparison["summary"] = self._generate_summary(comparison)
        
        return comparison
    
    def _load_manual_reference(self, reference_path: str) -> Dict:
        """Load manually structured reference document"""
        with open(reference_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract tables and figures from manual format
        tables = []
        figures = []
        
        # Find \Table Start ... \Table End blocks
        table_pattern = r'\\Table Start\n(.*?)\n\\Table End'
        for i, match in enumerate(re.finditer(table_pattern, content, re.DOTALL), 1):
            tables.append({
                "table_id": f"table_{i}",
                "text": match.group(1).strip()
            })
        
        # Find \Figure Start ... \Figure End blocks
        figure_pattern = r'\\Figure Start\n(.*?)\n\\Figure End'
        for i, match in enumerate(re.finditer(figure_pattern, content, re.DOTALL), 1):
            figures.append({
                "figure_id": f"figure_{i}",
                "text": match.group(1).strip()
            })
        
        return {
            "method": "manual",
            "source_file": reference_path,
            "tables": tables,
            "figures": figures,
            "text": content
        }
    
    def _analyze_result(self, result: Dict, method: str) -> Dict:
        """Analyze parsing result"""
        import time
        start_time = time.time()
        
        analysis = {
            "table_count": len(result.get("tables", [])),
            "figure_count": len(result.get("figures", [])),
            "text_length": len(result.get("full_text", result.get("text", ""))),
            "time": time.time() - start_time
        }
        
        return analysis
    
    def _compare_with_manual(self, automated_result: Dict, manual_result: Dict) -> Dict:
        """Compare automated result with manual reference"""
        # Compare table counts
        auto_tables = len(automated_result.get("tables", []))
        manual_tables = len(manual_result.get("tables", []))
        table_match = min(auto_tables, manual_tables) / max(auto_tables, manual_tables, 1)
        
        # Compare text similarity (simplified)
        auto_text = automated_result.get("full_text", automated_result.get("text", ""))
        manual_text = manual_result.get("text", "")
        
        # Simple character-based similarity
        if len(manual_text) > 0:
            text_similarity = min(len(auto_text), len(manual_text)) / max(len(auto_text), len(manual_text), 1)
        else:
            text_similarity = 0.0
        
        return {
            "table_match": table_match,
            "text_similarity": text_similarity
        }
    
    def _generate_summary(self, comparison: Dict) -> Dict:
        """Generate summary of comparison"""
        summary = {
            "methods_tested": list(comparison["methods"].keys()),
            "best_table_extraction": None,
            "best_figure_extraction": None,
            "fastest": None,
            "recommendation": ""
        }
        
        # Find best methods
        table_counts = {}
        figure_counts = {}
        
        for method, data in comparison["methods"].items():
            if "analysis" in data:
                analysis = data["analysis"]
                table_counts[method] = analysis["table_count"]
                figure_counts[method] = analysis["figure_count"]
        
        if table_counts:
            summary["best_table_extraction"] = max(table_counts, key=table_counts.get)
        if figure_counts:
            summary["best_figure_extraction"] = max(figure_counts, key=figure_counts.get)
        
        # Generate recommendation
        if "llamaparse" in comparison["methods"] and "unstructured" in comparison["methods"]:
            summary["recommendation"] = "Compare results and choose based on accuracy vs cost"
        elif "llamaparse" in comparison["methods"]:
            summary["recommendation"] = "LlamaParse recommended for production (high accuracy, fast)"
        elif "unstructured" in comparison["methods"]:
            summary["recommendation"] = "Unstructured.io good for open-source solution"
        
        return summary
    
    def save_comparison(self, comparison: Dict, output_path: str):
        """Save comparison results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Comparison saved to: {output_path}")


def main():
    """Main comparison function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compare_parsing_methods.py <docx_path> [manual_reference] [output]")
        print("\nExample:")
        print("  python compare_parsing_methods.py data/01_raw/test.docx data/01_raw/test_manual.txt")
        print("\nThis will:")
        print("  1. Test LlamaParse on the document")
        print("  2. Test Unstructured.io on the document")
        print("  3. Compare with manual reference (if provided)")
        print("  4. Generate comparison report")
        sys.exit(1)
    
    docx_path = sys.argv[1]
    manual_reference = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else "parsing_comparison.json"
    
    # Methods to test
    methods = ["llamaparse", "unstructured"]
    if manual_reference:
        methods.append("manual")
    
    # Run comparison
    comparator = ParsingMethodComparator()
    comparison = comparator.compare_methods(
        docx_path=docx_path,
        manual_reference_path=manual_reference,
        methods=methods
    )
    
    # Save results
    comparator.save_comparison(comparison, output_path)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nMethods tested: {', '.join(comparison['summary']['methods_tested'])}")
    if comparison['summary']['best_table_extraction']:
        print(f"Best table extraction: {comparison['summary']['best_table_extraction']}")
    if comparison['summary']['recommendation']:
        print(f"\nRecommendation: {comparison['summary']['recommendation']}")


if __name__ == "__main__":
    main()
