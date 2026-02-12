"""
Golden Dataset Preparation Pipeline
Based on RegNLP ObliQA methodology

This script implements the full pipeline for creating a golden dataset:
1. Document Collection & Preprocessing
2. Defining Passages
3. Question & Answer Generation (via GPT-4)
4. Validation via NLI Model
5. Dataset Assembly & Splitting
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")


@dataclass
class Passage:
    """Represents a single passage from a regulatory document"""
    document_id: str
    passage_id: str  # e.g., "M2.1.1" or "1.1.2"
    passage_text: str
    control_id: Optional[str] = None
    section: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class QuestionAnswerPair:
    """Represents a question-answer pair with associated passages"""
    question_id: str
    question: str
    passages: List[Passage]  # One or more passages needed to answer
    answer: str
    group: int  # Number of passages (1-6)
    topic: Optional[str] = None
    is_validated: bool = False
    nli_score: Optional[float] = None


class PassageExtractor:
    """Extract and structure passages from parsed regulatory documents"""
    
    def __init__(self, min_passage_length: int = 50, max_passage_length: int = 500):
        self.min_passage_length = min_passage_length
        self.max_passage_length = max_passage_length
        
    def extract_from_parsed_json(self, json_path: str) -> List[Passage]:
        """
        Extract passages from the parsed JSON structure.
        Looks for control IDs (M#.#.# or T#.#.#) and structures them as passages.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        document_id = data.get('meta', {}).get('doc_id', 'unknown')
        content = data.get('content', [])
        
        passages = []
        current_passage_text = []
        current_passage_id = None
        current_control_id = None
        current_section = None
        
        # Control ID pattern: M#.#.# or T#.#.#
        control_pattern = re.compile(r'([MT]\d+\.\d+(?:\.\d+)*(?:\.\w+)?)')
        
        for item in content:
            text = item.get('text', '').strip()
            item_type = item.get('type', 'text')
            
            # Check if this is a control ID
            control_match = control_pattern.search(text)
            if control_match:
                # Save previous passage if exists
                if current_passage_text and current_passage_id:
                    passage_text = ' '.join(current_passage_text)
                    if self._is_valid_passage(passage_text):
                        passages.append(Passage(
                            document_id=document_id,
                            passage_id=current_passage_id,
                            passage_text=passage_text,
                            control_id=current_control_id,
                            section=current_section,
                            metadata={'extracted_from': 'parsed_json'}
                        ))
                
                # Start new passage
                current_control_id = control_match.group(1)
                current_passage_id = current_control_id
                current_passage_text = [text]
            else:
                # Accumulate text for current passage
                if current_passage_id:
                    if item_type in ['text', 'header']:
                        current_passage_text.append(text)
                    
                    # Check if this is a section header
                    if item_type == 'header' and len(text) < 100:
                        current_section = text
                else:
                    # Before finding first control, try to identify sections
                    if item_type == 'header':
                        current_section = text
        
        # Don't forget the last passage
        if current_passage_text and current_passage_id:
            passage_text = ' '.join(current_passage_text)
            if self._is_valid_passage(passage_text):
                passages.append(Passage(
                    document_id=document_id,
                    passage_id=current_passage_id,
                    passage_text=passage_text,
                    control_id=current_control_id,
                    section=current_section,
                    metadata={'extracted_from': 'parsed_json'}
                ))
        
        return passages
    
    def extract_from_structured_controls(self, controls_json_path: str) -> List[Passage]:
        """
        Extract passages from structured control JSON (from auto_convert.py output).
        Each control becomes one or more passages.
        """
        with open(controls_json_path, 'r', encoding='utf-8') as f:
            controls = json.load(f)
        
        passages = []
        document_id = "uae_ia_regulation"
        
        for control in controls:
            control_id = control.get('control', {}).get('id', 'unknown')
            control_name = control.get('control', {}).get('name', '')
            description = control.get('control', {}).get('description', '')
            sub_controls = control.get('control', {}).get('sub_controls', [])
            guidelines = control.get('control', {}).get('implementation_guidelines', '')
            
            # Main control passage
            if description:
                passage_text = f"{control_name}. {description}"
                if self._is_valid_passage(passage_text):
                    passages.append(Passage(
                        document_id=document_id,
                        passage_id=control_id,
                        passage_text=passage_text,
                        control_id=control_id,
                        section=control.get('control_family', {}).get('name', ''),
                        metadata={
                            'control_name': control_name,
                            'family': control.get('control_family', {}).get('name', ''),
                            'subfamily': control.get('control_subfamily', {}).get('name', '')
                        }
                    ))
            
            # Sub-controls as separate passages
            for sub_control in sub_controls:
                if self._is_valid_passage(sub_control):
                    # Extract sub-control ID (e.g., "M2.1.1.a")
                    sub_id_match = re.search(r'([MT]\d+\.\d+(?:\.\d+)*\.\w+)', sub_control)
                    sub_id = sub_id_match.group(1) if sub_id_match else f"{control_id}.sub"
                    
                    passages.append(Passage(
                        document_id=document_id,
                        passage_id=sub_id,
                        passage_text=sub_control,
                        control_id=control_id,
                        section=control.get('control_family', {}).get('name', ''),
                        metadata={'is_subcontrol': True, 'parent_control': control_id}
                    ))
            
            # Implementation guidelines as separate passage
            if guidelines and self._is_valid_passage(guidelines):
                passages.append(Passage(
                    document_id=document_id,
                    passage_id=f"{control_id}.guidelines",
                    passage_text=f"Implementation Guidance for {control_name}: {guidelines}",
                    control_id=control_id,
                    section=control.get('control_family', {}).get('name', ''),
                    metadata={'is_guideline': True, 'parent_control': control_id}
                ))
        
        return passages
    
    def _is_valid_passage(self, text: str) -> bool:
        """Check if passage meets length requirements"""
        word_count = len(text.split())
        return self.min_passage_length <= len(text) <= self.max_passage_length * 10  # Allow longer for now
    
    def save_passages(self, passages: List[Passage], output_path: str):
        """Save passages to JSON file"""
        output_data = {
            'meta': {
                'created_at': datetime.now().isoformat(),
                'total_passages': len(passages),
                'document_ids': list(set(p.passage_id for p in passages))
            },
            'passages': [asdict(p) for p in passages]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(passages)} passages to {output_path}")


class QuestionGenerator:
    """Generate questions using GPT-4 following RegNLP methodology"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
        
        # Topics for multi-passage questions (based on compliance concerns)
        self.topics = {
            "Access Control": ["access", "authentication", "authorization", "user management", "privileges"],
            "Risk Management": ["risk assessment", "risk treatment", "vulnerability", "threat"],
            "Incident Management": ["incident", "breach", "security event", "response", "recovery"],
            "Compliance": ["compliance", "audit", "review", "assessment", "monitoring"],
            "Data Protection": ["data classification", "encryption", "backup", "retention", "privacy"],
            "Network Security": ["network", "firewall", "segmentation", "monitoring", "traffic"],
            "Physical Security": ["physical access", "facility", "equipment", "environmental"]
        }
    
    def generate_single_passage_question(self, passage: Passage) -> Optional[QuestionAnswerPair]:
        """Generate a question that can be answered from a single passage"""
        
        prompt = f"""You are an expert in regulatory compliance. Generate a question-answer pair based on the following regulatory passage.

Passage:
{passage.passage_text}

Instructions:
1. Generate a clear, specific question that can be answered using ONLY this passage
2. The question should be practical and relevant to compliance professionals
3. The answer should be directly extractable from the passage text
4. Format your response as JSON:
{{
    "question": "Your question here",
    "answer": "The answer extracted from the passage"
}}

Generate the question-answer pair:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in regulatory compliance and information security."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            question_id = f"Q_{passage.passage_id}_{len(result.get('question', ''))}"
            
            return QuestionAnswerPair(
                question_id=question_id,
                question=result.get('question', ''),
                passages=[passage],
                answer=result.get('answer', ''),
                group=1
            )
        except Exception as e:
            print(f"Error generating question for passage {passage.passage_id}: {e}")
            return None
    
    def generate_multi_passage_question(self, passages: List[Passage], topic: str) -> Optional[QuestionAnswerPair]:
        """Generate a question requiring multiple passages to answer"""
        
        if len(passages) < 2:
            return None
        
        passages_text = "\n\n".join([f"Passage {i+1} (ID: {p.passage_id}):\n{p.passage_text}" 
                                     for i, p in enumerate(passages)])
        
        prompt = f"""You are an expert in regulatory compliance. Generate a question-answer pair that requires information from MULTIPLE regulatory passages.

Topic: {topic}

Passages:
{passages_text}

Instructions:
1. Generate a question that requires integrating information from ALL {len(passages)} passages
2. The question should be practical and relevant to compliance professionals
3. The answer should synthesize information from multiple passages
4. Format your response as JSON:
{{
    "question": "Your question here",
    "answer": "The comprehensive answer synthesizing information from all passages"
}}

Generate the question-answer pair:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in regulatory compliance and information security."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            question_id = f"Q_MULTI_{topic.replace(' ', '_')}_{len(passages)}"
            
            return QuestionAnswerPair(
                question_id=question_id,
                question=result.get('question', ''),
                passages=passages,
                answer=result.get('answer', ''),
                group=len(passages),
                topic=topic
            )
        except Exception as e:
            print(f"Error generating multi-passage question: {e}")
            return None


class NLIValidator:
    """Validate question-passage pairs using Natural Language Inference"""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-xsmall"):
        if not NLI_AVAILABLE:
            raise ImportError("Transformers required. Install with: pip install transformers torch")
        
        print(f"Loading NLI model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Map model labels to NLI labels (entailment, contradiction, neutral)
        # This depends on the specific model - adjust based on model card
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    def validate(self, question: str, passage_text: str) -> Tuple[bool, float]:
        """
        Validate if question is entailed by passage.
        Returns: (is_valid, confidence_score)
        """
        # Format: premise (passage) and hypothesis (question)
        inputs = self.tokenizer(
            passage_text,
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get entailment probability (label 0 typically)
            entailment_prob = probs[0][0].item()
            
            # Consider valid if entailment probability > 0.5
            is_valid = entailment_prob > 0.5
        
        return is_valid, entailment_prob
    
    def validate_qa_pair(self, qa_pair: QuestionAnswerPair) -> QuestionAnswerPair:
        """Validate a QA pair against its passages"""
        if len(qa_pair.passages) == 1:
            # Single passage: direct validation
            is_valid, score = self.validate(qa_pair.question, qa_pair.passages[0].passage_text)
            qa_pair.is_validated = is_valid
            qa_pair.nli_score = score
        else:
            # Multi-passage: validate against combined passages
            combined_text = " ".join([p.passage_text for p in qa_pair.passages])
            is_valid, score = self.validate(qa_pair.question, combined_text)
            qa_pair.is_validated = is_valid
            qa_pair.nli_score = score
        
        return qa_pair


class GoldenDatasetBuilder:
    """Main pipeline for building golden dataset"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 nli_model: str = "microsoft/deberta-v3-xsmall",
                 gpt_model: str = "gpt-4-turbo-preview"):
        self.passage_extractor = PassageExtractor()
        self.question_generator = QuestionGenerator(api_key=openai_api_key, model=gpt_model)
        self.nli_validator = NLIValidator(model_name=nli_model) if NLI_AVAILABLE else None
    
    def build_dataset(self,
                     parsed_json_path: Optional[str] = None,
                     structured_controls_path: Optional[str] = None,
                     output_dir: str = "data/05_golden_dataset",
                     num_single_questions_per_passage: int = 1,
                     num_multi_questions: int = 50,
                     validation_enabled: bool = True,
                     train_split: float = 0.8,
                     dev_split: float = 0.1):
        """
        Main pipeline to build golden dataset
        
        Args:
            parsed_json_path: Path to parsed JSON from parser
            structured_controls_path: Path to structured controls JSON
            output_dir: Output directory for dataset
            num_single_questions_per_passage: How many single-passage questions per passage
            num_multi_questions: Total number of multi-passage questions to generate
            validation_enabled: Whether to use NLI validation
            train_split: Proportion for training set
            dev_split: Proportion for development set (test = 1 - train - dev)
        """
        print("=" * 60)
        print("Golden Dataset Preparation Pipeline")
        print("Based on RegNLP ObliQA Methodology")
        print("=" * 60)
        
        # Step 1: Extract Passages
        print("\n[Step 1] Extracting passages...")
        passages = []
        
        if structured_controls_path and Path(structured_controls_path).exists():
            print(f"  Loading from structured controls: {structured_controls_path}")
            passages.extend(self.passage_extractor.extract_from_structured_controls(structured_controls_path))
        
        if parsed_json_path and Path(parsed_json_path).exists():
            print(f"  Loading from parsed JSON: {parsed_json_path}")
            passages.extend(self.passage_extractor.extract_from_parsed_json(parsed_json_path))
        
        if not passages:
            raise ValueError("No passages extracted! Check input file paths.")
        
        print(f"  ✓ Extracted {len(passages)} passages")
        
        # Save passages
        passages_path = Path(output_dir) / "passages.json"
        self.passage_extractor.save_passages(passages, str(passages_path))
        
        # Step 2: Generate Questions
        print(f"\n[Step 2] Generating questions...")
        qa_pairs = []
        
        # Single-passage questions
        print(f"  Generating single-passage questions...")
        for i, passage in enumerate(passages):
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(passages)}")
            
            for _ in range(num_single_questions_per_passage):
                qa_pair = self.question_generator.generate_single_passage_question(passage)
                if qa_pair:
                    qa_pairs.append(qa_pair)
        
        print(f"  ✓ Generated {len(qa_pairs)} single-passage questions")
        
        # Multi-passage questions
        print(f"  Generating multi-passage questions...")
        topics = list(self.question_generator.topics.keys())
        passages_by_topic = self._group_passages_by_topic(passages, topics)
        
        for i in range(num_multi_questions):
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{num_multi_questions}")
            
            # Select topic and relevant passages
            topic = random.choice(topics)
            topic_passages = passages_by_topic.get(topic, [])
            
            if len(topic_passages) >= 2:
                # Randomly select 2-6 passages
                num_passages = random.randint(2, min(6, len(topic_passages)))
                selected_passages = random.sample(topic_passages, num_passages)
                
                qa_pair = self.question_generator.generate_multi_passage_question(
                    selected_passages, topic
                )
                if qa_pair:
                    qa_pairs.append(qa_pair)
        
        print(f"  ✓ Generated {len(qa_pairs)} total questions")
        
        # Step 3: NLI Validation
        if validation_enabled and self.nli_validator:
            print(f"\n[Step 3] Validating with NLI model...")
            validated_pairs = []
            for i, qa_pair in enumerate(qa_pairs):
                if (i + 1) % 50 == 0:
                    print(f"    Progress: {i+1}/{len(qa_pairs)}")
                
                validated = self.nli_validator.validate_qa_pair(qa_pair)
                if validated.is_validated:
                    validated_pairs.append(validated)
            
            qa_pairs = validated_pairs
            print(f"  ✓ Validated {len(qa_pairs)} questions (filtered invalid ones)")
        else:
            print(f"\n[Step 3] Skipping NLI validation (not available or disabled)")
        
        # Step 4: Dataset Assembly & Splitting
        print(f"\n[Step 4] Splitting dataset...")
        random.shuffle(qa_pairs)
        
        total = len(qa_pairs)
        train_end = int(total * train_split)
        dev_end = train_end + int(total * dev_split)
        
        train_set = qa_pairs[:train_end]
        dev_set = qa_pairs[train_end:dev_end]
        test_set = qa_pairs[dev_end:]
        
        print(f"  Train: {len(train_set)} questions")
        print(f"  Dev: {len(dev_set)} questions")
        print(f"  Test: {len(test_set)} questions")
        
        # Step 5: Save Dataset
        print(f"\n[Step 5] Saving dataset...")
        self._save_dataset(train_set, dev_set, test_set, passages, output_dir)
        
        # Statistics
        self._print_statistics(train_set, dev_set, test_set)
        
        print(f"\n✓ Golden dataset created successfully!")
        print(f"  Output directory: {output_dir}")
        
        return {
            'train': train_set,
            'dev': dev_set,
            'test': test_set,
            'passages': passages
        }
    
    def _group_passages_by_topic(self, passages: List[Passage], topics: List[str]) -> Dict[str, List[Passage]]:
        """Group passages by topic based on keywords"""
        grouped = defaultdict(list)
        
        for passage in passages:
            text_lower = passage.passage_text.lower()
            for topic, keywords in self.question_generator.topics.items():
                if any(keyword.lower() in text_lower for keyword in keywords):
                    grouped[topic].append(passage)
        
        return grouped
    
    def _save_dataset(self, train_set, dev_set, test_set, passages, output_dir):
        """Save dataset in ObliQA-like format"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        for split_name, split_data in [('train', train_set), ('dev', dev_set), ('test', test_set)]:
            split_path = output_path / f"{split_name}.json"
            split_data_dict = {
                'meta': {
                    'split': split_name,
                    'total_questions': len(split_data),
                    'created_at': datetime.now().isoformat()
                },
                'questions': [self._qa_pair_to_dict(qa) for qa in split_data]
            }
            
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(split_data_dict, f, indent=2, ensure_ascii=False)
        
        # Save full dataset
        full_dataset = {
            'meta': {
                'created_at': datetime.now().isoformat(),
                'total_questions': len(train_set) + len(dev_set) + len(test_set),
                'total_passages': len(passages),
                'train_size': len(train_set),
                'dev_size': len(dev_set),
                'test_size': len(test_set)
            },
            'passages': [asdict(p) for p in passages],
            'train': [self._qa_pair_to_dict(qa) for qa in train_set],
            'dev': [self._qa_pair_to_dict(qa) for qa in dev_set],
            'test': [self._qa_pair_to_dict(qa) for qa in test_set]
        }
        
        with open(output_path / "full_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(full_dataset, f, indent=2, ensure_ascii=False)
    
    def _qa_pair_to_dict(self, qa_pair: QuestionAnswerPair) -> Dict:
        """Convert QA pair to dictionary"""
        return {
            'question_id': qa_pair.question_id,
            'question': qa_pair.question,
            'answer': qa_pair.answer,
            'passages': [
                {
                    'document_id': p.document_id,
                    'passage_id': p.passage_id,
                    'passage_text': p.passage_text,
                    'control_id': p.control_id
                }
                for p in qa_pair.passages
            ],
            'group': qa_pair.group,
            'topic': qa_pair.topic,
            'is_validated': qa_pair.is_validated,
            'nli_score': qa_pair.nli_score
        }
    
    def _print_statistics(self, train_set, dev_set, test_set):
        """Print dataset statistics"""
        all_sets = train_set + dev_set + test_set
        
        # Count by group (number of passages)
        group_counts = defaultdict(int)
        for qa in all_sets:
            group_counts[qa.group] += 1
        
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Total questions: {len(all_sets)}")
        print(f"\nQuestions by number of passages:")
        for group in sorted(group_counts.keys()):
            print(f"  {group} passage(s): {group_counts[group]} questions")
        print(f"\nSplit distribution:")
        print(f"  Train: {len(train_set)} ({len(train_set)/len(all_sets)*100:.1f}%)")
        print(f"  Dev: {len(dev_set)} ({len(dev_set)/len(all_sets)*100:.1f}%)")
        print(f"  Test: {len(test_set)} ({len(test_set)/len(all_sets)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build golden dataset following RegNLP methodology")
    parser.add_argument("--parsed-json", type=str, help="Path to parsed JSON file")
    parser.add_argument("--structured-controls", type=str, 
                       default="data/02_processed/uae_ia_controls_structured.json",
                       help="Path to structured controls JSON")
    parser.add_argument("--output-dir", type=str, default="data/05_golden_dataset",
                       help="Output directory for dataset")
    parser.add_argument("--num-single", type=int, default=1,
                       help="Number of single-passage questions per passage")
    parser.add_argument("--num-multi", type=int, default=50,
                       help="Number of multi-passage questions to generate")
    parser.add_argument("--no-validation", action="store_true",
                       help="Skip NLI validation")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    builder = GoldenDatasetBuilder(openai_api_key=args.openai_key)
    
    builder.build_dataset(
        parsed_json_path=args.parsed_json,
        structured_controls_path=args.structured_controls,
        output_dir=args.output_dir,
        num_single_questions_per_passage=args.num_single,
        num_multi_questions=args.num_multi,
        validation_enabled=not args.no_validation
    )
