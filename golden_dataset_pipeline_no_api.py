"""
Golden Dataset Preparation Pipeline - NO API KEY REQUIRED
Template-based question generation that doesn't require OpenAI API

This version uses rule-based and template-based question generation
instead of GPT-4, making it free to use.
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


class TemplateQuestionGenerator:
    """Generate questions using templates and rules (no API required)"""
    
    # Question templates for different control types
    QUESTION_TEMPLATES = [
        "What are the requirements for {control_name}?",
        "What must entities do to comply with {control_name}?",
        "What are the key obligations under {control_name}?",
        "How should {control_name} be implemented?",
        "What does {control_name} require?",
        "What are the specific requirements of {control_name}?",
        "What is required for {control_name}?",
        "What obligations exist for {control_name}?",
    ]
    
    # Question patterns based on keywords in the passage
    KEYWORD_QUESTIONS = {
        "shall": "What must be done according to this requirement?",
        "must": "What is mandatory under this control?",
        "should": "What is recommended in this guidance?",
        "ensure": "What must be ensured according to this control?",
        "establish": "What must be established?",
        "implement": "What must be implemented?",
        "maintain": "What must be maintained?",
        "review": "What must be reviewed?",
        "assess": "What must be assessed?",
        "document": "What must be documented?",
        "communicate": "What must be communicated?",
        "monitor": "What must be monitored?",
        "protect": "What must be protected?",
        "manage": "What must be managed?",
        "control": "What must be controlled?",
    }
    
    def __init__(self):
        # Topics for multi-passage questions
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
        """Generate a question using templates and rules"""
        
        # Extract control name from passage ID or metadata
        control_name = self._extract_control_name(passage)
        passage_text = passage.passage_text
        
        # Try to find a keyword-based question
        question = None
        for keyword, template in self.KEYWORD_QUESTIONS.items():
            if keyword.lower() in passage_text.lower():
                question = template
                break
        
        # If no keyword match, use a template
        if not question:
            template = random.choice(self.QUESTION_TEMPLATES)
            question = template.format(control_name=control_name)
        
        # Generate answer by extracting key sentences
        answer = self._extract_answer(passage_text)
        
        if not answer or len(answer) < 10:
            return None
        
        question_id = f"Q_{passage.passage_id}_{hash(question) % 10000}"
        
        return QuestionAnswerPair(
            question_id=question_id,
            question=question,
            passages=[passage],
            answer=answer,
            group=1
        )
    
    def generate_multi_passage_question(self, passages: List[Passage], topic: str) -> Optional[QuestionAnswerPair]:
        """Generate a multi-passage question"""
        
        if len(passages) < 2:
            return None
        
        # Create a question that requires multiple passages
        question_templates = [
            f"What are the combined requirements for {topic} across these controls?",
            f"How do these controls work together for {topic}?",
            f"What are the comprehensive requirements for {topic}?",
            f"What must be done to ensure {topic} compliance across these controls?",
        ]
        
        question = random.choice(question_templates)
        
        # Combine answers from all passages
        answers = []
        for passage in passages:
            answer_part = self._extract_answer(passage.passage_text)
            if answer_part:
                answers.append(f"From {passage.passage_id}: {answer_part}")
        
        if not answers:
            return None
        
        combined_answer = " ".join(answers)
        question_id = f"Q_MULTI_{topic.replace(' ', '_')}_{len(passages)}_{hash(question) % 10000}"
        
        return QuestionAnswerPair(
            question_id=question_id,
            question=question,
            passages=passages,
            answer=combined_answer,
            group=len(passages),
            topic=topic
        )
    
    def _extract_control_name(self, passage: Passage) -> str:
        """Extract control name from passage"""
        # Try from metadata first
        if passage.metadata and 'control_name' in passage.metadata:
            return passage.metadata['control_name']
        
        # Try to extract from passage ID
        if passage.control_id:
            # Look for name pattern in passage text
            match = re.search(rf'{re.escape(passage.control_id)}\s+(.+?)(?:\n|Control|Priority)', passage.passage_text)
            if match:
                return match.group(1).strip()[:100]  # Limit length
        
        # Fallback to passage ID
        return passage.passage_id
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from passage text"""
        if not text or len(text.strip()) < 10:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Try to find the main statement (usually starts with "The entity shall" or similar)
        patterns = [
            r'(The entity shall[^.]*(?:\.[^.]*){0,3})',
            r'(Entities must[^.]*(?:\.[^.]*){0,3})',
            r'(The[^.]*(?:shall|must|should)[^.]*(?:\.[^.]*){0,2})',
            r'([^.]*(?:shall|must|should|required|ensure)[^.]*(?:\.[^.]*){0,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up answer
                answer = re.sub(r'\s+', ' ', answer)
                if len(answer) > 15:  # Lower threshold
                    return answer
        
        # If no pattern matches, take first sentence or first 200 chars
        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            answer = sentences[0].strip()
            answer = re.sub(r'\s+', ' ', answer)
            if len(answer) > 15:
                return answer
        
        # Last resort: first 200 characters, cleaned
        answer = text[:200].strip()
        answer = re.sub(r'\s+', ' ', answer)
        return answer if len(answer) > 10 else text[:100].strip()


class PassageExtractor:
    """Extract and structure passages from parsed regulatory documents"""
    
    def __init__(self, min_passage_length: int = 50, max_passage_length: int = 500):
        self.min_passage_length = min_passage_length
        self.max_passage_length = max_passage_length
        
    def extract_from_parsed_json(self, json_path: str) -> List[Passage]:
        """Extract passages from the parsed JSON structure."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        document_id = data.get('meta', {}).get('doc_id', 'unknown')
        content = data.get('content', [])
        
        passages = []
        current_passage_text = []
        current_passage_id = None
        current_control_id = None
        current_section = None
        
        control_pattern = re.compile(r'([MT]\d+\.\d+(?:\.\d+)*(?:\.\w+)?)')
        
        for item in content:
            text = item.get('text', '').strip()
            item_type = item.get('type', 'text')
            
            control_match = control_pattern.search(text)
            if control_match:
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
                
                current_control_id = control_match.group(1)
                current_passage_id = current_control_id
                current_passage_text = [text]
            else:
                if current_passage_id:
                    if item_type in ['text', 'header']:
                        current_passage_text.append(text)
                    if item_type == 'header' and len(text) < 100:
                        current_section = text
                else:
                    if item_type == 'header':
                        current_section = text
        
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
        """Extract passages from structured control JSON"""
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
        return self.min_passage_length <= len(text) <= self.max_passage_length * 10
    
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


class NLIValidator:
    """Validate question-passage pairs using Natural Language Inference"""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-xsmall"):
        if not NLI_AVAILABLE:
            raise ImportError("Transformers required. Install with: pip install transformers torch")
        
        print(f"Loading NLI model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    def validate(self, question: str, passage_text: str, threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Validate if question is entailed by passage.
        Lower threshold (0.3) to be less strict since template questions may not perfectly match.
        """
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
            entailment_prob = probs[0][0].item()
            # Use lower threshold for template-based questions
            is_valid = entailment_prob > threshold
        
        return is_valid, entailment_prob
    
    def validate_qa_pair(self, qa_pair: QuestionAnswerPair) -> QuestionAnswerPair:
        """Validate a QA pair against its passages"""
        if len(qa_pair.passages) == 1:
            is_valid, score = self.validate(qa_pair.question, qa_pair.passages[0].passage_text)
            qa_pair.is_validated = is_valid
            qa_pair.nli_score = score
        else:
            combined_text = " ".join([p.passage_text for p in qa_pair.passages])
            is_valid, score = self.validate(qa_pair.question, combined_text)
            qa_pair.is_validated = is_valid
            qa_pair.nli_score = score
        
        return qa_pair


class GoldenDatasetBuilder:
    """Main pipeline for building golden dataset (NO API REQUIRED)"""
    
    def __init__(self, 
                 nli_model: str = "microsoft/deberta-v3-xsmall"):
        self.passage_extractor = PassageExtractor()
        self.question_generator = TemplateQuestionGenerator()
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
        """Main pipeline to build golden dataset"""
        
        print("=" * 60)
        print("Golden Dataset Preparation Pipeline")
        print("Template-Based Question Generation (NO API REQUIRED)")
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
        
        passages_path = Path(output_dir) / "passages.json"
        self.passage_extractor.save_passages(passages, str(passages_path))
        
        # Step 2: Generate Questions (Template-based, no API)
        print(f"\n[Step 2] Generating questions using templates...")
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
            
            topic = random.choice(topics)
            topic_passages = passages_by_topic.get(topic, [])
            
            if len(topic_passages) >= 2:
                num_passages = random.randint(2, min(6, len(topic_passages)))
                selected_passages = random.sample(topic_passages, num_passages)
                
                qa_pair = self.question_generator.generate_multi_passage_question(
                    selected_passages, topic
                )
                if qa_pair:
                    qa_pairs.append(qa_pair)
        
        print(f"  ✓ Generated {len(qa_pairs)} total questions")
        
        # Step 3: NLI Validation (optional)
        if validation_enabled and self.nli_validator:
            print(f"\n[Step 3] Validating with NLI model (using relaxed threshold for template questions)...")
            validated_pairs = []
            rejected_count = 0
            for i, qa_pair in enumerate(qa_pairs):
                if (i + 1) % 50 == 0:
                    print(f"    Progress: {i+1}/{len(qa_pairs)}")
                
                validated = self.nli_validator.validate_qa_pair(qa_pair)
                if validated.is_validated:
                    validated_pairs.append(validated)
                else:
                    rejected_count += 1
            
            if len(validated_pairs) == 0 and len(qa_pairs) > 0:
                print(f"  ⚠️  WARNING: NLI validation rejected ALL {len(qa_pairs)} questions!")
                print(f"  This is likely because template questions don't match passage structure.")
                print(f"  Recommendation: Use --no-validation to skip NLI validation.")
                print(f"  Keeping all questions without validation...")
                # Don't filter - keep all questions
                for qa_pair in qa_pairs:
                    qa_pair.is_validated = False
                    qa_pair.nli_score = None
                validated_pairs = qa_pairs
            else:
                print(f"  ✓ Validated {len(validated_pairs)} questions (rejected {rejected_count})")
            
            qa_pairs = validated_pairs
        else:
            print(f"\n[Step 3] Skipping NLI validation")
            # Mark all as not validated
            for qa_pair in qa_pairs:
                qa_pair.is_validated = False
                qa_pair.nli_score = None
        
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
        
        for split_name, split_data in [('train', train_set), ('dev', dev_set), ('test', test_set)]:
            split_path = output_path / f"{split_name}.json"
            split_data_dict = {
                'meta': {
                    'split': split_name,
                    'total_questions': len(split_data),
                    'created_at': datetime.now().isoformat(),
                    'generation_method': 'template-based'
                },
                'questions': [self._qa_pair_to_dict(qa) for qa in split_data]
            }
            
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(split_data_dict, f, indent=2, ensure_ascii=False)
        
        full_dataset = {
            'meta': {
                'created_at': datetime.now().isoformat(),
                'total_questions': len(train_set) + len(dev_set) + len(test_set),
                'total_passages': len(passages),
                'train_size': len(train_set),
                'dev_size': len(dev_set),
                'test_size': len(test_set),
                'generation_method': 'template-based (no API required)'
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
        
        if len(all_sets) == 0:
            print("\n" + "=" * 60)
            print("Dataset Statistics")
            print("=" * 60)
            print("⚠️  WARNING: No questions in dataset!")
            print("This might be because:")
            print("  - NLI validation filtered out all questions (try --no-validation)")
            print("  - Question generation failed for all passages")
            print("  - Answer extraction returned empty results")
            return
        
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
    
    parser = argparse.ArgumentParser(description="Build golden dataset (NO API REQUIRED)")
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
                       help="Skip NLI validation (recommended for template-based questions)")
    
    args = parser.parse_args()
    
    # Default to no validation for template-based questions
    validation_enabled = not args.no_validation
    
    builder = GoldenDatasetBuilder()
    
    builder.build_dataset(
        parsed_json_path=args.parsed_json,
        structured_controls_path=args.structured_controls,
        output_dir=args.output_dir,
        num_single_questions_per_passage=args.num_single,
        num_multi_questions=args.num_multi,
        validation_enabled=validation_enabled
    )
