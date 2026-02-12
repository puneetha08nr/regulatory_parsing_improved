"""
Example script showing how to use the golden dataset pipeline programmatically
"""

from golden_dataset_pipeline import GoldenDatasetBuilder
import os
from pathlib import Path

def main():
    # Initialize the builder
    builder = GoldenDatasetBuilder(
        openai_api_key=os.getenv('OPENAI_API_KEY'),  # Or pass directly
        gpt_model="gpt-4-turbo-preview"
    )
    
    # Build the dataset
    dataset = builder.build_dataset(
        # Use structured controls (from auto_convert.py output)
        structured_controls_path="data/02_processed/uae_ia_controls_structured.json",
        
        # Optionally also use parsed JSON
        # parsed_json_path="data/02_intermediate/parsed_json/UAE_IA.json",
        
        # Output directory
        output_dir="data/05_golden_dataset",
        
        # Generation parameters
        num_single_questions_per_passage=1,  # 1 question per passage
        num_multi_questions=50,  # Generate 50 multi-passage questions
        
        # Validation
        validation_enabled=True,  # Use NLI validation
        
        # Splits
        train_split=0.8,
        dev_split=0.1
        # test_split = 1 - train_split - dev_split = 0.1
    )
    
    # Access the results
    print(f"\nDataset created successfully!")
    print(f"Train set: {len(dataset['train'])} questions")
    print(f"Dev set: {len(dataset['dev'])} questions")
    print(f"Test set: {len(dataset['test'])} questions")
    print(f"Total passages: {len(dataset['passages'])}")
    
    # Example: Access a question
    if dataset['train']:
        first_question = dataset['train'][0]
        print(f"\nExample question:")
        print(f"  ID: {first_question.question_id}")
        print(f"  Question: {first_question.question}")
        print(f"  Answer: {first_question.answer[:100]}...")
        print(f"  Passages needed: {first_question.group}")

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or create a .env file with OPENAI_API_KEY=your-key-here")
    
    main()
