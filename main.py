import sys
from src.pipeline.parser import extract_controls_for_label_studio

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py [parse | classify | map]")
        return

    command = sys.argv[1]

    if command == "parse":
        print("🚀 Starting Ingestion Pipeline...")
        extract_controls_for_label_studio()
    
    elif command == "classify":
        print("⏳ Classifier module coming in Step 3...")
        # from src.pipeline.classifier import run_classifier
        # run_classifier()
        
    elif command == "map":
        print("⏳ Mapping module coming in Step 4...")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()