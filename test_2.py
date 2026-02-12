from label_studio_sdk import Client
import requests

# 1. Update with your working API Key
API_KEY = "8aff3a565ab4778ee6fdc9c0259016088382e1cb"
URL = "http://localhost:8080"

def debug_create_project():
    ls = Client(url=URL, api_key=API_KEY)
    
    # Read your XML file
    try:
        with open('data/03_label_studio_input/uae_compliance_mapping.xml', 'r') as f:
            xml_content = f.read()
    except FileNotFoundError:
        print("❌ Error: XML file not found at 'data/03_label_studio_input/uae_compliance_mapping.xml'")
        return

    print("--- Attempting to create project ---")
    
    try:
        # We manually use the legacy method to catch the specific error
        response = ls.make_request(
            "POST", 
            "/api/projects", 
            json={"title": "DEBUG_PROJECT", "label_config": xml_content}
        )
        print("✅ Success! Project created.")
        print(f"ID: {response.json().get('id')}")
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ SERVER REJECTED THE XML (Error 400)")
        print("="*40)
        # THIS IS THE PART YOU NEED:
        print(e.response.text) 
        print("="*40)

if __name__ == "__main__":
    debug_create_project()