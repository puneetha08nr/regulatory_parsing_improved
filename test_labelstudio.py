# test_labelstudio.py

import requests
import json
from pathlib import Path

def test_label_studio_connection():
    """Test Label Studio connection with robust error handling"""
    
    BASE_URL = "http://localhost:8080"
    
    print("="*60)
    print("LABEL STUDIO CONNECTION TEST")
    print("="*60)
    
    # Step 1: Check if Label Studio is running (basic connectivity)
    print("\n1. Checking if Label Studio is running...")
    try:
        # Try a simple GET request first
        response = requests.get(BASE_URL, timeout=5)
        
        if response.status_code == 200:
            print(f"   ✓ Label Studio web interface is accessible")
            print(f"   URL: {BASE_URL}")
        else:
            print(f"   ⚠ Got response code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to {BASE_URL}")
        print("\n   Troubleshooting:")
        print("   1. Check if Label Studio is running:")
        print("      Run: label-studio start")
        print("   2. Check if port 8080 is in use:")
        print("      Run: lsof -i :8080")
        print("   3. Try accessing in browser: http://localhost:8080")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return False
    
    # Step 2: Check API endpoint
    print("\n2. Checking API endpoint...")
    try:
        # Try the API health check or projects endpoint
        response = requests.get(f"{BASE_URL}/api/projects", timeout=5)
        
        if response.status_code == 401:
            print("   ✓ API is accessible (needs authentication)")
        elif response.status_code == 200:
            print("   ✓ API is accessible (no auth required)")
        else:
            print(f"   ⚠ Unexpected status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"   ⚠ API check failed: {e}")
    
    # Step 3: Get API key from user
    print("\n3. API Key Setup")
    print("   " + "-"*56)
    print("   To get your API key:")
    print(f"   1. Open browser: {BASE_URL}")
    print("   2. Log in to Label Studio (create account if needed)")
    print("   3. Click your user avatar/icon (top right corner)")
    print("   4. Select 'Account & Settings'")
    print("   5. Copy the 'Access Token' value")
    print("   " + "-"*56)
    
    api_key = input("\n   Paste your API key here (or press Enter to skip): ").strip()
    
    if not api_key:
        print("\n   ⚠ No API key provided")
        print("   You can set it up later in: config/label_studio_config.json")
        
        # Create template config
        create_template_config(BASE_URL)
        return False
    
    # Step 4: Test API key
    print("\n4. Testing API key...")
    
    headers = {"Authorization": f"Token {api_key}"}
    
    try:
        response = requests.get(f"{BASE_URL}/api/projects", headers=headers, timeout=5)
        
        if response.status_code == 200:
            try:
                projects = response.json()
                print(f"   ✓ API key is VALID!")
                print(f"   ✓ You have access to {len(projects)} project(s)")
                
                # Save API key
                save_api_key(BASE_URL, api_key)
                return True
                
            except json.JSONDecodeError:
                print("   ⚠ Valid response but couldn't parse JSON")
                print(f"   Response: {response.text[:200]}")
                return False
                
        elif response.status_code == 401:
            print("   ✗ API key is INVALID")
            print("   Please check your token and try again")
            return False
            
        elif response.status_code == 403:
            print("   ✗ API key is valid but lacks permissions")
            return False
            
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Request failed: {e}")
        return False


def create_template_config(base_url: str):
    """Create a template config file"""
    
    config = {
        "label_studio": {
            "url": base_url,
            "api_key": "YOUR_API_KEY_HERE",
            "instructions": [
                "1. Get your API key from Label Studio UI",
                "2. Replace YOUR_API_KEY_HERE with your actual token",
                "3. Save this file"
            ]
        }
    }
    
    config_path = Path("config/label_studio_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n   ✓ Created template config: {config_path}")
    print("   Edit this file with your API key")


def save_api_key(base_url: str, api_key: str):
    """Save API key to config file"""
    
    config = {
        "label_studio": {
            "url": base_url,
            "api_key": api_key
        }
    }
    
    config_path = Path("config/label_studio_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n   ✓ Saved API key to: {config_path}")


def check_label_studio_process():
    """Check if Label Studio process is running"""
    import subprocess
    
    print("\n5. Checking Label Studio process...")
    try:
        # Check if label-studio process is running
        result = subprocess.run(
            ['pgrep', '-f', 'label-studio'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"   ✓ Label Studio process(es) running: {', '.join(pids)}")
            return True
        else:
            print("   ✗ No Label Studio process found")
            print("   Start it with: label-studio start")
            return False
            
    except FileNotFoundError:
        print("   ⚠ Cannot check process (pgrep not found)")
        return None


if __name__ == "__main__":
    print("\nPre-flight checks:")
    print("-" * 60)
    
    # Check if Label Studio is installed
    import subprocess
    try:
        result = subprocess.run(
            ['label-studio', '--version'],
            capture_output=True,
            text=True
        )
        version = result.stdout.strip() or result.stderr.strip()
        print(f"✓ Label Studio installed: {version}")
    except FileNotFoundError:
        print("✗ Label Studio not found!")
        print("  Install with: pip install label-studio")
        exit(1)
    
    # Check if process is running
    check_label_studio_process()
    
    print("-" * 60)
    
    # Run connection test
    success = test_label_studio_connection()
    
    print("\n" + "="*60)
    if success:
        print("✓ SETUP COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Extract controls: python src/pipeline/control_extractor_v2.py")
        print("2. Import to Label Studio: python scripts/import_to_labelstudio.py")
    else:
        print("⚠ SETUP INCOMPLETE")
        print("="*60)
        print("\nTroubleshooting steps:")
        print("1. Make sure Label Studio is running:")
        print("   label-studio start")
        print("\n2. Access Label Studio in browser:")
        print("   http://localhost:8080")
        print("\n3. Get your API key from Account Settings")
        print("\n4. Run this test again:")
        print("   python test_labelstudio.py")