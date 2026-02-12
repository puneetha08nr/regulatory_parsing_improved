# scripts/03_label_studio_mgmt/import_controls.py

from label_studio_sdk import Client

def import_controls_to_label_studio():
    """Import extracted controls to Label Studio"""
    
    # Connect
    ls = Client(url='http://localhost:8080', api_key='8aff3a565ab4778ee6fdc9c0259016088382e1cb')
    
    # Create/get project
    project = ls.start_project(
        title='UAE IA - Control Mapping',
        label_config=open('uae_compliance_mapping.xml').read()
    )
    
    # Load tasks
    import json
    with open('uae_ia_controls.json') as f:
        tasks = json.load(f)
    
    # Import
    project.import_tasks(tasks)
    
    print(f"✓ Imported {len(tasks)} controls to Label Studio")
    print(f"  Project URL: http://localhost:8080/projects/{project.id}")

if __name__ == "__main__":
    import_controls_to_label_studio()