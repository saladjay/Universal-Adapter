#!/usr/bin/env python3
"""
Check if Vertex AI environment is properly configured.

This script verifies:
1. GOOGLE_APPLICATION_CREDENTIALS environment variable is set
2. The credentials file exists
3. The credentials file is valid JSON
4. Required packages are installed
"""

import os
import sys
import json


def check_env_var():
    """Check if GOOGLE_APPLICATION_CREDENTIALS is set."""
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not creds_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS is not set")
        print("\nTo fix this, run:")
        print("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json")
        return False
    
    print(f"✅ GOOGLE_APPLICATION_CREDENTIALS is set: {creds_path}")
    return creds_path


def check_file_exists(creds_path):
    """Check if the credentials file exists."""
    if not os.path.exists(creds_path):
        print(f"❌ Credentials file does not exist: {creds_path}")
        return False
    
    print(f"✅ Credentials file exists")
    return True


def check_file_valid(creds_path):
    """Check if the credentials file is valid JSON."""
    try:
        with open(creds_path, 'r') as f:
            data = json.load(f)
        
        # Check for required fields
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"❌ Credentials file is missing required fields: {', '.join(missing_fields)}")
            return False
        
        if data.get('type') != 'service_account':
            print(f"❌ Credentials file type is '{data.get('type')}', expected 'service_account'")
            return False
        
        print(f"✅ Credentials file is valid")
        print(f"   Project ID: {data.get('project_id')}")
        print(f"   Service Account: {data.get('client_email')}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Credentials file is not valid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading credentials file: {e}")
        return False


def check_packages():
    """Check if required packages are installed."""
    packages = {
        'google-cloud-aiplatform': 'google.cloud.aiplatform',
        'vertexai': 'vertexai',
    }
    
    all_installed = True
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is not installed")
            print(f"   Install with: pip install {package_name}")
            all_installed = False
    
    return all_installed


def main():
    """Run all checks."""
    print("Checking Vertex AI Environment Configuration")
    print("=" * 50)
    print()
    
    # Check environment variable
    print("1. Checking GOOGLE_APPLICATION_CREDENTIALS...")
    creds_path = check_env_var()
    if not creds_path:
        print("\n❌ Environment check failed")
        sys.exit(1)
    print()
    
    # Check file exists
    print("2. Checking credentials file...")
    if not check_file_exists(creds_path):
        print("\n❌ Environment check failed")
        sys.exit(1)
    print()
    
    # Check file is valid
    print("3. Validating credentials file...")
    if not check_file_valid(creds_path):
        print("\n❌ Environment check failed")
        sys.exit(1)
    print()
    
    # Check packages
    print("4. Checking required packages...")
    if not check_packages():
        print("\n❌ Environment check failed")
        sys.exit(1)
    print()
    
    print("=" * 50)
    print("✅ All checks passed! Vertex AI environment is properly configured.")
    print()
    print("You can now use Vertex AI mode in your config.yaml:")
    print("""
  gemini:
    mode: vertex
    project_id: your-project-id
    location: asia-southeast1
    models:
      multimodal: gemini-2.5-flash
""")


if __name__ == "__main__":
    main()
