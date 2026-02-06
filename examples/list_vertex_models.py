"""
List available models in Vertex AI for a specific region.

This script uses the Vertex AI API to check which Gemini models are available
in different regions by attempting to initialize them.
"""

import os
import asyncio
import vertexai
from vertexai.generative_models import GenerativeModel




async def check_model_availability(project_id: str, location: str, model_name: str):
    """Check if a specific model is available by attempting a simple generation."""
    
    vertexai.init(project=project_id, location=location)
    
    try:
        model = GenerativeModel(model_name)
        # Try a minimal generation to verify availability
        response = await model.generate_content_async(
            "Hi",
            generation_config={"max_output_tokens": 5}
        )
        return True, None
    except Exception as e:
        error_msg = str(e)
        # Check if it's a 404 (model not found) or other error
        if "404" in error_msg or "not found" in error_msg.lower():
            return False, "Not available in this region"
        else:
            return False, f"Error: {error_msg[:80]}"



async def main():
    """Main function to check model availability across regions."""
    
    project_id = os.getenv("GCP_PROJECT_ID", "wingy-e87ee")
    
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("⚠ Warning: GOOGLE_APPLICATION_CREDENTIALS not set")
        return
    
    # Regions to check
    regions = [
        ("us-central1", "US Central (Iowa)"),
        ("us-east4", "US East (Virginia)"),
        ("europe-west1", "Europe West (Belgium)"),
        ("asia-southeast1", "Asia Southeast (Singapore)"),
        ("asia-northeast1", "Asia Northeast (Tokyo)"),
    ]
    
    # Models to check
    models_to_check = [
        # Gemini 2.5 models (latest)
        "gemini-2.5-flash",
        "gemini-2.5-flash-001",
        "gemini-2.5-flash-002",
        "gemini-2.5-pro",
        "gemini-2.5-pro-001",
        "gemini-2.5-pro-002",
        # Gemini 2.0 models
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
        "gemini-2.0-flash-001",
        "gemini-2.0-pro",
        "gemini-2.0-pro-exp",
        # Gemini 1.5 models
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro",
        "gemini-1.5-pro-001",
        "gemini-1.5-pro-002",
        # Gemini 1.0 models
        "gemini-1.0-pro",
        "gemini-1.0-pro-001",
        "gemini-1.0-pro-002",
        # Experimental/Preview models
        "gemini-exp-1206",
        "gemini-2.0-flash-thinking-exp",
        "gemini-2.5-flash-preview",
        "gemini-2.5-pro-preview",
    ]
    
    print("\n" + "="*80)
    print("Vertex AI Gemini Model Availability Check")
    print("="*80)
    
    # Check each model in each region
    results = {}
    
    for model in models_to_check:
        print(f"\n📦 Checking {model}...")
        results[model] = {}
        
        for region_code, region_name in regions:
            print(f"  Testing {region_name}...", end=" ", flush=True)
            available, error = await check_model_availability(project_id, region_code, model)
            results[model][region_code] = (available, error)
            
            if available:
                print("✓ Available")
            else:
                print(f"✗ {error}")
    
    # Print summary table
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"\n{'Model':<35} | {'Region':<25} | Status")
    print("-" * 80)
    
    for model in models_to_check:
        for region_code, region_name in regions:
            available, error = results[model][region_code]
            status = "✓ Available" if available else "✗ Not Available"
            print(f"{model:<35} | {region_name:<25} | {status}")
    
    # Print recommendations
    print("\n" + "="*80)
    print("Recommendations")
    print("="*80)
    
    for model in models_to_check:
        available_regions = [
            region_name for (region_code, region_name) in regions
            if results[model][region_code][0]
        ]
        
        if available_regions:
            print(f"\n{model}:")
            print(f"  Available in: {', '.join(available_regions)}")
        else:
            print(f"\n{model}:")
            print(f"  ⚠ Not available in any tested region")


if __name__ == "__main__":
    asyncio.run(main())
