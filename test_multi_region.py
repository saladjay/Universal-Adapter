import asyncio
import os
import traceback
from llm_adapter.adapters.gemini_adapter import GeminiAdapter

async def test():
    try:
        adapter = GeminiAdapter(
            api_key="dummy",
            mode="vertex",
            project_id="wingy-e87ee",
            location="global",  # Global endpoint - Google auto-selects best zone
            enable_region_fallback=False
        )
        
        print(f"Location: {adapter.location}")
        
        result = await adapter.generate("Say hi", "gemini-2.0-flash-lite-001")
        print(f"Success: {result.text}")
        
        await adapter.aclose()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

asyncio.run(test())
