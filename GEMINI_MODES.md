# Gemini Adapter Modes

The Gemini adapter supports three different modes for calling Google's Gemini models:

## 1. HTTP Mode (Default)

Direct HTTP API calls with minimal dependencies.

**Pros:**
- No additional dependencies
- Lightweight
- Fast initialization

**Cons:**
- Manual error handling
- Limited features

**Configuration:**
```yaml
providers:
  gemini:
    api_key: your-gemini-api-key
    mode: http  # or omit (default)
    models:
      cheap: gemini-2.5-flash
      normal: gemini-2.5-flash
      premium: gemini-2.5-flash
      multimodal: gemini-2.5-flash
```

**Usage:**
```python
from llm_adapter.adapters import GeminiAdapter

adapter = GeminiAdapter(api_key="your-key", mode="http")
result = await adapter.generate("Hello", "gemini-2.5-flash")
```

## 2. SDK Mode

Uses the official `google-generativeai` SDK.

**Pros:**
- Official SDK with better stability
- Automatic retries and error handling
- More features (safety settings, etc.)

**Cons:**
- Requires additional dependency
- Slightly heavier

**Installation:**
```bash
pip install google-generativeai
```

**Configuration:**
```yaml
providers:
  gemini:
    api_key: your-gemini-api-key
    mode: sdk
    models:
      cheap: gemini-2.5-flash
      normal: gemini-2.5-flash
      premium: gemini-2.5-flash
      multimodal: gemini-2.5-flash
```

**Usage:**
```python
from llm_adapter.adapters import GeminiAdapter

adapter = GeminiAdapter(api_key="your-key", mode="sdk")
result = await adapter.generate("Hello", "gemini-2.5-flash")
```

## 3. Vertex AI Mode

Uses Vertex AI SDK for GCP projects with regional deployment.

**Pros:**
- Regional deployment (lower latency)
- GCP integration
- Enterprise features
- Better quota management

**Cons:**
- Requires GCP project setup
- Requires additional dependency
- More complex configuration
- Requires service account credentials

**Installation:**
```bash
pip install google-cloud-aiplatform
```

**Setup:**

1. Create a GCP project and enable Vertex AI API
2. Create a service account with Vertex AI permissions
3. Download the service account JSON key file
4. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

**Configuration:**
```yaml
providers:
  gemini:
    api_key: ""  # Not used in vertex mode
    mode: vertex
    project_id: your-gcp-project-id
    location: asia-southeast1  # GCP region
    models:
      cheap: gemini-2.0-flash
      normal: gemini-2.5-flash
      premium: gemini-3-flash-preview
      multimodal: gemini-2.5-flash
```

**Usage:**
```python
from llm_adapter.adapters import GeminiAdapter

adapter = GeminiAdapter(
    api_key="",  # Not used
    mode="vertex",
    project_id="your-gcp-project-id",
    location="asia-southeast1"
)
result = await adapter.generate("你好", "gemini-2.0-flash")
```

**Environment Variables Required:**
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON key file

## Streaming Support

All three modes support streaming:

```python
async for chunk in adapter.stream("Count to 5", "gemini-2.5-flash"):
    print(chunk, end="", flush=True)
```

## Mode Comparison

| Feature | HTTP | SDK | Vertex AI |
|---------|------|-----|-----------|
| Dependencies | None | google-generativeai | google-cloud-aiplatform |
| Setup Complexity | Low | Low | High |
| Regional Deployment | No | No | Yes |
| GCP Integration | No | No | Yes |
| Streaming | ✅ | ✅ | ✅ |
| Token Counting | ✅ | ✅ | ✅ |
| Error Handling | Manual | Automatic | Automatic |

## Choosing a Mode

- **HTTP Mode**: Best for simple use cases, quick prototyping
- **SDK Mode**: Best for production with API key authentication
- **Vertex AI Mode**: Best for GCP-based deployments, enterprise use cases

## Vertex AI Setup Guide

### Prerequisites

1. **GCP Project**: Create a project in Google Cloud Console
2. **Enable Vertex AI API**: Enable the Vertex AI API for your project
3. **Service Account**: Create a service account with Vertex AI User role
4. **JSON Key**: Download the service account JSON key file

### Step-by-Step Setup

1. **Install required package:**
   ```bash
   pip install google-cloud-aiplatform
   ```

2. **Set environment variable:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

3. **Verify setup:**
   ```bash
   python core/llm_adapter/check_vertex_env.py
   ```

4. **Configure in config.yaml:**
   ```yaml
   gemini:
     mode: vertex
     project_id: your-gcp-project-id
     location: asia-southeast1
     models:
       multimodal: gemini-2.5-flash
   ```

### Troubleshooting

If you see errors about missing credentials:
- Check that `GOOGLE_APPLICATION_CREDENTIALS` is set: `echo $GOOGLE_APPLICATION_CREDENTIALS`
- Verify the file exists: `ls -l $GOOGLE_APPLICATION_CREDENTIALS`
- Run the check script: `python core/llm_adapter/check_vertex_env.py`

## Example

See `examples/gemini_modes_example.py` for complete working examples of all three modes.
