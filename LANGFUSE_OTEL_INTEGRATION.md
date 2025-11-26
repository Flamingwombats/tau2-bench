# Langfuse OTEL Integration (Recommended)

## Overview

The Langfuse OpenTelemetry (`langfuse_otel`) integration is the **recommended** way to integrate LiteLLM with Langfuse. It uses the OpenTelemetry protocol, which is more reliable and avoids compatibility issues.

## Advantages over `langfuse` callback

1. **No compatibility issues**: Uses OpenTelemetry protocol, avoiding `sdk_integration` parameter problems
2. **More reliable**: Standardized protocol with better error handling
3. **Better performance**: Optimized for high-throughput scenarios
4. **Future-proof**: OpenTelemetry is the industry standard

## Prerequisites

Install required packages:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

## Configuration

### Environment Variables

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_OTEL_HOST="https://us.cloud.langfuse.com"  # US region (default)
# OR
export LANGFUSE_OTEL_HOST="https://cloud.langfuse.com"  # EU region
```

### Endpoint Resolution

The integration automatically constructs the OTEL endpoint:
- **US (default)**: `https://us.cloud.langfuse.com/api/public/otel`
- **EU**: `https://cloud.langfuse.com/api/public/otel`
- **Self-hosted**: `{LANGFUSE_OTEL_HOST}/api/public/otel`

## Usage

### Basic Setup

```python
import os
import litellm

# Set credentials
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."

# Enable OTEL integration
litellm.callbacks = ["langfuse_otel"]

# Make LLM requests
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Implementation in tau2-bench

To use OTEL integration instead of the current `langfuse` callback:

1. **Install dependencies**:
   ```bash
   pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
   ```

2. **Update `src/tau2/utils/llm_utils.py`**:
   - Change `litellm.success_callback = ["langfuse"]` to `litellm.callbacks = ["langfuse_otel"]`
   - Use `LANGFUSE_OTEL_HOST` instead of `LANGFUSE_HOST`

3. **Set environment variables**:
   ```bash
   export LANGFUSE_PUBLIC_KEY=your-key
   export LANGFUSE_SECRET_KEY=your-key
   export LANGFUSE_OTEL_HOST=https://us.cloud.langfuse.com
   ```

## Current Status

- ✅ **Available**: LiteLLM supports `langfuse_otel` callback
- ⚠️ **Dependencies**: Requires OpenTelemetry packages
- 🔄 **Migration**: Can be added as an alternative to current implementation

## Recommendation

**Switch to `langfuse_otel`** for better reliability and to avoid compatibility issues. The current `langfuse` callback implementation works but requires compatibility patches. The OTEL integration is cleaner and more maintainable.

