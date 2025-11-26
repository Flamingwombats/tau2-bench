# Langfuse Tracing Troubleshooting

## ✅ **NEW: Using Langfuse OTEL Integration (Recommended)**

The code now **automatically uses `langfuse_otel` (OpenTelemetry) integration** by default, which:
- ✅ Avoids compatibility issues (no `sdk_integration` errors)
- ✅ More reliable and performant
- ✅ Uses standardized OpenTelemetry protocol

**No changes needed** - it works automatically! The code falls back to the regular `langfuse` callback if OTEL is not available.

## Issue: Traces Not Appearing in Dashboard

If you've enabled `USE_LANGFUSE=True` but don't see traces in your langfuse dashboard, check the following:

### 1. Verify Environment Variables

Make sure these are set **before** running your application:

```bash
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_BASE_URL=https://us.cloud.langfuse.com  # or your langfuse instance
# OR
export LANGFUSE_OTEL_HOST=https://us.cloud.langfuse.com  # for OTEL integration
```

**Note**: The code automatically sets `LANGFUSE_OTEL_HOST` based on `LANGFUSE_BASE_URL` if not explicitly set.

### 2. Check Langfuse Configuration

Verify in `src/tau2/config.py`:
```python
USE_LANGFUSE = True
```

### 3. Understand Langfuse Batching

**Important**: Langfuse batches traces and sends them asynchronously. Traces may take **1-2 seconds** to appear in the dashboard after an LLM call completes.

- Default flush interval: **1 second**
- Traces are batched for efficiency
- For faster visibility, set `LANGFUSE_FLUSH_INTERVAL=0.1` (but this increases API calls)

### 4. Check Application Logs

Look for these log messages:
- `✓ Langfuse OTEL callbacks enabled (host: ...)` ← **Preferred (OTEL integration)**
- `✓ Langfuse callbacks enabled (host: ...)` ← Fallback (regular callback)
- `✓ Applied langfuse compatibility patch` ← Only if using fallback

**If using OTEL integration** (default), you should see:
```
Langfuse OTEL callbacks enabled (host: https://us.cloud.langfuse.com)
Using OpenTelemetry integration - more reliable and avoids compatibility issues
```

**If using fallback**, you might see `sdk_integration` errors - these are handled by the compatibility patch.

### 5. Verify Traces Are Being Sent

Add debug logging to see if callbacks are invoked:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Look for LiteLLM callback logs during LLM calls.

### 6. Test with a Simple Call

Use the test script to verify tracing works:

```bash
cd tau2-bench/test
source ../.venv/bin/activate
export LANGFUSE_SECRET_KEY=your-key
export LANGFUSE_PUBLIC_KEY=your-key
export LANGFUSE_HOST=https://us.cloud.langfuse.com
python test_langfuse.py
```

Then check your langfuse dashboard - you should see a trace within 1-2 seconds.

### 7. Check Dashboard Filters

In the langfuse dashboard:
- Check the time range filter (make sure it includes recent time)
- Check project filter (make sure you're looking at the right project)
- Refresh the page

### 8. Network Issues

If traces still don't appear:
- Check network connectivity to langfuse host
- Verify firewall/proxy settings
- Check if langfuse host is accessible: `curl https://us.cloud.langfuse.com`

### 9. Force Immediate Flush (for testing)

For testing purposes, you can reduce flush interval:

```bash
export LANGFUSE_FLUSH_INTERVAL=0.1
```

This makes traces appear faster but increases API calls.

### 10. Verify Credentials

Make sure your `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` are correct:
- They should start with `sk-lf-` and `pk-lf-` respectively
- They should match the credentials in your langfuse project settings

## Common Issues

### "sdk_integration" errors
- **Fixed**: The compatibility patch should handle this automatically
- If you still see this error, the patch might not be applied - check logs

### Traces appear but are empty
- This might indicate the LLM call succeeded but callback data wasn't captured
- Check if LiteLLM is actually invoking callbacks

### Traces appear with delay
- **Normal**: Langfuse batches traces (default 1 second)
- Wait 1-2 seconds after LLM calls complete

## Still Not Working?

1. Check application logs for errors
2. Verify all environment variables are set
3. Test with the `test_langfuse.py` script
4. Check langfuse dashboard project settings
5. Verify network connectivity to langfuse host

