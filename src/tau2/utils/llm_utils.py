import json
import re
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()


# Apply langfuse compatibility patch BEFORE setting callbacks
# This fixes the sdk_integration parameter issue for langfuse 3.x
def _patch_litellm_langfuse_compatibility():
    """Patch LiteLLM's langfuse integration to work with langfuse 3.x"""
    try:
        import langfuse
        from packaging.version import Version

        # Get version from langfuse.version.__version__ (not langfuse.__version__)
        try:
            langfuse_version = langfuse.version.__version__
        except AttributeError:
            # Fallback to checking if langfuse is installed
            langfuse_version = None

        if not langfuse_version:
            return False

        # Only patch if langfuse is 3.x
        if Version(langfuse_version) >= Version("3.0.0"):
            try:
                import litellm.integrations.langfuse.langfuse as langfuse_module

                # Store original safe_init_langfuse_client method
                original_safe_init = (
                    langfuse_module.LangFuseLogger.safe_init_langfuse_client
                )

                def patched_safe_init(self, parameters: dict):
                    """Patched safe_init_langfuse_client that removes sdk_integration for langfuse 3.x"""
                    # Remove sdk_integration if present (langfuse 3.x doesn't support it)
                    if "sdk_integration" in parameters:
                        logger.debug(
                            "Removing sdk_integration parameter for langfuse compatibility "
                            "(not supported in langfuse 3.x)"
                        )
                        parameters = parameters.copy()  # Don't modify original dict
                        del parameters["sdk_integration"]

                    # Call original method with filtered parameters
                    return original_safe_init(self, parameters)

                # Apply the patch
                langfuse_module.LangFuseLogger.safe_init_langfuse_client = (
                    patched_safe_init
                )

                # Also patch LangfusePromptManagement if it exists
                try:
                    import litellm.integrations.langfuse.langfuse_prompt_management as prompt_mgmt_module
                    from packaging.version import Version as VersionCheck

                    # Patch langfuse_client_init function
                    if hasattr(prompt_mgmt_module, "langfuse_client_init"):

                        def patched_client_init(
                            langfuse_public_key=None,
                            langfuse_secret=None,
                            langfuse_host=None,
                            flush_interval=1,
                        ):
                            """Patched langfuse_client_init that removes sdk_integration for langfuse 3.x"""
                            import os
                            import langfuse as langfuse_pkg
                            from langfuse import Langfuse

                            # Build parameters (same as original)
                            parameters = {
                                "public_key": langfuse_public_key
                                or os.getenv("LANGFUSE_PUBLIC_KEY"),
                                "secret_key": langfuse_secret
                                or os.getenv("LANGFUSE_SECRET_KEY"),
                                "host": langfuse_host
                                or os.getenv(
                                    "LANGFUSE_HOST", "https://cloud.langfuse.com"
                                ),
                            }

                            # Only add sdk_integration for langfuse 2.6.0 - 2.x (not 3.x)
                            current_version = langfuse_pkg.version.__version__
                            if VersionCheck(current_version) >= VersionCheck(
                                "2.6.0"
                            ) and VersionCheck(current_version) < VersionCheck("3.0.0"):
                                parameters["sdk_integration"] = "litellm"

                            # Create client without sdk_integration for 3.x
                            client = Langfuse(**parameters)
                            return client

                        prompt_mgmt_module.langfuse_client_init = patched_client_init
                        logger.debug(
                            "Patched LangfusePromptManagement for compatibility"
                        )
                except (ImportError, AttributeError) as e:
                    # LangfusePromptManagement might not be available or have different structure
                    logger.debug(f"Could not patch LangfusePromptManagement: {e}")

                logger.debug(
                    f"Applied langfuse compatibility patch for version {langfuse_version}"
                )
                return True
            except ImportError:
                # langfuse not installed, no patch needed
                return False
            except Exception as patch_error:
                logger.debug(
                    f"Could not patch LiteLLM langfuse integration: {patch_error}"
                )
                return False
    except ImportError:
        # langfuse not installed, no patch needed
        return False
    except Exception:
        # Any other error, skip patch
        return False


# Apply patch early, before USE_LANGFUSE check
_patch_litellm_langfuse_compatibility()

if USE_LANGFUSE:
    # Check if langfuse environment variables are set before enabling callbacks
    import os

    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    # Support both LANGFUSE_HOST and LANGFUSE_BASE_URL (LiteLLM uses LANGFUSE_HOST)
    langfuse_host = (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )

    if langfuse_secret_key and langfuse_public_key:
        try:
            # IMPORTANT: Set langfuse environment variables BEFORE importing/using langfuse
            # LiteLLM reads these from environment when initializing langfuse callbacks
            os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret_key
            # LiteLLM uses LANGFUSE_HOST (not LANGFUSE_BASE_URL)
            if os.getenv("LANGFUSE_HOST") is None:
                if os.getenv("LANGFUSE_BASE_URL"):
                    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL")
                else:
                    os.environ["LANGFUSE_HOST"] = langfuse_host

            # Verify langfuse package is installed and compatible
            try:
                import langfuse

                # Get version from langfuse.version.__version__
                try:
                    langfuse_version = langfuse.version.__version__
                except AttributeError:
                    langfuse_version = "unknown"

                logger.debug(f"Langfuse package found, version: {langfuse_version}")

                # Compatibility patch is already applied at module level (see _patch_litellm_langfuse_compatibility)
                # This ensures it's applied before any callbacks are set

            except ImportError:
                logger.error(
                    "Langfuse package not installed. Install it with: pip install langfuse"
                )
                raise
            except Exception as import_error:
                # Handle compatibility issues (e.g., Python 3.14 with Pydantic V1)
                import sys

                python_version = sys.version_info
                error_msg = str(import_error)

                if "pydantic" in error_msg.lower() or "ConfigError" in error_msg:
                    logger.error(
                        f"Langfuse is not compatible with Python {python_version.major}.{python_version.minor}. "
                        f"Error: {error_msg[:200]}. "
                        "Langfuse uses Pydantic V1 which is not compatible with Python 3.14+. "
                        "Please use Python 3.13 or earlier, or wait for langfuse to support Pydantic V2."
                    )
                else:
                    logger.error(
                        f"Failed to import langfuse: {import_error}. "
                        "Langfuse callbacks will not be enabled."
                    )
                raise

            # Test langfuse connectivity (optional, but helpful for debugging)
            # Note: This is a simple check - actual callback errors will be handled in generate()
            try:
                # Try to create a langfuse client to verify credentials
                # This will fail fast if credentials are wrong, rather than failing during LLM calls
                from langfuse import Langfuse

                test_client = Langfuse(
                    public_key=langfuse_public_key,
                    secret_key=langfuse_secret_key,
                    host=langfuse_host,
                )
                logger.debug("Langfuse client created successfully")
                # Don't call auth_check() as it makes a network request and might block
            except Exception as client_error:
                logger.warning(
                    f"Failed to create Langfuse client: {client_error}. "
                    "Callbacks will still be enabled, but may fail during LLM calls."
                )

            # Set langfuse environment variables for LiteLLM
            # IMPORTANT: These must be set BEFORE setting callbacks
            os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret_key

            # Try to use langfuse_otel (OpenTelemetry) integration first - it's more reliable
            # Falls back to langfuse callback if OTEL is not available
            use_otel = os.getenv("LANGFUSE_USE_OTEL", "true").lower() == "true"

            if use_otel:
                # Use OTEL integration (recommended - avoids compatibility issues)
                # Set OTEL host - defaults to US region, can be overridden
                if not os.getenv("LANGFUSE_OTEL_HOST"):
                    # Convert LANGFUSE_HOST to LANGFUSE_OTEL_HOST format
                    if "us.cloud" in langfuse_host or "us.cloud" in os.getenv(
                        "LANGFUSE_BASE_URL", ""
                    ):
                        os.environ["LANGFUSE_OTEL_HOST"] = (
                            "https://us.cloud.langfuse.com"
                        )
                    elif "cloud.langfuse.com" in langfuse_host:
                        os.environ["LANGFUSE_OTEL_HOST"] = (
                            "https://cloud.langfuse.com"  # EU
                        )
                    else:
                        # Custom/self-hosted - use the host directly
                        os.environ["LANGFUSE_OTEL_HOST"] = langfuse_host

                try:
                    litellm.callbacks = ["langfuse_otel"]
                    logger.info(
                        f"Langfuse OTEL callbacks enabled (host: {os.getenv('LANGFUSE_OTEL_HOST')})"
                    )
                    logger.debug(
                        "Using OpenTelemetry integration - more reliable and avoids compatibility issues"
                    )
                except Exception as otel_error:
                    logger.warning(
                        f"Failed to set langfuse_otel callback: {otel_error}. "
                        "Falling back to langfuse callback."
                    )
                    use_otel = False

            if not use_otel:
                # Fallback to regular langfuse callback (requires compatibility patches)
                # LiteLLM uses LANGFUSE_HOST (not LANGFUSE_BASE_URL)
                os.environ["LANGFUSE_HOST"] = langfuse_host

                # Configure langfuse flush settings for faster trace visibility
                flush_interval = float(os.getenv("LANGFUSE_FLUSH_INTERVAL", "1.0"))
                if flush_interval < 1.0:
                    os.environ["LANGFUSE_FLUSH_INTERVAL"] = str(flush_interval)
                    logger.debug(f"Langfuse flush interval set to {flush_interval}s")

                # set callbacks with error handling
                try:
                    litellm.success_callback = ["langfuse"]
                    litellm.failure_callback = ["langfuse"]
                    logger.info(f"Langfuse callbacks enabled (host: {langfuse_host})")
                    logger.debug(
                        "Note: Using langfuse callback (not OTEL). "
                        "Consider using langfuse_otel for better reliability."
                    )
                except Exception as callback_error:
                    logger.warning(
                        f"Failed to set langfuse callbacks: {callback_error}. "
                        "This might be a version compatibility issue between LiteLLM and langfuse. "
                        "The agent will continue without langfuse tracing."
                    )
                    # Don't raise - let the agent work without langfuse

            # Configure LiteLLM to handle callback errors gracefully
            # Set suppress_debug_info to reduce noise, but this doesn't affect callback error handling
            litellm.suppress_debug_info = True
        except Exception as e:
            logger.error(
                f"Failed to enable Langfuse callbacks: {e}. "
                "Continuing without Langfuse tracing. "
                "LLM calls will still work normally."
            )
            import traceback

            logger.debug(traceback.format_exc())
    else:
        logger.warning(
            "USE_LANGFUSE is True but langfuse environment variables are not set. "
            "Langfuse callbacks will not be enabled. "
            "Please set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY environment variables."
        )

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        # silence this dogshit code that doesn't even work
        # logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    # Auto-detect Nebius API and load API key from environment if needed
    base_url = kwargs.get("base_url")
    # Auto-detect Nebius from model name if base_url not provided
    if not base_url and "nebius" in model.lower():
        base_url = "https://api.tokenfactory.nebius.com/v1/"
        kwargs["base_url"] = base_url
        logger.debug(f"Auto-detected Nebius API from model name: {model}")

    if base_url and "nebius" in base_url.lower():
        import os

        # Ensure api_key is set - prefer kwargs (if non-empty), then environment
        api_key_from_kwargs = kwargs.get("api_key")
        if not api_key_from_kwargs or (
            isinstance(api_key_from_kwargs, str) and not api_key_from_kwargs.strip()
        ):
            # api_key is missing or empty in kwargs, try environment
            nebius_api_key = os.getenv("NEBIUS_API_KEY")
            if nebius_api_key:
                kwargs["api_key"] = nebius_api_key
                logger.debug("Auto-loaded NEBIUS_API_KEY from environment for LiteLLM")
            else:
                logger.warning(
                    "Nebius API detected but no api_key provided in kwargs or NEBIUS_API_KEY environment variable. "
                    "LLM call will likely fail."
                )
        else:
            logger.debug(
                f"Using api_key from kwargs for Nebius API (length: {len(str(api_key_from_kwargs))})"
            )

        # LiteLLM strips provider prefix when using custom base_url, but Nebius API needs full model name
        # Solution: Use OpenAI provider format and pass full model name via extra_body to override
        # Handle both "nebius/openai/gpt-oss-120b" and "openai/gpt-oss-120b" formats
        if "/" in model:
            # Extract the model part (e.g., "gpt-oss-120b" from "nebius/openai/gpt-oss-120b" or "openai/gpt-oss-120b")
            model_parts = model.split("/")
            if len(model_parts) >= 2:
                # Get the last part as the actual model name
                model_part = model_parts[-1]  # "gpt-oss-120b"
                # Store the original full model name that Nebius API expects
                # For "nebius/openai/gpt-oss-120b", use "openai/gpt-oss-120b"
                # For "openai/gpt-oss-120b", use as-is
                if model.startswith("nebius/"):
                    original_model_name = "/".join(
                        model_parts[1:]
                    )  # "openai/gpt-oss-120b"
                else:
                    original_model_name = model
                # Use openai provider format so LiteLLM uses OpenAI client with correct endpoint
                model = f"openai/{model_part}"
                # Pass the full model name via extra_body - this should override the model in request body
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"]["model"] = original_model_name
                logger.debug(
                    f"Using OpenAI provider with model override: {model} -> {original_model_name} (via extra_body)"
                )

        # Ensure api_base is set explicitly (LiteLLM uses api_base parameter, not base_url)
        kwargs["api_base"] = base_url
        # Remove base_url if present to avoid confusion
        if "base_url" in kwargs:
            del kwargs["base_url"]

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    # Check if langfuse callbacks are causing issues before making the call
    # If langfuse is enabled but has compatibility issues, disable it temporarily
    langfuse_enabled_before_call = False
    if USE_LANGFUSE and (
        litellm.success_callback == ["langfuse"]
        or (
            isinstance(litellm.success_callback, list)
            and "langfuse" in litellm.success_callback
        )
    ):
        langfuse_enabled_before_call = True

    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        # Flush langfuse traces to ensure they're sent to the dashboard
        # Langfuse batches traces and sends them asynchronously, so we need to flush
        if USE_LANGFUSE and (
            litellm.success_callback == ["langfuse"]
            or (
                isinstance(litellm.success_callback, list)
                and "langfuse" in litellm.success_callback
            )
        ):
            try:
                # Get the langfuse logger instance and flush it
                # LiteLLM stores logger instances in a callback list
                import litellm.integrations.langfuse.langfuse as langfuse_module

                # Try to flush any active langfuse clients
                # Langfuse clients flush automatically, but we can force flush
                # Check if there are any initialized clients
                if hasattr(langfuse_module, "litellm"):
                    # Try to access the logger instance
                    # This is a bit hacky, but LiteLLM doesn't expose a clean way to flush
                    pass

                # Langfuse flushes automatically based on flush_interval (default: 1 second)
                # Traces are batched and sent asynchronously
                # For immediate visibility, traces may take a few seconds to appear in dashboard
                logger.debug(
                    "Langfuse traces will be sent automatically (batched, ~1s delay)"
                )
            except Exception as flush_error:
                logger.debug(f"Could not flush langfuse traces: {flush_error}")
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__

        # Check if langfuse callbacks might be causing the issue
        langfuse_enabled = USE_LANGFUSE and (
            litellm.success_callback == ["langfuse"]
            or (
                isinstance(litellm.success_callback, list)
                and "langfuse" in litellm.success_callback
            )
        )

        # Log the error with details
        logger.error(f"LLM call failed: {error_type}: {error_msg}")

        # If langfuse is enabled and the error might be callback-related, try without it
        if langfuse_enabled:
            # Check if error might be related to langfuse (network errors, auth errors, compatibility issues, etc.)
            langfuse_related_errors = [
                "langfuse",
                "connection",
                "timeout",
                "authentication",
                "unauthorized",
                "forbidden",
                "callback",
                "sdk_integration",  # Specific compatibility issue between LiteLLM and langfuse
                "unexpected keyword argument",
            ]
            might_be_langfuse_error = any(
                err_keyword.lower() in error_msg.lower()
                for err_keyword in langfuse_related_errors
            )

            # Also check for TypeError which is what the sdk_integration error raises
            is_type_error = error_type == "TypeError"

            if (
                might_be_langfuse_error
                or "langfuse" in error_msg.lower()
                or (is_type_error and langfuse_enabled_before_call)
            ):
                logger.warning(
                    f"LLM call failed with langfuse enabled (error: {error_type}). "
                    "This might be a langfuse callback issue. "
                    "Disabling langfuse callbacks and retrying..."
                )
                # Temporarily disable langfuse callbacks
                original_success_callback = litellm.success_callback
                original_failure_callback = litellm.failure_callback
                litellm.success_callback = []
                litellm.failure_callback = []
                try:
                    response = completion(
                        model=model,
                        messages=litellm_messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        **kwargs,
                    )
                    logger.warning(
                        "LLM call succeeded after disabling langfuse callbacks. "
                        "Langfuse tracing is disabled for this session."
                    )
                    # Keep callbacks disabled since they're causing issues
                    # Don't restore them to avoid repeated failures
                except Exception as retry_error:
                    # Restore callbacks before re-raising
                    litellm.success_callback = original_success_callback
                    litellm.failure_callback = original_failure_callback
                    logger.error(
                        f"LLM call failed even after disabling langfuse: {retry_error}"
                    )
                    raise retry_error
            else:
                # Error doesn't seem related to langfuse, raise it normally
                logger.error(
                    f"LLM call failed (not langfuse-related): {error_type}: {error_msg}"
                )
                raise e
        else:
            # Langfuse not enabled, just raise the error
            raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
