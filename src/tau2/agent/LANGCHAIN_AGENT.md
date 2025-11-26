# LangChain Agent Implementation

## Overview

The `LangChainAgent` is a conversational agent implementation for the `tau2-bench` framework that leverages LangGraph's `create_react_agent` to provide a robust, tool-using agent capable of interacting with domain-specific tools.

## Architecture

### Class Hierarchy

```
LocalAgent (base.py)
  └── LangChainAgent (langchain_agent.py)
```

The `LangChainAgent` inherits from `LocalAgent` and implements the required methods for integration with the `tau2-bench` framework.

### Key Components

1. **LangGraph Agent**: Uses `create_react_agent` from `langgraph.prebuilt` for the core agent logic
2. **LangChain LLM**: Wraps the LLM using `ChatOpenAI` from `langchain_openai`
3. **Tool Conversion**: Converts `tau2.environment.tool.Tool` objects to `langchain_core.tools.StructuredTool`
4. **State Management**: Maintains conversation history in both `tau2` and LangChain message formats
5. **Isolated Environment**: Uses a separate environment instance for LangGraph's internal tool executions

## Key Features

### 1\. Tool Call Extraction

The agent extracts **all** tool calls from LangGraph's responses, not just the first one. This ensures that when LangGraph executes multiple tools in sequence during a single `invoke()` call, all tool calls are properly captured and sent to the orchestrator.

**Implementation Details:**

- Collects all `AIMessage` objects with `tool_calls` from new messages
- Combines all tool calls into a single `AssistantMessage`
- Ensures all expected actions are recorded in the trajectory

### 2\. Isolated Tool Execution

LangGraph's `create_react_agent` executes tools internally during its reasoning loop. To prevent state mismatches during evaluation replay, the agent uses an isolated environment instance for LangGraph's tool executions.

**Why This Matters:**

- LangGraph executes tools synchronously during `agent.invoke()`
- Without isolation, these executions would modify the main environment state
- During evaluation replay, the state would differ, causing mismatches
- The isolated environment ensures LangGraph gets correct responses without affecting the main environment

### 3\. API Key Handling

The agent provides robust API key handling with automatic fallback to environment variables:

- Checks for `api_key` parameter
- Falls back to `NEBIUS_API_KEY` environment variable for Nebius API
- Provides clear error messages if no API key is found
- Handles `None` and empty string values gracefully

### 4\. Message Format Conversion

The agent maintains conversation history in both formats:

- **tau2 format**: For integration with the framework
- **LangChain format**: For LangGraph's internal processing

## How It Works

### Initialization Flow

1. **Tool Conversion**: Converts `tau2` tools to LangChain `StructuredTool` objects
2. **LLM Setup**: Creates `ChatOpenAI` instance with proper API key and base URL
3. **Agent Creation**: Creates LangGraph agent using `create_react_agent`
4. **State Initialization**: Sets up initial state with system messages and empty conversation history

### Message Processing Flow

```
User Message
    ↓
Add to state (both tau2 and LangChain formats)
    ↓
Invoke LangGraph agent
    ↓
LangGraph executes tools internally (through isolated environment)
    ↓
Extract ALL tool calls from LangGraph response
    ↓
Return AssistantMessage with tool calls to orchestrator
    ↓
Orchestrator executes tools through main environment
    ↓
Tool responses added to state
    ↓
Repeat
```

### Tool Execution Flow

```
LangGraph wants to execute tool
    ↓
Tool wrapper function called
    ↓
If tool_executor provided:
    Create ToolCall object
    Execute through tool_executor (isolated environment)
    Return response to LangGraph
Else:
    Execute directly (fallback)
    Return response to LangGraph
    ↓
LangGraph continues reasoning
    ↓
Tool calls extracted and sent to orchestrator
    ↓
Orchestrator executes through main environment
```

## State Management

### LangChainAgentState

```python
class LangChainAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]  # tau2 format
    langchain_messages: list  # LangChain format
    tool_execution_cache: dict = {}  # For evaluation replay
```

### State Synchronization

The agent maintains two parallel message histories:

- **`messages`**: tau2 format messages for framework integration
- **`langchain_messages`**: LangChain format messages for LangGraph

When a new message arrives:

1. It's added to `messages` (tau2 format)
2. It's converted and added to `langchain_messages` (LangChain format)
3. LangGraph processes `langchain_messages`
4. Tool calls are extracted and converted back to tau2 format

## Integration with tau2-bench

### Registration

The agent is registered in `src/tau2/registry.py`:

```python
from tau2.agent.langchain_agent import LangChainAgent
registry.register_agent(LangChainAgent, "langchain_agent")
```

### Instantiation

The agent is instantiated in `src/tau2/run.py` with special handling:

```python
elif issubclass(AgentConstructor, LangChainAgent):
    # Extract base_url and api_key
    base_url = llm_args_agent.pop("base_url", None)
    api_key = llm_args_agent.pop("api_key", None)

    # Create isolated environment for LangGraph
    langgraph_environment = environment_constructor()
    # ... initialize environment ...

    # Create tool_executor callback
    def tool_executor(tool_call):
        return langgraph_environment.get_response(tool_call)

    # Instantiate agent
    agent = AgentConstructor(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=llm_agent,
        tool_executor=tool_executor,
        base_url=base_url,
        api_key=api_key,
        ...
    )
```

## Usage Examples

### Basic Usage

```bash
tau2 run \
  --domain retail \
  --agent langchain_agent \
  --agent-llm openai/gpt-oss-120b \
  --agent-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key": "your-key"}' \
  --user-llm openai/gpt-oss-120b \
  --user-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key": "your-key"}'
```

### Using Environment Variable for API Key

```bash
export NEBIUS_API_KEY="your-api-key"
tau2 run \
  --domain retail \
  --agent langchain_agent \
  --agent-llm openai/gpt-oss-120b \
  --agent-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/"}' \
  --user-llm openai/gpt-oss-120b \
  --user-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/"}'
```

## Implementation Details

### Tool Conversion

The `_convert_tools_to_langchain` method converts `tau2` tools to LangChain tools:

1. Creates a wrapper function for each tool
2. If `tool_executor` is provided, routes tool calls through it
3. Otherwise, executes tools directly (fallback)
4. Handles errors and converts responses to strings

### Tool Call Extraction

The `generate_next_message` method extracts tool calls from LangGraph responses:

1. Invokes LangGraph agent with current message history
2. Gets new messages from LangGraph response
3. Collects all `AIMessage` objects with `tool_calls`
4. Combines all tool calls into a single `AssistantMessage`
5. Converts to tau2 format and returns

### Message Format Conversion

The agent converts between tau2 and LangChain message formats:

- **tau2 → LangChain**: `_tau2_to_langchain_message`
- **LangChain → tau2**: `_langchain_to_tau2_assistant_message`

Key conversions:

- `UserMessage` → `HumanMessage`
- `AssistantMessage` → `AIMessage`
- `ToolMessage` → `ToolMessage`
- `SystemMessage` → `SystemMessage`

## Known Limitations

### 1\. Evaluation Mismatch

The agent may receive a reward of 0.0 even when it correctly follows user choices that differ from the expected golden actions. This is a limitation of the evaluation framework, not the agent implementation.

**Example:**

- Expected action: Exchange with keyboard `7706410293` (no backlight)
- User chooses: Keyboard `6342039236` (white backlight)
- Agent correctly executes user's choice
- Evaluation fails because action doesn't match expected

### 2\. State Isolation

The isolated environment approach ensures evaluation replay works correctly, but means LangGraph's internal tool executions don't affect the main environment. This is intentional but worth noting.

### 3\. Tool Execution Order

LangGraph may execute multiple tools in sequence during a single `invoke()` call. The agent extracts all tool calls, but the order may differ from what the orchestrator expects.

## Troubleshooting

### API Key Errors

**Error**: `The api_key client option must be set`

**Solution**:

- Provide `api_key` in `llm_args`
- Or set `NEBIUS_API_KEY` environment variable
- Or pass `api_key` as a separate parameter

### Tool Calls Not Extracted

**Symptom**: Expected actions not found in trajectory

**Solution**:

- Check that tool calls are being extracted from all `AIMessage` objects
- Verify that `tool_executor` is properly configured
- Check logs for tool call extraction debug messages

### State Mismatch During Evaluation

**Symptom**: DB check fails during evaluation replay

**Solution**:

- Verify that isolated environment is being used for LangGraph
- Check that tool calls are being executed through the orchestrator
- Ensure state is properly synchronized

## Code Structure

```
src/tau2/agent/
├── langchain_agent.py      # Main implementation
├── base.py                  # Base classes (LocalAgent)
└── registry.py             # Agent registration

src/tau2/run.py             # Agent instantiation logic
```

## Key Methods

### `__init__`

- Initializes the agent
- Converts tools to LangChain format
- Sets up LLM and LangGraph agent

### `get_init_state`

- Returns initial state with system messages
- Initializes empty conversation history

### `generate_next_message`

- Processes new messages
- Invokes LangGraph agent
- Extracts and returns tool calls or text response

### `_convert_tools_to_langchain`

- Converts tau2 tools to LangChain tools
- Sets up tool execution routing

### `_create_langchain_llm`

- Creates ChatOpenAI instance
- Handles API key and base URL configuration

### `_tau2_to_langchain_message`

- Converts tau2 messages to LangChain format

### `_langchain_to_tau2_assistant_message`

- Converts LangChain AIMessage to tau2 AssistantMessage
- Extracts tool calls and content

## Future Improvements

1. **Better Error Handling**: More robust error handling for tool execution failures
2. **State Synchronization**: Better synchronization between isolated and main environments
3. **Evaluation Flexibility**: Support for flexible evaluation that accounts for user choices
4. **Streaming Support**: Support for streaming responses from LangGraph
5. **Custom Prompts**: Allow custom system prompts per domain

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [tau2-bench Framework](../README.md)
- [Agent Base Classes](./base.py)
