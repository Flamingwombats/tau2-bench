# Agent Module Documentation

## Overview

The `agent` module provides the core abstraction and implementations for conversational agents in the τ²-bench framework. Agents are responsible for generating responses to user messages and performing actions through tool calls in domain-specific environments.

## Architecture

### Class Hierarchy

```
BaseAgent (ABC)
    └── LocalAgent
            ├── LLMAgent
            ├── LLMGTAgent (Ground Truth)
            └── LLMSoloAgent
```

### Core Components

1. **BaseAgent**: Abstract base class defining the agent interface
2. **LocalAgent**: Base implementation for local agents with tools and domain policy
3. **LLMAgent**: Standard LLM-powered conversational agent
4. **LLMGTAgent**: Ground truth agent with oracle action guidance
5. **LLMSoloAgent**: Solo agent that works without user interaction

---

## BaseAgent Interface

### Purpose
`BaseAgent` defines the contract that all agents must implement. It uses Python's `ABC` (Abstract Base Class) to enforce the interface.

### Key Methods

#### `generate_next_message(message, state) -> tuple[AssistantMessage, AgentState]`
**Purpose**: Generate the next assistant message based on user/tool input and current state.

**Parameters**:
- `message`: A `ValidAgentInputMessage` (UserMessage, ToolMessage, or MultiToolMessage)
- `state`: The agent's current state (type: `AgentState`)

**Returns**: Tuple of:
- `AssistantMessage`: The generated response
- `AgentState`: Updated agent state

**Behavior**: This is the core method that processes input and generates responses. The agent can either:
- Send a text message to the user
- Make tool calls to interact with the environment
- **Cannot do both simultaneously** (enforced by validation)

#### `get_init_state(message_history) -> AgentState`
**Purpose**: Initialize the agent's state, optionally from a message history.

**Parameters**:
- `message_history`: Optional list of previous messages (for resuming conversations)

**Returns**: Initial `AgentState` object

**Use Case**: Enables rerunning agents from any point in a conversation by reconstructing state.

#### `stop(message, state) -> None`
**Purpose**: Clean up resources when stopping the agent.

**Parameters**:
- `message`: Optional last message received
- `state`: Optional final state

**Note**: Most implementations are no-ops, but can be overridden for cleanup.

#### `is_stop(message) -> bool` (class method)
**Purpose**: Determine if an assistant message indicates the agent should stop.

**Default**: Returns `False` (agents don't stop by default)

**Override**: `LLMSoloAgent` overrides this to detect stop tokens.

### Message Validation

#### `is_valid_agent_history_message(message) -> bool`
Validates that a message can be part of agent history:
- ✅ `AssistantMessage` (agent responses)
- ✅ `UserMessage` (non-tool-call user messages)
- ✅ `ToolMessage` (tool responses to agent requests)
- ❌ Other message types

#### `validate_message_format(message, solo=False) -> tuple[bool, str]`
Validates assistant message format:

**Default Mode** (`solo=False`):
- Must have **either** text content **or** tool calls
- **Cannot** have both
- **Cannot** be empty

**Solo Mode** (`solo=True`):
- Must have tool calls
- **Cannot** have text content
- Used for `LLMSoloAgent`

---

## LocalAgent

### Purpose
Base implementation for agents that run locally with access to domain tools and policies.

### Initialization

```python
LocalAgent(tools: list[Tool], domain_policy: str)
```

**Parameters**:
- `tools`: List of available tools (domain API functions)
- `domain_policy`: Domain-specific policy document (rules and constraints)

### Key Features
- Provides common infrastructure for tool-based agents
- Validates message formats
- Manages domain policy access

---

## LLMAgent

### Purpose
Standard conversational agent powered by an LLM. This is the primary agent implementation for most use cases.

### Architecture

```
┌─────────────────────────────────────────┐
│         LLMAgent                        │
├─────────────────────────────────────────┤
│ • System Prompt (policy + instructions) │
│ • Message History                        │
│ • Tool Definitions                       │
│ • LLM Configuration                      │
└─────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │  LLM (via     │
    │  generate())  │
    └───────────────┘
```

### Initialization

```python
LLMAgent(
    tools: List[Tool],
    domain_policy: str,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None
)
```

**Parameters**:
- `tools`: Available domain tools
- `domain_policy`: Domain policy document
- `llm`: LLM model name (e.g., `"gpt-4"`, `"openai/gpt-oss-120b"`)
- `llm_args`: Additional LLM parameters (temperature, seed, etc.)

### System Prompt Structure

The agent uses a structured system prompt:

```
<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
</instructions>
<policy>
{domain_policy}
</policy>
```

### State Management

**LLMAgentState**:
```python
class LLMAgentState(BaseModel):
    system_messages: list[SystemMessage]  # System prompt
    messages: list[APICompatibleMessage]   # Conversation history
```

### Message Generation Flow

1. **Receive Input**: User message or tool response(s)
2. **Update State**: Append input to message history
3. **Build Context**: Combine system messages + conversation history
4. **Generate**: Call LLM with context and tools
5. **Update State**: Append assistant response to history
6. **Return**: Assistant message + updated state

### Example Usage

```python
from tau2.agent.llm_agent import LLMAgent
from tau2.environment.tool import Tool

# Initialize agent
agent = LLMAgent(
    tools=[tool1, tool2, tool3],
    domain_policy=policy_text,
    llm="gpt-4",
    llm_args={"temperature": 0.7}
)

# Get initial state
state = agent.get_init_state()

# Generate response to user message
user_msg = UserMessage(role="user", content="I need help with my order")
assistant_msg, state = agent.generate_next_message(user_msg, state)

# Generate response to tool result
tool_msg = ToolMessage(...)
assistant_msg, state = agent.generate_next_message(tool_msg, state)
```

---

## LLMGTAgent (Ground Truth Agent)

### Purpose
An agent that receives oracle action guidance (ground truth steps) to test user simulator correctness. Used for ablation studies and evaluation.

### Key Differences from LLMAgent

1. **Oracle Guidance**: Receives expected action sequence from task
2. **Testing Focus**: Designed to verify user simulator behavior
3. **Action Formatting**: Can include or exclude function arguments in guidance

### Initialization

```python
LLMGTAgent(
    tools: List[Tool],
    domain_policy: str,
    task: Task,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    provide_function_args: bool = True
)
```

**Parameters**:
- `task`: Task object containing evaluation criteria with expected actions
- `provide_function_args`: Whether to include function arguments in action guidance

### System Prompt Structure

```
<instructions>
You are testing that our user simulator is working correctly.
User simulator will have an issue for you to solve.
You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
...
</instructions>
<policy>
{domain_policy}
</policy>
<resolution_steps>
[Step 1] Perform the following action: {action1}
[Step 2] User action: {action2}
...
</resolution_steps>
```

### Action Formatting

Actions are formatted as numbered steps:
- **User Actions**: `"User action: {action_name}"` or `"Instruct the user to perform: {action.get_func_format()}"`
- **Agent Actions**: `"Assistant action: {action_name}"` or `"Perform the following action: {action.get_func_format()}"`

### Task Validation

Only tasks with:
- Non-empty evaluation criteria
- At least one expected action

are valid for `LLMGTAgent`.

### Use Case

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_gt \
  --agent-llm gpt-4 \
  --user-llm gpt-4 \
  ...
```

---

## LLMSoloAgent

### Purpose
An agent that operates without user interaction. Receives a ticket upfront and must solve it entirely through tool calls. Used for ablation studies (no-user mode).

### Key Characteristics

1. **No User Messages**: Cannot receive or send user messages
2. **Tool Calls Only**: Must make tool calls (no text responses)
3. **Stop Mechanism**: Uses a special `done()` tool to signal completion
4. **Ticket-Based**: Receives full problem description in system prompt

### Initialization

```python
LLMSoloAgent(
    tools: List[Tool],
    domain_policy: str,
    task: Task,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None
)
```

**Parameters**:
- `task`: Must contain a `ticket` field with problem description

### System Prompt Structure

```
<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `done` tool.
...
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket_content}
</ticket>
```

### Stop Mechanism

1. **Stop Tool**: Automatically adds a `done()` tool to available tools
2. **Stop Detection**: When agent calls `done()`, message content is replaced with `"###STOP###"`
3. **Validation**: `is_stop()` checks for stop token in message content

### Tool Requirements

- **Required**: `done` tool (automatically added)
- **Recommended**: `transfer_to_human_agents` tool (for escalation)

### Task Validation

Tasks must have:
- A `ticket` field with problem description
- Evaluation criteria with expected actions
- If `initial_state` exists, message history must only contain tool calls/responses (no user messages)

### Message Generation

- **Tool Choice**: Always uses `tool_choice="required"` to force tool calls
- **Validation**: Raises error if LLM generates text instead of tool calls
- **User Messages**: Raises error if user message is received

### Use Case

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_solo \
  --user dummy_user \
  ...
```

---

## Message Types

### ValidAgentInputMessage
Union type for messages agents can receive:
- `UserMessage`: User text messages (non-tool-call)
- `ToolMessage`: Tool execution results
- `MultiToolMessage`: Multiple tool results bundled together

### AssistantMessage
Agent responses can contain:
- **Text Content**: Message to send to user
- **Tool Calls**: Actions to perform in environment
- **Not Both**: Validation enforces mutual exclusivity

---

## State Management

### AgentState (TypeVar)
Generic type parameter allowing agents to define custom state types.

### LLMAgentState
Standard state for LLM-based agents:
- `system_messages`: System prompts (typically one)
- `messages`: Full conversation history

### State Persistence
The `get_init_state()` method enables:
- **Resumability**: Restart agents from any conversation point
- **Reproducibility**: Recreate exact agent state
- **Debugging**: Inspect agent state at any turn

---

## Error Handling

### AgentError
Base exception for agent-related errors.

### Common Validation Errors

1. **Empty Message**: Message has neither text nor tool calls
2. **Both Content Types**: Message has both text and tool calls
3. **Invalid History**: Message history contains invalid message types
4. **Invalid Task**: Task doesn't meet agent requirements
5. **Missing Tools**: Required tools not found (e.g., `done` for SoloAgent)

---

## Integration with Framework

### Registration

Agents are registered in `src/tau2/registry.py`:

```python
from tau2.registry import registry
from tau2.agent.llm_agent import LLMAgent

# Agents are auto-registered, but you can register custom agents:
registry.register_agent(MyCustomAgent, "my_custom_agent")
```

### Usage in Orchestrator

The orchestrator uses agents via the `BaseAgent` interface:

1. **Initialize**: `agent.get_init_state(message_history)`
2. **Generate**: `agent.generate_next_message(message, state)`
3. **Check Stop**: `agent.is_stop(assistant_message)`
4. **Cleanup**: `agent.stop(message, state)` (on completion)

### CLI Integration

Agents are selected via command-line:

```bash
tau2 run --agent llm_agent --agent-llm gpt-4 ...
```

---

## Best Practices

### 1. Message Format Validation
Always validate assistant messages before sending:
```python
is_valid, error_msg = validate_message_format(assistant_message)
if not is_valid:
    raise ValueError(error_msg)
```

### 2. State Immutability
Treat agent state as immutable. Create new state objects rather than mutating:
```python
# Good: Create new state
new_state = LLMAgentState(
    system_messages=state.system_messages,
    messages=state.messages + [new_message]
)

# Bad: Mutate existing state
state.messages.append(new_message)  # Avoid
```

### 3. Tool Definition
Ensure tools are properly defined with:
- Clear names and descriptions
- Correct parameter schemas
- Appropriate return types

### 4. Policy Clarity
Domain policies should be:
- Clear and unambiguous
- Include all constraints
- Specify required behaviors

### 5. Error Handling
Implement robust error handling:
- Validate inputs
- Handle LLM failures gracefully
- Provide clear error messages

---

## Extending the Agent System

### Creating Custom Agents

1. **Inherit from LocalAgent**:
```python
from tau2.agent.base import LocalAgent, AgentState

class MyCustomAgent(LocalAgent[MyCustomState]):
    def __init__(self, tools, domain_policy, **kwargs):
        super().__init__(tools, domain_policy)
        # Custom initialization
    
    def get_init_state(self, message_history=None):
        # Return initial state
    
    def generate_next_message(self, message, state):
        # Generate response
        return assistant_message, updated_state
```

2. **Register Your Agent**:
```python
from tau2.registry import registry
registry.register_agent(MyCustomAgent, "my_custom_agent")
```

3. **Use Your Agent**:
```bash
tau2 run --agent my_custom_agent ...
```

### Custom State Types

Define custom state by creating a Pydantic model:

```python
from pydantic import BaseModel

class MyCustomState(BaseModel):
    messages: list[Message]
    custom_field: str
    metadata: dict
```

---

## Testing

### Unit Testing

Test individual agent methods:
```python
def test_agent_initialization():
    agent = LLMAgent(tools=[], domain_policy="...", llm="gpt-4")
    state = agent.get_init_state()
    assert state.system_messages is not None

def test_message_generation():
    agent = LLMAgent(...)
    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Hello")
    assistant_msg, new_state = agent.generate_next_message(user_msg, state)
    assert assistant_msg is not None
```

### Integration Testing

Test agents with full orchestrator:
```python
from tau2.run import run_simulation

results = run_simulation(
    domain="airline",
    agent="llm_agent",
    agent_llm="gpt-4",
    ...
)
```

---

## Troubleshooting

### Common Issues

1. **"Empty message" error**
   - **Cause**: Agent generated message with no content or tool calls
   - **Fix**: Check LLM output, ensure proper prompt formatting

2. **"Both text and tool calls" error**
   - **Cause**: Agent generated message with both text and tools
   - **Fix**: Validate message format, check LLM configuration

3. **"Invalid message history" error**
   - **Cause**: Message history contains invalid message types
   - **Fix**: Filter history using `is_valid_agent_history_message()`

4. **Tool not found errors**
   - **Cause**: Required tool missing from tools list
   - **Fix**: Ensure all required tools are provided (e.g., `done` for SoloAgent)

5. **LLM authentication errors**
   - **Cause**: Invalid API key or configuration
   - **Fix**: Check environment variables, verify API key format

---

## Summary

The agent module provides a flexible, extensible framework for building conversational agents:

- **BaseAgent**: Abstract interface for all agents
- **LLMAgent**: Standard conversational agent
- **LLMGTAgent**: Ground truth agent with oracle guidance
- **LLMSoloAgent**: Solo agent without user interaction

All agents follow the same interface, enabling easy swapping and testing. The system supports custom agents, state management, and robust error handling.

For more information, see:
- `README.md` in this directory for developer guide
- `src/tau2/orchestrator/` for how agents are orchestrated
- `src/tau2/run.py` for agent execution
- `src/tau2/registry.py` for agent registration

