# Testing LangChain Agent

## Quick Test Command

```bash
# Set your Nebius API key
export NEBIUS_API_KEY=your-nebius-api-key

# Run a quick test (1 task, 5 steps)
tau2 run \
  --domain retail \
  --agent langchain_agent \
  --agent-llm openai/gpt-oss-120b \
  --agent-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key": "'"$NEBIUS_API_KEY"'", "temperature": 0}' \
  --user-llm openai/gpt-oss-120b \
  --user-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key": "'"$NEBIUS_API_KEY"'", "temperature": 0}' \
  --task-set-name retail \
  --num-tasks 1 \
  --max-steps 5
```

## Using the Test Script

A test script is available at `test_langchain_agent.sh`:

```bash
# Make sure NEBIUS_API_KEY is set
export NEBIUS_API_KEY=your-key

# Run the test script
bash test_langchain_agent.sh
```

## What the LangChain Agent Does

The `langchain_agent` uses:
- **LangGraph's `create_react_agent`**: For conversational agent capabilities
- **LangChain tools**: Converts domain tools to LangChain StructuredTool format
- **ChatOpenAI**: Uses OpenAI-compatible API (works with Nebius)

## Expected Output

You should see:
1. Simulation configuration displayed
2. Agent interacting with the retail domain
3. Tool calls being made
4. Conversation between agent and user simulator
5. Final results and metrics

## Troubleshooting

### Authentication Error
- Make sure `NEBIUS_API_KEY` is set correctly
- Check that the API key is valid

### Import Errors
- Ensure all dependencies are installed: `pip install langchain langgraph langchain-openai`

### Tool Call Errors
- Verify the domain server is running (if needed)
- Check that tools are properly registered

## Full Test Run

For a more comprehensive test:

```bash
tau2 run \
  --domain retail \
  --agent langchain_agent \
  --agent-llm openai/gpt-oss-120b \
  --agent-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key": "'"$NEBIUS_API_KEY"'", "temperature": 0}' \
  --user-llm openai/gpt-oss-120b \
  --user-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key": "'"$NEBIUS_API_KEY"'", "temperature": 0}' \
  --task-set-name retail \
  --num-tasks 5 \
  --max-steps 20 \
  --save-to test_langchain_retail.json
```
