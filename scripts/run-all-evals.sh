#!/bin/bash
# Run complete evaluation for all domains using LangChain agent with Nebius API

# Check if NEBIUS_API_KEY is set
if [ -z "$NEBIUS_API_KEY" ]; then
  echo "Error: NEBIUS_API_KEY environment variable is not set"
  echo "Please set it before running: export NEBIUS_API_KEY='your-actual-api-key'"
  exit 1
fi

# Build LLM args with api_key included
LLM_ARGS="{\"base_url\": \"https://api.tokenfactory.nebius.com/v1/\", \"api_key\": \"$NEBIUS_API_KEY\", \"temperature\": 0}"

# Generate timestamp for unique filenames
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Retail domain
# tau2 run \
#   --domain retail \
#   --agent langchain_agent \
#   --agent-llm openai/gpt-oss-120b \
#   --agent-llm-args "$LLM_ARGS" \
#   --user-llm openai/gpt-oss-120b \
#   --user-llm-args "$LLM_ARGS" \
#   --num-trials 1 \
#   --save-to "langchain_agent_retail_${TIMESTAMP}"

# # Airline domain
# tau2 run \
#   --domain airline \
#   --agent langchain_agent \
#   --agent-llm openai/gpt-oss-120b \
#   --agent-llm-args "$LLM_ARGS" \
#   --user-llm openai/gpt-oss-120b \
#   --user-llm-args "$LLM_ARGS" \
#   --num-trials 1 \
#   --save-to "langchain_agent_airline_${TIMESTAMP}"

# Telecom domain
tau2 run \
  --domain telecom \
  --agent langchain_agent \
  --agent-llm openai/gpt-oss-120b \
  --agent-llm-args "$LLM_ARGS" \
  --user-llm openai/gpt-oss-120b \
  --user-llm-args "$LLM_ARGS" \
  --num-trials 1 \
  --save-to "langchain_agent_telecom_${TIMESTAMP}"