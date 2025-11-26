#!/bin/bash
# Test script for langchain_agent

export NEBIUS_API_KEY=${NEBIUS_API_KEY:-"your-nebius-api-key-here"}

tau2 run \
  --domain retail \
  --agent langchain_agent \
  --agent-llm openai/gpt-oss-120b \
  --agent-llm-args "{\"base_url\": \"https://api.tokenfactory.nebius.com/v1/\", \"api_key\": \"$NEBIUS_API_KEY\", \"temperature\": 0}" \
  --user-llm openai/gpt-oss-120b \
  --user-llm-args "{\"base_url\": \"https://api.tokenfactory.nebius.com/v1/\", \"api_key\": \"$NEBIUS_API_KEY\", \"temperature\": 0}" \
  --task-set-name retail \
  --num-tasks 1 \
  --max-steps 20

