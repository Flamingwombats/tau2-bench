#!/bin/bash
# Script to clean all tau2/LLM related environment variables

echo "Cleaning tau2/LLM environment variables..."

# Unset all related environment variables
unset OPENAI_API_KEY
unset NEBIUS_API_KEY
unset ANTHROPIC_API_KEY
unset TAU2_LLM_AGENT
unset TAU2_LLM_USER
unset TAU2_LLM_NL_ASSERTIONS
unset TAU2_LLM_ENV_INTERFACE
unset TAU2_LLM_TEMPERATURE_AGENT
unset TAU2_LLM_TEMPERATURE_USER
unset TAU2_LLM_NL_ASSERTIONS_TEMPERATURE
unset TAU2_LLM_ENV_INTERFACE_TEMPERATURE
unset TAU2_TEST_LLM_AGENT
unset TAU2_TEST_LLM_USER
unset TAU2_TEST_LLM

# Verify they're unset
echo ""
echo "Checking remaining variables..."
remaining=$(env | grep -E "(TAU2|OPENAI|NEBIUS|ANTHROPIC)" || true)
if [ -z "$remaining" ]; then
    echo "✅ All tau2/LLM environment variables cleared!"
else
    echo "⚠️  Still set:"
    echo "$remaining"
fi

echo ""
echo "Note: This only affects the current shell session."
echo "If variables are set in ~/.zshrc, ~/.bashrc, or ~/.profile,"
echo "you'll need to remove them from those files to make it permanent."


