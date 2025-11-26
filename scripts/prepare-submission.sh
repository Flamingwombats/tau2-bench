#!/bin/bash
# Prepare submission from langchain_agent simulation results

SIMULATIONS_DIR="data/simulations"
OUTPUT_DIR="./langchain_agent_submission"

# Prepare submission with the 3 exact files
tau2 submit prepare \
  ${SIMULATIONS_DIR}/langchain_agent_airline_2025-11-26_00-37-06.json \
  ${SIMULATIONS_DIR}/langchain_agent_retail_2025-11-26_00-37-06.json \
  ${SIMULATIONS_DIR}/langchain_agent_telecom_2025-11-26_01-17-46.json \
  --output ${OUTPUT_DIR}

