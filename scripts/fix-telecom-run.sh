#!/bin/bash
# Fix telecom run by rerunning missing tasks and merging results

set -e

OLD_JSON="data/simulations/langchain_agent_telecom_2025-11-26_01-17-46.json"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
NEW_JSON="data/simulations/langchain_agent_telecom_${TIMESTAMP}.json"
MERGED_JSON="data/simulations/langchain_agent_telecom_merged_${TIMESTAMP}.json"

# Check if NEBIUS_API_KEY is set
if [ -z "$NEBIUS_API_KEY" ]; then
  echo "Error: NEBIUS_API_KEY environment variable is not set"
  echo "Please set it before running: export NEBIUS_API_KEY='your-actual-api-key'"
  exit 1
fi

# Build LLM args with api_key included
LLM_ARGS="{\"base_url\": \"https://api.tokenfactory.nebius.com/v1/\", \"api_key\": \"$NEBIUS_API_KEY\", \"temperature\": 0}"

# Extract task IDs that are missing simulations and write to temp file
echo "Extracting task IDs missing simulations from ${OLD_JSON}..."
TASK_IDS_FILE=$(mktemp)
python3 << 'PYTHON_SCRIPT'
import json
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)
    
    # Get task IDs that have simulations
    simulations = data.get('simulations', [])
    tasks_with_sims = {str(sim.get('task_id')) for sim in simulations}
    
    # Find missing task IDs
    missing_task_ids = []
    for task in data['tasks']:
        task_id = str(task['id'])  # Convert to string for comparison
        if task_id not in tasks_with_sims:
            missing_task_ids.append(task['id'])  # Keep original format
    
    # Write to file, one per line
    with open(sys.argv[2], 'w') as out:
        for tid in missing_task_ids:
            out.write(f"{tid}\n")
    
    print(f"Found {len(missing_task_ids)} tasks missing simulations")
PYTHON_SCRIPT
"${OLD_JSON}" "${TASK_IDS_FILE}"

TASK_COUNT=$(wc -l < "${TASK_IDS_FILE}" | tr -d ' ')
if [ "$TASK_COUNT" -eq 0 ]; then
  echo "No tasks missing simulations. All tasks have been completed."
  rm -f "${TASK_IDS_FILE}"
  exit 0
fi

# Read task IDs into array
mapfile -t TASK_IDS_ARRAY < "${TASK_IDS_FILE}"
rm -f "${TASK_IDS_FILE}"

# Run tau2 with specific task IDs
echo "Running tau2 for ${TASK_COUNT} tasks..."
tau2 run \
  --domain telecom \
  --agent langchain_agent \
  --agent-llm openai/gpt-oss-120b \
  --agent-llm-args "$LLM_ARGS" \
  --user-llm openai/gpt-oss-120b \
  --user-llm-args "$LLM_ARGS" \
  --num-trials 1 \
  --task-ids "${TASK_IDS_ARRAY[@]}" \
  --save-to "langchain_agent_telecom_${TIMESTAMP}"

# Wait for the file to be created
sleep 2

# Check if new JSON exists
if [ ! -f "${NEW_JSON}" ]; then
  echo "Error: New JSON file not found: ${NEW_JSON}"
  exit 1
fi

# Merge old and new JSON files
echo "Merging old and new results..."
python3 << 'PYTHON_SCRIPT'
import json
import sys

old_file = sys.argv[1]
new_file = sys.argv[2]
merged_file = sys.argv[3]

# Load old JSON
with open(old_file, 'r') as f:
    old_data = json.load(f)

# Load new JSON
with open(new_file, 'r') as f:
    new_data = json.load(f)

# Get existing simulations from old file (keep any that were completed)
old_simulations = old_data.get('simulations', [])
old_sim_task_ids = {sim.get('task_id') for sim in old_simulations}

# Get new simulations
new_simulations = new_data.get('simulations', [])
new_sim_task_ids = {sim.get('task_id') for sim in new_simulations}

# Combine simulations: keep old ones, add new ones
# If a task_id exists in both, prefer the new one
merged_simulations = []
seen_task_ids = set()

# First add old simulations (for tasks that weren't rerun)
for sim in old_simulations:
    task_id = sim.get('task_id')
    if task_id not in new_sim_task_ids:
        merged_simulations.append(sim)
        seen_task_ids.add(task_id)

# Then add all new simulations
for sim in new_simulations:
    task_id = sim.get('task_id')
    merged_simulations.append(sim)
    seen_task_ids.add(task_id)

# Merge tasks (use new tasks, they should be the same structure)
merged_tasks = new_data.get('tasks', old_data.get('tasks', []))

# Create merged data structure
merged_data = {
    "timestamp": new_data.get("timestamp", old_data.get("timestamp")),
    "info": new_data.get("info", old_data.get("info")),
    "tasks": merged_tasks,
    "simulations": merged_simulations
}

# Write merged JSON
with open(merged_file, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"Merged {len(merged_tasks)} tasks")
print(f"Merged {len(merged_simulations)} simulations")
print(f"  Old simulations kept: {len(old_simulations) - len([s for s in old_simulations if s.get('task_id') in new_sim_task_ids])}")
print(f"  New simulations added: {len(new_simulations)}")
print(f"Output: {merged_file}")
PYTHON_SCRIPT
"${OLD_JSON}" "${NEW_JSON}" "${MERGED_JSON}"

echo ""
echo "✓ Successfully merged results!"
echo "  Merged file: ${MERGED_JSON}"
echo ""
echo "You can now use this merged file for submission:"
echo "  tau2 submit prepare ${MERGED_JSON} --output ./langchain_agent_submission"

