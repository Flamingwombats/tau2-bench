# Why LangChain Agent Simulation Has No Rewards/Metrics

## Issue Summary

The simulation results show:
- `reward: 0.0`
- All checks are `null` (db_check, action_checks, etc.)
- `termination_reason: "max_steps"`
- Note: "Simulation terminated prematurely. Termination reason: max_steps"

## Root Cause

The simulation was run with `--max-steps 5`, which cut off the conversation before the task could be completed. 

### How Rewards Are Computed

Looking at `src/tau2/evaluator/evaluator.py` (lines 31-41):

```python
if simulation.termination_reason not in {
    TerminationReason.AGENT_STOP,
    TerminationReason.USER_STOP,
}:
    return RewardInfo(
        reward=0.0,
        reward_basis=None,
        info={
            "note": f"Simulation terminated prematurely. Termination reason: {simulation.termination_reason.value}"
        },
    )
```

**Key Point**: Rewards are only computed if the simulation terminates with:
- `AGENT_STOP` - Agent completed the task and stopped
- `USER_STOP` - User completed the task and stopped

If the simulation terminates for any other reason (like `max_steps`), it returns `reward=0.0` with no actual evaluation.

### What Happened in Your Simulation

Looking at the conversation flow:
1. ✅ Agent greeted user
2. ✅ User explained the exchange request
3. ✅ Agent found user ID and order details
4. ✅ Agent got product details for both items
5. ⚠️ Agent explained the exchange but **didn't make the tool call** (`exchange_delivered_order_items`)
6. ⚠️ User confirmed but asked follow-up questions
7. ❌ Simulation cut off at step 5 (max_steps reached)

The agent was making progress but needed more steps to:
- Actually call `exchange_delivered_order_items` tool
- Get user confirmation
- Complete the task

## Solution

Run the simulation with more steps to allow the task to complete:

```bash
tau2 run \
  --domain retail \
  --agent langchain_agent \
  --agent-llm openai/gpt-oss-120b \
  --agent-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key": "'"$NEBIUS_API_KEY"'", "temperature": 0}' \
  --user-llm openai/gpt-oss-120b \
  --user-llm-args '{"base_url": "https://api.tokenfactory.nebius.com/v1/", "api_key": "'"$NEBIUS_API_KEY"'", "temperature": 0}' \
  --task-set-name retail \
  --num-tasks 1 \
  --max-steps 20  # Increased from 5 to 20
```

## Expected Behavior

When the simulation completes successfully:
- `termination_reason` will be `AGENT_STOP` or `USER_STOP`
- `reward_info` will contain:
  - `reward`: Actual reward value (0.0 to 1.0)
  - `db_check`: Database state check results
  - `action_checks`: Action execution checks
  - `communicate_checks`: Communication checks
  - `reward_breakdown`: Detailed breakdown of reward components

## Metrics

Metrics are computed from completed simulations:
- `avg_reward`: Average reward across all simulations
- `pass_hat_k`: Pass rate metrics for different k values
- `avg_agent_cost`: Average cost per simulation

These metrics require simulations that complete successfully (not terminated early).

## Recommendation

For testing, use:
- `--max-steps 20` or higher (default is usually 100-200)
- This gives the agent enough turns to complete the task
- The agent will naturally stop when the task is complete

The langchain_agent implementation is correct - it just needs more steps to complete the task!

