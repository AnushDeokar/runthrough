# Simulated conversations

`simulated_conversation` is an Inspect solver. For each scenario it holds a conversation
between a **simulator model**, which plays the user, and the **target**, your agent.

```python
from inspect_ai import Task, task
from runthrough import scenarios, simulated_conversation


@task
def support():
    return Task(
        dataset=scenarios("scenarios/"),
        solver=simulated_conversation(target=my_agent),
    )
```

## One turn

1. The user message comes from the scenario's `opening` list while it lasts, then from the
   simulator model.
2. The target sends it to your agent. Targets with `history = "full"` receive the whole
   conversation; `"session"` targets receive only the new message.
3. The agent's reply is added to the transcript as standard Inspect messages: an assistant
   message with `tool_calls`, one tool message per result, then the reply text. `inspect view`
   and scorers read these like any other Inspect transcript.

The conversation ends when:

| `stop_reason` | When |
|---|---|
| `simulator_stopped` | The simulator replies with `###STOP###` (goal met or gave up) or an empty message. |
| `max_turns` | The scenario's `max_turns` user messages have been sent. |
| `agent_error` | The target returned a turn with `error` set (for example an HTTP 500). |

If the target raises an exception (agent unreachable, timeout), the sample fails with that
error. Inspect's `--retry-on-error` can retry it.

The outcome is stored in each sample's metadata under `runthrough`:

```json
{"conversation_id": "5f0c…", "turns": 3, "stop_reason": "simulator_stopped"}
```

`conversation_id` is new for every sample and epoch, so each run is a fresh chat for the agent.

## What the simulator sees

A system prompt built from the scenario's `persona` and `goal`, then one message with the
conversation so far as a labelled transcript:

```
Conversation so far:

You: Hi, my order arrived broken and I'd like my money back.

Assistant: Sorry to hear that! Could you give me your order number?

Write your next message.
```

Tool calls are left out, because a real user never sees them. Pass `prompt=` to use your own
system prompt; it must contain `{persona}`, `{goal}` and `{stop_token}`.

## Choosing the simulator model

The solver uses the `simulator` model role, then falls back to the eval's main model:

```bash
inspect eval task.py --model-role simulator=openai/gpt-4.1-mini
inspect eval task.py --model ollama/qwen3:8b            # one model for everything
```

Or fix it in code with `simulated_conversation(target, simulator="anthropic/claude-haiku-4-5")`.

Pick a model that follows instructions well and stays in character; small local models are
fine for building a suite but tend to be too agreeable or to stop early.

## Trying a suite without an agent

`runthrough.testing.ScriptedTarget` returns pre-written replies: reply *n* answers turn *n*.
It's useful for checking scenarios and scorers before connecting a real endpoint. See
`examples/suites/support/task.py`.
