# Scenario files

A scenario describes one simulated user and what the agent should do for them. Write them in
YAML: one scenario per file, or a list of scenarios in one file. Point runthrough at files or
directories; directories are searched recursively for `.yaml` and `.yml` files.

```yaml
id: refund-damaged-item
description: A clear refund request should be looked up, then refunded.
persona: Priya, a regular customer. Polite but busy.
goal: Get a refund for a mug that arrived broken (order 48213).
opening: "Hi, my order arrived broken and I'd like my money back."
max_turns: 6
context:
  locale: en-GB
expect:
  tools_called: [lookup_order, issue_refund]
  tool_order: [lookup_order, issue_refund]
  tools_not_called: [request_human_handoff]
  language: en
rubrics: [../rubrics/tone.md]
tags: [refunds]
```

## Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique within the suite. Letters, digits, `_`, `-`, `.`. |
| `persona` | yes | Who the simulated user is and how they behave. |
| `goal` | yes | What the simulated user wants to get done. |
| `description` | no | What the scenario tests, for people reading the suite. |
| `opening` | no | Exact user message(s) to send first, before the simulator takes over. A string or a list. |
| `max_turns` | no | Maximum user messages, 1 to 50. Default 8. |
| `context` | no | Values passed to the target adapter on every turn (profile data, prompt keys, locale). |
| `expect.tools_called` | no | Tools that must be called at least once. |
| `expect.tools_not_called` | no | Tools that must never be called. |
| `expect.tool_order` | no | Tools that must be called in this relative order. |
| `expect.language` | no | ISO 639-1 code every agent reply must use. |
| `rubrics` | no | Rubric files for the LLM judge, relative to the scenario file. |
| `tags` | no | Labels for filtering, e.g. `scenarios("suite/", tags=["handoff"])`. |

Unknown fields are rejected, so a typo like `max_turn` fails loudly instead of being ignored.

## Check a suite

```bash
runthrough validate examples/suites/support/scenarios
```

All problems in all files are reported together, with the file and field for each.

## Use in an Inspect task

```python
from inspect_ai import Task, task
from runthrough import scenarios


@task
def support():
    return Task(dataset=scenarios("suites/support/scenarios", tags=["refunds"]))
```

Each scenario becomes one Inspect sample whose id is the scenario id.
