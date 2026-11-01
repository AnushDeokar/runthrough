# CLAUDE.md

Guidance for AI coding assistants working in this repository.

## What this is

runthrough is an open-source Inspect AI extension for evaluating **deployed** chat agents.
Inspect normally owns the model and the agent loop; runthrough instead treats the agent as a
black box behind HTTP, drives it with an LLM-simulated user, and scores the whole conversation.

## Commands

```bash
uv sync                  # install (dev group included)
uv run pytest            # tests
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## Layout

- `src/runthrough/adapters/` - `Target` protocol and one adapter per endpoint protocol.
  Each adapter turns the agent's response into an `AgentTurn`.
- `src/runthrough/scenario/` - scenario YAML schema (pydantic) and loader to Inspect `Sample`s.
- `src/runthrough/simulator/` - `simulated_conversation` solver: simulator model <-> target loop,
  appending user / assistant / tool messages to `TaskState.messages`.
- `src/runthrough/scorers/` - exact trajectory checks and the rubric judge (Inspect `grader` role).
- `src/runthrough/calibrate/` - labels, Cohen's kappa, labelling page.
- `src/runthrough/report/` - regression report between two `.eval` logs.
- `src/runthrough/testing.py` - `ScriptedTarget`, a fake agent for tests and dry runs.
- `src/runthrough/_registry.py` - Inspect entry point; import new solvers/scorers here.

## Conventions

- Map everything the agent does into standard Inspect message types
  (`ChatMessageAssistant` with `tool_calls`, `ChatMessageTool`) so `inspect view` and scorers
  work without special cases.
- The simulator model uses the Inspect model role `simulator`; the judge uses `grader`.
- Keep the core free of anything specific to one company or agent. Product-specific suites
  live in the consumer's repo, not here.
- Adapter tests run against recorded response fixtures in `tests/fixtures/`, never live endpoints.
- End-to-end solver tests run real `inspect_ai.eval()` with a `mockllm` simulator
  (`custom_outputs` callable) and `ScriptedTarget`; see `tests/test_simulator.py`.
- Python 3.11+, full type hints, ruff + pyright clean.
