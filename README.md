# runthrough

Evaluate deployed chat agents with simulated users.

runthrough is an extension for [Inspect AI](https://inspect.aisi.org.uk/). It holds full
conversations with your agent over HTTP, using an LLM to play the user. It records every
reply, tool call and tool result, and scores them with exact checks and an LLM judge. You
can measure how often that judge agrees with your own labels.

> **Status:** pre-alpha. The package structure is in place and the features are being built.

## What it will do

- **Talk to any deployed agent.** Adapters for the Vercel AI SDK UI-message stream,
  OpenAI-compatible chat endpoints and plain JSON over HTTP.
- **Simulate users.** Scenario files describe a persona and a goal. A simulator model plays
  that user, turn by turn, until the goal is met or a turn limit is reached.
- **Check what the agent did.** Whether a tool was called, the order of calls, the arguments,
  and whether the agent's claims match what the tools returned.
- **Judge with rubrics you can trust.** Rubric-based LLM judging, plus a labelling page and a
  report on how often the judge agrees with you.
- **Catch regressions.** Compare two runs (prompt v1 vs v2, model A vs B) scenario by scenario.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create .venv and install dev dependencies
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pyright               # type check
uv run runthrough --version
```

## License

MIT
