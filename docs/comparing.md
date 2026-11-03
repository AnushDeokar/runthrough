# Comparing runs

`runthrough compare` puts several runs of the same suite side by side: pass rate per
scorer, reply latency, tokens and cost. Use it to compare models, prompt versions or any
other change to the agent.

```bash
runthrough compare logs/ --prices prices.yaml
runthrough compare logs/ --prices prices.yaml --json > results.json
```

## Runs and labels

Logs are grouped into runs by label, taken from (in order):

1. the task's `metadata={"runthrough_label": "..."}`,
2. the task arguments (`-T model=x` becomes `model=x`),
3. the log file name.

Logs with the same label are merged, so you can add epochs to a run later. If the same
sample and epoch appear in several logs (after `inspect eval-retry`, for example), it is
counted once: the copy without an error, otherwise the one from the newest log.

## What's reported

| Field | From |
|---|---|
| Pass rate per scorer | Every scorer's verdicts across conversations and epochs (not-scored samples excluded) |
| Overall | Mean of all applicable verdicts |
| Latency p50 / p95 | `latency_ms` of each agent turn, measured by the adapter |
| Tokens, cost | Usage the agent reports in its reply metadata (below) |
| Per-scenario pass rate | In `--json`, for finding where runs differ |

## Token usage and cost

runthrough can't see the agent's model calls, so the agent has to report usage. With the
AI SDK adapter, put it in the UI-message stream's message metadata, for example:

```ts
result.pipeUIMessageStreamToResponse(res, {
  messageMetadata: ({ part }) =>
    part.type === "finish" ? { model: "gpt-x", usage: part.totalUsage } : undefined,
});
```

`usage` uses the AI SDK names (`inputTokens`, `cachedInputTokens`, `outputTokens`,
`reasoningTokens`). Prices are a YAML or JSON file, USD per 1M tokens, keyed by the reported
model (or the run label):

```yaml
gpt-x: {input: 0.10, cached_input: 0.01, output: 0.50}
```

Cost = uncached input × input + cached input × cached_input + output × output. Output
tokens already include reasoning tokens.
