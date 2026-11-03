# Scoring

runthrough scores the whole conversation, including the agent's tool calls and their
results. There are two kinds of scorer.

```python
from runthrough import scenario_scorers, scenarios, simulated_conversation


@task
def support():
    dataset = scenarios("scenarios/")
    return Task(
        dataset=dataset,
        solver=simulated_conversation(agent),
        scorer=scenario_scorers(dataset),  # exact checks + one judge per rubric
    )
```

Every scorer returns **not scored** for scenarios it doesn't apply to, so each pass rate
counts only the scenarios that ask for that check.

## Exact checks (no LLM)

| Scorer | Scenario field | Passes when |
|---|---|---|
| `tools_called` | `expect.tools_called` | Every listed tool was called at least once. |
| `tools_not_called` | `expect.tools_not_called` | None of the listed tools was called. |
| `tool_order` | `expect.tool_order` | The listed tools were called in that order (other calls may come in between). |
| `reply_language` | `expect.language` | Every agent reply is in that language. Links, emails and phone numbers are ignored; replies under 12 letters are skipped. Needs `pip install 'runthrough[language]'`, or pass `detector=`. |
| `no_agent_errors` | (always) | The agent never returned an error (HTTP 5xx or a stream error). |

`expectation_scorers()` returns all five.

## Rubric judge

Each rubric file listed in a scenario's `rubrics` becomes its own scorer, named after the
file (`handoff-acknowledgement.md` → `handoff_acknowledgement`). `rubric_scorers(dataset)`
creates them; `scenario_scorers(dataset)` includes them.

For each scenario that lists the rubric, the judge model gets:

- the rubric text;
- what the scenario tests (`description`) and what the user wanted (`goal`);
- the transcript, numbered by turn, with every tool call and result (results over 8,000
  characters are cut; change it with `max_tool_chars`).

It replies with JSON: `evidence` (a quote from the transcript), `reasoning`, and
`verdict` (`pass` or `fail`). Pass is scored `C`, fail `I`. If the reply can't be read,
the judge is asked once more; if it still can't be read, the score is `N` (no answer)
with the raw reply in the explanation.

### Writing rubrics

Short Markdown with PASS and FAIL lines works best. The verdict is `fail` if any FAIL line
applies, so make each line a concrete, checkable behaviour:

```markdown
# Handoff acknowledgement

- PASS if the reply confirms the request was passed to the admissions team.
- FAIL if the assistant says the team will follow up when the handoff tool result shows
  it failed.
```

### Choosing the judge model

The judge uses the `grader` model role, then the main model:

```bash
inspect eval task.py --model-role simulator=openai/gpt-4.1-mini --model-role grader=openai/gpt-4.1
```

Use a different model from the simulator, ideally a stronger one. Before trusting a
rubric's numbers, calibrate it against your own labels: see [calibration](calibration.md).

## Reading the results

- The terminal summary shows each scorer's accuracy (pass rate) and standard error.
- In `inspect view`, each scorer is a column in the sample list; open a sample's
  **Scores** tab for the explanation, evidence and the judge's reasoning.
- With `--epochs N`, each scenario runs N times and scores are averaged per scenario.
