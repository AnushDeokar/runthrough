# Judge calibration

An LLM judge is only useful if it agrees with you. Calibration measures that: you label a
sample of judged conversations yourself, and runthrough reports how often the judge
matches, per rubric.

## 1. Run a suite with rubric scorers

```bash
inspect eval task.py --model-role grader=openai/gpt-4.1 --epochs 3
```

## 2. Label conversations

```bash
runthrough label logs/                 # opens a local page; labels go to labels.jsonl
runthrough label logs/ --scorer tone   # one rubric at a time
```

The page shows one conversation at a time: the transcript with tool calls and results on
the left, and on the right one row per rubric the judge scored it on. A scenario with three
rubrics gives one conversation with three rows to label, so the header counts both:
`Conversation 2 of 3 · 4 of 5 verdicts labelled`. The rubric dropdown filters to the
conversations (and rows) of one rubric, with the number of verdicts for each.

For each row, read the rubric and press **Pass**, **Fail** or **Skip** (keys `P`, `F`, `S`
act on the highlighted row, and the next unlabelled row is highlighted after you label).
Add a note to explain a tricky call.

The judge's verdict on a row stays hidden until you've labelled it, so it doesn't sway you;
press `J` to show it anyway. Use `↑` / `↓` to move between rubrics, `←` / `→` between
conversations, and `U` to jump to the next unlabelled verdict.

The page runs on `127.0.0.1` only. Labels are appended to `labels.jsonl` (JSON Lines); if
you change a label, the latest line wins. Commit the file next to your suite so labels
build up over time.

## 3. Read the agreement report

```bash
runthrough calibrate logs/
```

```
rubric                    labelled  agree  kappa  strength     judge ✓ you ✗  judge ✗ you ✓
grounded_in_retrieval        34/40    88%   0.74  substantial              3              1
handoff_acknowledgement      30/30    97%   0.93  almost perfect           0              1
tone                         31/38    71%   0.31  fair                     8              1
```

- **labelled**: conversations you labelled pass or fail, out of those the judge scored.
  Skips and unlabelled items are left out.
- **agree**: share of labelled conversations where you and the judge gave the same verdict.
- **kappa**: Cohen's kappa, which corrects agreement for chance. With mostly-pass data a
  judge can agree 90% of the time just by always saying pass; kappa shows that. **strength**
  uses the usual Landis & Koch bands (0.61–0.80 substantial, above 0.80 almost perfect).
- **judge ✓ you ✗ / judge ✗ you ✓**: which way the judge errs. A lenient judge has a high
  first column; a strict one, the second.

Below the table, each rubric's disagreements are listed with the judge's reasoning and your
note. Use `--json` for machine-readable output.

## 4. Improve weak rubrics

When kappa is low, read the disagreements and rewrite the rubric: make each PASS/FAIL line
a concrete behaviour, and add a FAIL line for the mistake the judge keeps missing. Then run
the suite again and calibrate on the new log. Labels are tied to a specific conversation
(its sample id in the log), so a new run needs a fresh round of labelling.

A rough guide: aim for kappa above 0.6 before using a rubric's pass rate to make decisions,
and label at least 30 conversations per rubric, mixing passes and fails.
