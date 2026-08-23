# Verification

## How to run

From this directory:

```bash
python -m unittest discover -s tests -t .
```

## Environment used

- Python 3.11
- `pydantic` and `pyyaml` available
- No GPU, no network, no API keys, no datasets, no model checkpoints

## Result

```
Ran 35 tests in 0.030s
OK
```

All 35 tests pass offline.

## What the tests check

| Test module | Behavior verified |
|---|---|
| `tests/test_plan.py` | Plan object construction; tool-sequence extraction; terminal-Response invariant; Response only at the end; JSON round-trip serialization; rejection of invalid JSON and unknown tool names |
| `tests/test_tools.py` | Image/text search tools run offline via stub providers; input validation; Requery and Response tools invoke the provider-independent model and report its model name |
| `tests/test_planner_executor.py` | Planner performs exactly one model call (single-pass) and parses a plan; invalid plan text is rejected; executor runs a full plan to a terminal Response; missing-tool and invalid-plan error handling; end-to-end plan-then-execute via `EAgent` |
| `tests/test_data_and_evaluation.py` | RemPlan instance schema validation; per-tool precision/recall; Plan-acc exact-match; Param-acc and Param-sim raise without an injected criterion and compute with one; answer-judge 0–2 scale |
| `tests/test_config_and_training.py` | Development config loads (records `benchmark_size: 200`, `planner_train_samples: 10000`); planner/executor models build from the config via the Step 5 layer; training scaffold reports paper values; `train()` raises `NotImplementedError` |

## Independence from unavailable assets

The tests exercise only: the plan representation, the planner/executor control flow, tool dispatch, the model-layer integration through the `stub` provider (plus a documented `ScriptedVisionLanguageModel` development double), and the data/evaluation/training interfaces. They do not touch InternVL2-8B, Qwen2-VL-72B, Baidu Image Search, Tavily, the RemPlan benchmark, or the 10K training set.

## Comment-policy check

A scan for `#` across `*.py` and `*.yaml` in this directory returns no matches: generated source and configuration contain no comments. Paper references live in `PAPER_SPEC.md`, `REPRODUCTION_NOTES.md`, and this file.

## Reproduction status

**NOT REPRODUCED.** No paper experiment was executed; no paper metric was computed against paper data. Passing tests verify software behavior of the baseline scaffold only.
