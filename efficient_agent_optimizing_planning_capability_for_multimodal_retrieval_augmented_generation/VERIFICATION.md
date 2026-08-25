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
Ran 77 tests in 0.122s
OK
```

All 77 tests pass offline.

The repo-root model layer (`python -m unittest discover -s tests` from the repository root) has **7** tests: stub planner/executor construction, multimodal stub requests, independent instances, and `UnsupportedProviderError` for non-`stub` providers. HuggingFace / CPU VLM code is not in that suite.

A separate functional-extension suite (8 mocked tests, not paper fidelity) lives in `functional_extension/` and is documented in `functional_extension/VERIFICATION.md`.

## End-to-end execution

The smallest faithful E-Agent path was executed with the deterministic development components (scripted planner + `stub` executor model + offline stub search providers), with no GPU, network, API keys, or RemPlan data:

```
multimodal input -> MRAGPlanner (1 call) -> MRAGPlan
                 -> TaskExecutor -> image_search -> requery -> text_search -> response
                 -> terminal Response -> final response
```

Observed: `plan_call_count = 1`; execution trace `['image_search', 'requery', 'text_search', 'response']`; a non-empty terminal response string. This path is pinned by `tests/test_end_to_end_smoke.py`.

## Strict-fidelity audit rule

A **PASS** is granted only when the implementation demonstrably matches the paper requirement and the evidence is sufficient to establish paper fidelity. A **UNVERIFIED** is used whenever the code works but the paper requirement cannot be established as faithfully implemented because the real model/service is unavailable, the paper leaves the detail unspecified, only a stub/development substitute exists, the metric formula is an implementation assumption, the actual dataset is unavailable, or the test verifies execution rather than paper fidelity. A passing local test is NOT sufficient evidence for PASS when the implementation uses a stub or an assumption.

PASS entries in the matrix below verify the offline architecture/interface contract only — planner/executor separation, single-pass planner interface, structured plan object, terminal-Response invariant, tool-dispatch architecture, deterministic offline execution, and development testability. They do not assert fidelity to the paper's trained models (InternVL2-8B, Qwen2-VL-72B), live services (Baidu, Tavily), the RemPlan or 10K datasets, undisclosed prompts, or the reported metrics; those are all UNVERIFIED.

## Requirement-by-requirement matrix

| # | Requirement | Paper reference | Code location | Test(s) that verify | Status | Evidence / Why |
|---|-------------|----------------|---------------|----------------------|--------|----------------|
| 1 | **E-Agent architecture** (planner + executor) | §4, Fig. 4 | `eagent_baseline/agent.py`, `eagent_baseline/planner.py`, `eagent_baseline/executor.py` | `tests/test_planner_executor.py` (plan -> executor flow) | **PASS** | The code implements two distinct modules that correspond exactly to the "mRAG planner" and "Task Executor" described in the paper. |
| 2 | **Planner / Executor separation** | §4 | Same files as #1 | `tests/test_planner_executor.py` | **PASS** | Separate classes (`MRAGPlanner`, `TaskExecutor`) with a clear interface, matching the paper's modular description. |
| 3 | **Single-pass planning** | §4.1 (single forward pass) | `eagent_baseline/planner.py` (`plan_call_count`) | `tests/test_planner_executor.py` (asserts `plan_call_count == 1`) | **PASS** | The planner calls the model exactly once per question, as required. |
| 4 | **Planner inputs (text + image)** | §4.1 | `eagent_baseline/planner.py` (`build_request`) | `tests/test_planner_executor.py` (synthetic `Question` with images) | **PASS** | The request includes both the question text and the image list, matching the paper's multimodal input. |
| 5 | **Dynamic tool selection** | §4.1 (planner decides which tools to use) | `eagent_baseline/planner.py` (stub model) | *None* - the stub model does not demonstrate context-dependent selection. | **UNVERIFIED** | No evidence that the planner's output varies with the multimodal context; only a deterministic stub is used. |
| 6 | **Structured mRAG plan** | §3.3, Fig. 4 | `eagent_baseline/plan.py` (`MRAGPlan`) | `tests/test_plan.py` (JSON round-trip) | **PASS** | The paper specifies an ordered, structured sequence of tool/MLLM actions (Fig. 4, §3.3). The repository uses JSON as an implementation-level serialization choice for that sequence; the paper does not specify JSON as the canonical serialization format. |
| 7 | **Plan step representation** | §3.3 | `eagent_baseline/plan.py` (`PlanStep`) | `tests/test_plan.py` (step parsing) | **PASS** | Each step stores a tool name and arguments, matching the paper's description. |
| 8 | **Plan ordering** | §3.3 | Same as #6 | `tests/test_plan.py` (order preserved) | **PASS** | The plan list preserves the order of steps; the validator enforces the terminal-Response invariant. |
| 9 | **Terminal Response invariant** | §4.2 (plan must end with RESPONSE) | `eagent_baseline/plan.py` (`MRAGPlan.validate`) | `tests/test_plan.py` (invalid plan rejected) | **PASS** | Validation rejects any plan not ending with a RESPONSE step. |
| 10 | **Task Executor behavior** (dispatch, sequencing) | §4.2 | `eagent_baseline/executor.py` | `tests/test_planner_executor.py`, `tests/test_end_to_end_smoke.py` | **PASS** | Executor runs each step in order, invoking the registered tool, and returns the final response. |
| 11 | **Requery tool** (MLLM-driven query synthesis) | §4.2 (Requery description) | `eagent_baseline/tools/requery.py` | `tests/test_tools.py` (requery execution) | **UNVERIFIED** | The tool is dispatched and invoked, but its paper-defined function — synthesizing a concise search query from the image, question, and image-search results — depends on the undisclosed Requery prompt template and the Qwen2-VL-72B model. Only the stub model and an assumed prompt template (`ASSUMED_REQUERY_PROMPT_TEMPLATE`) are used, so the query-synthesis behavior is not reproduced. |
| 12 | **Response tool** (final answer generation) | §4.2 (Response description) | `eagent_baseline/tools/response.py` | `tests/test_tools.py` (response execution) | **UNVERIFIED** | The tool is dispatched and aggregates inputs, but its paper-defined function — producing a coherent, user-oriented response from the image, query, and search results — depends on the undisclosed Response prompt template and the Qwen2-VL-72B model. Only the stub model and an assumed prompt template (`ASSUMED_RESPONSE_PROMPT_TEMPLATE`) are used, so the answer-generation behavior is not reproduced. |
| 13 | **Image Search** (Baidu Image Search) | §4.2, §5.2 | `eagent_baseline/tools/image_search.py` (stub) | `tests/test_tools.py` (image_search execution) | **UNVERIFIED** | The implementation uses an offline deterministic stub; Baidu's behavior is not reproduced. |
| 14 | **Text Search** (Tavily) | §4.2, §5.2 | `eagent_baseline/tools/text_search.py` (stub) | `tests/test_tools.py` (text_search execution) | **UNVERIFIED** | Stub does not call Tavily; paper-specified service is absent. |
| 15 | **Planner model role** (InternVL2-8B) | §5.2 | `eagent_baseline/models.py` (factory -> stub) | `tests/test_model_layer.py` (ProviderRejectionTests) | **UNVERIFIED** | The real InternVL2-8B model is not used; a stub model provides deterministic output. |
| 16 | **Executor MLLM role** (Qwen2-VL-72B) | §5.2 | Same as #15 | `tests/test_model_layer.py` (ProviderRejectionTests) | **UNVERIFIED** | Qwen2-VL-72B is not instantiated; only a stub provider runs. |
| 17 | **RemPlan question taxonomy** (four types) | §3.2.1 | `eagent_baseline/data.py` (`QuestionType` enum) | `tests/test_data_and_evaluation.py` (schema validation) | **PASS** | The enum matches the four question types defined in the paper. |
| 18 | **RemPlan dataset facts** (200 pairs) | §3.1, §5.1 | `configs/base.yaml` (`benchmark_size: 200`) | *None* - no dataset shipped. | **UNVERIFIED** | The benchmark data are not present; only a config value exists. |
| 19 | **10K training-data distinction** | §5.2 | `configs/base.yaml` (`planner_train_samples: 10000`) | *None* - training set not provided. | **UNVERIFIED** | No training data or recipe is available to verify the claim. |
| 20 | **IS-P (image-search precision)** | §3.3 | `eagent_baseline/evaluation.py` (`ToolPrecision`) | `tests/test_data_and_evaluation.py` (precision on stub data) | **UNVERIFIED** | The paper does not give a concrete formula; the stub implementation is an assumption. |
| 21 | **IS-R (image-search recall)** | §3.3 | Same as #20 | `tests/test_data_and_evaluation.py` (recall on stub data) | **UNVERIFIED** | Same reasoning as #20. |
| 22 | **TS-P (text-search precision)** | §3.3 | Same as #20 | `tests/test_data_and_evaluation.py` (precision on stub data) | **UNVERIFIED** | Same reasoning as #20. |
| 23 | **TS-R (text-search recall)** | §3.3 | Same as #20 | `tests/test_data_and_evaluation.py` (recall on stub data) | **UNVERIFIED** | Same reasoning as #20. |
| 24 | **Plan-acc** (plan-accuracy metric) | §3.3 | `eagent_baseline/evaluation.py` (`PlanAccuracy`) | `tests/test_data_and_evaluation.py` (exact-match) | **UNVERIFIED** | The paper does not specify that exact-sequence matching is the metric; this is an implementation assumption. |
| 25 | **Param-acc** (parameter correctness) | §3.3 | `eagent_baseline/evaluation.py` (`ParamAccuracy`) | `tests/test_data_and_evaluation.py` (raises if no validator) | **UNVERIFIED** | Real validation criteria are missing. |
| 26 | **Param-sim** (semantic similarity) | §3.3 | Same as #25 | `tests/test_data_and_evaluation.py` (raises if no similarity fn) | **UNVERIFIED** | Same as #25. |
| 27 | **Answer evaluation** (GPT-4o judge) | §5.1 | `eagent_baseline/evaluation.py` (`StubAnswerJudge`) | `tests/test_data_and_evaluation.py` (judge test) | **UNVERIFIED** | The real GPT-4o judge prompt and scoring are not implemented. |
| 28 | **GPT-4o judge** (score 0-2) | §5.1 | Same as #27 | `tests/test_data_and_evaluation.py` (judge test) | **UNVERIFIED** | Same as #27. |
| 29 | **Training configuration** (model-training hyper-parameters) | §5.2 | `configs/base.yaml` (records values) | *None* - no training run. | **UNVERIFIED** | The paper's training recipe (loss, optimizer, schedule, etc.) is not reproduced. |
| 30 | **Experimental configuration** (runtime settings, tool choices) | §5.2 | `configs/base.yaml` (records tool names, model names) | *None* - no real services/models used. | **UNVERIFIED** | Config loads, but the actual experimental setup (Baidu, Tavily, InternVL2-8B, Qwen2-VL-72B) is missing. |
| 31 | **Stated models / tools** (InternVL2-8B, Qwen2-VL-72B, Baidu, Tavily) | §5.2 | `eagent_baseline/models.py` (factory -> stub) | `tests/test_model_layer.py` (ProviderRejectionTests) | **UNVERIFIED** | Real models and services are not present. |
| 32 | **Major implementation assumptions** | - | - | - | **UNVERIFIED** (documented separately) | See the "Implementation Assumptions" section below. |

## Counts

- **PASS:** 10
- **FAIL:** 0
- **UNVERIFIED:** 22

## Implementation Assumptions (explicitly documented, not counted as PASS)

| Component | Assumption (paper-unspecified or partially specified) | Reason it is an assumption |
|-----------|------------------------------------------------------|----------------------------|
| Planner prompt template | Minimal hand-crafted template (`ASSUMED_PLANNER_PROMPT_TEMPLATE`) | Paper only says "manually written task-specific prompts"; exact text is not given. |
| Planner multimodal input handling | `build_request` passes `question.images` to `ModelRequest`, but `ASSUMED_PLANNER_PROMPT_TEMPLATE` embeds only `question.text`; the stub model performs no image+text contextual analysis | richer multimodal prompt / vision encoding | Paper says "contextual analysis of both textual queries and visual inputs" (§4.1); the assumed prompt only embeds text, and the stub model does not perform multimodal fusion. |
| Requery / Response prompts | Minimal templates in `tools/requery.py` & `tools/response.py` | Same as above - exact prompts undisclosed. |
| Tool identifier strings | Fixed identifiers (`image_search`, `text_search`, `requery`, `response`) | Paper shows free-text names; mapping is an implementation choice. |
| Result threading between steps | Image-search result -> Requery -> Text-search -> Response | Paper describes this flow but does not prescribe exact data-passing mechanics. |
| Metric formulas (IS-P/IS-R/TS-P/TS-R/Plan-acc) | Simple set-based precision/recall and exact-match plan accuracy | Paper defines metrics conceptually but does not give concrete formulas or matching granularity. |
| Parameter-correctness & semantic-similarity | No concrete validator or similarity function; `NotImplementedError` placeholders | Paper mentions these metrics but provides no implementation details. |
| Answer-judge (GPT-4o) | Stub judge returning 0/2 based on exact match | Real GPT-4o prompt and scoring rubric are not disclosed. |
| Training data & recipe | No 10K dataset, no loss/optimizer/schedule | Paper states a 10K fine-tuning set but does not release it. |
| Search services (Baidu Image Search, Tavily) | Offline deterministic stubs | Paper specifies these services; no network access in the baseline. |
| Model backbones (InternVL2-8B, Qwen2-VL-72B) | Stub provider (`ScriptedVisionLanguageModel`) | Real models are not available in the offline repository. |

These assumptions are documented to keep a clear record of what is not verified against the paper, but they do not count as PASS entries.

## What the tests check

| Test module | Behavior verified |
|---|---|
| `tests/test_plan.py` | Plan object construction; tool-sequence extraction; terminal-Response invariant; Response only at the end; JSON round-trip serialization; rejection of invalid JSON and unknown tool names |
| `tests/test_tools.py` | Image/text search tools run offline via stub providers; input validation; Requery and Response tools invoke the provider-independent model and report its model name |
| `tests/test_planner_executor.py` | Planner performs exactly one model call (single-pass) and parses a plan; invalid plan text is rejected; executor runs a full plan to a terminal Response; missing-tool and invalid-plan error handling; end-to-end plan-then-execute via `EAgent` |
| `tests/test_data_and_evaluation.py` | RemPlan instance schema validation; per-tool precision/recall; Plan-acc exact-match; Param-acc and Param-sim raise without an injected criterion and compute with one; answer-judge 0-2 scale |
| `tests/test_config_and_training.py` | Development config loads (records `benchmark_size: 200`, `planner_train_samples: 10000`); planner/executor models build from the config via the layer-1 factory; training scaffold reports paper values; `train()` raises `NotImplementedError` |
| `tests/test_end_to_end_smoke.py` | The smallest faithful end-to-end path: a synthetic multimodal `Question` drives the single-pass `MRAGPlanner`, and the produced `MRAGPlan` runs through `TaskExecutor` dispatching all four tools in order (`image_search -> requery -> text_search -> response`) to a terminal Response with a non-empty final response |
| `tests/test_model_layer.py` (repo root; 7 tests) | The `stub` provider builds; the `research` provider raises `UnsupportedProviderError` (model names only recorded). No HuggingFace provider is wired into `src/eagent`. |

## Independence from unavailable assets

The tests exercise only: the plan representation, the planner/executor control flow, tool dispatch, the model-layer integration through the `stub` provider (plus a documented `ScriptedVisionLanguageModel` development double), and the data/evaluation/training interfaces. They do not touch InternVL2-8B, Qwen2-VL-72B, Baidu Image Search, Tavily, the RemPlan benchmark, or the 10K training set.

## Comment-policy check

A scan for `#` across `*.py` and `*.yaml` in this directory returns no matches: generated source and configuration contain no comments. Paper references live in `PAPER_SPEC.md`, `REPRODUCTION_NOTES.md`, and this file.

## Reproduction status

**NOT REPRODUCED.** No paper experiment was executed; no paper metric was computed against paper data. Passing tests verify software behavior of the baseline scaffold only.
