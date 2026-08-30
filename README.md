# E-Agent — Paper2Code Skeleton & Functional Extension

A deterministic, offline implementation of the **plan-then-execute** architecture from
*Efficient Agent: Optimizing Planning Capability for Multimodal Retrieval Augmented Generation*
(arXiv [`2508.08816`](https://arxiv.org/abs/2508.08816)), plus an isolated layer that exercises the
same architecture with real substitute models.

> **Status: NOT a reproduction of the paper's reported results.**
> This repository reproduces the paper's *architecture and control-flow contract*, verified by
> offline tests. It does **not** reproduce any paper metric. The paper-specific assets required for
> a faithful reproduction — the fine-tuned InternVL2-8B planner, the Qwen2-VL-72B tool backbone, the
> RemPlan benchmark, the 10K planner-training set, the undisclosed prompts, the exact metric
> formulas, and the live Baidu/Tavily services — were **not located** in public sources (see
> [PAPER_SPECIFIC_ASSET_AUDIT.md](PAPER_SPECIFIC_ASSET_AUDIT.md)). Public base checkpoints and
> substitute models are **not** the paper's trained system and are never presented as such.

## The 30-second version

The project is organized into three layers that are kept strictly separate:

| Layer | What it is | Runs real models? | Reproduces the paper? |
|-------|-----------|-------------------|-----------------------|
| **1. Paper baseline** | Faithful skeleton of the E-Agent architecture, anchored line-by-line to the paper via `PAPER_SPEC.md`. Only a deterministic `stub` provider is executable. | No — stub only | Architecture/control-flow only. **Not** the reported results. |
| **2. Functional extension** | An isolated layer that injects real *substitute* VLMs into the same planner interface to check the architecture runs end-to-end with a live model. | Yes — substitutes | No. Substitute-model validation, explicitly not paper reproduction. |
| **3. Research extension** | Planned work: controlled planner evaluation and model comparison beyond exact reproduction. | Future | No — separate research track. |

The paper baseline never imports the functional extension; substitute models are wired in by
dependency injection only and are never registered in the reproduction's model factory.

## Architecture

E-Agent decouples *planning* from *execution*: a single-pass planner emits a complete structured
plan, and a tool-aware executor carries it out to a terminal response.

```
multimodal question (text + image)
        │
        ▼
  MRAGPlanner.plan()          single model call — no re-planning mid-run
        │
        ▼
  MRAGPlan  =  [ PlanStep(tool, arguments), ... ]   ends with RESPONSE (invariant)
        │
        ▼
  TaskExecutor.execute()      routes each step through registered Tools
        │
        ├─ image_search ─┐
        ├─ text_search   │  results threaded through a mutable ExecutionState
        ├─ requery       │
        └─ response  ────┴─▶  terminal response
```

Component roles:

- **`VisionLanguageModel`** (`src/eagent/models/protocols.py`) — provider-independent ABC:
  a `model_name` property and `generate(ModelRequest) -> ModelResponse`. Every model, real or
  stub, is wired in through this one interface.
- **`MRAGPlan` / `PlanStep` / `ToolName`** (`eagent_baseline/plan.py`) — the structured plan.
  `MRAGPlan.validate()` enforces the invariant: non-empty, must **end with `RESPONSE`**, and
  `RESPONSE` may appear only as the terminal step. JSON shape: `{"steps":[{"tool","arguments"}]}`.
  Malformed or misordered output raises `PlanValidationError` — no repair, no retry.
- **`MRAGPlanner`** (`eagent_baseline/planner.py`) — single-pass: calls the model exactly once
  (asserted via `plan_call_count`).
- **`TaskExecutor`** (`eagent_baseline/executor.py`) — dispatches each step to a registered `Tool`,
  threading intermediate results through `ExecutionState`. Missing tools or missing queries raise
  rather than silently degrade.
- **Isolation** — the functional extension injects substitute models into this same interface. It
  is never registered in the reproduction's `eagent.models.factory`, which stays stub-only.

## The three layers in detail

### 1. Paper baseline (faithful skeleton)

Two code areas wired at runtime via `sys.path` injection (there is no packaging):

- **`src/eagent/`** — the provider-independent model layer: the `VisionLanguageModel` ABC,
  `ModelRequest`/`ModelResponse`, provider-neutral `Image`/`Question`, typed config, and a factory.
  Only the `stub` provider is executable; the `research` provider raises `UnsupportedProviderError`.
- **`efficient_agent_.../eagent_baseline/`** — the plan-then-execute baseline that consumes the
  model layer: planner, executor, plan representation, four tools, and metric/data/training
  *interfaces* that the paper supports but that are intentionally not runnable reproductions.

Every behavior is mapped to a paper section in
[`PAPER_SPEC.md`](efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation/PAPER_SPEC.md).
The baseline fails loudly by design instead of inventing missing research details.

### 2. Functional extension (substitute-model validation)

`functional_extension/` checks that the architecture runs end-to-end with a *real* model, using
public **substitute** VLMs — never the paper's trained system. Models are injected into a
`FunctionalValidationPlanner`; the fail-loud parser and single-pass invariant are unchanged.

Two substitutes were exercised:

- **`HuggingFaceTB/SmolVLM-256M-Instruct` — tested and FAILED the structured-planning requirement.**
  Wiring worked (one real CPU call, correct model id, image reached the provider, no stub fallback),
  but the model could not satisfy the plan-ordering invariant: it produced syntactically correct
  JSON yet placed `response` first, raising `PlanValidationError`. This is a model capability
  ceiling, not a wiring bug. It is recorded, not hidden — see
  [MODEL_SELECTION.md](functional_extension/MODEL_SELECTION.md).
- **`gemini-3.6-flash` — substitute planner validated (not paper reproduction).** A Gemini provider
  (`eagent_functional/gemini_provider.py`, official `google-genai` SDK, API key from
  `GEMINI_API_KEY`, structured JSON schema, no retry/repair/fallback) produces a valid,
  `RESPONSE`-terminal plan through the same harness. The in-repo real end-to-end test is gated
  behind an env flag **and** the API key, so it is skipped during ordinary offline runs; all other
  Gemini tests are mocked.

A remote substitute (`Qwen/Qwen2-VL-2B-Instruct`) was investigated but **not** implemented: no
turnkey serverless OpenAI-compatible endpoint for the exact checkpoint could be verified (see
[REMOTE_MODEL_ENDPOINT_AUDIT.md](functional_extension/REMOTE_MODEL_ENDPOINT_AUDIT.md)). No model was
silently substituted.

### 3. Research extension (future work)

Beyond exact reproduction: controlled planner evaluation and cross-model comparison. The evaluation
*harness* exists (`eagent_functional/planner_eval.py`, model-agnostic, records per case: id,
question type, raw output, parsed plan, validity, tool sequence, plan length, planner latency,
failure reason). The evaluation *dataset* and quantitative metrics do not yet exist — see
[Roadmap](#roadmap). Nothing here reproduces a paper metric.

## What is and isn't reproduced

**Reproduced (architecture only):** the plan-then-execute control-flow contract, the single-pass
planner, the structured `MRAGPlan` representation and its validation invariant, the tool interfaces
and executor dispatch, and the evaluation/data/training *interfaces* the paper supports.

**Not reproduced:** the fine-tuned InternVL2-8B planner, the Qwen2-VL-72B tool backbone, the RemPlan
benchmark, the 10K planner-training set, the undisclosed prompt templates, the exact metric formulas,
the GPT-4o judge configuration, and the live Baidu Image Search / Tavily services. All are
UNVERIFIED / NOT LOCATED per [PAPER_SPECIFIC_ASSET_AUDIT.md](PAPER_SPECIFIC_ASSET_AUDIT.md). Public
base checkpoints exist but are not the paper's trained system.

## Testing & verification

All suites use stdlib `unittest`, run fully offline, and require no GPU, keys, or datasets. The
model layer and baseline use different working directories because of the `sys.path` bootstrap.

| Suite | Tests | Notes |
|-------|------:|-------|
| Model layer (`tests/`, repo root) | 7 | Provider-independent layer; `research` provider rejection. |
| Baseline (`efficient_agent_.../tests/`) | 77 | Plan/planner/executor/tools/agent/data/eval/training + deterministic end-to-end smoke path. |
| Functional extension (`functional_extension/tests/`) | 62 | 58 run offline; **4 gated real tests skip** (2 real Gemini, 2 real SmolVLM) unless an env flag + key are set. All non-gated tests are mocked — no network. |

```bash
# Model layer — from repo root
python -m unittest discover -s tests

# Baseline — from inside the baseline directory
cd efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation
python -m unittest discover -s tests -t .

# Functional extension — from repo root
python -m unittest discover -s functional_extension/tests
```

### Completed milestones

- Paper2Code architecture reconstruction (baseline skeleton anchored to `PAPER_SPEC.md`).
- Baseline test hardening (77 offline tests).
- Deterministic end-to-end smoke path (stub providers).
- Paper-specific asset audit (assets not located; documented).
- Functional-extension isolation (injection-only; factory stays stub-only).
- Real SmolVLM-256M multimodal inference wired and run on CPU.
- SmolVLM-256M planner failure identified and documented (capability ceiling).
- Gemini planner integration (SDK-based provider, structured JSON, fail-loud).
- Gemini substitute planner validated end-to-end (gated real test).
- Model-agnostic planner evaluation harness (`planner_eval.py`).

## Current status

- **Works:** the deterministic offline baseline and all three test suites; substitute-model
  injection into the planner interface; the Gemini substitute planner (via the gated real test).
- **Blocked:** faithful paper reproduction — the required trained models, benchmark, training set,
  prompts, metric formulas, and live services are not available.
- **Experimentally validated (substitute only):** the architecture runs end-to-end with a real
  planner model; a small captioner-class VLM (SmolVLM-256M) is insufficient for the ordering
  invariant.
- **Remains research:** an evaluation dataset, quantitative planner metrics, and cross-model
  comparison.

## Roadmap

Planned, not yet implemented — do not read these as existing features:

1. A controlled planner-evaluation protocol on top of the existing harness.
2. An initial 20-case evaluation dataset spanning the question taxonomy.
3. Quantitative planner metrics (plan validity, tool-sequence shape, latency).
4. Cross-model comparison of substitute planners.
5. A research extension beyond exact reproduction, only after baseline measurements exist.

## Repository structure

```
e-agent-reproduction/
├── README.md                       # this file
├── CLAUDE.md                       # working rules
├── PAPER_SPECIFIC_ASSET_AUDIT.md   # why faithful reproduction is blocked
├── FUNCTIONAL_REPRODUCTION_PLAN.md # hardware + functional-validation plan
├── configs/                        # layer-1 model config (development/research)
├── src/eagent/                     # provider-independent model layer (stub executable)
│   ├── common/types.py             # Image, Question, ...
│   └── models/                     # protocols, config, factory, providers/stub
├── efficient_agent_.../            # paper baseline package + PAPER_SPEC.md + tests
│   └── eagent_baseline/            # plan, planner, executor, agent, tools/, data, evaluation, training
├── functional_extension/           # isolated substitute-model layer
│   ├── eagent_functional/          # gemini_provider, transformers_provider, planner_integration, planner_eval
│   └── tests/                      # mocked tests + gated real tests
└── tests/                          # model-layer tests
```

## Setup

Baseline (minimal deps; no GPU/network/keys):

```bash
pip install -r efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation/requirements.txt
```

Functional extension (adds the substitute-model dependencies):

```bash
pip install -r functional_extension/requirements.txt
```

Optional real-model API access (only for the gated tests). Provide credentials via environment
variables — never hardcode a key, and never commit one:

```bash
export GEMINI_API_KEY=...   # required only to run the gated real Gemini test
```

## Reproducibility & honesty

- The baseline is deterministic and offline; passing tests verify software behavior of the
  skeleton, not any paper number.
- Functional experiments use public substitute models, clearly labeled as substitutes and kept out
  of the reproduction's model factory.
- Missing paper-specific assets are documented, not worked around; the code raises rather than
  fabricating research details.
- Experimental (substitute-model) claims are stated separately from paper claims, and no result
  here should be read as reproducing a paper metric.

## Citation

```bibtex
@article{wang2025eagent,
  title={Efficient Agent: Optimizing Planning Capability for Multimodal Retrieval Augmented Generation},
  author={Wang, Yuechen and Qiao, Yuming and Meng, Dan and Yang, Jun and Lu, Haonan and Yang, Zhenyu and Zhang, Xudong},
  journal={arXiv preprint arXiv:2508.08816},
  year={2025}
}
```

Paper: [arXiv:2508.08816](https://arxiv.org/abs/2508.08816).

