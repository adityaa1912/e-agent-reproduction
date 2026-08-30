# E-Agent

A research-grade implementation and experimental testbed for studying **planning in multimodal
retrieval-augmented generation**, based on the E-Agent architecture from
[arXiv:2508.08816](https://arxiv.org/abs/2508.08816).

The repository provides a deterministic, offline implementation of E-Agent's *plan-then-execute*
control flow, together with an isolated layer for running the same architecture against real
multimodal models. It is built to make planning behavior easy to inspect, test, and compare — not
to restate the paper's published numbers.

## Overview

E-Agent separates *planning* from *execution*. A planner reads a multimodal question and emits one
complete, structured plan; a tool-aware executor then carries that plan out to a terminal answer.
This repository reconstructs that contract faithfully and wires it to a provider-independent model
interface, so any vision-language model — a deterministic stub or a live API — can drive the planner
through the same seam.

```
Image + Question
      │
      ▼
   Planner ───────────▶ Structured MRAG Plan        (single model call)
                              │
                              ▼
                           Executor
                              │
        ┌───────────┬─────────┴─────────┬───────────┐
        ▼           ▼                   ▼           ▼
   image_search  text_search         requery     response
        └───────────┴─────────┬─────────┴───────────┘
                              ▼
                        Final Response
```

## Design

A few architectural decisions carry the project:

- **Planner/executor separation.** `MRAGPlanner` produces a full plan; `TaskExecutor` runs it. No
  model call happens mid-execution to re-plan.
- **Single-pass planning.** The planner calls its model exactly once per question, asserted through
  `plan_call_count`.
- **Provider-independent model interface.** Every model implements one ABC —
  `VisionLanguageModel` (`model_name` + `generate(ModelRequest) -> ModelResponse`) — so stubs and
  real providers are interchangeable behind the same call.
- **Structured `MRAGPlan`.** A plan is an ordered list of `PlanStep(tool, arguments)`, serialized as
  `{"steps":[{"tool","arguments"}]}`.
- **Terminal `Response` invariant.** `MRAGPlan.validate()` requires a non-empty plan that ends with
  `RESPONSE`, with `RESPONSE` allowed only as the final step. Malformed or misordered output raises
  `PlanValidationError` — there is no repair or retry.
- **Dependency-injected experiments.** Real models are injected into the planner interface; they are
  never registered in the reproduction's `eagent.models.factory`.
- **Strict isolation.** The paper baseline never imports the functional extension. Substitute-model
  code lives entirely under `functional_extension/`.

## What the repository contains

Two code areas wired together at runtime via `sys.path` injection (there is no packaging):

- `src/eagent/` — the provider-independent model layer: the `VisionLanguageModel` ABC,
  `ModelRequest`/`ModelResponse`, provider-neutral `Image`/`Question`, typed config, and a factory.
  The `stub` provider is executable; the `research` provider raises `UnsupportedProviderError`.
- `efficient_agent_.../eagent_baseline/` — the plan-then-execute baseline: plan representation,
  planner, executor, four tools, and metric/data/training *interfaces* the paper supports. Each
  behavior is mapped to a paper section in
  [`PAPER_SPEC.md`](efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation/PAPER_SPEC.md).
- `functional_extension/` — an isolated layer that injects real substitute VLMs into the planner
  interface and a model-agnostic evaluation harness.

## Three layers

| Layer | Scope | Real models | State |
|-------|-------|-------------|-------|
| **Paper baseline** | E-Agent architecture and control flow, deterministic and offline | Stub only | Architecture reproduced; runs and tested |
| **Functional extension** | Real multimodal provider experiments through the same planner interface | Substitutes (SmolVLM, Gemini) | Working; substitute-model validation |
| **Research extension** | Planner evaluation and cross-model comparison | Substitutes | Harness in place; dataset and metrics in progress |

The baseline reproduces the paper's *architecture*, not its results. The functional and research
layers use public **substitute** models and never stand in for the paper's trained system.

## Validated experiments

Two real models were run through the functional planner interface. Both are substitutes; neither
reproduces the paper's trained system or its metrics.

- **SmolVLM-256M** (`HuggingFaceTB/SmolVLM-256M-Instruct`) performed real multimodal inference on
  CPU — one real call, correct model id, image reaching the provider, no stub fallback — but failed
  the structured-planning requirement: it emitted syntactically valid JSON yet placed `response`
  first, violating the terminal-`Response` invariant. This is a capability ceiling of a small
  captioner-class model, documented in
  [MODEL_SELECTION.md](functional_extension/MODEL_SELECTION.md).
- **Gemini 3.6 Flash** (`gemini-3.6-flash`) passed the gated real planner validation, producing a
  valid `RESPONSE`-terminal plan through the same harness. The provider
  (`eagent_functional/gemini_provider.py`) uses the official `google-genai` SDK, reads its key from
  `GEMINI_API_KEY`, requests a structured JSON schema, and does no retry, repair, or fallback.

## Current status

- **Reproduced:** the plan-then-execute control-flow contract, single-pass planner, structured plan
  and its validation invariant, tool interfaces, and executor dispatch — all deterministic and
  offline.
- **Working experiments:** real substitute models drive the planner interface; Gemini 3.6 Flash
  produces valid plans; SmolVLM-256M's failure is characterized and recorded.
- **In progress:** an evaluation dataset and quantitative planner metrics for cross-model
  comparison.
- **Out of scope here:** the paper's reported numbers (see
  [Reproduction boundaries](#reproduction-boundaries)).

## Components

| Area | Technology |
|------|-----------|
| Language | Python 3.11 |
| Baseline deps | `pydantic>=2.0`, `pyyaml>=6.0` |
| Tests | stdlib `unittest`, fully offline |
| Local substitute VLM | `transformers` (`AutoModelForVision2Seq`, CPU/FP32) |
| Remote substitute VLM | `google-genai` SDK (Gemini) |
| Config | YAML, typed via pydantic |

## Repository structure

```
e-agent-reproduction/
├── CLAUDE.md                       # working rules for this repo
├── PAPER_SPECIFIC_ASSET_AUDIT.md   # evidence: which paper assets exist / are missing
├── FUNCTIONAL_REPRODUCTION_PLAN.md # hardware snapshot + functional-validation plan
├── configs/                        # layer-1 model config (development / research)
├── src/eagent/                     # provider-independent model layer
│   ├── common/types.py             # Image, Question
│   └── models/                     # protocols, config, factory, providers/stub
├── efficient_agent_.../            # paper baseline package
│   ├── eagent_baseline/            # plan, planner, executor, agent, tools/, data, evaluation, training
│   ├── PAPER_SPEC.md               # paper-claim → code mapping
│   ├── REPRODUCTION_NOTES.md       # assumptions, deviations, blockers
│   ├── VERIFICATION.md             # what the tests check
│   └── tests/
├── functional_extension/           # isolated substitute-model layer
│   ├── eagent_functional/          # gemini_provider, transformers_provider, planner_integration, planner_eval
│   ├── MODEL_SELECTION.md          # SmolVLM failure + candidate analysis
│   ├── REMOTE_MODEL_ENDPOINT_AUDIT.md
│   └── tests/                      # mocked tests + gated real tests
└── tests/                          # model-layer tests
```

## Installation

Baseline only — no GPU, network, or keys:

```bash
pip install -r efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation/requirements.txt
```

Functional extension — adds the substitute-model dependencies:

```bash
pip install -r functional_extension/requirements.txt
```

Real-model access is optional and used only by the gated tests. Supply credentials through the
environment; never hardcode or commit a key:

```bash
export GEMINI_API_KEY=...
```

## Running tests

All suites use stdlib `unittest` and run offline. The model layer and baseline use different working
directories because of the `sys.path` bootstrap.

```bash
# Model layer (7 tests) — from repo root
python -m unittest discover -s tests

# Paper baseline (77 tests) — from inside the baseline directory
cd efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation
python -m unittest discover -s tests -t .

# Functional extension (62 tests) — from repo root
python -m unittest discover -s functional_extension/tests
```

The functional suite has **62 tests**: 58 run offline with mocked responses, and **4 gated
real-model tests** (2 Gemini, 2 SmolVLM) skip automatically unless an env flag and API key are set.
Ordinary offline runs make no network calls.

## Reproduction boundaries

This repository reproduces E-Agent's architecture and control flow, not the paper's experimental
results. The paper-specific assets required for a faithful reproduction — the fine-tuned InternVL2-8B
planner, the Qwen2-VL-72B tool backbone, the RemPlan benchmark, the 10K planner-training set, the
prompt templates, the exact metric formulas, the judge configuration, and the live Baidu / Tavily
services — could not be located in public sources. Public base checkpoints and the substitute models
used here are not the paper's trained system.

Evidence and detail:
[PAPER_SPECIFIC_ASSET_AUDIT.md](PAPER_SPECIFIC_ASSET_AUDIT.md) ·
[REPRODUCTION_NOTES.md](efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation/REPRODUCTION_NOTES.md) ·
[REMOTE_MODEL_ENDPOINT_AUDIT.md](functional_extension/REMOTE_MODEL_ENDPOINT_AUDIT.md).

## Roadmap

Planned work, not yet implemented:

1. A controlled planner-evaluation protocol on top of the existing harness.
2. An initial 20-case evaluation dataset spanning the question taxonomy.
3. Quantitative planner metrics (plan validity, tool-sequence shape, latency).
4. Cross-model comparison of substitute planners.
5. A research extension beyond exact reproduction, once baseline measurements exist.

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

