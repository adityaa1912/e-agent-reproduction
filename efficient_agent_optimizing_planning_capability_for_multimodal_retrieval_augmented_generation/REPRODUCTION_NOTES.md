# Reproduction Notes: Efficient Agent (E-Agent)

> Read this before using the code. It records every implementation choice, whether the paper specified it, and what remains unavailable.

## Paper

- **Title:** Efficient Agent: Optimizing Planning Capability for Multimodal Retrieval Augmented Generation
- **Authors:** Yuechen Wang, Yuming Qiao, Dan Meng, Jun Yang, Haonan Lu, Zhenyu Yang, Xudong Zhang
- **Year:** 2025
- **ArXiv:** https://arxiv.org/abs/2508.08816
- **Official code:** None found (`paper_metadata.json`: `official_code: []`)

## Reproduction status

**NOT REPRODUCED.** No experiment was executed. No paper metric was computed against paper data.

## What this implements

The E-Agent plan-then-execute control flow: a single-pass mRAG planner contract, a structured mRAG plan representation with a terminal-Response invariant, a Task Executor that routes plan steps to tools, the four tool interfaces (image search, text search, Requery, Response), and the evaluation/training/data interfaces the paper supports. See `PAPER_SPEC.md` for section references.

## Verified against

- [x] Paper prose (§4, §4.1, §4.2), Figure 4 plan format, §5.6 example plans, §5.1/§5.2 settings.
- [ ] Paper Algorithm box — none exists.
- [ ] Official code — none found.
- [ ] Well-known reimplementation — none used.

## No-comments policy

Per the Stage 4 instruction, generated source and configuration files contain **no** explanatory/inline/paper-reference comments. All paper references live in `PAPER_SPEC.md`, this file, and `VERIFICATION.md`. Short descriptive docstrings are used consistent with the existing `src/eagent` project convention and contain no paper citations.

## Unspecified choices (assumptions)

| Component | Our choice | Alternatives | Paper quote (if partial) | Section |
|---|---|---|---|---|
| Plan serialization | JSON `{"steps":[{"tool","arguments"}]}` | YAML, DSL, free text | Figure 4 shows free-text step examples only | §3.3/Fig 4 |
| Tool identifier strings | `image_search`,`text_search`,`requery`,`response` | any labels | placeholder names `search_tool_name`,`MLLM_tool_name` | §4.2/Fig 4 |
| Planner input prompt | minimal template in `planner.py` | any prompt | — (planner prompt not given) | §4.1 |
| Requery prompt template | minimal template in `tools/requery.py` | authors' hidden template | "manually written task-specific prompt templates" | §4.2 |
| Response prompt template | minimal template in `tools/response.py` | authors' hidden template | "manually written task-specific prompt templates" | §4.2 |
| Planner multimodal input handling | `build_request` passes `question.images` to `ModelRequest`, but `ASSUMED_PLANNER_PROMPT_TEMPLATE` embeds only `question.text`; the stub model performs no image+text contextual analysis | richer multimodal prompt / vision encoding | "contextual analysis of both textual queries and visual inputs" | §4.1 |
| Result threading between steps | image results → requery/response; requery query → text search | other routing | — | §4.2/§5.6 |
| Executor error/timeout/retry | none (fail fast with clear errors) | retries, timeouts | — | §4.2 |
| Tool precision/recall matching | set membership of tool usage per plan | span/parameter-level matching | precision/recall named only | §3.3 |
| Plan-acc comparator | exact tool-sequence match (default; pluggable) | partial/weighted match | "complete and correct plan" | §3.3 |
| Param-acc validity | not defaulted; caller must inject a validator | any validity rule | "validity of the parameters" | §3.3 |
| Param-sim similarity | not defaulted; caller must inject a similarity_fn | embedding cosine, etc. | "semantic consistency" | §3.3 |
| Answer judge | `StubAnswerJudge` (exact-substring 0/2) for tests | GPT-4o judge | GPT-4o scores 0–2 | §5.1 |

## Known deviations

| Deviation | Paper says | We do | Reason |
|---|---|---|---|
| Planner model | InternVL2-8B fine-tuned on 10K | development `stub` provider; `ScriptedVisionLanguageModel` for deterministic runs | model + training recipe + data unavailable |
| MLLM tools model | Qwen2-VL-72B | development `stub` provider | model unavailable; not runnable locally |
| Search tools | Baidu Image Search, Tavily | offline stub providers behind an abstraction | no credentials/network in this baseline |

## Critical unspecified training details

The paper states only: InternVL2-8B, fine-tuned, 10K samples of (image, question, plan) (§5.2). It does **not** specify objective/loss, optimizer, learning rate, scheduler, epochs, batch size, precision, seed, hardware, checkpointing, or a train/validation split. `eagent_baseline/training.py` therefore exposes a labelled scaffold whose `train()` raises `NotImplementedError`. It does not fabricate a training loop.

## Expected results (context only; not reproduced)

| Metric | Paper's number | Dataset | Conditions |
|---|---|---|---|
| Ans. (All) | 1.25 | RemPlan | Table 2, E-Agent-sft |
| Plan-acc | 0.86 | RemPlan | Table 2, E-Agent-sft |
| Redundant-search reduction | 37% | — | Abstract |
| Accuracy gain vs SOTA mRAG | 13% | — | Abstract |

Exact reproduction requires the unreleased data, the undisclosed training recipe, the undisclosed prompts/metric definitions, and live search services. None are available.

## Scope decisions

### Implemented
- Planner contract, plan representation, executor, four tool interfaces — core contribution (§4).
- Question taxonomy, metric interfaces, data schema, training scaffold — paper-supported interfaces (§3.1–§3.3, §5.1–§5.2).

### Intentionally excluded
- Baselines (raw Qwen2-VL-72B, MMSearch, OmniSearch) — comparison methods, not the contribution.
- RemPlan data collection/annotation and GPT-4o plan generation — dataset creation, not the method.
- Real model serving/downloads and live search API clients — out of scope for an offline baseline.

### Needed for full reproduction (not included)
- The 10K planner training set and the fine-tuning recipe.
- The 200-pair RemPlan benchmark with plan/answer annotations.
- The authors' Requery/Response prompts and GPT-4o judge prompt.
- Exact planning-metric formulas and the Param-sim similarity model.
- Access to Baidu Image Search and Tavily.

## Reproduction blockers

| Blocker | Severity | Consequence |
|---|---|---|
| RemPlan dataset unreleased | Critical | No planning/answer evaluation possible |
| 10K training set + recipe unavailable | Critical | Planner (E-Agent-sft) cannot be trained |
| Tool prompts undisclosed | High | Requery/Response behavior not reproducible |
| Metric formulas incomplete | High | Plan-acc/Param-acc/Param-sim not exactly reproducible |
| Judge prompt undisclosed | High | Ans. scores not reproducible |
| Live search APIs, no params | High | Non-deterministic, time-dependent results |
| No official code / appendix | Medium | Nothing to resolve unspecified items |

## References

- Paper §§3–5, Figure 4 — architecture, plan format, metrics, settings.
- `.paper2code_work/2508.08816/contribution.md`, `ambiguity_audit.md` — Stage 2/3 analyses this baseline follows.
