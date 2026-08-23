# Paper Specification Map

Every claim this baseline relies on, mapped to the paper. arXiv `2508.08816`. This file holds the paper references so that the source code can stay comment-free.

## Framework (§4)

- E-Agent has two interconnected modules: the **mRAG planner** and the **Task Executor** (§4). The planner "determines the sequence of actions, deciding when to employ search tools and when to rely on the MLLMs directly"; the executor "then carries out these actions" (§4).
  - Code: `eagent_baseline/planner.py`, `eagent_baseline/executor.py`, `eagent_baseline/agent.py`.

## mRAG Planner (§4.1)

- The planner performs "contextual analysis of both textual queries and visual inputs through a single forward pass to formulate a comprehensive mRAG plan" (§4.1) — single-pass planning.
- The plan "simultaneously determines three critical components: (1) optimal selection of multimodal search tools …, (2) adaptive configuration of auxiliary MLLM function, and (3) generation of specialized instructions and parameters for various tool invocation" (§4.1).
  - Code: `MRAGPlanner.plan` calls the model exactly once (`plan_call_count`).
- Planner backbone is InternVL2-8B (§5.2); "operates effectively with an 8B parameter model" (§1).

## Structured mRAG plan (§3.3 Figure 4, §4.1, §5.6)

- Figure 4 shows a plan as ordered steps: `step 1: search_tool_name (image, search query)` and `step 2: MLLM_tool_name (prompt, instruction, references)`.
- §5.6 shows a concrete plan order: image search; requery w/ search results; text search; response w/ search results.
- Response is "the terminal processing unit" (§4.2) — plans end with a Response step.
  - Code: `eagent_baseline/plan.py` (`MRAGPlan.validate` enforces the terminal-Response invariant).

## Task Executor (§4.2)

- "translates the structured plan into executable actions"; "invokes designated search tools and MLLMs according to parameter specifications in the generated plan"; "dynamically selects context-appropriate prompt templates for the MLLM tools" (§4.2).
  - Code: `eagent_baseline/executor.py`.

## Tools (§4.2, §5.2)

- MLLM tools use the Qwen2-VL-72B model, "with manually written task-specific prompt templates" (§4.2).
- **Requery tool**: "MLLM-driven component synthesizes visual inputs (original image and possible image search results) and textual queries to formulate optimized search strings for subsequent text retrieval … concise phrase structures" (§4.2). Code: `tools/requery.py`.
- **Response tool**: "aggregates the input image, query, potential image search, and text search results to produce coherent, user-oriented responses" (§4.2). Code: `tools/response.py`.
- **Image search tool**: "reverse image search services, returning relevant webpage content through similarity-based visual matching"; provider Baidu Image Search (§4.2, §5.2). Code: `tools/image_search.py`.
- **Text search tool**: "keyword-based web queries using compact text phrases"; engine Tavily (§4.2, §5.2). Code: `tools/text_search.py`.

## Question taxonomy (§3.2.1)

- Type 1 Fundamental (no search); Type 2 Visual-Recognition (image search); Type 3 Information-Seeking (text search); Type 4 Multi-Faceted (both) (§3.2.1).
  - Code: `eagent_baseline/data.py` (`QuestionType`).

## Planning metrics (§3.3)

- IS-P/IS-R, TS-P/TS-R (per-tool precision/recall), Plan-acc, Param-acc, Param-sim are defined *conceptually* in §3.3. No formulas, matching procedure, thresholds, aggregation, or (for Param-sim) similarity model are given.
  - Code: `eagent_baseline/evaluation.py` (precision/recall implemented with standard set definitions; matching granularity is an assumption; Param-acc and Param-sim require injected criteria).

## Answer evaluation (§5.1)

- GPT-4o judge scores answers 0–2 given "the corresponding image, query, ground-truth answer, and the model's response" (§5.1). A-OKVQA uses answer accuracy (§5.1). The exact judge prompt/settings are not provided.
  - Code: `eagent_baseline/evaluation.py` (`AnswerJudge`; `StubAnswerJudge` is a development double, not the paper's judge).

## Datasets (§3.1, §5.2)

- Final RemPlan benchmark: 200 image-question pairs with plan trajectories and answers (§3.1).
- Planner training set: 10K samples containing images, questions, and plan annotations; "same as in Section 3.1 while the human verification and answer annotation phase are excluded" (§5.2).
- Relationship between the 10K set and the 200-pair benchmark is not specified in the paper.
  - Code: `configs/base.yaml` records `benchmark_size: 200` and `planner_train_samples: 10000`; `eagent_baseline/data.py` provides the instance schema only. No benchmark data is included.

## Reported results (context only; not reproduced)

- Abstract: "13% accuracy gain over state-of-the-art mRAG methods while reducing redundant searches by 37%."
- Table 2 (RemPlan) and Table 3 (tool-call counts) report E-Agent-fewshot and E-Agent-sft numbers. These are recorded for reference and are not reproduced here.

## Not present in the paper

- No formal Algorithm box (verified: zero occurrences of "Algorithm" in the paper text).
- No official code repository (`paper_metadata.json`: `official_code: []`).
- No appendix in the extracted artifacts.
