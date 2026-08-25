# Functional Planner Model Selection

This document selects a **more capable public multimodal VLM** to act as the
planner model in the `functional_extension/` pipeline, after the current
baseline (`HuggingFaceTB/SmolVLM-256M-Instruct`) was shown unable to produce a
valid `MRAGPlan`.

Scope and hard constraints (unchanged from the failed run):

- This is **FUNCTIONAL VALIDATION only** — it is **not** a reproduction of the
  paper's models, prompts, data, or metrics. Every candidate here is a
  **substitute** VLM, not the paper's planner (InternVL2-8B) or executor
  (Qwen2-VL-72B).
- No change to `MRAGPlan.validate()`, no output-repair layer, no plan
  reordering, no planner retry, no weakening of the single-pass invariant, no
  change to `src/eagent` or `eagent_baseline`. The model is swapped by
  **dependency injection** into `FunctionalValidationPlanner`, exactly as today.
- The parser stays fail-loud: malformed or mis-ordered output raises; there is
  no stub fallback. A candidate "passes" only by emitting a genuinely valid,
  `RESPONSE`-terminal plan on its own.
- **Nothing is installed or downloaded in this phase.** This is a selection
  document only.

Machine (verified in `FUNCTIONAL_REPRODUCTION_PLAN.md` §2): Windows 11,
CPU-only (no CUDA; integrated Intel UHD), **~7.6 GiB total RAM**, ~26 GiB free
disk, Python 3.11, `transformers>=4.49,<5.0` (4.57 present), torch CPU.

---

## Failed Baseline

| Attribute | Value |
|-----------|-------|
| Checkpoint | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Params | ~256 M (Idefics3: SmolLM2 backbone + SigLIP vision) |
| Wiring status | **Works.** Real CPU/FP32 model injected into `FunctionalValidationPlanner`; single real call (`plan_call_count == 1`, `generate_call_count == 1`); correct model id; multimodal `Question` reaches the provider; non-empty output; no stub fallback. |
| Why it failed | **Model capability ceiling, not a wiring/prompt-format bug.** Two deterministic (temp 0) real calls: (1) with a `"<name>"` placeholder in the prompt, the model copied the placeholder as a tool name → `PlanValidationError: Unknown tool name: '<name>'`; (2) with a corrected prompt (concrete example, explicit tool list, explicit "last step must be response"), it produced valid JSON with correct step shape and valid tool names but placed `response` **first** → `PlanValidationError: Plan must terminate with a Response step.` |
| Conclusion | SmolVLM-256M gets JSON **syntax** right but cannot satisfy the **compositional ordering** invariant. A substantially stronger instruction-follower is required. Do not re-attempt 256M or add repair/retry/fallback. |

The bar a replacement must clear is therefore **instruction-following /
compositional ordering** ("the last step must be `response`"), not raw JSON
syntax — the 256 M model already had the syntax.

---

## Candidate Models

Three realistic candidates spanning the local↔remote space. `qwen-vl-utils`,
`flash_attention_2`, `bitsandbytes`, and any GPU path are **not** assumed
(none are available on this machine).

| Model | Local CPU | Remote | Multimodal | JSON/Instruction | Cost | Recommendation |
|-------|-----------|--------|------------|------------------|------|----------------|
| `HuggingFaceTB/SmolVLM-500M-Instruct` | **Yes** — FP32 ≈ 2 GiB, fits with headroom | Rarely hosted (too small; self-host is the norm) | Yes (image+text) | **Weak–Moderate** — same family that failed at 256 M, only 2× larger; ordering still uncertain | Free (offline) | Offline hedge / fallback only |
| `Qwen/Qwen2-VL-2B-Instruct` | **Marginal** — FP32 ≈ 8.8 GiB (exceeds RAM); BF16 ≈ 4.4 GiB (OOM-prone, slow) | **Yes** — cheap hosted endpoints (verify 2B availability; 7B/72B are more universally hosted) | Yes (image+text/video) | **Strong** — instruction-tuned "visual agent"; reliable short-JSON | ~cents for validation volume (verify $/Mtok) | **RECOMMENDED — remote-primary, local as marginal offline fallback** |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | **No** — BF16 ≈ 21 GiB, far over RAM | **Yes** — widely hosted (Together / Fireworks / OpenRouter / HF Inference Providers) | Yes (image+text) | **Strong** — reliable structured output | low-$/Mtok; still cents for our volume | Escalation only if 2B proves insufficient; overkill |

### Candidate 1 — `HuggingFaceTB/SmolVLM-500M-Instruct`

- **Exact checkpoint:** `HuggingFaceTB/SmolVLM-500M-Instruct`
- **Params:** ~0.5 B (0.5B params; SmolLM2-360M language backbone + SigLIP vision; Idefics3 architecture).
- **Local CPU feasibility:** **Yes.** Same Idefics3 architecture and loader (`AutoModelForVision2Seq`) as the current 256 M model; CPU inference is officially supported (device auto-selects `cpu`, switches to "eager" attention on CPU).
- **Approx RAM:** FP32 weights ≈ 2.0 GiB; peak working set (weights + SigLIP image encoding + KV cache + PyTorch/OS) ≈ 2.5–3.0 GiB. Comfortably under 7.6 GiB. (Model card cites ~1.23 GiB to run one image, which is a BF16 runtime figure.)
- **Remote availability:** Minimal — 0.5 B models are usually self-hosted, not offered on paid endpoints. Treat as a local-only option.
- **API/provider requirements:** None (local).
- **Expected cost:** Free (one-time weight download ≈ a few hundred MiB, offline thereafter).
- **Transformers compatibility:** Native under the installed `transformers>=4.49`. `AutoModelForVision2Seq` + `AutoProcessor` + `apply_chat_template` with interleaved image/text — **identical** to the existing provider.
- **Structured/JSON reliability:** Uncertain. It is the same SmolVLM family and training recipe that failed the ordering invariant at 256 M; doubling parameters is an incremental hedge, not a capability-tier jump. It may still place `response` non-terminally.
- **Implementation complexity:** **Trivial.** `RealTransformersVisionLanguageModel` already takes a `model_id` argument — pass `"HuggingFaceTB/SmolVLM-500M-Instruct"`. Zero provider-code change; reuse `FunctionalValidationPlanner` and the existing prompt unchanged.
- **Can it realistically run our functional planner prompt?** Mechanically yes; the open question is whether it satisfies `RESPONSE`-terminal ordering, which the 256 M sibling could not.

### Candidate 2 — `Qwen/Qwen2-VL-2B-Instruct`  *(recommended)*

- **Exact checkpoint:** `Qwen/Qwen2-VL-2B-Instruct`
- **Params:** ~2.2 B (repo labeled "2B"; instruction-tuned; text backbone + ViT vision encoder with Naive Dynamic Resolution + M-ROPE).
- **Local CPU feasibility:** **Marginal / not recommended.** FP32 ≈ 8.8 GiB exceeds total RAM (7.6 GiB). BF16 ≈ 4.4 GiB weights plus vision-encoder activations, KV cache, processor, and Windows/Python overhead realistically pushes peak to ~5.5–6.5+ GiB on a machine that was observed with only a few hundred MiB free — high risk of swap/OOM, and CPU BF16 matmul is slow (many minutes per generation). This is exactly the "don't call it local just because a smaller dtype exists" case: assessed honestly, local is not a reliable path here.
- **Approx RAM:** FP32 ≈ 8.8 GiB (won't fit); BF16 ≈ 4.4 GiB weights, ~5.5–6.5 GiB peak (marginal). Remote: negligible (HTTP client only).
- **Remote availability:** **Yes.** Qwen2-VL is broadly served; confirm a provider hosts the **2B** variant specifically (many list 7B/72B — `Qwen/Qwen2-VL-7B-Instruct` is a same-family, same-interface drop-in escalation if 2B is unavailable). Candidate hosts: HuggingFace Inference Providers, OpenRouter, Together, Alibaba Model Studio/DashScope.
- **API/provider requirements:** An OpenAI-compatible chat/completions endpoint (or HF Inference Providers). API key from an environment variable (e.g. `OPENROUTER_API_KEY` / `TOGETHER_API_KEY` / `HF_TOKEN`), never hardcoded; fail loudly if absent (per `FUNCTIONAL_REPRODUCTION_PLAN.md` §8). Image passed as a base64 `data:` URL in the message content.
- **Expected cost:** Pay-per-token, images billed separately. For functional validation (a handful of short calls) total cost is a **fraction of a cent**; 2 B-class hosted rates are roughly ~$0.05–$0.30 / Mtok. **Verify current pricing/availability at implementation time** (live pricing could not be confirmed in this phase).
- **Transformers compatibility (for the local fallback):** Supported by the installed `transformers` (Qwen2-VL support landed in ≥ 4.45; 4.57 is present). Loader is `Qwen2VLForConditionalGeneration` (or `AutoModelForImageTextToText`), **not** `AutoModelForVision2Seq` — a one-line loader change in the provider. Chat-templating with image+text content is the same pattern already used.
- **Structured/JSON reliability:** **Strong.** Qwen2-VL is explicitly instruction-tuned with "visual agent" behavior and is a dependable emitter of short JSON given an explicit schema and rule set — precisely our prompt's shape. It is the smallest candidate that represents a genuine capability-tier jump over SmolVLM.
- **Implementation complexity:** Remote = **moderate**: implement `RealRemoteAPIVisionLanguageModel(VisionLanguageModel)` (already designed in `FUNCTIONAL_REPRODUCTION_PLAN.md` §5) — build OpenAI-style messages (image data URL + the existing `FUNCTIONAL_VALIDATION_PLANNER_PROMPT`), POST at `temperature=0`, map `choices[0].message.content` → `ModelResponse.text`. New dependency is a thin HTTP client (`httpx`, or stdlib `urllib`). Local fallback = **low**: reuse `RealTransformersVisionLanguageModel` with a loader-class + dtype tweak.
- **Can it realistically run our functional planner prompt?** **Yes, very likely** — no prompt change and no repair needed; the existing prompt already gives a concrete example, the tool list, and the "last step must be `response`" rule.

### Candidate 3 — `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Exact checkpoint:** `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Params:** ~11 B (`MllamaForConditionalGeneration`).
- **Local CPU feasibility:** **No.** BF16 ≈ 21 GiB, FP32 ≈ 42 GiB — both far exceed 7.6 GiB. Not runnable locally under any dtype available here.
- **Approx RAM (remote):** Negligible on this machine (HTTP client only).
- **Remote availability:** **Yes** — widely hosted (Together, Fireworks, OpenRouter, HF Inference Providers).
- **API/provider requirements:** Same remote provider pattern as Candidate 2. Note the checkpoint is **gated** on HuggingFace (Llama 3.2 community license; historically some regional/EU availability limits) — remote hosts typically abstract this away, but self-serving weights requires license acceptance.
- **Expected cost:** low single-digit $/Mtok for an 11 B-class model; still only cents at validation volume. **Verify current pricing.**
- **Transformers compatibility:** `MllamaForConditionalGeneration` (transformers ≥ 4.45) — relevant only if self-hosted, which is infeasible here; remote use needs no local transformers.
- **Structured/JSON reliability:** **Strong** — reliable structured output and instruction following.
- **Implementation complexity:** Same remote provider as Candidate 2 (moderate). No local path.
- **Can it realistically run our functional planner prompt?** **Yes, near-certain** — but it is heavier, costlier, gated, and has no local fallback, making it overkill for emitting a tiny ordered JSON plan.

---

## Recommended Model

**`Qwen/Qwen2-VL-2B-Instruct`, executed via a remote OpenAI-compatible endpoint
as the primary path, with local CPU (BF16) documented as a marginal,
offline-only fallback.**

It is the **smallest** candidate that is a genuine capability-tier jump over
SmolVLM (strong, "visual-agent" instruction following and reliable short-JSON),
it is a real public Apache-2.0 checkpoint, remote inference is cheap and
accessible while keeping this machine a thin client (the plan's Option C /
`RealRemoteAPIVisionLanguageModel`, already designed), and — critically — it is
**realistically usable on this hardware** the way the task requires (remotely),
without pretending an 8 B+ model or a RAM-marginal local BF16 load is
practical here.

If a chosen provider does not host the 2B variant, `Qwen/Qwen2-VL-7B-Instruct`
is a same-family, same-interface drop-in at slightly higher cost. Candidate 3
(`Llama-3.2-11B-Vision-Instruct`) is the escalation only if 2B/7B prove
insufficient at the ordering task.

SmolVLM-500M is retained only as a zero-cost, zero-interface-change **offline
hedge**; it is not the recommendation because, being the same family that
already failed at 256 M, it is not "substantially more capable" in the
dimension that matters (compositional ordering).

---

## Why It Is Better Than SmolVLM-256M

- **The exact failure mode is instruction-following, and this is where Qwen2-VL
  is strong.** 256 M produced correct JSON but put `response` first; a 2 B
  instruction-tuned "visual agent" reliably honors an explicit "last step must
  be `response`" rule and a concrete schema example — the ordering constraint
  the 256 M model could not respect.
- **~9× the parameters and a stronger training recipe** (agentic/instruction
  tuning, dynamic-resolution vision, M-ROPE), versus a 256 M general captioner.
- **Reliable short-JSON emission** given a schema, so the fail-loud parser
  should pass on the model's own output — no repair, retry, or reordering, all
  of which remain prohibited.
- **Realistic on this machine** via remote inference (sub-second to a few
  seconds per call) instead of minutes of CPU generation, enabling actual
  iterative validation.

## Implementation Path

No paper/core changes; injection only; parser and single-pass invariant
untouched. When the user authorizes the next step:

1. **Add a remote provider** `RealRemoteAPIVisionLanguageModel(VisionLanguageModel)`
   under `functional_extension/eagent_functional/` — implements the same ABC
   (`model_name`, `generate(ModelRequest) -> ModelResponse`). Build
   OpenAI-style `messages` (a base64 image `data:` URL + the existing
   `FUNCTIONAL_VALIDATION_PLANNER_PROMPT`), POST with `temperature=0` and
   `max_tokens=FUNCTIONAL_VALIDATION_PLANNER_MAX_TOKENS`, map the returned text
   into `ModelResponse.text`. Credentials from an env var; raise clearly if
   missing. New dep: a thin HTTP client (`httpx` or stdlib `urllib`) in
   `functional_extension/requirements.txt` only.
2. **Reuse everything else unchanged:** inject the remote model into
   `FunctionalValidationPlanner` (override of `build_request` only), keep
   `run_functional_pipeline`, the `RawCapturingModel` seam, the existing
   `MRAGPlan` parser, `MRAGPlan.validate()`, and the stub-backed executor. The
   executor tools (`requery`, `response`) stay on the deterministic stub
   executor model — only the **planner** model changes.
3. **Add a gated real test** (analogous to `RealSmolVlmEndToEndTest`), skipped
   unless an env flag **and** the API key are present, so ordinary offline
   `unittest` discovery never makes a network call. Assert
   `plan_call_count == 1`, correct `model_name`, non-empty raw output, no stub
   fallback, a parseable `RESPONSE`-terminal `MRAGPlan`, and a non-empty
   terminal response.
4. **Optional offline fallback (marginal):** run `Qwen/Qwen2-VL-2B-Instruct`
   through the existing `RealTransformersVisionLanguageModel` with the loader
   class changed to `Qwen2VLForConditionalGeneration` /
   `AutoModelForImageTextToText` and dtype `bfloat16`. Expect slow generation
   and RAM pressure; suitable only for a one-off local check, not iteration.

## What It Still Does NOT Reproduce

- **Not the paper's planner.** The paper's planner is a fine-tuned InternVL2-8B
  (UNVERIFIED / NOT LOCATED per `PAPER_SPECIFIC_ASSET_AUDIT.md`); Qwen2-VL-2B is
  an unrelated **substitute** used only to exercise the architecture.
- **Not the paper's executor.** Qwen2-VL here is a 2 B substitute for the
  *planner* role; it is not the paper's Qwen2-VL-**72B** executor, and the
  executor tools remain deterministic stubs.
- **Not the paper's prompt.** The planner prompt stays the explicitly-labeled
  engineering `FUNCTIONAL_VALIDATION_PLANNER_PROMPT`, not the paper's undisclosed
  template.
- **No paper metrics or data.** RemPlan (BLOCKED), the 10 K planner-training set
  (BLOCKED), and the metric formulas (UNVERIFIED) are untouched; no Plan-acc /
  IS-P/R / TS-P/R / Param-acc / Param-sim is computed.
- **No live retrieval.** Baidu Image Search / Tavily remain stubs.

This is FUNCTIONAL E-AGENT VALIDATION with a substitute VLM — proof the
architecture runs end-to-end with a real model — **not** a reproduction of the
paper's results.

## Risks

- **Provider hosts 7B/72B but not 2B.** Mitigation: use `Qwen2-VL-7B-Instruct`
  (same family/interface) or another listed host; confirm before wiring.
- **Network / API key / cost dependency** (new for this repo, which is
  otherwise offline). Mitigation: env-var credentials, fail-loud on absence,
  tiny call volume; keep the offline SmolVLM-500M hedge available.
- **Weaker determinism than local greedy decoding.** A remote endpoint at
  `temperature=0` is effectively greedy but the provider may not guarantee
  byte-identical output across calls. Acceptable for a short JSON plan; note it
  and do not add retries to "stabilize" output.
- **A capable model can still occasionally deviate** (e.g., wrap JSON in a code
  fence). The parser stays fail-loud by design — that is a correct outcome to
  record, **not** a trigger to add repair/retry/fallback. A single clean pass
  is sufficient to validate the architecture.
- **Local Qwen2-VL-2B may OOM or be unusably slow** on 7.6 GiB CPU. Mitigation:
  treat local as a marginal fallback only; prefer remote.
- **Pricing/availability not verified live in this phase.** Re-check the
  provider's model list and per-token (and per-image) pricing before
  implementation.
- **Substitution risk / honesty.** None of this narrows the gap to the paper's
  models; keep the "not reproduced" status explicit in all docs.
