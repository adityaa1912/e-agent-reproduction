# Remote Model Endpoint Audit — `Qwen/Qwen2-VL-2B-Instruct`

**Milestone:** M1 STEP 2C — verify whether a real hosted endpoint for the
**exact** selected model is currently available before implementing a remote
provider.

**Verdict up front:** **No turnkey serverless (pay-per-token), OpenAI-compatible
hosted endpoint for the exact `Qwen/Qwen2-VL-2B-Instruct` could be verified.**
The exact 2B is available only via (a) **Fireworks AI on-demand dedicated GPU**
(serverless explicitly *not* supported; GPU-hour cost; OpenAI-compat
unconfirmed) or (b) **self-hosting the open weights** (vLLM/Transformers).
Neither is the lightweight thin-client endpoint the remote path assumed.
Per the task rules: **do not implement a remote provider yet**; the candidate's
turnkey hosted availability **remains UNVERIFIED**; closest technically valid
substitutes are identified below and clearly marked as substitutes.

This is FUNCTIONAL VALIDATION scoping only — not paper reproduction. Nothing was
installed; no API keys were added; no paid API calls were made.

---

## Methodology & tooling limitations (read this — it bounds confidence)

- **Live web search was non-functional this session** (returned fallback text /
  "temporarily unavailable"). Findings therefore rely on directly fetched
  authoritative pages, not search snippets.
- **JS-heavy catalog pages did not render** (OpenRouter model list, Together
  model list). Their fetched content contained implausible/hallucinated model
  names (e.g. "Qwen3.8-2.4T-A95B", "Kimi K3", "Gemma 4 31B") and was
  **discarded, not used**. Those providers are marked **UNVERIFIED**, not
  "confirmed absent."
- **High-confidence sources** (reliably server-rendered or direct model-slug
  fetches): HuggingFace model cards, the official QwenLM GitHub README, the
  Alibaba Model Studio English docs, and direct DeepInfra/Fireworks model URLs.
- The Alibaba **Chinese** model-list page refused connection (`ECONNREFUSED`);
  the English Model Studio docs were used instead.

---

## Provider audit table

| Provider | Exact 2B model available? | Multimodal input | API format | Authentication | Current availability | Cost | OpenAI-compatible? |
|----------|---------------------------|------------------|------------|----------------|----------------------|------|--------------------|
| **Hugging Face Inference Providers** | **No** — card states *"This model isn't deployed by any Inference Provider"* | n/a | n/a | n/a | Not served by any HF partner provider (same for the 7B) | n/a | n/a |
| **Fireworks AI** | **Yes** — `accounts/fireworks/models/qwen2-vl-2b-instruct` (2.44B, "Ready") | **Yes** (image "Supported") | **On-demand dedicated-GPU deployment** — **serverless "Not supported"** | Fireworks API key **+ provision an on-demand GPU deployment** | On-demand dedicated GPU only (no serverless per-token) | **GPU-hour** (no per-token price listed; verify) | **Not stated on page — must confirm** |
| **Alibaba Model Studio / DashScope** | **No** — `qwen2-vl-2b-instruct` not listed; lineup is Qwen3-VL / Qwen2.5-VL | Yes (for current VL models) | OpenAI-compatible `.../compatible-mode/v1` | DashScope API key | Superseded by Qwen2.5-VL / Qwen3-VL; the 2B is not an API model | Per-token (successors, **not** the 2B) | Yes (for successors, **not** the 2B) |
| **OpenRouter** | **UNVERIFIED** — page did not render; no Qwen2-VL entry observed | ? | (OpenAI-compatible if present) | API key | Could not confirm | Could not confirm | (yes if present) |
| **Together AI** | **UNVERIFIED** — partial/garbled render; no Qwen2-VL / 2.5-VL observed | ? | (OpenAI-compatible if present) | API key | Could not confirm | Could not confirm | (yes if present) |
| **DeepInfra** | **No** — direct model URL returned **HTTP 404** | n/a | n/a | n/a | Not hosted | n/a | n/a |
| **Self-host (vLLM / SGLang / Transformers)** | **Yes** — open weights; card shows `vllm serve "Qwen/Qwen2-VL-2B-Instruct"` | Yes | vLLM exposes **OpenAI-compatible** `/v1` | none (your server) | Requires **user-provisioned** compute; marginal on this 7.6 GiB CPU box | Your own compute | Yes (vLLM) |

---

## Evidence

- **HF card (exact 2B):** `https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct` —
  Inference Providers widget: *"This model isn't deployed by any Inference
  Provider."* Only local/self-host apps listed (Transformers, vLLM, SGLang,
  Docker).
- **HF card (7B, for the fallback question):**
  `https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct` — same string, *"This model
  isn't deployed by any Inference Provider."* → the whole Qwen2-VL family is
  self-host-only on HF.
- **Official Qwen GitHub:** `https://github.com/QwenLM/Qwen2-VL` — now presents
  **Qwen3-VL**; Qwen2-VL appears only as a past release (HF weights). The only
  documented hosted API model ID is `qwen3-vl-235b-a22b-instruct` via DashScope
  `https://dashscope.aliyuncs.com/compatible-mode/v1`. No Qwen2-VL hosted API,
  no 2B API model ID.
- **Alibaba Model Studio (English):**
  `https://www.alibabacloud.com/help/en/model-studio/vision` — current VL
  lineup is Qwen3-VL (`qwen3-vl-plus`, `qwen3-vl-flash`,
  `qwen3-vl-235b-a22b-*`) and Qwen2.5-VL. `qwen2-vl-2b-instruct` (and 7B/72B)
  **not present**.
- **Fireworks AI:** `https://fireworks.ai/models/fireworks/qwen2-vl-2b-instruct`
  — valid page, "Ready". Model ID `accounts/fireworks/models/qwen2-vl-2b-instruct`,
  params **2.44B**, context 32.7k, **image input Supported**, **Serverless "Not
  supported"** (on-demand dedicated GPU only), function-calling/embeddings/
  fine-tuning "Not supported", created 12/2/2024, provider Qwen. No per-token
  price and no OpenAI-compat statement on the page.
- **DeepInfra:** `https://deepinfra.com/Qwen/Qwen2-VL-2B-Instruct` — **HTTP 404**.

---

## Why the exact 2B has no clean hosted endpoint

By August 2026 the **Qwen2-VL** generation has been **superseded** by
**Qwen2.5-VL** and **Qwen3-VL**. Consequently:

- HuggingFace Inference Providers dropped the whole Qwen2-VL family (2B *and*
  7B show "not deployed by any Inference Provider").
- Alibaba's own Model Studio now lists only the successor families.
- The exact 2B survives as **open weights** (self-host) and in Fireworks'
  catalog **only as an on-demand dedicated-GPU deployment**, not a serverless
  per-token endpoint.

The remote path was chosen precisely to be a cheap, turnkey thin client. That
option does not exist for this exact checkpoint today.

---

## Closest technically valid alternatives (clearly marked as SUBSTITUTES — not selected, not a silent swap)

These are **not** `Qwen2-VL-2B-Instruct`. They are recorded so the user can
choose a direction; each would need its own STEP-2C-style verification of a
concrete endpoint + price before implementation.

1. **Preserve the EXACT 2B, accept provisioned compute:**
   - **Fireworks AI on-demand** (`accounts/fireworks/models/qwen2-vl-2b-instruct`)
     — exact model, image supported; requires an on-demand GPU deployment
     (GPU-hour cost) and confirmation of OpenAI-compatibility. Overkill/costly
     for a handful of validation calls.
   - **Self-host via vLLM** — exact model, OpenAI-compatible `/v1`; requires a
     user-provisioned GPU (the 7.6 GiB CPU box remains marginal, per
     `MODEL_SELECTION.md`).
2. **Turnkey serverless, same Qwen VL lineage (successor generation):**
   - **`Qwen2.5-VL-3B-Instruct`** — closest size to 2B, direct architectural
     successor; **`Qwen2.5-VL-7B-Instruct`** — wider hosting; or Alibaba's
     current small hosted VL model **`qwen3-vl-flash`**. Hosting is *likely* via
     Alibaba Model Studio (OpenAI-compatible `.../compatible-mode/v1`) and
     third-party providers, but a specific endpoint + price was **not verified**
     this pass.
3. **Turnkey serverless, cross-family (already vetted in `MODEL_SELECTION.md`):**
   - **`meta-llama/Llama-3.2-11B-Vision-Instruct`** — widely hosted serverless;
     heavier and costlier, but the most reliably available managed endpoint.

Providers not confirmable this pass (search down / SPA render failed) that are
worth re-checking if pursuing a substitute: OpenRouter, Together, plus
Chinese-market hosts (e.g. SiliconFlow) that historically served small Qwen
models.

---

## Final report

1. **Exact provider selected:** **None** as a turnkey serverless endpoint. The
   only provider carrying the *exact* 2B is **Fireworks AI**, and only via
   **on-demand dedicated GPU** (serverless not supported) — recorded but **not
   greenlit** for this lightweight use case.
2. **Exact model ID:** target `Qwen/Qwen2-VL-2B-Instruct`; on Fireworks it is
   `accounts/fireworks/models/qwen2-vl-2b-instruct` (2.44B).
3. **Evidence URLs:** HF 2B card; HF 7B card; QwenLM/Qwen2-VL GitHub (→Qwen3-VL);
   Alibaba Model Studio vision doc; Fireworks model page; DeepInfra 404 (all
   listed under **Evidence** above).
4. **API format:** no turnkey serverless OpenAI-compatible endpoint verified for
   the exact 2B. Fireworks path = on-demand deployment API (**OpenAI-compat
   unconfirmed**); self-host/vLLM = OpenAI-compatible `/v1`; Alibaba successors =
   OpenAI-compatible `.../compatible-mode/v1` (but not the 2B).
5. **Authentication requirement:** none applies to a (nonexistent) turnkey 2B
   endpoint. Fireworks = API key **+** on-demand GPU deployment; DashScope
   successors = DashScope API key; self-host = none.
6. **Current availability:** the exact 2B is **NOT** available as serverless
   pay-per-token on any verified provider (HF: none; Alibaba: superseded;
   DeepInfra: 404; OpenRouter/Together: unverified). It is available only via
   Fireworks on-demand GPU or self-hosting. The Qwen2-VL family is superseded by
   Qwen2.5-VL / Qwen3-VL.
7. **Estimated cost (if publicly documented):** none documented for the exact 2B
   (no serverless host lists a per-token price). Fireworks on-demand = GPU-hour
   billing (figure not on the model page; must verify). Substitute pricing not
   verified this pass.
8. **Can we proceed to implementation?** **No** for the intended lightweight,
   per-token remote provider — no such endpoint exists for the exact 2B.
   **Conditional** otherwise: proceed **only if** the user opts into either
   (a) Fireworks on-demand GPU (paid, hourly) *and* we first confirm its
   OpenAI-compatible API, or (b) self-hosting via vLLM — or **explicitly
   approves a clearly-marked substitute** (Qwen2.5-VL-3B/7B, Qwen3-VL-Flash, or
   Llama-3.2-11B-Vision), which then needs its own endpoint+price verification.
9. **If not, why not:** the exact `Qwen/Qwen2-VL-2B-Instruct` has no verified
   turnkey serverless OpenAI-compatible hosted endpoint; it survives only as
   open weights (self-host) or a Fireworks on-demand GPU deployment. Task rules
   forbid implementing a remote provider without a verified usable endpoint and
   forbid silently substituting another model — so the candidate is reported
   **UNVERIFIED for managed hosting**, and the substitutes above are surfaced for
   an explicit user decision rather than chosen automatically.

**STOP** — audit complete; no implementation performed.
