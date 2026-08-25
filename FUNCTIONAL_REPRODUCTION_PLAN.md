# Functional Reproduction Plan

## 0. Purpose & Scope

This document evaluates the runtime feasibility of executing a real multimodal vision-language model on the development machine and derives a practical next milestone. It does **not** modify production code, tests, configs, or documentation. It only records findings and corrects the earlier (overly optimistic) planning guidance.

---

## 1. Current Architecture

The repository preserves the paper's plan-then-execute control flow:

```
Question
  → MRAGPlanner.plan()
  → MRAGPlan (ordered step sequence)
  → TaskExecutor.execute()
  → Tool dispatch (image_search, requery, text_search, response)
  → terminal Response
```

The only component targeted for replacement in this phase is the **VisionLanguageModel provider** — currently the deterministic `StubVisionLanguageModel` / `ScriptedVisionLanguageModel` doubles in `src/eagent/models/providers/stub.py`. The `VisionLanguageModel` ABC (`ModelRequest` / `ModelResponse`) and the `EAgentModelConfig` + `ModelSpec` schema already insulate downstream code from provider-specific details. The existing `factory.py` raises `UnsupportedProviderError` for anything other than `stub`, which is the intended fail-fast boundary.

---

## 2. Hardware Snapshot (verified, not assumed)

| Component | Value | Verification method |
|-----------|-------|---------------------|
| OS | Windows 11 Home Single Language (10.0.26200) | `systeminfo` |
| CPU | AMD64, 1 logical processor (exact model not reported by `systeminfo`) | PowerShell |
| System RAM | **7 834 MiB total** (~7.6 GiB) | `Win32_OperatingSystem.TotalVisibleMemorySize` |
| Free RAM at time of check | **~425 MiB** | `FreePhysicalMemory` |
| GPU | **Intel UHD Graphics** (integrated) | `Win32_VideoController` |
| GPU VRAM | **~128 MiB** (shared system memory) | `Win32_VideoController.AdapterRAM` |
| CUDA | **Not present** | `torch.cuda.is_available()` → N/A (torch not installed) |
| Python | 3.11.15 | `python --version` |
| Python ML stack | `numpy 1.26.4`; **PyTorch not installed** | `pip list` |
| C: drive free space | **~26.4 GiB** | `Win32_LogicalDisk.FreeSpace` |

### Engineering inferences drawn from the snapshot

1. **No CUDA-capable GPU exists.** The Intel UHD is a shared-memory integrated GPU with 128 MiB of dedicated VRAM — effectively no discrete GPU. All inference must be CPU-only.
2. **Total RAM is 7.6 GiB.** The OS, Python interpreter, and any running tooling consume a portion of this before model loading begins. Available headroom for a single large model is **well under 4 GiB**.
3. **Free RAM at check time was ~425 MiB** — effectively zero headroom for any additional allocation.
4. **PyTorch is not installed.** Intalling `torch` for CPU-only use adds ~1 GiB of wheel download and unpacked size, plus all its transitive dependencies (`numpy` already present, but `typing-extensions`, `networkx`, `jinja2`, etc. add overhead).
5. **Disk space is ~26 GiB** — sufficient for small model downloads (≤ 5 GiB) but tight for an 8B-parameter model even at aggressive quantization when you account for download staging, unpacking, and PyTorch itself.

---

## 3. Model Feasibility (corrected)

The earlier plan claimed that InternVL2-8B at 8-bit quantization "could run on a modest GPU (≥ 8 GiB)." This is **incorrect for the current machine**. The corrections below reflect the verified hardware.

| Option | Execution target | Model size (FP16) | Model size (4-bit) | VRAM needed | RAM needed | Feasible on this machine? | Practical latency | Multimodal image input? |
|--------|-----------------|-------------------|---------------------|-------------|------------|---------------------------|-------------------|------------------------|
| **A.** Local CPU, InternVL2-8B (FP16) | CPU only | ~15 GiB | — | — | ~15 GiB | **NO** — exceeds total RAM by ~2× | — | — |
| **B.** Local CPU, InternVL2-8B (4-bit quantized) | CPU only | ~15 GiB | ~4 GiB | ~4–5 GiB (model + KV cache + image encoder) | ~6–8 GiB (model + Python + OS + image buffers + generation overhead) | **MARGINAL / LIKELY NOT** — 4 GiB fits within 7.6 GiB theoretical total, but with only ~425 MiB free at check time and shared GPU memory stealing from system RAM, peak usage during image encoding + autoregressive generation will likely exceed available RAM, trigger heavy swap, and/or OOM. | **Minutes per generation** on CPU (no GPU acceleration; 8B model at 4-bit still requires substantial GEMM work). A single `generate()` call producing a ~200-token plan could take 2–10+ minutes. | **Yes, in theory** — InternVL2 supports images. But CPU image encoding is slow and memory-hungry. |
| **C.** Remote inference (hosted endpoint) | HTTP client only | N/A | N/A | N/A | Minimal (~tens of MB for request/response) | **YES** — the machine only needs network access and an API key. The inference happens elsewhere. | Sub-second to a few seconds (depends on provider). | **Yes** — depends on provider's model offering. |
| **D.** Smaller local VLM (e.g. SmolVLM-256M, Qwen2-VL-2B) | CPU only | — | SmolVLM-256M: ~0.5 GiB (4-bit); Qwen2-VL-2B: ~1.5–2 GiB (4-bit) | < 1 GiB | ~1–3 GiB total | **YES for very small VLMs** (SmolVLM-256M, Florence-2). **Marginal for 2B-class VLMs** (Qwen2-VL-2B at 4-bit may fit but leaves little headroom). | Sub-second to low-single-digit seconds for generation (small model on CPU). | **Yes** — all candidate small VLMs support image input. |
| **E.** Keep the stub (deterministic doubles) | CPU only | — | — | — | Negligible | **YES** — already works; 84/84 tests pass. | Sub-millisecond per call. | **No** — stub does not perform real inference, but supports the provider interface contract. |

### Key corrected conclusions

1. **Option A is impossible** (15 GiB > 7.6 GiB total RAM).
2. **Option B is impractical** on this machine — even at 4-bit quantization the model nearly exhausts available RAM, and CPU inference latency makes it unsuitable for iterative functional validation.
3. **Option C is feasible** but requires an external API key and ongoing cost; the machine is only an HTTP client.
4. **Option D is feasible for very small VLMs** (≤ 256M parameters); 2B-class models are marginal.
5. **Option E is trivially feasible** and already operational.

---

## 4. Recommended Inference Backend (corrected)

The earlier plan recommended 🤗 Transformers for all options. That recommendation **still stands for Option C and Option D**, but **does not apply to Option B** on this hardware (the model simply will not load reliably).

| Backend | Windows CPU feasibility | Quantization support | Implementation complexity | Verdict for this machine |
|---------|------------------------|---------------------|--------------------------|--------------------------|
| 🤗 **Transformers** (`pipeline` / `AutoModelForCausalLM`) | ✅ Pure Python, no CUDA required | ✅ `bitsandbytes` (NF4/int8), GPTQ | Low — straightforward wrapper around `VisionLanguageModel` | Recommended for **Option D** (small local VLM). |
| **vLLM** | ❌ Officially Linux-only; Windows requires WSL2 | ✅ (GPU only) | Medium — server process + async client | Not applicable (no GPU, no Linux). |
| **ONNX Runtime** | ✅ Good Windows support | ✅ int8, float16 | High — requires model export and custom pipeline | Overkill for a small local VLM; considered but not recommended. |
| **Remote API client** (HTTP) | ✅ Any OS with Python `httpx` / `requests` | N/A | Low — thin HTTP wrapper around `VisionLanguageModel` | Recommended for **Option C**. |

---

## 5. Provider Architecture (real implementation design)

The existing abstraction already supports the design:

```
VisionLanguageModel (ABC)
  ├── StubVisionLanguageModel         ← already implemented
  ├── ScriptedVisionLanguageModel     ← already implemented (baseline package)
  ├── RealTransformersVisionLanguageModel   ← new: Transformers backend, CPU-compatible
  └── RealRemoteAPIVisionLanguageModel  ← new: HTTP client to hosted endpoint
```

**Design constraints (preserved):**
- The ABC (`ModelRequest` / `ModelResponse`) does **not** reference any provider-specific concepts (no HF, no CUDA, no API client types).
- The paper skeleton factory (`src/eagent/models/factory.py`) stays **stub-only**. HuggingFace CPU inference belongs in `functional_extension/` and is injected into planner/executor constructors. Do not add `real_transformers` to the reproduction factory.
- Planner and executor code call only `VisionLanguageModel.generate(request)`, never inspect the concrete provider type.
- No new model abstractions are duplicated; the `VisionLanguageModel` ABC is the single interface.

**`RealTransformersVisionLanguageModel`** (for Option D):
```python
class RealTransformersVisionLanguageModel(VisionLanguageModel):
    def __init__(self, model_id: str, quant: str = "nf4"):
        # load model_id via transformers; quantization via bitsandbytes if requested
        ...
    @property
    def model_name(self) -> str: ...
    def generate(self, request: ModelRequest) -> ModelResponse:
        # build processor inputs (text + images if present)
        # run inference on CPU
        # return ModelResponse(text=..., model_name=self.model_id, usage=..., raw=...)
```

**`RealRemoteAPIVisionLanguageModel`** (for Option C):
```python
class RealRemoteAPIVisionLanguageModel(VisionLanguageModel):
    def __init__(self, endpoint: str, api_key: str, model_name: str):
        # endpoint = e.g. "https://api.together.xyz/v1/chat/completions"
        # credentials from env vars, never hardcoded
        ...
    @property
    def model_name(self) -> str: ...
    def generate(self, request: ModelRequest) -> ModelResponse:
        # build HTTP request; stream or non-stream response
        # return ModelResponse(text=..., model_name=self.model_name, usage=..., raw=...)
```

Neither provider is registered in `src/eagent` `factory.py`. Construct them in `functional_extension` and inject the `VisionLanguageModel` instance.

---

## 6. Configuration Strategy

Extend `EAgentModelConfig` / `ModelSpec` — **do not duplicate schemas**. The existing `ModelSpec(provider, model_name)` fields are sufficient:

```yaml
# configs/development.yaml (unchanged — stub only)
mode: development
planner:
  provider: stub
  model_name: stub-planner
executor:
  provider: stub
  model_name: stub-executor
```

```yaml
# functional_extension example (not a paper config; inject the model in code)
# HuggingFaceTB/SmolVLM-256M-Instruct via RealTransformersVisionLanguageModel
# executor remains stub
```

```yaml
# configs/remote.yaml (NEW — remote API)
mode: development
planner:
  provider: remote_api
  model_name: internvl2-8b    # or whatever the provider alias is
  endpoint: https://api.together.xyz/v1/chat/completions
  api_key_env: TOGETHER_API_KEY
executor:
  provider: stub
  model_name: stub-executor
```

The `BaselineConfig` → `EAgentModelConfig` down-conversion in `eagent_baseline/models.py` passes through the new fields unchanged; the factory selects the provider. **No schema duplication.**

---

## 7. Retrieval Provider Strategy (Baidu & Tavily)

| Service | Public API status | Required interface | Paper-specific gap |
|---------|-------------------|--------------------|--------------------|
| **Baidu Image Search** | Public web service (`image.baidu.com`); no official public REST API documented. Community wrappers exist but are unofficial. | `ImageSearchTool.run(image: Image) → ImageSearchResult` — currently a deterministic stub. | **Missing**: query construction, result parsing, access credentials (if using an unofficial API endpoint). |
| **Tavily Text Search** | Public REST API (`https://api.tavily.com`) — requires API key. | `TextSearchTool.run(query: str) → TextSearchResult` — currently a deterministic stub. | **Missing**: API key, request throttling, result parsing. |

**Design note** — keep tool signatures unchanged. Future `BaiduImageSearchProvider` and `TavilyTextSearchProvider` implementations can be injected via the existing tool registration mechanism, with credentials read from environment variables (`BAIDU_API_KEY`, `TAVILY_API_KEY`). No credentials are added to this plan.

---

## 8. Security / Secrets Strategy

- **Never** hard-code API keys in source or config.
- Load credentials from environment variables (`os.getenv`); fail loudly with a clear message if the required variable is absent.
- Document required variable names in a `README` (outside the repo) and in the config's `secrets:` block (which is git-ignored).
- For local debugging, a `.env` file may be used but must be excluded from the repository (already covered by `.gitignore`).

---

## 9. Corrected Functional Milestones

| Milestone | Description | Required assets | Feasibility on this machine | Can be called "FUNCTIONAL E-AGENT VALIDATION"? |
|-----------|-------------|----------------|------------------------------|----------------------------------------------|
| **M0 — Provider-integration scaffold** | Add `RealTransformersVisionLanguageModel` + `RealRemoteAPIVisionLanguageModel` providers; wire them through the factory; run all 84 existing tests (all must pass) + new provider-wiring tests. | None (stubs still used in tests). | **YES** — zero new runtime deps. | No — still stubs. But it is the **prerequisite** for any real-model validation. |
| **M1 — Small-VLM local validation** | Run the full E-Agent pipeline with a **real** small public VLM (≤ 256M params, 4-bit quantized) as the planner. Executor remains stub. Produce a non-empty terminal response for a synthetic multimodal question. | Public VLM checkpoint (≈ 0.5 GiB download); `transformers` + `bitsandbytes` installed. | **YES** — model fits in 7.6 GiB RAM; CPU inference is slow but functional. | **Yes** — but **only if** the VLM used is explicitly documented as a **substitute**, not the paper's InternVL2-8B. Call it "FUNCTIONAL E-AGENT VALIDATION (substitute small VLM)." |
| **M2 — Remote-API validation** | Run the full pipeline with a **remote** InternVL2-8B endpoint. | API key + network + cost. | **YES** (externally) — machine is just an HTTP client. | **Yes, conditionally** — if the hosted endpoint uses InternVL2-8B. Still a **substitute** for the paper's fine-tuned checkpoint (base checkpoint ≠ fine-tuned checkpoint). |
| **M3 — Full local (paper-claimed models)** | InternVL2-8B (4-bit) locally + Qwen2-VL-72B (4-bit) locally. | ≥ 32 GiB RAM, ≥ 80 GiB GPU VRAM, or multi-GPU setup. | **NO** — hardware insufficient. | N/A until hardware is available. |

---

## 10. Risks & Blockers

| Risk | Category | Severity | Mitigation |
|------|----------|----------|------------|
| **7.6 GiB RAM insufficient for 8B model** | Hardware limitation (observed) | Critical | Defer 8B local inference; use small VLM or remote API. |
| **No CUDA GPU** | Hardware limitation (observed) | Critical | CPU-only paths only; accept slower inference. |
| **Paper-specific fine-tuned checkpoint UNVERIFIED / NOT LOCATED** | Asset audit (PAPER_SPECIFIC_ASSET_AUDIT.md) | High | Use public base checkpoint as substitute; document clearly. |
| **RemPlan benchmark BLOCKED** | Asset audit | Critical | Cannot compute paper metrics; functional validation uses synthetic questions only. |
| **10K training data BLOCKED** | Asset audit | High | Cannot train the paper's planner; use base checkpoint or small VLM. |
| **Prompts UNVERIFIED** | Asset audit | Medium | Use assumed templates (`ASSUMED_*_PROMPT_TEMPLATE`); note limitation. |
| **Metric formulas UNVERIFIED** | Asset audit | Medium | Stub implementations remain; cannot verify paper numbers. |
| **Windows + large-model quantization** | Engineering (inference) | Medium | `bitsandbytes` has Windows support but is less battle-tested than Linux; pin versions and test. |
| **CPU inference latency** | Engineering (inference) | Low | Accept for M1/M2 validation; not a blocker for functional correctness. |
| **API key cost (Option C)** | External dependency | Medium | User must provision; cost is per-token. |

---

## 11. What This Does NOT Reproduce

Regardless of which milestone is chosen, the following remain **unreproduced**:

- The **paper-specific fine-tuned InternVL2-8B** planner checkpoint (UNVERIFIED / NOT LOCATED).
- The **Qwen2-VL-72B** executor model (hardware-infeasible locally; remote option uses base checkpoint, not paper-configured).
- The **RemPlan** 200-pair benchmark (BLOCKED).
- The **10K planner-training dataset** (BLOCKED).
- The **paper's exact prompt templates** (Planner, Requery, Response, GPT-4o judge) — UNVERIFIED.
- The **exact metric formulas** (IS-P/R, TS-P/R, Plan-acc, Param-acc, Param-sim, GPT-4o judge rubric) — UNVERIFIED.
- The **live Baidu / Tavily APIs** with paper-specific configuration — BLOCKED (credentials/config missing).

A "FUNCTIONAL E-AGENT VALIDATION" milestone **explicitly does not claim paper reproduction**. It validates that the **architecture** (planner → plan → executor → tools → response) works end-to-end with a real model, using a **public base checkpoint or substitute VLM** as documented.

---

## 12. Recommended First Implementation Milestone

**M0 — Provider-integration scaffold** is the recommended first step, because:

1. It requires **zero unavailable resources** (no model download, no GPU, no API key, no network).
2. It **de-risks the provider-swap architecture** — the actual code change (new provider classes + factory extension) is tested in isolation before any real inference is attempted.
3. It is a **prerequisite** for every subsequent milestone (M1, M2, M3 all depend on the factory wiring being correct).
4. It preserves the existing 84/84 passing test baseline; no tests are broken.
5. It keeps the reproduction status **honest** ("NOT REPRODUCED") — no fake inference, no overstated feasibility.

**After M0 is complete**, the next milestone depends on a **user decision** (not on this machine's capabilities):

| If the user has… | Then proceed to… |
|------------------|-----------------|
| A desire to run a **real small VLM locally** with no external cost | **M1** — download a ≤ 256M parameter VLM (e.g. `HuggingFaceTB/SmolVLM-256M-Instruct`), install `transformers` + `bitsandbytes`, run FUNCTIONAL E-AGENT VALIDATION (substitute small VLM). |
| An **API key** for a hosted endpoint (Together AI, HF Inference Endpoints, etc.) | **M2** — use `RealRemoteAPIVisionLanguageModel` to call a remote InternVL2-8B endpoint; run FUNCTIONAL E-AGENT VALIDATION (remote, base checkpoint). |
| A **hardware upgrade** (≥ 32 GiB RAM, ≥ 80 GiB GPU VRAM) | **M3** — run the paper-claimed models locally (InternVL2-8B 4-bit + Qwen2-VL-72B 4-bit). |

---

## 13. Summary

### 1. Hardware snapshot
- **7.6 GiB system RAM**, ~425 MiB free at check time.
- **No CUDA GPU** — Intel UHD Graphics (integrated, ~128 MiB VRAM shared).
- **Python 3.11.15**, `numpy 1.26.4`; **PyTorch not installed**.
- **~26 GiB free disk** on C:.

### 2. Model feasibility
- **InternVL2-8B (FP16):** IMPOSSIBLE — 15 GiB exceeds total RAM.
- **InternVL2-8B (4-bit quantized):** MARGINAL / LIKELY NOT — ~4–5 GiB model RAM + OS + Python + image buffers pushes against 7.6 GiB limit; CPU latency is minutes per generation.
- **Small VLM (≤ 256M):** FEASIBLE — ~0.5 GiB fits comfortably.
- **Remote API:** FEASIBLE — machine is only an HTTP client.
- **Stub:** ALREADY WORKS — 84/84 tests pass.

### 3. Recommended backend
**🤗 Transformers** (with `bitsandbytes` for quantization) for local small-VLM execution; **HTTP client** (`httpx` / `requests`) for remote API execution. vLLM is excluded (Linux/GPU-only). ONNX Runtime is excluded (overkill).

### 4. Proposed provider architecture
Two new concrete providers under the existing `VisionLanguageModel` ABC:
- `RealTransformersVisionLanguageModel` — CPU-compatible, supports quantization flag, no CUDA required.
- `RealRemoteAPIVisionLanguageModel` — thin HTTP wrapper; credentials from env vars only.
Factory stays stub-only in `src/eagent`; CPU Transformers lives in `functional_extension`.

### 5. Proposed configuration strategy
Extend existing `ModelSpec(provider, model_name)` with optional provider-specific fields (`quant`, `endpoint`, `api_key_env`) via Pydantic optional fields — **no schema duplication**. New configs: `functional.yaml` (local small VLM), `remote.yaml` (API). `development.yaml` unchanged.

### 6. First achievable functional milestone
**M0 — Provider-integration scaffold** — zero hardware dependencies, de-risks the architecture, preserves 84/84 passing tests. **M1** (small VLM locally) is the first milestone that executes real multimodal inference on this machine, gated only on downloading a ≤ 256M parameter model and installing `transformers` + `bitsandbytes`.

### 7. Major risks
- 7.6 GiB RAM makes 8B-model local inference infeasible (hardware limitation, not code).
- No CUDA GPU means all inference is CPU-bound; latency is high.
- Paper-specific assets (fine-tuned checkpoint, RemPlan, 10K training data, prompts) remain BLOCKED / UNVERIFIED.
- `bitsandbytes` Windows support is less battle-tested than Linux.
- Remote API option requires user-provisioned credentials and ongoing cost.

### 8. Files created
- **`FUNCTIONAL_REPRODUCTION_PLAN.md`** — this document (created / corrected in this session).

### 9. Confirmation of no implementation
- **No production code modified** (`src/eagent/`, `eagent_baseline/`, `tests/`, `configs/`, `scripts/` untouched).
- **No tests created or modified.**
- **No model downloads initiated.**
- **No packages installed.**
- **No API calls made.**
- **No credentials added.**
- **No existing documentation modified** (only `FUNCTIONAL_REPRODUCTION_PLAN.md` created/updated).

---

## 14. M1 Runtime Decision

### 14.1 Chosen model

| Attribute | Value |
|-----------|-------|
| **Model ID** | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| **Architecture** | `Idefics3ForConditionalGeneration` (text backbone: VLlama3ForCausalLM; vision encoder: SigLIP-base) |
| **Parameter count** | ~256 M (≈ 0.3 B total) |
| **Weight file** | `model.safetensors` – **513 MiB** (single file) |
| **Repository size** | 3.54 GiB total (weights + tokenizer + configs) |
| **Dtype** | `bfloat16` (model checkpoint) |
| **Source** | [HuggingFace model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |

### 14.2 Runtime feasibility findings

| Question | Answer | Evidence |
|----------|--------|----------|
| **1. Exact architecture** | VLlama3ForCausalLM backbone (30 layers, hidden=576, 9 heads, vocab=49280) + SigLIP-base vision encoder (12 layers, hidden=768, 12 heads, image_size=512, patch_size=16) | [config.json](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/config.json) |
| **2. Exact checkpoint ID** | `HuggingFaceTB/SmolVLM-256M-Instruct` | HuggingFace model card |
| **3. CPU inference support** | **YES** – official example uses `device = "cuda" if torch.cuda.is_available() else "cpu"` | [README](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/README.md) |
| **4. Windows support** | **YES** – standard Python/PyTorch/Transformers stack; no platform restriction noted | Model card / general transformers docs |
| **5. Python requirements** | Python ≥ 3.8 (standard for transformers) | Implicit from `transformers` ecosystem |
| **6. Transformers requirements** | `transformers` (latest stable); uses `AutoProcessor`, `AutoModelForVision2Seq` | Model card |
| **7. bitsandbytes required?** | **NO** – quantization is explicitly optional: *"You can also load SmolVLM with 4/8‑bit quantization using bitsandbytes…"* | Model card |
| **8. bitsandbytes Windows CPU support** | **YES** – bitsandbytes lists Windows 11 / Server 2022+ (x86‑64, AVX2) as supported for `LLM.int8()`, `QLoRA4‑bit`, and 8‑bit optimizers. However, Windows CPU paths are noted to *lack performance optimizations*. | [GitHub repository](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| **9. Is quantization necessary for a ~256 M model?** | **NO** – the FP16 weight file is only **513 MiB**. Even FP32 (~1 GiB) fits comfortably within 7.6 GiB total RAM. Quantization would halve/in‑quarter the weight size but adds a dependency (`bitsandbytes`) with potential Windows‑CPU performance caveats. The simplicity gain of skipping quantization outweighs the memory saving. | Model card (513 MiB safetensors) |
| **10. Approximate CPU RAM requirement** | **~1.2–1.8 GiB peak** estimated: <br>• Model weights (FP16): ~513 MiB<br>• PyTorch runtime overhead: ~200–300 MiB<br>• Image preprocessing (SigLIP encoder, 512×512): ~100–200 MiB<br>• Generation buffers / KV cache: ~100–200 MiB<br>• Python + dependencies: ~100–200 MiB<br><br>**Total well within 7.6 GiB** with substantial margin. | Engineering estimate based on model size + typical PyTorch CPU overhead |
| **11. Is plain Transformers CPU simpler and more reliable than quantization?** | **YES** – plain `AutoModelForVision2Seq.from_pretrained(..., torch_dtype=torch.float32)` (or `torch.bfloat16` with fallback to float32 for CPU) requires only `transformers` + `torch` + `PIL`. No `bitsandbytes` install, no quantization config, no platform-specific performance caveats. | Principle of least surprise; avoids an additional dependency with Windows‑CPU notes |

### 14.3 Chosen M1 stack

| Attribute | Value | Rationale |
|-----------|-------|-----------|
| **Chosen model** | `HuggingFaceTB/SmolVLM-256M-Instruct` | Smallest public multimodal VLM; fits in RAM; officially supports CPU inference |
| **Chosen backend** | 🤗 **Transformers** (`AutoModelForVision2Seq`) | Official inference path; CPU-compatible; no GPU required |
| **Chosen precision** | **FP32** (explicitly forced on CPU) | Avoids bfloat16/CPU edge-case issues; 256 M model makes FP32 only ~1 GiB vs ~0.5 GiB FP16 – negligible difference on this machine. If memory is tight, FP16 is the next option. |
| **Quantization** | **None** | Not needed for a 256 M model; avoids `bitsandbytes` dependency and its Windows‑CPU performance notes |
| **Required dependencies** | `torch` (CPU), `transformers`, `Pillow` | Already have `numpy`; these are the minimal additions |
| **Image input** | `PIL.Image.open()` / `load_image()` from transformers | Model natively supports image + text input via `processor()` |
| **Generation** | `model.generate(**inputs, max_new_tokens=…)` | Standard transformers API |
| **Estimated peak RAM** | ~1.2–1.8 GiB | Well within 7.6 GiB total, with ~425 MiB free headroom at baseline (model download and install will temporarily increase memory, but peak inference usage stays well under total RAM) |

### 14.4 Expected execution path (M1)

```
synthetic multimodal Question (text + image URL)
  → RealTransformersVisionLanguageModel.generate(ModelRequest)
      ├─ Load HuggingFaceTB/SmolVLM-256M-Instruct (FP32 CPU) on first call
      ├─ Process image via SigLIP encoder + text via tokenizer
      └─ Return ModelResponse(text=<plan JSON string>, model_name="HuggingFaceTB/SmolVLM-256M-Instruct")
  → MRAGPlanner.plan(question)
      ├─ Build ModelRequest(prompt=<ASSUMED_PLANNER_PROMPT_TEMPLATE>, images=[question.images])
      ├─ Call model.generate(request)  ← real model inference
      └─ Parse returned JSON into MRAGPlan
  → TaskExecutor.execute(plan, question)
      ├─ image_search tool (stub)
      ├─ requery tool → RealTransformersVisionLanguageModel.generate()  ← real model inference
      ├─ text_search tool (stub)
      └─ response tool → RealTransformersVisionLanguageModel.generate()  ← real model inference
  → terminal Response (non-empty string)
```

**Note**: In the first implementation, only the **planner** will use the real model. The executor tools (`requery`, `response`) will continue to use the `ScriptedVisionLanguageModel` stub to keep the test scope focused. This is the smallest achievable functional milestone.

### 14.5 M1 Output Strategy

#### 14.5.1 The problem

`HuggingFaceTB/SmolVLM-256M-Instruct` is a general-purpose image‑text‑to‑text model. It is **not** fine‑tuned to emit the specific `MRAGPlan` JSON format required by the planner. A raw `generate()` call will almost certainly produce natural‑language text, not a parseable JSON plan. The milestone must therefore include a concrete strategy for making the model produce parseable output, and must define what happens when it does not.

#### 14.5.2 Prompt strategy

We do **not** invent or pretend to have the paper's undisclosed planner prompt. We label all prompt text used in this milestone as **ENGINEERING PROMPT / FUNCTIONAL VALIDATION PROMPT** to make clear it is a validation construct, not a paper artifact.

The existing `ASSUMED_PLANNER_PROMPT_TEMPLATE` in `eagent_baseline/planner.py` will be **replaced** (or supplemented) by a new `FUNCTIONAL_VALIDATION_PLANNER_PROMPT` constant, placed in the same module, that:

- Is a short, explicit JSON‑output instruction: *"Given the question and image, output only a JSON object with a single key `steps` whose value is an ordered array of objects with keys `tool` and `arguments`. Use only these tool names: `image_search`, `requery`, `text_search`, `response`. The last step must be `response`."*
- Embeds the question text and image description (the model receives the image directly; the prompt text describes the image if needed).
- Contains no paper-specific terminology that would imply fidelity to the paper's prompt.

The constant is clearly named and documented as an engineering prompt, not a paper prompt.

#### 14.5.3 Parser behaviour

The existing `MRAGPlan.from_json()` parser is kept **unchanged**. It continues to:
- Parse a JSON string into a list of `PlanStep` objects.
- Enforce `tool` and `arguments` keys.
- Validate the terminal‑Response invariant (`MRAGPlan.validate()`).

The planner calls `MRAGPlan.from_json(response.text)` and then `.validate()`. **No silent fallback to the stub planner is implemented.** If parsing fails or validation fails, the planner raises a clear, informative exception (e.g. `PlanParseError(response.text, error=...)`) that includes the raw model output. This preserves the paper's single‑pass contract — the planner makes exactly one model call and does not retry with a different prompt or fall back to deterministic behaviour.

#### 14.5.4 Failure handling

When the model emits malformed plan text:

| Scenario | Behaviour |
|----------|-----------|
| Non‑JSON output (plain text) | `PlanParseError` raised with raw text included; test records the failure. |
| JSON but missing required keys | `PlanParseError` raised with validation details. |
| JSON with valid keys but fails `validate()` (no terminal RESPONSE) | `PlanParseError` raised; execution does not proceed. |
| Empty or null output | `PlanParseError` raised. |

In every failure case, `plan_call_count` is still exactly **1** (the invariant is measured before any parsing attempt), and the error is documented rather than silently swallowed.

#### 14.5.5 What this does and does not claim

| Aspect | Statement |
|--------|-----------|
| Prompt origin | **ENGINEERING PROMPT** — a functional‑validation construct, not the paper's planner prompt. |
| Paper fidelity | **Not claimed.** The prompt is designed to elicit structured JSON from a general VLM; it does not reproduce the paper's planning behaviour. |
| Architecture preservation | **Yes.** The plan‑then‑execute flow, single‑pass planner contract, and `MRAGPlan` parser are all preserved unchanged. |
| No silent fallback | **Enforced.** Malformed output raises; the stub is never invoked as a fallback. |

#### 14.5.6 Definition of successful validation

A run is considered **successful** when all of the following are true:

1. The full E‑Agent pipeline executes end‑to‑end without `UnsupportedProviderError` or runtime crash.
2. The planner provider loads the real `HuggingFaceTB/SmolVLM-256M-Instruct` model on CPU.
3. The planner receives a `ModelRequest` with at least one image (multimodal) and returns a `ModelResponse` whose `text` field contains a valid JSON plan (parseable into an `MRAGPlan`).
4. The `TaskExecutor` dispatches all plan steps to their registered tools.
5. The terminal `Response` is a non‑empty string.
6. A test (new, in the test suite) asserts that `plan_call_count == 1` **and** the model name in the response is `"HuggingFaceTB/SmolVLM-256M-Instruct"` (or whatever `model_name` the provider reports), proving the real model was invoked exactly once, not the stub.

If any of these conditions fail, the milestone is **not** achieved and the failure mode is recorded (including the raw model output for post‑hoc analysis).

### 14.6 Why this is the safest CPU path

| Factor | Choice | Why |
|--------|--------|-----|
| Model size | 256 M params, 513 MiB weights | Smallest publicly available multimodal VLM; FP32 loads in ~1 GiB total, far under 7.6 GiB |
| No quantization | Skip bitsandbytes | Avoids Windows‑CPU performance caveats and an additional dependency for a model that does not need it |
| Plain Transformers | No vLLM / ONNX export | Single dependency chain; official inference code works out‑of‑the‑box |
| FP32 precision | Explicit `torch.float32` on CPU | Avoids bfloat16‑on‑CPU edge cases; modern CPUs handle float32 efficiently |
| No external API | Purely local | No network latency, no API key cost, fully offline after model download |
| Minimal config change | Inject `RealTransformersVisionLanguageModel` from `functional_extension`; do not extend the paper factory |

### 14.7 Exact dependencies that will be needed (not yet installed)

| Package | Purpose | Approx. download size |
|---------|---------|-----------------------|
| `torch` (CPU wheel) | PyTorch runtime for model loading & inference | ~2 GiB (CPU-only wheel) |
| `transformers` ≥ 4.45 | Model loading, tokenizer, processor | ~30 MiB + transitive deps |
| `Pillow` | Image loading (already required by `transformers`) | Already available or ~10 MiB |
| `HuggingFaceTB/SmolVLM-256M-Instruct` (model weights) | The real multimodal VLM | 3.54 GiB (total repo; safetensors = 513 MiB) |

**Total disk needed**: ~5.5 GiB (torch wheel + model weights). **Total RAM at peak inference**: ~1.2–1.8 GiB.

### 14.8 What this does NOT reproduce

This milestone is explicitly **not** a paper reproduction. It validates the **functional architecture** with a real multimodal model, but:

- The model is a **substitute** (`SmolVLM-256M-Instruct`, 256 M), not the paper's claimed planner model (`InternVL2-8B`, 8 B).
- The checkpoint is a **public base model**, not a fine‑tuned checkpoint (the paper's fine‑tuned planner is UNVERIFIED / NOT LOCATED).
- The prompts remain the assumed templates (`ASSUMED_PLANNER_PROMPT_TEMPLATE`, etc.).
- The executor tools still use deterministic stubs in this milestone.
- No paper metrics (Plan‑acc, IS‑P/R, etc.) are computed against the RemPlan benchmark (which is BLOCKED).

This is a **FUNCTIONAL E‑AGENT VALIDATION** – proof that the architecture can run with a real model – not a reproduction of the paper's experimental results.

---

*Prepared by Claude Code (analysis of publicly-available information, verified hardware snapshot, and repository state).*
