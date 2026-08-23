# PAPER‑SPECIFIC ASSET AUDIT

**Purpose** – Determine whether the assets that the paper *Efficient Agent: Optimizing Planning Capability for Multimodal Retrieval Augmented Generation* (arXiv:2508.08816) requires for a faithful reproduction are publicly available.

---

## A. Scope of the audit
The audit focuses on **paper‑specific** assets (i.e., items that the authors explicitly mention as part of the experimental setup) and distinguishes them from publicly‑available base resources. The assets examined are:

1. **Fine‑tuned planner checkpoint** – InternVL2‑8B fine‑tuned on the 10 K planner‑training set.
2. **Executor model checkpoint** – Qwen2‑VL‑72B configuration used for the MLLM tools.
3. **RemPlan benchmark** – 200‑pair multimodal QA dataset with planning trajectories.
4. **10 K planner‑training dataset** – Image‑question‑plan triples used for fine‑tuning the planner.
5. **Prompt templates** – Planner, Requery, Response, and GPT‑4o judge prompts.
6. **Live search services** – Baidu Image Search API and Tavily Text‑Search API.
7. **Metric implementations** – Exact formulas for IS‑P/R, TS‑P/R, Plan‑acc, Param‑acc, Param‑sim, and the GPT‑4o judge scoring rubric.
8. **Any supplementary code repository** – Official implementation released by the authors.

---

## B. Methodology
* **Web‑scraping** of the arXiv abstract/HTML page for any "Code / Data / Media" links.
* **GitHub search** for "Efficient Agent", "E‑Agent", "RemPlan", "InternVL2‑8B fine‑tuned".
* **HuggingFace search** for model checkpoints or datasets matching the paper‑specific names.
* **DagsHub / OpenGVLab** checks for the RemPlan benchmark.
* **General web search** for any author‑hosted pages or supplemental material.

All searches were performed in August 2026. No private or pay‑walled sources were accessed.

---

## C. Asset inventory & status
| # | Asset | Status | Evidence / Source |
|---|-------|--------|--------------------|
| 1 | **Fine‑tuned planner checkpoint** (InternVL2‑8B + 10 K fine‑tune) | **UNVERIFIED / NOT LOCATED** | No URL found on arXiv page, GitHub, or HuggingFace. The arXiv "Code, Data, Media" panel is empty【WebFetch 1】. GitHub search returned no matching repo【WebSearch 1】. |
| 2 | **Executor model checkpoint** (Qwen2‑VL‑72B + paper‑specific config) | **UNVERIFIED / NOT LOCATED** | Same as above – no repository or model card found. |
| 3 | **RemPlan benchmark (200‑pair QA)** | **BLOCKED / NOT AVAILABLE** | No dataset link on arXiv, DagsHub, or HuggingFace. Search for "RemPlan" returns no public dataset【WebSearch 2】. |
| 4 | **10 K planner‑training data** | **BLOCKED / NOT AVAILABLE** | No public release; the paper only mentions the size. |
| 5 | **Prompt templates** (Planner, Requery, Response, GPT‑4o judge) | **UNVERIFIED** | Prompt text is not disclosed in the paper; the repository uses assumed placeholders (e.g., `ASSUMED_REQUERY_PROMPT_TEMPLATE`). |
| 6 | **Baidu Image Search API** | **BLOCKED** (service exists; paper‑specific config/credentials unavailable) | Public service exists (image.baidu.com). The paper’s specific usage configuration, query parameters, and access credentials are not disclosed. Reproduction blocked by missing paper‑specific implementation details and credentials. |
| 7 | **Tavily Text‑Search API** | **BLOCKED** (PUBLIC API available; paper‑specific config/credentials unavailable) | Public Tavily API exists (tavily.com). The paper’s specific configuration, API credentials, and result‑handling behavior are not disclosed. Reproduction blocked by missing paper‑specific configuration/credentials. |
| 8 | **Metric formulas** (IS‑P/R, TS‑P/R, Plan‑acc, Param‑acc, Param‑sim) | **UNVERIFIED** | Paper defines metrics conceptually but does not publish exact formulas; the code uses stub implementations. |
| 9 | **Official code repository** | **NOT AVAILABLE** | The arXiv page lists no repository; GitHub search for the authors (Yuechen Wang, Yuming Qiao, etc.) yields no E‑Agent repo【WebSearch 3】. |
|10| **Public base checkpoints** (InternVL2‑8B, Qwen2‑VL‑72B) | **PUBLIC BASE AVAILABLE** | HuggingFace hosts the base models: `OpenGVLab/InternVL2-8B`【https://huggingface.co/OpenGVLab/InternVL2-8B】 and `Qwen/Qwen2-VL-72B`【https://huggingface.co/Qwen/Qwen2-VL-72B】. These are **not** the paper‑specific fine‑tuned versions. |

---

## D. Summary of findings
* **No paper‑specific code, checkpoints, or datasets were located in the authoritative/public sources searched during this audit.**
* The only publicly available assets are the *base* language‑vision models (InternVL2‑8B, Qwen2‑VL‑72B) and the paper’s PDF/LaTeX source.
* All live services (Baidu, Tavily) and the GPT‑4o judge are proprietary and require credentials that are not disclosed.
* Prompt templates and metric formulas remain undisclosed, making a faithful functional reproduction impossible without author‑provided material.

---

## E. Evidence (hyperlinks)
- **arXiv abstract / HTML** – shows empty "Code, Data, Media" panel【WebFetch 1】
- **GitHub search** for "Efficient Agent" / "E‑Agent" – no relevant repositories【WebSearch 1】
- **GitHub search** for author names – no E‑Agent repo【WebSearch 3】
- **HuggingFace base model** InternVL2‑8B – https://huggingface.co/OpenGVLab/InternVL2-8B
- **HuggingFace base model** Qwen2‑VL‑72B – https://huggingface.co/Qwen/Qwen2-VL-72B
- **Search for RemPlan** – no public dataset found【WebSearch 2】

---

## F. Conclusion
The paper’s **paper‑specific assets** (fine‑tuned checkpoint, executor configuration, RemPlan benchmark, training data, prompts, metric definitions, and live service access) are **not publicly available** as of August 2026. Consequently, a **faithful reproduction** of the experimental results is **blocked**. The repository only provides a deterministic skeleton that mirrors the architecture and control‑flow contract.

---

*Prepared by Claude Code (analysis of publicly‑available information).*
