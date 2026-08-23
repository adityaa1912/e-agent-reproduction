# Efficient Agent: Optimizing Planning Capability for Multimodal Retrieval Augmented Generation

Paper2Code baseline for [Efficient Agent (E-Agent)](https://arxiv.org/abs/2508.08816) — Yuechen Wang, Yuming Qiao, Dan Meng, Jun Yang, Haonan Lu, Zhenyu Yang, Xudong Zhang.

arXiv: `2508.08816`

## Reproduction status

**NOT REPRODUCED.** No experiment from the paper has been run. This repository is a faithful *skeleton* of the E-Agent plan-then-execute architecture where the paper specifies it, plus clearly labelled scaffolding and development doubles where it does not. See `REPRODUCTION_NOTES.md` before using anything here.

## What this implements

E-Agent is a decoupled *plan-then-execute* multimodal-RAG agent: an mRAG planner produces a complete structured plan in a single pass, and a tool-aware Task Executor carries it out via search tools and MLLM tools (Requery, Response), ending in a terminal Response. This baseline implements that control-flow contract, the plan representation, the tool interfaces, and the evaluation/training/data interfaces that the paper supports. It does **not** implement the trained InternVL2-8B planner or the Qwen2-VL-72B tool backbone (those are referenced through the existing provider-independent model layer).

```
Multimodal input -> mRAG Planner -> single-pass structured mRAG plan
                 -> Task Executor -> tool-aware execution (search + MLLM tools) -> final response
```

The RemPlan benchmark is a separate contribution and is represented here only as a data schema and metric interfaces — no benchmark data is included.

## Architecture / file structure

```
efficient_agent_.../
├── README.md
├── REPRODUCTION_NOTES.md      # scope, assumptions, deviations, blockers
├── PAPER_SPEC.md             # every paper-derived claim with a section reference
├── VERIFICATION.md           # what the tests check and how to run them
├── requirements.txt
├── configs/
│   ├── base.yaml             # development configuration (stub providers)
│   └── research.yaml         # records the paper's research setup (not runnable)
├── eagent_baseline/
│   ├── _bootstrap.py         # puts the repo's src/ (Step 5 model layer) on sys.path
│   ├── config.py             # typed configuration
│   ├── models.py             # integration with the Step 5 eagent model layer
│   ├── plan.py               # structured mRAG plan representation
│   ├── planner.py            # mRAG planner contract (single-pass)
│   ├── executor.py           # Task Executor
│   ├── agent.py              # plan-then-execute composition
│   ├── data.py               # RemPlan instance schema + question taxonomy
│   ├── evaluation.py         # metric interfaces (IS/TS-P/R, Plan-acc, Param-acc/sim, Ans.)
│   ├── training.py           # planner training scaffold (no fabricated recipe)
│   └── tools/                # base, image_search, text_search, requery, response
└── tests/                    # offline tests (no GPU/network/keys/data)
```

## Model layer integration

This baseline reuses the existing Step 5 provider-independent model layer (`src/eagent`). Planner and executor models are constructed via `eagent.models.factory` and, in development mode, resolve to the `stub` provider. The real research providers (InternVL2-8B, Qwen2-VL-72B) are not implemented and are not runnable.

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

```python
from eagent_baseline.config import BaselineConfig, default_config_path
from eagent_baseline.models import build_executor_model, ScriptedVisionLanguageModel
from eagent_baseline.planner import MRAGPlanner
from eagent_baseline.executor import TaskExecutor
from eagent_baseline.plan import ToolName
from eagent_baseline.tools import ImageSearchTool, TextSearchTool, RequeryTool, ResponseTool
from eagent.common.types import Image, Question

config = BaselineConfig.from_yaml(default_config_path())
executor_model = build_executor_model(config)

planner_model = ScriptedVisionLanguageModel(
    "dev-planner",
    ['{"steps": [{"tool": "image_search", "arguments": {}}, {"tool": "response", "arguments": {}}]}'],
)
planner = MRAGPlanner(planner_model)
executor = TaskExecutor({
    ToolName.IMAGE_SEARCH: ImageSearchTool(),
    ToolName.TEXT_SEARCH: TextSearchTool(),
    ToolName.REQUERY: RequeryTool(executor_model),
    ToolName.RESPONSE: ResponseTool(executor_model),
})

question = Question(text="Who is this person?", images=[Image(url="http://example.invalid/a.png")])
plan = planner.plan(question)
state = executor.execute(plan, question)
print(state.final_response)
```

## Citation

```bibtex
@article{wang2025eagent,
  title={Efficient Agent: Optimizing Planning Capability for Multimodal Retrieval Augmented Generation},
  author={Wang, Yuechen and Qiao, Yuming and Meng, Dan and Yang, Jun and Lu, Haonan and Yang, Zhenyu and Zhang, Xudong},
  journal={arXiv preprint arXiv:2508.08816},
  year={2025}
}
```
