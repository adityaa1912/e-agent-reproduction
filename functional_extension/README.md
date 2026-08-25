# Functional extension (not the paper skeleton)

CPU HuggingFace inference for a **substitute** small VLM (`HuggingFaceTB/SmolVLM-256M-Instruct`). This package is **not** part of the E-Agent paper reproduction.

The paper-only code (`src/eagent`, `eagent_baseline`) remains stub-only. Inject `RealTransformersVisionLanguageModel` into `MRAGPlanner` / MLLM tools; do not add a HuggingFace provider to `eagent.models.factory`.

```bash
pip install -r efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation/requirements.txt
pip install -r functional_extension/requirements.txt
python -m unittest discover -s functional_extension/tests -t functional_extension
```

`transformers>=4.49.0` is required for Idefics3 / SmolVLM processors. Weights download only on the first real `generate()` call.
