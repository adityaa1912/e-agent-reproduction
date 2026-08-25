from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
_BASELINE = (
    _REPO_ROOT
    / "efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation"
)
_HF_CACHE = _HERE / ".venv" / "hf_cache"

for _extra in (str(_SRC), str(_BASELINE), str(_HERE)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_HF_CACHE)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from PIL import Image as PILImage

from eagent.common.types import Image, Question

from eagent_baseline.config import BaselineConfig, default_config_path
from eagent_baseline.models import build_executor_model

from eagent_functional.planner_integration import run_functional_pipeline
from eagent_functional.transformers_provider import RealTransformersVisionLanguageModel

_FIXTURE_PATH = _HF_CACHE / "planner_integration_fixture.png"
_QUESTION_TEXT = "What is shown in this image?"


def _write_fixture() -> None:
    canvas = PILImage.new("RGB", (128, 128), (30, 60, 160))
    for x in range(32, 96):
        for y in range(32, 96):
            canvas.putpixel((x, y), (200, 40, 40))
    canvas.save(_FIXTURE_PATH, format="PNG")


def main() -> int:
    print("M1 STEP 2B real SmolVLM planner integration smoke test")
    print("planner model = real SmolVLM; executor tools = deterministic stub; no fallback")
    print()

    _write_fixture()
    question = Question(text=_QUESTION_TEXT, images=[Image(path=str(_FIXTURE_PATH))])
    print(f"fixture_path: {_FIXTURE_PATH}")
    print(f"question_text: {question.text}")
    print(f"question_is_multimodal: {question.is_multimodal}")
    print(f"question_image_count: {len(question.images)}")
    print()

    planner_model = RealTransformersVisionLanguageModel()
    load_start = time.perf_counter()
    planner_model._ensure_loaded()
    load_seconds = time.perf_counter() - load_start
    print(f"planner_model_load_seconds: {load_seconds:.2f}")

    executor_model = build_executor_model(BaselineConfig.from_yaml(default_config_path()))
    print(f"executor_model_name: {executor_model.model_name}")
    print()

    result = run_functional_pipeline(planner_model, executor_model, question)

    print("=== 2. PROMPT USED ===")
    print(result.prompt)
    print()
    print("=== 3. RAW PLANNER OUTPUT (repr) ===")
    print(repr(result.raw_output))
    print("--- raw planner output (rendered) ---")
    print(result.raw_output)
    print()
    print(f"planner_response_model_name: {result.model_name}")
    print(f"raw_output_classification: {result.classification}")
    print(f"raw_output_nonempty: {bool(result.raw_output and result.raw_output.strip())}")
    print()
    print(f"=== 4. json_parse_succeeded: {result.parse_ok}")
    print(f"parse_error: {result.parse_error}")
    print()

    if result.parse_ok:
        print("=== 5. PLAN CONTENTS ===")
        print(result.plan.to_json())
        print(f"tool_sequence: {result.tool_sequence}")
    else:
        print("=== 5. PLAN CONTENTS === (none; parsing failed)")
    print()

    print(f"=== 6. plan_call_count: {result.plan_call_count}")
    print(f"generate_call_count: {result.generate_call_count}")
    print()

    print("=== 7. EXECUTOR RESULT ===")
    print(f"executor_error: {result.executor_error}")
    print(f"executor_tool_trace: {result.executor_tool_trace}")
    print()
    print("=== 8. TERMINAL RESPONSE ===")
    print(repr(result.terminal_response))
    print()

    print(f"=== 9. total_pipeline_latency_seconds: {result.latency_seconds:.2f}")
    print(f"planner_model_load_seconds: {load_seconds:.2f}")
    print()
    print(f"=== 10. stub_fallback_occurred: {result.stub_fallback_occurred}")
    print()

    model_name_ok = (
        result.model_name == RealTransformersVisionLanguageModel.DEFAULT_MODEL_ID
    )
    terminal_ok = bool(result.terminal_response and result.terminal_response.strip())
    raw_ok = bool(result.raw_output and result.raw_output.strip())
    success = (
        result.parse_ok
        and result.plan is not None
        and result.plan_call_count == 1
        and result.generate_call_count == 1
        and model_name_ok
        and result.executor_error is None
        and terminal_ok
        and raw_ok
        and not result.stub_fallback_occurred
    )

    print("=== SUCCESS-CRITERIA CHECK ===")
    print(f"multimodal_question_reached_provider: {question.is_multimodal and result.generate_call_count == 1}")
    print(f"exactly_one_planner_call: {result.plan_call_count == 1 and result.generate_call_count == 1}")
    print(f"model_name_is_smolvlm_256m: {model_name_ok}")
    print(f"model_output_nonempty: {raw_ok}")
    print(f"output_parsed_into_mragplan: {result.parse_ok}")
    print(f"executor_completed_plan: {result.parse_ok and result.executor_error is None}")
    print(f"terminal_response_nonempty: {terminal_ok}")
    print(f"no_stub_planner_fallback: {not result.stub_fallback_occurred}")
    print()
    print(f"M1_STEP_2B_SUCCESS: {success}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
