import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eagent_baseline

from eagent.common.types import Image, Question
from eagent_baseline.config import BaselineConfig, default_config_path
from eagent_baseline.executor import TaskExecutor
from eagent_baseline.models import ScriptedVisionLanguageModel, build_executor_model
from eagent_baseline.plan import MRAGPlan, ToolName
from eagent_baseline.planner import MRAGPlanner
from eagent_baseline.tools import ImageSearchTool, RequeryTool, ResponseTool, TextSearchTool


FULL_PLAN_JSON = (
    '{"steps": ['
    '{"tool": "image_search", "arguments": {}},'
    '{"tool": "requery", "arguments": {}},'
    '{"tool": "text_search", "arguments": {}},'
    '{"tool": "response", "arguments": {}}]}'
)


def _synthetic_question() -> Question:
    return Question(
        text="Which club is this person playing for?",
        images=[Image(url="http://example.invalid/person.png")],
    )


def _build_executor() -> TaskExecutor:
    model = build_executor_model(BaselineConfig.from_yaml(default_config_path()))
    return TaskExecutor(
        {
            ToolName.IMAGE_SEARCH: ImageSearchTool(),
            ToolName.TEXT_SEARCH: TextSearchTool(),
            ToolName.REQUERY: RequeryTool(model),
            ToolName.RESPONSE: ResponseTool(model),
        }
    )


def _mean_stdev(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def _bench_planner(iterations: int) -> tuple[float, float, MRAGPlan]:
    question = _synthetic_question()
    planner = MRAGPlanner(ScriptedVisionLanguageModel("dev-planner", [FULL_PLAN_JSON]))
    times_ms: list[float] = []
    plan: MRAGPlan | None = None
    for _ in range(iterations):
        start = time.perf_counter()
        plan = planner.plan(question)
        times_ms.append((time.perf_counter() - start) * 1000)
    assert plan is not None
    return (*_mean_stdev(times_ms), plan)


def _bench_parse_validate(iterations: int) -> tuple[float, float]:
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        MRAGPlan.from_json(FULL_PLAN_JSON).validate()
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _bench_executor(iterations: int, plan: MRAGPlan) -> tuple[float, float]:
    executor = _build_executor()
    question = _synthetic_question()
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        executor.execute(plan, question)
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _bench_image_search(iterations: int) -> tuple[float, float]:
    tool = ImageSearchTool()
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        tool.run(image=Image(url="http://example.invalid/a.png"))
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _bench_text_search(iterations: int) -> tuple[float, float]:
    tool = TextSearchTool()
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        tool.run(query="basketball finals")
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _bench_requery(iterations: int) -> tuple[float, float]:
    model = build_executor_model(BaselineConfig.from_yaml(default_config_path()))
    tool = RequeryTool(model)
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        tool.run(question="who is this?", image=Image(url="x"), image_search_results=[])
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _bench_response(iterations: int) -> tuple[float, float]:
    model = build_executor_model(BaselineConfig.from_yaml(default_config_path()))
    tool = ResponseTool(model)
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        tool.run(question="who is this?", image=Image(url="x"), image_search_results=[], text_search_results=[])
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _bench_serialization(iterations: int) -> tuple[float, float]:
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = MRAGPlan.from_json(FULL_PLAN_JSON).to_json()
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _bench_config_build(iterations: int) -> tuple[float, float]:
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        BaselineConfig.from_yaml(default_config_path())
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _bench_executor_rebuild(iterations: int) -> tuple[float, float]:
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _build_executor()
        times_ms.append((time.perf_counter() - start) * 1000)
    return _mean_stdev(times_ms)


def _run_test_suite(cwd: str, args: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "unittest"] + args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return (time.perf_counter() - start) * 1000


def main() -> None:
    print("deterministic development/stub baseline profiling")
    print("no network, no GPU, no API keys; stub providers only")
    print()
    N = 2000
    planner_mean, planner_sd, plan = _bench_planner(N)
    parse_mean, parse_sd = _bench_parse_validate(N)
    executor_mean, executor_sd = _bench_executor(N, plan)
    image_mean, image_sd = _bench_image_search(N)
    text_mean, text_sd = _bench_text_search(N)
    requery_mean, requery_sd = _bench_requery(N)
    response_mean, response_sd = _bench_response(N)
    serial_mean, serial_sd = _bench_serialization(N)
    config_mean, config_sd = _bench_config_build(N)
    rebuild_mean, rebuild_sd = _bench_executor_rebuild(N)
    baseline_time = _run_test_suite(Path(__file__).resolve().parents[1] / "efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation", ["discover", "-s", "tests", "-t", "."])
    model_time = _run_test_suite(Path(__file__).resolve().parents[1], ["discover", "-s", "tests", "-t", "."])
    total_mean = planner_mean + parse_mean + executor_mean
    total_sd = (planner_sd ** 2 + parse_sd ** 2 + executor_sd ** 2) ** 0.5
    print(f"iterations: {N}")
    print()
    print(f"planner_mean_ms: {planner_mean:.3f} (±{planner_sd:.3f})")
    print(f"plan_parse_validate_mean_ms: {parse_mean:.3f} (±{parse_sd:.3f})")
    print(f"executor_mean_ms: {executor_mean:.3f} (±{executor_sd:.3f})")
    print()
    print(f"per-tool mean ms:")
    print(f"  image_search: {image_mean:.3f} (±{image_sd:.3f})")
    print(f"  text_search: {text_mean:.3f} (±{text_sd:.3f})")
    print(f"  requery: {requery_mean:.3f} (±{requery_sd:.3f})")
    print(f"  response: {response_mean:.3f} (±{response_sd:.3f})")
    print()
    print(f"serialization_mean_ms: {serial_mean:.3f} (±{serial_sd:.3f})")
    print(f"config_build_mean_ms: {config_mean:.3f} (±{config_sd:.3f})")
    print(f"executor_rebuild_mean_ms: {rebuild_mean:.3f} (±{rebuild_sd:.3f})")
    print()
    print(f"total_end_to_end_ms (planner + parse + executor): {total_mean:.3f} (±{total_sd:.3f})")
    print()
    print(f"test_suite_baseline_ms: {baseline_time:.1f}")
    print(f"test_suite_model_layer_ms: {model_time:.1f}")


if __name__ == "__main__":
    main()
