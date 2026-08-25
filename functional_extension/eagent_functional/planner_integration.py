from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_SRC = _REPO_ROOT / "src"
_BASELINE = (
    _REPO_ROOT
    / "efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation"
)
for _extra in (str(_SRC), str(_BASELINE)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from eagent.common.types import Question
from eagent.models.protocols import ModelRequest, ModelResponse, VisionLanguageModel

from eagent_baseline.executor import ExecutionState, ExecutorError, TaskExecutor
from eagent_baseline.plan import MRAGPlan, PlanValidationError, ToolName
from eagent_baseline.planner import MRAGPlanner
from eagent_baseline.tools import (
    ImageSearchTool,
    RequeryTool,
    ResponseTool,
    TextSearchTool,
)


FUNCTIONAL_VALIDATION_PLANNER_PROMPT = (
    "You output a JSON plan for a multimodal retrieval agent.\n"
    "You are given one image and one question about it.\n"
    "Reply with only a JSON object of this exact form:\n"
    "{{\"steps\": [{{\"tool\": \"response\", \"arguments\": {{}}}}]}}\n"
    "Rules:\n"
    "- \"steps\" is a list of step objects.\n"
    "- Each step object has a \"tool\" string and an \"arguments\" object.\n"
    "- \"tool\" must be one of: image_search, text_search, requery, response.\n"
    "- The last step must have \"tool\" set to response.\n"
    "- Output only the JSON object with no markdown, no code fences, no extra text.\n"
    "Question: {question}"
)

FUNCTIONAL_VALIDATION_PLANNER_MAX_TOKENS = 200


class FunctionalValidationPlanner(MRAGPlanner):
    def build_request(self, question: Question) -> ModelRequest:
        prompt = FUNCTIONAL_VALIDATION_PLANNER_PROMPT.format(question=question.text)
        return ModelRequest(
            prompt=prompt,
            images=list(question.images),
            temperature=0.0,
            max_tokens=FUNCTIONAL_VALIDATION_PLANNER_MAX_TOKENS,
        )


class RawCapturingModel(VisionLanguageModel):
    def __init__(self, inner: VisionLanguageModel) -> None:
        self._inner = inner
        self.last_response: Optional[ModelResponse] = None
        self.generate_call_count = 0

    @property
    def inner(self) -> VisionLanguageModel:
        return self._inner

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def generate(self, request: ModelRequest) -> ModelResponse:
        response = self._inner.generate(request)
        self.last_response = response
        self.generate_call_count += 1
        return response


def build_stub_backed_executor(executor_model: VisionLanguageModel) -> TaskExecutor:
    return TaskExecutor(
        {
            ToolName.IMAGE_SEARCH: ImageSearchTool(),
            ToolName.TEXT_SEARCH: TextSearchTool(),
            ToolName.REQUERY: RequeryTool(executor_model),
            ToolName.RESPONSE: ResponseTool(executor_model),
        }
    )


def classify_raw_output(raw: Optional[str]) -> str:
    if raw is None:
        return "no_output"
    stripped = raw.strip()
    if not stripped:
        return "empty"
    if stripped.startswith("```"):
        return "markdown_fenced"
    try:
        data = json.loads(stripped)
    except ValueError:
        return "non_json_natural_language_or_invalid"
    if isinstance(data, dict) and "steps" in data:
        return "valid_json_plan"
    return "valid_json_wrong_shape"


def detect_stub_fallback(raw: Optional[str], model_name: Optional[str]) -> bool:
    if model_name is not None and "stub" in model_name.lower():
        return True
    if raw is not None and raw.lstrip().startswith("[stub:"):
        return True
    return False


@dataclass
class FunctionalRunResult:
    prompt: str
    raw_output: Optional[str]
    model_name: Optional[str]
    parse_ok: bool
    parse_error: Optional[str]
    classification: str
    plan: Optional[MRAGPlan]
    tool_sequence: Optional[List[str]]
    plan_call_count: int
    generate_call_count: int
    executor_state: Optional[ExecutionState]
    executor_tool_trace: Optional[List[str]]
    executor_error: Optional[str]
    terminal_response: Optional[str]
    stub_fallback_occurred: bool
    latency_seconds: float


def run_functional_pipeline(
    planner_model: VisionLanguageModel,
    executor_model: VisionLanguageModel,
    question: Question,
) -> FunctionalRunResult:
    capturing = RawCapturingModel(planner_model)
    planner = FunctionalValidationPlanner(capturing)
    executor = build_stub_backed_executor(executor_model)

    prompt = planner.build_request(question).prompt

    start = time.perf_counter()
    plan: Optional[MRAGPlan] = None
    parse_ok = False
    parse_error: Optional[str] = None
    try:
        plan = planner.plan(question)
        parse_ok = True
    except PlanValidationError as error:
        parse_error = f"{type(error).__name__}: {error}"

    raw = capturing.last_response.text if capturing.last_response is not None else None
    model_name = (
        capturing.last_response.model_name
        if capturing.last_response is not None
        else None
    )
    classification = classify_raw_output(raw)

    executor_state: Optional[ExecutionState] = None
    executor_trace: Optional[List[str]] = None
    executor_error: Optional[str] = None
    terminal: Optional[str] = None
    if parse_ok and plan is not None:
        try:
            executor_state = executor.execute(plan, question)
            terminal = executor_state.final_response
            executor_trace = [result.tool for result in executor_state.trace]
        except ExecutorError as error:
            executor_error = f"{type(error).__name__}: {error}"

    latency = time.perf_counter() - start

    return FunctionalRunResult(
        prompt=prompt,
        raw_output=raw,
        model_name=model_name,
        parse_ok=parse_ok,
        parse_error=parse_error,
        classification=classification,
        plan=plan,
        tool_sequence=(
            [tool.value for tool in plan.tool_sequence()] if plan is not None else None
        ),
        plan_call_count=planner.plan_call_count,
        generate_call_count=capturing.generate_call_count,
        executor_state=executor_state,
        executor_tool_trace=executor_trace,
        executor_error=executor_error,
        terminal_response=terminal,
        stub_fallback_occurred=detect_stub_fallback(raw, model_name),
        latency_seconds=latency,
    )
