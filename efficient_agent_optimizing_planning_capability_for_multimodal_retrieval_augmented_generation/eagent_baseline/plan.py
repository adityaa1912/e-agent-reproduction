from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ToolName(str, Enum):
    IMAGE_SEARCH = "image_search"
    TEXT_SEARCH = "text_search"
    REQUERY = "requery"
    RESPONSE = "response"


class PlanValidationError(ValueError):
    pass


@dataclass
class PlanStep:
    tool: ToolName
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool.value, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        if "tool" not in data:
            raise PlanValidationError("Plan step is missing the 'tool' field.")
        try:
            tool = ToolName(data["tool"])
        except ValueError as error:
            raise PlanValidationError(f"Unknown tool name: {data['tool']!r}") from error
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise PlanValidationError("Plan step 'arguments' must be a mapping.")
        return cls(tool=tool, arguments=dict(arguments))


@dataclass
class MRAGPlan:
    steps: List[PlanStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MRAGPlan":
        if not isinstance(data, dict) or "steps" not in data:
            raise PlanValidationError("Plan must be a mapping containing 'steps'.")
        raw_steps = data["steps"]
        if not isinstance(raw_steps, list):
            raise PlanValidationError("Plan 'steps' must be a list.")
        return cls(steps=[PlanStep.from_dict(step) for step in raw_steps])

    @classmethod
    def from_json(cls, text: str) -> "MRAGPlan":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise PlanValidationError("Plan text is not valid JSON.") from error
        return cls.from_dict(data)

    def tool_sequence(self) -> List[ToolName]:
        return [step.tool for step in self.steps]

    def validate(self) -> None:
        if not self.steps:
            raise PlanValidationError("Plan must contain at least one step.")
        if self.steps[-1].tool is not ToolName.RESPONSE:
            raise PlanValidationError("Plan must terminate with a Response step.")
        for step in self.steps[:-1]:
            if step.tool is ToolName.RESPONSE:
                raise PlanValidationError("Response may only appear as the terminal step.")
