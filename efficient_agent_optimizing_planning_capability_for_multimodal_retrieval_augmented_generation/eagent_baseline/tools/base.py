from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


class ToolError(RuntimeError):
    pass


@dataclass
class ToolResult:
    tool: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError
