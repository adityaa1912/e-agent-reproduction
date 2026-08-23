from __future__ import annotations

import tests._paths

from eagent_baseline.config import BaselineConfig, default_config_path


def _config() -> BaselineConfig:
    return BaselineConfig.from_yaml(default_config_path())


FULL_PLAN_JSON = (
    '{"steps": ['
    '{"tool": "image_search", "arguments": {}},'
    '{"tool": "requery", "arguments": {}},'
    '{"tool": "text_search", "arguments": {}},'
    '{"tool": "response", "arguments": {}}]}'
)

EXPECTED_TOOL_SEQUENCE = ["image_search", "requery", "text_search", "response"]
