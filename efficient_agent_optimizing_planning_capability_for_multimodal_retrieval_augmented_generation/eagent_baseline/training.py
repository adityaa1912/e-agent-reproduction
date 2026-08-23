from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from eagent_baseline.data import RemPlanInstance


@dataclass(frozen=True)
class PlannerTrainingScaffold:
    base_model: str
    train_samples: int
    data_fields: Tuple[str, ...] = ("image", "question", "plan")

    def describe(self) -> dict:
        return {
            "base_model": self.base_model,
            "train_samples": self.train_samples,
            "data_fields": list(self.data_fields),
            "recipe_available": False,
        }

    def prepare(self, instances: Iterable[RemPlanInstance]) -> int:
        count = 0
        for instance in instances:
            instance.validate()
            count += 1
        return count

    def train(self) -> None:
        raise NotImplementedError(
            "The planner fine-tuning recipe is not specified by the paper. "
            "Objective, optimizer, learning rate, scheduler, epochs, batch size, "
            "precision, seed, hardware, and data split are all unspecified, and "
            "neither the 10K training set nor a training/validation split is "
            "available. This is a labelled scaffold, not a faithful training loop."
        )
