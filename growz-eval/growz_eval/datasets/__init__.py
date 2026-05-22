"""Dataset acquisition: registry, downloaders, manager."""

from growz_eval.datasets.manager import DatasetManager
from growz_eval.datasets.registry import DATASETS, DatasetSpec

__all__ = ["DATASETS", "DatasetSpec", "DatasetManager"]
