"""Typed registry of source datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Source = Literal["kaggle", "github"]
Priority = Literal["high", "medium"]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    id: str
    source: Source
    locator: str  # kaggle slug or github URL
    crop: str
    priority: Priority
    note: str


DATASETS: dict[str, DatasetSpec] = {
    spec.id: spec
    for spec in [
        DatasetSpec(
            id="cotton_disease",
            source="kaggle",
            locator="dhamur/cotton-plant-disease",
            crop="cotton",
            priority="high",
            note="Cotton plant diseases (~2,500 images)",
        ),
        DatasetSpec(
            id="cotton_leaf",
            source="kaggle",
            locator="seroshkarim/cotton-leaf-disease-dataset",
            crop="cotton",
            priority="high",
            note="Cotton leaf: curl virus, bacterial blight, fusarium wilt, healthy (~2,300)",
        ),
        DatasetSpec(
            id="grape_original",
            source="kaggle",
            locator="rm1000/grape-disease-dataset-original",
            crop="grape",
            priority="high",
            note="4 classes: ESCA, Leaf Blight, Black Rot, Healthy (~9,000 images)",
        ),
        DatasetSpec(
            id="plantvillage",
            source="kaggle",
            locator="mohitsingh1804/plantvillage",
            crop="mixed",
            priority="medium",
            note="PlantVillage — 38 classes incl. tomato/potato/grape (~54k images)",
        ),
        DatasetSpec(
            id="plantdoc",
            source="github",
            locator="https://github.com/pratikkayal/PlantDoc-Dataset.git",
            crop="mixed",
            priority="medium",
            note="2,598 real-world field images across 13 species",
        ),
    ]
}
