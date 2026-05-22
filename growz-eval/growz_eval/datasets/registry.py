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
            id="ccdphd11",
            source="kaggle",
            locator="dhamur/cotton-plant-diseases",
            crop="cotton",
            priority="high",
            note="11 classes of cotton diseases/pests, ~19k field images",
        ),
        DatasetSpec(
            id="ldd_grape",
            source="github",
            locator="https://github.com/lab-rossi/LDD",
            crop="grape",
            priority="high",
            note="Grape diseases with instance segmentation",
        ),
        DatasetSpec(
            id="grape_original",
            source="kaggle",
            locator="rm1000/grape-disease-dataset-original",
            crop="grape",
            priority="high",
            note="4 classes: ESCA, Leaf Blight, Black Rot, Healthy",
        ),
        DatasetSpec(
            id="fieldplant",
            source="kaggle",
            locator="khaledelsayedibrahim/fieldplant",
            crop="mixed",
            priority="medium",
            note="5,170 field photos under plant pathologist supervision",
        ),
        DatasetSpec(
            id="plantdoc",
            source="github",
            locator="https://github.com/pratikkayal/PlantDoc-Dataset",
            crop="mixed",
            priority="medium",
            note="2,598 field images across 13 species",
        ),
    ]
}
