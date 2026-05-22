"""Heuristic classifiers: folder-name -> crop, folder-name -> disease tier."""

from __future__ import annotations

from growz_eval.config import COMMON_DISEASES, CROP_DISTRIBUTION, HEALTHY_KEYWORDS, KNOWN_CROPS


class CropClassifier:
    """Decides which crop bucket a folder-name belongs to."""

    def classify(self, label: str, crop_hint: str) -> str:
        if crop_hint != "mixed":
            return crop_hint if crop_hint in CROP_DISTRIBUTION else "other"

        label_l = label.lower()
        for crop in KNOWN_CROPS:
            if crop in label_l:
                resolved = "corn" if crop == "maize" else crop
                return resolved if resolved in CROP_DISTRIBUTION else "other"
        return "other"


class TierClassifier:
    """Decides whether a folder-name implies healthy/common/rare."""

    def classify(self, label: str, crop: str) -> str:
        label_l = label.lower()
        if any(keyword in label_l for keyword in HEALTHY_KEYWORDS):
            return "healthy"
        if any(disease in label_l for disease in COMMON_DISEASES.get(crop, ())):
            return "common"
        return "rare"
