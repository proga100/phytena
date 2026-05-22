from growz_eval.golden_set.classifier import CropClassifier, TierClassifier


class TestCropClassifier:
    def test_uses_hint_when_not_mixed(self):
        assert CropClassifier().classify("anything", "cotton") == "cotton"

    def test_falls_back_to_other_for_unknown_hint(self):
        assert CropClassifier().classify("anything", "unicorn") == "other"

    def test_mixed_hint_uses_label_keywords(self):
        assert CropClassifier().classify("tomato_leaf_mold", "mixed") == "tomato"

    def test_mixed_hint_unknown_label_becomes_other(self):
        assert CropClassifier().classify("strawberry_blight", "mixed") == "other"

    def test_maize_is_remapped_but_not_in_distribution_so_other(self):
        # 'corn' is not in CROP_DISTRIBUTION -> fallback to other
        assert CropClassifier().classify("maize_rust", "mixed") == "other"


class TestTierClassifier:
    def test_healthy_keyword_wins(self):
        assert TierClassifier().classify("healthy_leaf", "cotton") == "healthy"

    def test_common_disease_keyword(self):
        assert TierClassifier().classify("bacterial_blight_x", "cotton") == "common"

    def test_unknown_disease_is_rare(self):
        assert TierClassifier().classify("weirdspot_42", "cotton") == "rare"

    def test_unknown_crop_has_no_common_diseases(self):
        assert TierClassifier().classify("anything", "moon_fruit") == "rare"
