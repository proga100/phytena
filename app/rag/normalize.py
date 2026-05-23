from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    language_detected: str
    normalized_ru: str
    entities: dict[str, str | list[str]] = field(default_factory=dict)


UZ_LATIN_TERMS = {
    "pomidor": "томат",
    "bug'doy": "пшеница",
    "bugdoy": "пшеница",
    "barg": "лист",
    "barglari": "листья",
    "sargay": "желтеют",
    "kasal": "болезнь",
}

RU_CROP_TERMS = {
    "пшеница": "пшеница",
    "томат": "томат",
    "помидор": "томат",
    "хлопок": "хлопок",
    "виноград": "виноград",
}


def detect_language_script(text: str) -> str:
    lowered = f" {text.lower()} "
    if any(term in lowered for term in UZ_LATIN_TERMS):
        return "uz_latn"
    if any("а" <= char <= "я" or "А" <= char <= "Я" for char in text):
        return "ru"
    return "unknown"


def extract_entities(text: str) -> dict[str, str | list[str]]:
    lowered = text.lower()
    entities: dict[str, str | list[str]] = {}
    for term, canonical in RU_CROP_TERMS.items():
        if term in lowered:
            entities["crop"] = canonical
            break
    if "желт" in lowered or "sargay" in lowered:
        entities["symptoms"] = ["пожелтение листьев"]
    if "сох" in lowered:
        entities.setdefault("symptoms", [])
        symptoms = entities["symptoms"]
        if isinstance(symptoms, list):
            symptoms.append("усыхание листьев")
    return entities


def normalize_query_to_russian(text: str, language: str | None = None) -> NormalizedQuery:
    detected = language if language and language != "auto" else detect_language_script(text)
    normalized = text
    if detected == "uz_latn":
        normalized = text.lower()
        for source, target in UZ_LATIN_TERMS.items():
            normalized = normalized.replace(source, target)
        if "nima qilish kerak" in normalized:
            normalized = normalized.replace("nima qilish kerak", "что делать")
    entities = extract_entities(normalized)
    return NormalizedQuery(
        original=text,
        language_detected=detected,
        normalized_ru=normalized,
        entities=entities,
    )
