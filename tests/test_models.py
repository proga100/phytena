from app import models  # noqa: F401
from app.db import Base


def test_core_tables_are_registered() -> None:
    expected_tables = {
        "kb_sources",
        "kb_documents",
        "kb_chunks",
        "kb_entities",
        "kb_chunk_entities",
        "images",
        "conversations",
        "messages",
        "pipeline_runs",
        "retrieval_hits",
        "traces",
        "golden_items",
        "eval_runs",
        "eval_results",
        "llm_judgments",
        "feedback",
        "human_reviews",
        "prompt_versions",
    }

    assert expected_tables.issubset(Base.metadata.tables.keys())
