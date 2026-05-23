"""create core agronomy prototype schema

Revision ID: 20260523_0001
Revises:
Create Date: 2026-05-23
"""
from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260523_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSION = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    op.create_table(
        "kb_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("url", sa.Text()),
        sa.Column("file_uri", sa.Text()),
        sa.Column("publisher", sa.Text()),
        sa.Column("published_at", sa.Date()),
        sa.Column("version", sa.Text()),
        sa.Column("trust_level", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("checksum", sa.Text()),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "kb_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("kb_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text()),
        sa.Column("canonical_language", sa.Text(), nullable=False),
        sa.Column("language_original", sa.Text()),
        sa.Column("crop", sa.Text()),
        sa.Column("region", sa.Text()),
        sa.Column("country", sa.Text()),
        sa.Column("topic", sa.Text()),
        sa.Column("document_type", sa.Text()),
        sa.Column("published_at", sa.Date()),
        sa.Column("valid_from", sa.Date()),
        sa.Column("valid_to", sa.Date()),
        sa.Column("safety_critical", sa.Boolean(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "kb_entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("canonical_name_ru", sa.Text(), nullable=False),
        sa.Column("canonical_name_uz", sa.Text()),
        sa.Column("aliases_ru", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("aliases_uz", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("aliases_latin", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("aliases_cyrillic", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "images",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("original_sha256", sa.Text()),
        sa.Column("normalized_sha256", sa.Text()),
        sa.Column("mime_type", sa.Text()),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("quality", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("analysis", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True)),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "kb_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("kb_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("kb_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text_ru", sa.Text(), nullable=False),
        sa.Column("text_original", sa.Text()),
        sa.Column("language_original", sa.Text()),
        sa.Column("tokens", sa.Integer()),
        sa.Column("heading_path", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("section_title", sa.Text()),
        sa.Column("crop", sa.Text()),
        sa.Column("region", sa.Text()),
        sa.Column("topic", sa.Text()),
        sa.Column("problem_type", sa.Text()),
        sa.Column("phenological_stage", sa.Text()),
        sa.Column("chemical_names", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("pest_names", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("disease_names", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("nutrient_names", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("safety_flags", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIMENSION)),
        sa.Column("fts_ru", postgresql.TSVECTOR()),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_kb_chunks_document_index"),
    )

    op.create_table(
        "kb_chunk_entities",
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("kb_chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("kb_entities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("match_type", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("image_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("images.id", ondelete="SET NULL")),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("language_detected", sa.Text()),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "pipeline_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="SET NULL"),
        ),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("messages.id", ondelete="SET NULL")),
        sa.Column("pipeline", sa.Text(), nullable=False),
        sa.Column("query_original", sa.Text(), nullable=False),
        sa.Column("query_normalized_ru", sa.Text()),
        sa.Column("query_language", sa.Text()),
        sa.Column("detected_entities", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("image_analysis", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("fusion", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("filters", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("retrieval_params", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("model_name", sa.Text()),
        sa.Column("embedding_model", sa.Text()),
        sa.Column("reranker_model", sa.Text()),
        sa.Column("prompt_version", sa.Text()),
        sa.Column("answer", sa.Text()),
        sa.Column("answer_json", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("citations", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("cost_estimate", sa.Numeric()),
        sa.Column("safety_flags", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("refusal_reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "retrieval_hits",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("kb_chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rank_initial", sa.Integer()),
        sa.Column("rank_final", sa.Integer()),
        sa.Column("vector_score", sa.Numeric()),
        sa.Column("lexical_score", sa.Numeric()),
        sa.Column("metadata_score", sa.Numeric()),
        sa.Column("rrf_score", sa.Numeric()),
        sa.Column("reranker_score", sa.Numeric()),
        sa.Column("included_in_prompt", sa.Boolean(), nullable=False),
        sa.Column("text_preview", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
        ),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_json", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "golden_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("language", sa.Text()),
        sa.Column("image_path", sa.Text()),
        sa.Column("crop", sa.Text()),
        sa.Column("region", sa.Text()),
        sa.Column("growth_stage", sa.Text()),
        sa.Column("expected_answer", sa.Text()),
        sa.Column("expected_diagnoses", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("expected_actions", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("required_caveats", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("gold_chunk_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), server_default="{}", nullable=False),
        sa.Column("safety_critical", sa.Boolean(), nullable=False),
        sa.Column("difficulty", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("eval_set_name", sa.Text()),
        sa.Column("pipeline_ids", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("judge_model", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("metrics", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "eval_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "eval_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("eval_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "golden_item_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("golden_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "pipeline_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
        ),
        sa.Column("answer", sa.Text()),
        sa.Column("retrieved_chunk_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), server_default="{}", nullable=False),
        sa.Column("metrics", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("human_score", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "llm_judgments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("eval_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("eval_runs.id", ondelete="CASCADE")),
        sa.Column(
            "golden_item_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("golden_items.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "pipeline_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
        ),
        sa.Column("judge_model", sa.Text(), nullable=False),
        sa.Column("rubric_version", sa.Text()),
        sa.Column("scores", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("winner_pipeline", sa.Text()),
        sa.Column("reasoning_summary", sa.Text()),
        sa.Column("safety_flags", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
        ),
        sa.Column("rating", sa.Text()),
        sa.Column("comment", sa.Text()),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "human_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
        ),
        sa.Column("eval_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("eval_runs.id", ondelete="CASCADE")),
        sa.Column(
            "golden_item_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("golden_items.id", ondelete="CASCADE"),
        ),
        sa.Column("reviewer_id", postgresql.UUID(as_uuid=True)),
        sa.Column("review_status", sa.Text(), nullable=False),
        sa.Column("winner_pipeline", sa.Text()),
        sa.Column("accuracy_score", sa.Integer()),
        sa.Column("usefulness_score", sa.Integer()),
        sa.Column("safety_score", sa.Integer()),
        sa.Column("citation_score", sa.Integer()),
        sa.Column("notes", sa.Text()),
        sa.Column("corrected_answer", sa.Text()),
        sa.Column("failure_tags", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "prompt_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("purpose", sa.Text()),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column("model_defaults", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_index("ix_kb_documents_source_id", "kb_documents", ["source_id"])
    op.create_index("ix_kb_documents_crop", "kb_documents", ["crop"])
    op.create_index("ix_kb_documents_topic", "kb_documents", ["topic"])
    op.create_index("ix_kb_chunks_document_id", "kb_chunks", ["document_id"])
    op.create_index("ix_kb_chunks_source_id", "kb_chunks", ["source_id"])
    op.create_index("ix_kb_chunks_crop", "kb_chunks", ["crop"])
    op.create_index("ix_kb_chunks_topic", "kb_chunks", ["topic"])
    op.create_index("ix_kb_chunks_region", "kb_chunks", ["region"])
    op.create_index("ix_kb_chunks_metadata", "kb_chunks", ["metadata"], postgresql_using="gin")
    op.create_index("ix_kb_chunks_fts_ru", "kb_chunks", ["fts_ru"], postgresql_using="gin")
    op.create_index("ix_kb_chunks_chemical_names", "kb_chunks", ["chemical_names"], postgresql_using="gin")
    op.create_index("ix_kb_chunks_pest_names", "kb_chunks", ["pest_names"], postgresql_using="gin")
    op.create_index("ix_kb_chunks_disease_names", "kb_chunks", ["disease_names"], postgresql_using="gin")
    op.create_index("ix_kb_chunks_nutrient_names", "kb_chunks", ["nutrient_names"], postgresql_using="gin")
    op.create_index(
        "ix_kb_chunks_embedding_hnsw",
        "kb_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_pipeline_runs_created_at", "pipeline_runs", ["created_at"])
    op.create_index("ix_pipeline_runs_pipeline", "pipeline_runs", ["pipeline"])
    op.create_index("ix_retrieval_hits_pipeline_run_id", "retrieval_hits", ["pipeline_run_id"])
    op.create_index("ix_retrieval_hits_chunk_id", "retrieval_hits", ["chunk_id"])
    op.create_index("ix_traces_pipeline_run_id", "traces", ["pipeline_run_id"])
    op.create_index("ix_traces_request_id", "traces", ["request_id"])
    op.create_index("ix_eval_results_eval_run_id", "eval_results", ["eval_run_id"])
    op.create_index("ix_human_reviews_review_status", "human_reviews", ["review_status"])


def downgrade() -> None:
    op.drop_index("ix_human_reviews_review_status", table_name="human_reviews")
    op.drop_index("ix_eval_results_eval_run_id", table_name="eval_results")
    op.drop_index("ix_traces_request_id", table_name="traces")
    op.drop_index("ix_traces_pipeline_run_id", table_name="traces")
    op.drop_index("ix_retrieval_hits_chunk_id", table_name="retrieval_hits")
    op.drop_index("ix_retrieval_hits_pipeline_run_id", table_name="retrieval_hits")
    op.drop_index("ix_pipeline_runs_pipeline", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_created_at", table_name="pipeline_runs")
    op.drop_index("ix_kb_chunks_embedding_hnsw", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_nutrient_names", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_disease_names", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_pest_names", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_chemical_names", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_fts_ru", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_metadata", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_region", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_topic", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_crop", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_source_id", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_document_id", table_name="kb_chunks")
    op.drop_index("ix_kb_documents_topic", table_name="kb_documents")
    op.drop_index("ix_kb_documents_crop", table_name="kb_documents")
    op.drop_index("ix_kb_documents_source_id", table_name="kb_documents")

    op.drop_table("prompt_versions")
    op.drop_table("human_reviews")
    op.drop_table("feedback")
    op.drop_table("llm_judgments")
    op.drop_table("eval_results")
    op.drop_table("eval_runs")
    op.drop_table("golden_items")
    op.drop_table("traces")
    op.drop_table("retrieval_hits")
    op.drop_table("pipeline_runs")
    op.drop_table("messages")
    op.drop_table("kb_chunk_entities")
    op.drop_table("kb_chunks")
    op.drop_table("conversations")
    op.drop_table("images")
    op.drop_table("kb_entities")
    op.drop_table("kb_documents")
    op.drop_table("kb_sources")

    op.execute("DROP EXTENSION IF EXISTS unaccent")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS vector")
