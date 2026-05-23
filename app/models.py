from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Numeric, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.config import get_settings
from app.db import Base


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def created_at_column() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def jsonb_object() -> Mapped[dict[str, Any]]:
    return mapped_column(JSONB, server_default="{}", nullable=False)


def text_array() -> Mapped[list[str]]:
    return mapped_column(ARRAY(Text), server_default="{}", nullable=False)


class KbSource(Base):
    __tablename__ = "kb_sources"

    id: Mapped[uuid.UUID] = uuid_pk()
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False, default="ru")
    url: Mapped[str | None] = mapped_column(Text)
    file_uri: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[date | None] = mapped_column(Date)
    version: Mapped[str | None] = mapped_column(Text)
    trust_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    checksum: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    ingested_at: Mapped[datetime] = created_at_column()
    created_at: Mapped[datetime] = created_at_column()

    documents: Mapped[list[KbDocument]] = relationship(back_populates="source")
    chunks: Mapped[list[KbChunk]] = relationship(back_populates="source")


class KbDocument(Base):
    __tablename__ = "kb_documents"

    id: Mapped[uuid.UUID] = uuid_pk()
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kb_sources.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    canonical_language: Mapped[str] = mapped_column(Text, nullable=False, default="ru")
    language_original: Mapped[str | None] = mapped_column(Text)
    crop: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(Text)
    country: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(Text)
    document_type: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[date | None] = mapped_column(Date)
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    safety_critical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()

    source: Mapped[KbSource] = relationship(back_populates="documents")
    chunks: Mapped[list[KbChunk]] = relationship(back_populates="document")


class KbChunk(Base):
    __tablename__ = "kb_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_kb_chunks_document_index"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kb_documents.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kb_sources.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text_ru: Mapped[str] = mapped_column(Text, nullable=False)
    text_original: Mapped[str | None] = mapped_column(Text)
    language_original: Mapped[str | None] = mapped_column(Text)
    tokens: Mapped[int | None] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = text_array()
    section_title: Mapped[str | None] = mapped_column(Text)
    crop: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(Text)
    problem_type: Mapped[str | None] = mapped_column(Text)
    phenological_stage: Mapped[str | None] = mapped_column(Text)
    chemical_names: Mapped[list[str]] = text_array()
    pest_names: Mapped[list[str]] = text_array()
    disease_names: Mapped[list[str]] = text_array()
    nutrient_names: Mapped[list[str]] = text_array()
    safety_flags: Mapped[list[str]] = text_array()
    embedding: Mapped[list[float] | None] = mapped_column(Vector(get_settings().embedding_dimension))
    fts_ru: Mapped[str | None] = mapped_column(TSVECTOR)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()

    document: Mapped[KbDocument] = relationship(back_populates="chunks")
    source: Mapped[KbSource] = relationship(back_populates="chunks")
    entity_links: Mapped[list[KbChunkEntity]] = relationship(back_populates="chunk")
    retrieval_hits: Mapped[list[RetrievalHit]] = relationship(back_populates="chunk")


class KbEntity(Base):
    __tablename__ = "kb_entities"

    id: Mapped[uuid.UUID] = uuid_pk()
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name_ru: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name_uz: Mapped[str | None] = mapped_column(Text)
    aliases_ru: Mapped[list[str]] = text_array()
    aliases_uz: Mapped[list[str]] = text_array()
    aliases_latin: Mapped[list[str]] = text_array()
    aliases_cyrillic: Mapped[list[str]] = text_array()
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()

    chunk_links: Mapped[list[KbChunkEntity]] = relationship(back_populates="entity")


class KbChunkEntity(Base):
    __tablename__ = "kb_chunk_entities"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kb_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kb_entities.id", ondelete="CASCADE"), primary_key=True
    )
    match_type: Mapped[str] = mapped_column(Text, nullable=False, default="manual")
    confidence: Mapped[Decimal | None] = mapped_column(Numeric)
    created_at: Mapped[datetime] = created_at_column()

    chunk: Mapped[KbChunk] = relationship(back_populates="entity_links")
    entity: Mapped[KbEntity] = relationship(back_populates="chunk_links")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = uuid_pk()
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    original_sha256: Mapped[str | None] = mapped_column(Text)
    normalized_sha256: Mapped[str | None] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(Text)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    quality: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    analysis: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()

    messages: Mapped[list[Message]] = relationship(back_populates="image")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()

    messages: Mapped[list[Message]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = uuid_pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="SET NULL")
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language_detected: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    image: Mapped[Image | None] = relationship(back_populates="messages")
    pipeline_runs: Mapped[list[PipelineRun]] = relationship(back_populates="message")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = uuid_pk()
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL")
    )
    pipeline: Mapped[str] = mapped_column(Text, nullable=False)
    query_original: Mapped[str] = mapped_column(Text, nullable=False)
    query_normalized_ru: Mapped[str | None] = mapped_column(Text)
    query_language: Mapped[str | None] = mapped_column(Text)
    detected_entities: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    image_analysis: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    fusion: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    retrieval_params: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    model_name: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str | None] = mapped_column(Text)
    reranker_model: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text)
    answer_json: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_estimate: Mapped[Decimal | None] = mapped_column(Numeric)
    safety_flags: Mapped[list[str]] = text_array()
    refusal_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_column()

    message: Mapped[Message | None] = relationship(back_populates="pipeline_runs")
    retrieval_hits: Mapped[list[RetrievalHit]] = relationship(back_populates="pipeline_run")
    traces: Mapped[list[Trace]] = relationship(back_populates="pipeline_run")


class RetrievalHit(Base):
    __tablename__ = "retrieval_hits"

    id: Mapped[uuid.UUID] = uuid_pk()
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kb_chunks.id", ondelete="CASCADE"), nullable=False
    )
    rank_initial: Mapped[int | None] = mapped_column(Integer)
    rank_final: Mapped[int | None] = mapped_column(Integer)
    vector_score: Mapped[Decimal | None] = mapped_column(Numeric)
    lexical_score: Mapped[Decimal | None] = mapped_column(Numeric)
    metadata_score: Mapped[Decimal | None] = mapped_column(Numeric)
    rrf_score: Mapped[Decimal | None] = mapped_column(Numeric)
    reranker_score: Mapped[Decimal | None] = mapped_column(Numeric)
    included_in_prompt: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    text_preview: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_column()

    pipeline_run: Mapped[PipelineRun] = relationship(back_populates="retrieval_hits")
    chunk: Mapped[KbChunk] = relationship(back_populates="retrieval_hits")


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = uuid_pk()
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="CASCADE")
    )
    request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    trace_json: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="created")
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_column()

    pipeline_run: Mapped[PipelineRun | None] = relationship(back_populates="traces")


class GoldenItem(Base):
    __tablename__ = "golden_items"

    id: Mapped[uuid.UUID] = uuid_pk()
    question: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(Text)
    image_path: Mapped[str | None] = mapped_column(Text)
    crop: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(Text)
    growth_stage: Mapped[str | None] = mapped_column(Text)
    expected_answer: Mapped[str | None] = mapped_column(Text)
    expected_diagnoses: Mapped[list[str]] = text_array()
    expected_actions: Mapped[list[str]] = text_array()
    required_caveats: Mapped[list[str]] = text_array()
    gold_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), server_default="{}")
    safety_critical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    difficulty: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    eval_set_name: Mapped[str | None] = mapped_column(Text)
    pipeline_ids: Mapped[list[str]] = text_array()
    judge_model: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="created")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at_column()


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = uuid_pk()
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    golden_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("golden_items.id", ondelete="CASCADE"), nullable=False
    )
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="SET NULL")
    )
    answer: Mapped[str | None] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), server_default="{}"
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    human_score: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()


class LlmJudgment(Base):
    __tablename__ = "llm_judgments"

    id: Mapped[uuid.UUID] = uuid_pk()
    eval_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE")
    )
    golden_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("golden_items.id", ondelete="CASCADE")
    )
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="SET NULL")
    )
    judge_model: Mapped[str] = mapped_column(Text, nullable=False)
    rubric_version: Mapped[str | None] = mapped_column(Text)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    winner_pipeline: Mapped[str | None] = mapped_column(Text)
    reasoning_summary: Mapped[str | None] = mapped_column(Text)
    safety_flags: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = uuid_pk()
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="SET NULL")
    )
    rating: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}")
    created_at: Mapped[datetime] = created_at_column()


class HumanReview(Base):
    __tablename__ = "human_reviews"

    id: Mapped[uuid.UUID] = uuid_pk()
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="SET NULL")
    )
    eval_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE")
    )
    golden_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("golden_items.id", ondelete="CASCADE")
    )
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    review_status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    winner_pipeline: Mapped[str | None] = mapped_column(Text)
    accuracy_score: Mapped[int | None] = mapped_column(Integer)
    usefulness_score: Mapped[int | None] = mapped_column(Integer)
    safety_score: Mapped[int | None] = mapped_column(Integer)
    citation_score: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)
    corrected_answer: Mapped[str | None] = mapped_column(Text)
    failure_tags: Mapped[list[str]] = text_array()
    created_at: Mapped[datetime] = created_at_column()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str | None] = mapped_column(Text)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    model_defaults: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = created_at_column()
