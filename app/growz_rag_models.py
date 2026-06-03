"""Models for the dedicated Growz RAG database (growz_rag).

Two embedded document types, kept separate so symptom search stays clean:

* GrowzRagDisease  -- one per disease; embeds crop_name + disease name + type +
                      symptoms/description + biology. Drives symptom -> disease search.
* GrowzRagCrop     -- one per crop that has a description; embeds the crop essay.
                      Drives crop-info / agronomy queries only.

Treatments are NOT embedded. They are stored verbatim (linked by source disease id)
and fetched after a disease is retrieved.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Numeric, Text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.config import get_settings

_EMBED_DIM = get_settings().embedding_dimension


class GrowzRagBase(DeclarativeBase):
    pass


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class GrowzRagDisease(GrowzRagBase):
    __tablename__ = "rag_diseases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Source disease id in the growz DB; lets us fetch treatments after retrieval.
    source_disease_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, unique=True)
    crop_name: Mapped[str | None] = mapped_column(Text)
    disease_name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str | None] = mapped_column(Text)
    biology_name: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    # The exact text that was embedded (composed Uzbek doc), kept for FTS + debugging.
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBED_DIM))
    fts: Mapped[str | None] = mapped_column(TSVECTOR)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = _created_at()

    treatments: Mapped[list[GrowzRagTreatment]] = relationship(back_populates="disease")


class GrowzRagCrop(GrowzRagBase):
    __tablename__ = "rag_crops"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_crop_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, unique=True)
    crop_name: Mapped[str] = mapped_column(Text, nullable=False)
    biology_name: Mapped[str | None] = mapped_column(Text)
    crop_category_name: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBED_DIM))
    fts: Mapped[str | None] = mapped_column(TSVECTOR)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = _created_at()


class GrowzRagTreatment(GrowzRagBase):
    __tablename__ = "rag_treatments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_treatment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, unique=True)
    rag_disease_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rag_diseases.id", ondelete="CASCADE")
    )
    drug_name: Mapped[str | None] = mapped_column(Text)
    drug_description: Mapped[str | None] = mapped_column(Text)
    dose_min: Mapped[Decimal | None] = mapped_column(Numeric)
    dose_max: Mapped[Decimal | None] = mapped_column(Numeric)
    type: Mapped[str | None] = mapped_column(Text)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = _created_at()

    disease: Mapped[GrowzRagDisease | None] = relationship(back_populates="treatments")
