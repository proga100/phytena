from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Numeric, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.growz_db import GrowzBase
from app.models import created_at_column


class GrowzCrop(GrowzBase):
    __tablename__ = "growz_crops"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    biology_name: Mapped[str | None] = mapped_column(Text)
    crop_category_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    crop_category_name: Mapped[str | None] = mapped_column(Text)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}", nullable=False)
    imported_at: Mapped[datetime] = created_at_column()

    diseases: Mapped[list[GrowzDisease]] = relationship(back_populates="crop")


class GrowzDrug(GrowzBase):
    __tablename__ = "growz_drugs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at_src: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}", nullable=False)
    imported_at: Mapped[datetime] = created_at_column()

    treatments: Mapped[list[GrowzTreatment]] = relationship(back_populates="drug")


class GrowzDisease(GrowzBase):
    __tablename__ = "growz_diseases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    biology_name: Mapped[str | None] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    crop_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("growz_crops.id", ondelete="SET NULL")
    )
    created_at_src: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}", nullable=False)
    imported_at: Mapped[datetime] = created_at_column()

    crop: Mapped[GrowzCrop | None] = relationship(back_populates="diseases")
    treatments: Mapped[list[GrowzTreatment]] = relationship(back_populates="disease")


class GrowzTreatment(GrowzBase):
    __tablename__ = "growz_treatments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    disease_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("growz_diseases.id", ondelete="CASCADE"), nullable=False
    )
    drug_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("growz_drugs.id", ondelete="CASCADE"), nullable=False
    )
    dose_min: Mapped[Decimal | None] = mapped_column(Numeric)
    dose_max: Mapped[Decimal | None] = mapped_column(Numeric)
    type: Mapped[str | None] = mapped_column(Text)
    weeds: Mapped[list[Any]] = mapped_column(JSONB, server_default="[]", nullable=False)
    created_at_src: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}", nullable=False)
    imported_at: Mapped[datetime] = created_at_column()

    disease: Mapped[GrowzDisease] = relationship(back_populates="treatments")
    drug: Mapped[GrowzDrug] = relationship(back_populates="treatments")
