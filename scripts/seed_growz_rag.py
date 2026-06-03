"""Seed the dedicated growz_rag vector DB from the imported Growz data.

Two embedded document types (kept separate so symptom search stays clean):
  * disease docs  -> symptom -> disease retrieval
  * crop docs     -> crop-info / agronomy queries

Treatments are copied verbatim (linked to their disease) but never embedded; they
are fetched by disease after retrieval.

Reads from the `growz` DB, embeds with Gemini, writes to `growz_rag`.
"""

import argparse
import asyncio
import os
import sys

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

sys.path.append(os.getcwd())

from app.clients.embeddings import EmbeddingsClient
from app.config import get_settings
from app.growz_models import GrowzCrop, GrowzDisease, GrowzTreatment
from app.growz_rag_models import (
    GrowzRagBase,
    GrowzRagCrop,
    GrowzRagDisease,
    GrowzRagTreatment,
)

EMBED_BATCH = 50  # Gemini batchEmbedContents call size


def _mask_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, _, host = rest.partition("@")
    return f"{scheme}://{creds.split(':', 1)[0]}:***@{host}"


def compose_disease_text(crop_name: str | None, name: str, type_: str | None,
                         description: str | None, biology_name: str | None) -> str:
    """Build the Uzbek doc embedded for symptom -> disease search.

    Crop *name* (clean) is included; the crop *description* essay is deliberately
    excluded so it doesn't dilute the symptom signal.
    """
    lines: list[str] = []
    if crop_name:
        lines.append(f"Ekin: {crop_name}.")
    lines.append(f"Kasallik yoki zararkunanda: {name}.")
    if type_:
        lines.append(f"Turi: {type_}.")
    if description:
        lines.append(f"Belgilari va tavsifi: {description}")
    if biology_name:
        lines.append(f"Biologiyasi: {biology_name}")
    return "\n".join(lines)


def compose_crop_text(name: str, biology_name: str | None,
                      category: str | None, description: str | None) -> str:
    """Build the Uzbek doc embedded for crop-info queries."""
    lines = [f"Ekin: {name}."]
    if biology_name:
        lines.append(f"Biologik nomi: {biology_name}.")
    if category:
        lines.append(f"Toifasi: {category}.")
    if description:
        lines.append(f"Tavsifi: {description}")
    return "\n".join(lines)


async def _embed_in_batches(client: EmbeddingsClient, texts: list[str], dim: int,
                            label: str) -> list[list[float]]:
    out: list[list[float]] = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc=f"Embedding {label}", unit="batch"):
        batch = texts[i:i + EMBED_BATCH]
        out.extend(await client.get_embeddings_batch(batch, output_dimensionality=dim))
    return out


async def seed(*, limit_diseases: int | None, skip_treatments: bool) -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required to embed (set it in .env).")

    src_engine = create_async_engine(settings.growz_database_url)
    rag_engine = create_async_engine(settings.growz_rag_database_url)
    src_session = sessionmaker(src_engine, class_=AsyncSession, expire_on_commit=False)
    rag_session = sessionmaker(rag_engine, class_=AsyncSession, expire_on_commit=False)

    print(f"Source : {_mask_url(settings.growz_database_url)}")
    print(f"RAG    : {_mask_url(settings.growz_rag_database_url)}")

    async with rag_engine.begin() as conn:
        await conn.run_sync(GrowzRagBase.metadata.create_all)

    client = EmbeddingsClient(api_key=settings.gemini_api_key, model=settings.embeddings_model)
    dim = settings.embedding_dimension

    # ---- 1. Crops (join clean name; only those that exist) ----
    async with src_session() as s:
        crops = (await s.execute(select(GrowzCrop))).scalars().all()
    crop_texts = [compose_crop_text(c.name, c.biology_name, c.crop_category_name, c.description) for c in crops]
    print(f"Crops: {len(crops)}")
    crop_vectors = await _embed_in_batches(client, crop_texts, dim, "crops")

    async with rag_session() as s:
        async with s.begin():
            for c, txt, vec in zip(crops, crop_texts, crop_vectors):
                s.add(GrowzRagCrop(
                    source_crop_id=c.id, crop_name=c.name, biology_name=c.biology_name,
                    crop_category_name=c.crop_category_name, description=c.description,
                    search_text=txt, embedding=vec, raw=c.raw or {},
                ))

    # ---- 2. Diseases (join crop name) ----
    async with src_session() as s:
        stmt = (
            select(GrowzDisease, GrowzCrop.name)
            .outerjoin(GrowzCrop, GrowzDisease.crop_id == GrowzCrop.id)
            .order_by(GrowzDisease.id)
        )
        if limit_diseases is not None:
            stmt = stmt.limit(limit_diseases)
        rows = (await s.execute(stmt)).all()
    diseases = [(d, crop_name) for d, crop_name in rows]
    dis_texts = [
        compose_disease_text(crop_name, d.name, d.type, d.description, d.biology_name)
        for d, crop_name in diseases
    ]
    print(f"Diseases: {len(diseases)}")
    dis_vectors = await _embed_in_batches(client, dis_texts, dim, "diseases")

    # map source disease id -> rag disease id, for linking treatments
    src_to_rag: dict = {}
    async with rag_session() as s:
        async with s.begin():
            for (d, crop_name), txt, vec in zip(diseases, dis_texts, dis_vectors):
                row = GrowzRagDisease(
                    source_disease_id=d.id, crop_name=crop_name, disease_name=d.name,
                    type=d.type, biology_name=d.biology_name, description=d.description,
                    search_text=txt, embedding=vec, raw=d.raw or {},
                )
                s.add(row)
                await s.flush()
                src_to_rag[d.id] = row.id

    # ---- 3. Treatments (linked, NOT embedded) ----
    if not skip_treatments:
        async with src_session() as s:
            stmt = select(GrowzTreatment, GrowzDisease.id).outerjoin(
                GrowzDisease, GrowzTreatment.disease_id == GrowzDisease.id
            )
            t_rows = (await s.execute(stmt)).all()
        # only treatments whose disease we actually seeded
        linkable = [(t, dis_id) for t, dis_id in t_rows if dis_id in src_to_rag]
        print(f"Treatments to link: {len(linkable)}")
        async with rag_session() as s:
            async with s.begin():
                for t, dis_id in tqdm(linkable, desc="Linking treatments", unit="tx"):
                    drug = (t.raw or {}).get("drug") or {}
                    s.add(GrowzRagTreatment(
                        source_treatment_id=t.id,
                        rag_disease_id=src_to_rag.get(dis_id),
                        drug_name=drug.get("name"),
                        drug_description=drug.get("description"),
                        dose_min=t.dose_min, dose_max=t.dose_max, type=t.type, raw=t.raw or {},
                    ))

    # ---- 4. FTS (simple config; Uzbek isn't a built-in stemmer) ----
    async with rag_engine.begin() as conn:
        await conn.execute(text("UPDATE rag_diseases SET fts = to_tsvector('simple', search_text) WHERE fts IS NULL"))
        await conn.execute(text("UPDATE rag_crops SET fts = to_tsvector('simple', search_text) WHERE fts IS NULL"))

    await src_engine.dispose()
    await rag_engine.dispose()
    print("Seeding complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the growz_rag vector DB.")
    parser.add_argument("--limit-diseases", type=int, default=None, help="Trial: only first N diseases.")
    parser.add_argument("--skip-treatments", action="store_true", help="Skip linking treatments (faster trial).")
    args = parser.parse_args()
    asyncio.run(seed(limit_diseases=args.limit_diseases, skip_treatments=args.skip_treatments))
