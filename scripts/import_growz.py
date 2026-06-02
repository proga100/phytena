import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from sqlalchemy import func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# Add the project root to sys.path to import app modules
sys.path.append(os.getcwd())

from app.clients.growz import GrowzApiClient
from app.config import get_settings
from app.growz_db import GrowzBase
from app.growz_models import GrowzCrop, GrowzDisease, GrowzDrug, GrowzTreatment

DEFAULT_ILLNESS = "scripts/growz_illness.json"
DEFAULT_TREATMENTS = "scripts/treaments.json"


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _to_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        # API timestamps are ISO-8601 with a trailing 'Z' (e.g. 2023-08-29T09:59:15.112Z).
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _mask_url(url: str) -> str:
    """Hide the password component of a database URL for safe logging."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, _, host = rest.partition("@")
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("data", [])


async def _upsert(
    session: AsyncSession,
    table: Any,
    values: dict[str, Any],
    update_cols: dict[str, Any],
) -> bool:
    """Run an INSERT ... ON CONFLICT (id) DO UPDATE. Returns True if a row was inserted.

    Detects insert-vs-update via Postgres' ``xmax = 0`` predicate, which holds
    only for freshly inserted (not updated) tuples.
    """
    stmt = (
        pg_insert(table)
        .values(**values)
        .on_conflict_do_update(index_elements=["id"], set_=update_cols)
        .returning(literal_column("(xmax = 0)").label("inserted"))
    )
    result = await session.execute(stmt)
    row = result.first()
    return bool(row.inserted) if row is not None else False


async def _import_treatment_records(
    session: AsyncSession,
    records: list[dict[str, Any]],
    tally: Any,
) -> None:
    """Upsert drugs, merge diseases, and upsert treatments for a batch of records.

    Shared by both the file-based and per-disease API import paths so the
    catalog-vs-treatment merge semantics stay identical.
    """
    for rec in records:
        disease = rec.get("disease") or {}
        drug = rec.get("drug") or {}
        disease_id = disease.get("id")
        drug_id = drug.get("id")

        if drug_id:
            inserted = await _upsert(
                session,
                GrowzDrug,
                {
                    "id": drug_id,
                    "name": drug.get("name") or "",
                    "description": drug.get("description"),
                    "created_at_src": _to_datetime(drug.get("createdAt")),
                    "raw": drug,
                },
                {
                    "name": pg_insert(GrowzDrug).excluded.name,
                    "description": pg_insert(GrowzDrug).excluded.description,
                    "created_at_src": pg_insert(GrowzDrug).excluded.created_at_src,
                    "raw": pg_insert(GrowzDrug).excluded.raw,
                },
            )
            tally("drugs", inserted)

        if disease_id:
            excluded = pg_insert(GrowzDisease).excluded
            inserted = await _upsert(
                session,
                GrowzDisease,
                {
                    "id": disease_id,
                    "name": disease.get("name") or "",
                    "description": disease.get("description"),
                    "biology_name": disease.get("biology_name"),
                    # type is NOT NULL; provide a fallback for treatment-only diseases.
                    "type": rec.get("type") or "disease",
                    "crop_id": None,
                    "created_at_src": _to_datetime(disease.get("createdAt")),
                    "raw": disease,
                },
                {
                    # Merge: keep catalog-provided values; only fill gaps from treatments.
                    "name": func.coalesce(excluded.name, GrowzDisease.name),
                    "description": func.coalesce(excluded.description, GrowzDisease.description),
                    "biology_name": func.coalesce(
                        excluded.biology_name, GrowzDisease.biology_name
                    ),
                    "type": func.coalesce(GrowzDisease.type, excluded.type),
                    "crop_id": func.coalesce(GrowzDisease.crop_id, excluded.crop_id),
                    "created_at_src": func.coalesce(
                        GrowzDisease.created_at_src, excluded.created_at_src
                    ),
                },
            )
            tally("diseases", inserted)

        inserted = await _upsert(
            session,
            GrowzTreatment,
            {
                "id": rec["id"],
                "disease_id": disease_id,
                "drug_id": drug_id,
                "dose_min": _to_decimal(rec.get("dose_min")),
                "dose_max": _to_decimal(rec.get("dose_max")),
                "type": rec.get("type"),
                "weeds": rec.get("weeds") or [],
                "created_at_src": _to_datetime(rec.get("createdAt")),
                "raw": rec,
            },
            {
                "disease_id": pg_insert(GrowzTreatment).excluded.disease_id,
                "drug_id": pg_insert(GrowzTreatment).excluded.drug_id,
                "dose_min": pg_insert(GrowzTreatment).excluded.dose_min,
                "dose_max": pg_insert(GrowzTreatment).excluded.dose_max,
                "type": pg_insert(GrowzTreatment).excluded.type,
                "weeds": pg_insert(GrowzTreatment).excluded.weeds,
                "created_at_src": pg_insert(GrowzTreatment).excluded.created_at_src,
                "raw": pg_insert(GrowzTreatment).excluded.raw,
            },
        )
        tally("treatments", inserted)


async def import_growz(
    illness_path: Path,
    treatments_path: Path,
    from_api: bool,
    *,
    delay: float = 0.3,
    limit: int = 50,
    max_diseases: int | None = None,
    resume: bool = False,
) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.growz_database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    print(f"Connected to Growz database: {_mask_url(settings.growz_database_url)}")

    # Bootstrap growz tables (managed via create_all, not Alembic).
    async with engine.begin() as conn:
        await conn.run_sync(GrowzBase.metadata.create_all)

    counts = {
        "crops": [0, 0],
        "diseases": [0, 0],
        "drugs": [0, 0],
        "treatments": [0, 0],
    }

    def tally(key: str, inserted: bool) -> None:
        counts[key][0 if inserted else 1] += 1

    async with async_session() as session:
        async with session.begin():
            # 1. Illness catalog: crops + diseases (authoritative for crop_id/type)
            illness = _load_json_records(illness_path)
            print(f"Loaded {len(illness)} illness catalog records from {illness_path}")
            for rec in illness:
                crop = rec.get("crop") or {}
                crop_id = crop.get("id")
                if crop_id:
                    # The diseases/illness endpoints stuff a crop *description* into
                    # crop.name; the authoritative clean name comes from /ai/crops
                    # (scripts/import_crops.py). So only fill the row when absent and
                    # never overwrite an existing name/raw with the essay version.
                    inserted = await _upsert(
                        session,
                        GrowzCrop,
                        {"id": crop_id, "name": crop.get("name") or "", "raw": crop},
                        {
                            "name": func.coalesce(GrowzCrop.name, pg_insert(GrowzCrop).excluded.name),
                            "raw": func.coalesce(GrowzCrop.raw, pg_insert(GrowzCrop).excluded.raw),
                        },
                    )
                    tally("crops", inserted)

                inserted = await _upsert(
                    session,
                    GrowzDisease,
                    {
                        "id": rec["id"],
                        "name": rec.get("name") or "",
                        "description": rec.get("description"),
                        "biology_name": None,
                        "type": rec.get("type") or "disease",
                        "crop_id": crop_id,
                        "created_at_src": _to_datetime(rec.get("createAt")),
                        "raw": rec,
                    },
                    {
                        "name": pg_insert(GrowzDisease).excluded.name,
                        "description": pg_insert(GrowzDisease).excluded.description,
                        "type": pg_insert(GrowzDisease).excluded.type,
                        "crop_id": pg_insert(GrowzDisease).excluded.crop_id,
                        "created_at_src": pg_insert(GrowzDisease).excluded.created_at_src,
                        "raw": pg_insert(GrowzDisease).excluded.raw,
                    },
                )
                tally("diseases", inserted)

    # 2. Treatments: drugs + diseases (merge) + treatments
    diseases_processed = 0
    diseases_skipped = 0

    if from_api:
        # Fail fast before the long throttled loop if the token is missing.
        if not settings.growz_api_token:
            raise RuntimeError(
                "Growz API token is required for --from-api (set GROWZ_API_TOKEN in .env)."
            )
        client = GrowzApiClient(
            base_url=settings.growz_api_base_url,
            token=settings.growz_api_token,
            page_size=limit,
            delay_seconds=delay,
        )

        async with async_session() as session:
            # Disease ids come from the catalog we just imported (deterministic order).
            result = await session.execute(select(GrowzDisease.id).order_by(GrowzDisease.id))
            disease_ids = [row[0] for row in result.all()]

            skip_ids: set[Any] = set()
            if resume:
                done = await session.execute(select(GrowzTreatment.disease_id).distinct())
                skip_ids = {row[0] for row in done.all()}
                print(f"Resume: {len(skip_ids)} diseases already have treatments and will be skipped")

        if max_diseases is not None:
            disease_ids = disease_ids[:max_diseases]

        total_diseases = len(disease_ids)
        print(f"Fetching treatments for {total_diseases} diseases (delay={delay}s, limit={limit})")

        progress = tqdm(disease_ids, desc="Treatments", unit="disease")
        for disease_id in progress:
            if resume and disease_id in skip_ids:
                diseases_skipped += 1
                progress.set_postfix(skipped=diseases_skipped, refresh=False)
                continue

            records = await client.fetch_treatments_for_disease(str(disease_id))
            # Commit per-disease so a crash mid-run keeps prior progress durable.
            async with async_session() as session:
                async with session.begin():
                    await _import_treatment_records(session, records, tally)
            diseases_processed += 1
            # tqdm.write keeps the bar intact while still emitting a per-disease line.
            progress.write(f"disease id={disease_id} treatments={len(records)}")
            progress.set_postfix(done=diseases_processed, last=len(records), refresh=False)
        progress.close()
    else:
        treatments = _load_json_records(treatments_path)
        print(f"Loaded {len(treatments)} treatment records from {treatments_path}")
        async with async_session() as session:
            async with session.begin():
                await _import_treatment_records(session, treatments, tally)

    await engine.dispose()

    print("Import complete (inserted / updated):")
    for key, (ins, upd) in counts.items():
        print(f"  {key:<11} inserted={ins} updated={upd}")
    if from_api:
        print(f"  diseases    processed={diseases_processed} skipped={diseases_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Growz disease/pest/treatment data.")
    parser.add_argument("--illness", default=DEFAULT_ILLNESS, help="Path to illness catalog JSON")
    parser.add_argument("--treatments", default=DEFAULT_TREATMENTS, help="Path to treatments JSON")
    parser.add_argument(
        "--from-api",
        action="store_true",
        help="Fetch treatments live from the Growz API (per-disease) instead of the file.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3, help="Seconds to sleep between API requests."
    )
    parser.add_argument("--limit", type=int, default=50, help="API page size.")
    parser.add_argument(
        "--max-diseases",
        type=int,
        default=None,
        help="Only fetch treatments for the first N diseases (testing/trial runs).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip diseases that already have at least one treatment row (checkpointing).",
    )
    args = parser.parse_args()

    asyncio.run(
        import_growz(
            illness_path=Path(args.illness),
            treatments_path=Path(args.treatments),
            from_api=args.from_api,
            delay=args.delay,
            limit=args.limit,
            max_diseases=args.max_diseases,
            resume=args.resume,
        )
    )
