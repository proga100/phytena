import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import literal_column, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# Add the project root to sys.path to import app modules
sys.path.append(os.getcwd())

from app.config import get_settings
from app.growz_db import GrowzBase
from app.growz_models import GrowzCrop

DEFAULT_CROPS = "scripts/crops.json"

# growz tables are managed via create_all (not Alembic). create_all is a no-op on
# tables that already exist, so new columns must be added explicitly and idempotently.
_ADD_COLUMNS = (
    "ALTER TABLE growz_crops ADD COLUMN IF NOT EXISTS biology_name text",
    "ALTER TABLE growz_crops ADD COLUMN IF NOT EXISTS crop_category_id uuid",
    "ALTER TABLE growz_crops ADD COLUMN IF NOT EXISTS crop_category_name text",
)


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
    values: dict[str, Any],
    update_cols: dict[str, Any],
) -> bool:
    """INSERT ... ON CONFLICT (id) DO UPDATE. Returns True if a row was inserted.

    Insert-vs-update is detected via Postgres' ``xmax = 0`` predicate, which holds
    only for freshly inserted (not updated) tuples.
    """
    stmt = (
        pg_insert(GrowzCrop)
        .values(**values)
        .on_conflict_do_update(index_elements=["id"], set_=update_cols)
        .returning(literal_column("(xmax = 0)").label("inserted"))
    )
    result = await session.execute(stmt)
    row = result.first()
    return bool(row.inserted) if row is not None else False


async def import_crops(crops_path: Path) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.growz_database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    print(f"Connected to Growz database: {_mask_url(settings.growz_database_url)}")

    # Bootstrap the table, then add any new columns the model gained over time.
    async with engine.begin() as conn:
        await conn.run_sync(GrowzBase.metadata.create_all)
        for ddl in _ADD_COLUMNS:
            await conn.execute(text(ddl))

    records = _load_json_records(crops_path)
    print(f"Loaded {len(records)} crop records from {crops_path}")

    excluded = pg_insert(GrowzCrop).excluded
    inserted = updated = skipped = 0

    async with async_session() as session:
        async with session.begin():
            for rec in tqdm(records, desc="Crops", unit="crop"):
                crop_id = rec.get("id")
                if not crop_id:
                    skipped += 1
                    continue

                category = rec.get("crop_category") or {}
                was_inserted = await _upsert(
                    session,
                    {
                        "id": crop_id,
                        "name": rec.get("name") or "",
                        "biology_name": rec.get("biology_name"),
                        "crop_category_id": category.get("id"),
                        "crop_category_name": category.get("name"),
                        "raw": rec,
                    },
                    {
                        "name": excluded.name,
                        "biology_name": excluded.biology_name,
                        "crop_category_id": excluded.crop_category_id,
                        "crop_category_name": excluded.crop_category_name,
                        "raw": excluded.raw,
                    },
                )
                if was_inserted:
                    inserted += 1
                else:
                    updated += 1

    await engine.dispose()

    print("Import complete:")
    print(f"  crops inserted={inserted} updated={updated} skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Growz crop catalog from a JSON file.")
    parser.add_argument("--crops", default=DEFAULT_CROPS, help="Path to crops JSON")
    args = parser.parse_args()

    asyncio.run(import_crops(crops_path=Path(args.crops)))
