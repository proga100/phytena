"""Import ALL treatments from the unfiltered Growz /ai/treatments endpoint.

The per-disease importer (import_growz.py --from-api) only fetches treatments it
can reach via ?disease_id=X, looping the catalog diseases. That structurally
misses:
  * treatments with disease = null (no disease_id to query under), and
  * treatments under disease ids absent from the local illness catalog.

This script pages the unfiltered endpoint to load every treatment, reusing the
same upsert/merge semantics as import_growz so drugs and diseases stay consistent.
Treatments with a null disease are stored with disease_id = NULL, which requires
growz_treatments.disease_id to be nullable (handled idempotently below).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add the project root to sys.path to import app modules
sys.path.append(os.getcwd())

from app.clients.growz import GrowzApiClient
from app.config import get_settings
from app.growz_db import GrowzBase

# Reuse the shared upsert/merge logic so catalog-vs-treatment semantics stay identical.
from scripts.import_growz import _import_treatment_records, _mask_url  # noqa: E402

# growz tables use create_all (not Alembic); create_all won't relax an existing
# NOT NULL, so drop it explicitly and idempotently for the null-disease rows.
_RELAX_DISEASE_NOT_NULL = "ALTER TABLE growz_treatments ALTER COLUMN disease_id DROP NOT NULL"


async def import_all_treatments(*, delay: float = 0.3, limit: int = 50) -> None:
    settings = get_settings()
    if not settings.growz_api_token:
        raise RuntimeError("Growz API token is required (set GROWZ_API_TOKEN in .env).")

    engine = create_async_engine(settings.growz_database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    print(f"Connected to Growz database: {_mask_url(settings.growz_database_url)}")

    async with engine.begin() as conn:
        await conn.run_sync(GrowzBase.metadata.create_all)
        await conn.execute(text(_RELAX_DISEASE_NOT_NULL))

    client = GrowzApiClient(
        base_url=settings.growz_api_base_url,
        token=settings.growz_api_token,
        page_size=limit,
        delay_seconds=delay,
    )

    print(f"Fetching ALL treatments (unfiltered, delay={delay}s, limit={limit})")
    # fetch_treatments pages the whole endpoint until `total`; throttled per request.
    records = await client.fetch_treatments()
    print(f"Fetched {len(records)} treatment records from the API")

    counts = {"crops": [0, 0], "diseases": [0, 0], "drugs": [0, 0], "treatments": [0, 0]}

    def tally(key: str, inserted: bool) -> None:
        counts[key][0 if inserted else 1] += 1

    async with async_session() as session:
        async with session.begin():
            await _import_treatment_records(session, records, tally)

    await engine.dispose()

    print("Import complete (inserted / updated):")
    for key, (ins, upd) in counts.items():
        print(f"  {key:<11} inserted={ins} updated={upd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import ALL Growz treatments (unfiltered).")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between API requests.")
    parser.add_argument("--limit", type=int, default=50, help="API page size.")
    args = parser.parse_args()

    asyncio.run(import_all_treatments(delay=args.delay, limit=args.limit))
