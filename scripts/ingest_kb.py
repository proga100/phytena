import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Add the project root to sys.path to import app modules
sys.path.append(os.getcwd())

from app.config import get_settings
from app.clients.embeddings import EmbeddingsClient
from app.models import Base, KbSource, KbDocument, KbChunk

async def ingest_file(file_path: Path, crop: str | None = None, topic: str | None = None):
    settings = get_settings()
    if not settings.gemini_api_key:
        print("Error: GEMINI_API_KEY is not set.")
        return

    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    print(f"Reading file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple chunking by paragraphs (can be improved later)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    print(f"Split into {len(paragraphs)} chunks.")

    embed_client = EmbeddingsClient(api_key=settings.gemini_api_key, model=settings.embeddings_model)

    async with async_session() as session:
        async with session.begin():
            # 1. Create Source
            source = KbSource(
                title=file_path.name,
                source_type="file",
                language="ru",
                file_uri=str(file_path),
            )
            session.add(source)
            await session.flush()

            # 2. Create Document
            doc = KbDocument(
                source_id=source.id,
                title=file_path.stem,
                crop=crop,
                topic=topic,
            )
            session.add(doc)
            await session.flush()

            # 3. Process Chunks
            print("Generating embeddings and saving chunks...")
            for i, p in enumerate(paragraphs):
                embedding = await embed_client.get_embedding(p)
                chunk = KbChunk(
                    document_id=doc.id,
                    source_id=source.id,
                    chunk_index=i,
                    text_ru=p,
                    embedding=embedding,
                    crop=crop,
                    topic=topic,
                    metadata_={"file_path": str(file_path)}
                )
                session.add(chunk)
                
                # Update FTS (Postgres Full-Text Search)
                # Note: In a real app, this might be a DB trigger or a background task
                # Here we manually set it if possible or rely on the DB
                
            await session.commit()
            print(f"Successfully ingested {len(paragraphs)} chunks into the database.")

    # Update FTS vector manually for all new chunks
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE kb_chunks SET fts_ru = to_tsvector('russian', text_ru) WHERE fts_ru IS NULL"))
    
    await engine.dispose()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a text file into the Knowledge Base.")
    parser.add_argument("file", help="Path to the text file")
    parser.add_argument("--crop", help="Crop name (e.g., tomato)")
    parser.add_argument("--topic", help="Topic (e.g., diseases)")
    
    args = parser.parse_args()
    
    asyncio.run(ingest_file(Path(args.file), crop=args.crop, topic=args.topic))
