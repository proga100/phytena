# AI Agronomy Assistant Prototype

FastAPI prototype for comparing AI agronomy assistant architectures for Growz-style plant and crop diagnosis workflows.

Farmers eventually send:

- a plant/crop photo;
- a question in Russian, Uzbek Latin, Uzbek Cyrillic, or mixed language;
- optional context such as crop, region, and growth stage.

The prototype is designed to compare architecture options before building a production diagnosis system. It currently uses stubbed LLM/RAG/Vision behavior so the API, admin UI, traces, confidence routing, safety validation, and evaluation flow can be tested without external API keys.

The full architecture plan is in [docs/ai-agronomy-assistant-architecture-plan.md](docs/ai-agronomy-assistant-architecture-plan.md).

## Current Architecture

The system is a modular FastAPI monolith:

```text
User/Admin
  -> FastAPI
  -> image quality pipeline
  -> text normalization / RAG utilities
  -> pipeline A/B/C runner
  -> confidence fusion
  -> safety validation
  -> admin comparison UI
  -> evaluation runner
```

Implemented comparison pipelines:

- Pipeline A: pure LLM / Vision LLM baseline.
- Pipeline B: hybrid RAG with Postgres full-text search + pgvector.
- Pipeline C: hybrid RAG + reranker.

Current pipeline behavior is stubbed but structured. Each pipeline returns:

- diagnosis candidates;
- confidence;
- farmer-facing answer;
- actions;
- warnings;
- citations if applicable;
- trace metadata;
- safety validation flags.

## What Is Implemented

- Docker Compose with FastAPI API and Postgres + pgvector.
- FastAPI health/readiness endpoints.
- SQLAlchemy models and Alembic migration for the core schema.
- Admin query playground at `/admin`.
- Side-by-side A/B/C pipeline comparison view.
- `/v1/query` and `/v1/query/compare`.
- `/v1/retrieve` debug endpoint with query normalization and RRF merge.
- `/v1/image/quality-check` for local image quality metrics.
- `/v1/evals/run` stub golden-set evaluation runner.
- RU/UZ/mixed query normalization scaffolding.
- RRF utility for hybrid retrieval.
- Deterministic confidence fusion and decision rules.
- Safety validator for unsupported chemical and dosage advice.
- Pytest and Ruff checks.

## What Is Not Implemented Yet

- Real LLM provider integration.
- Real Gemini/Claude Vision API integration.
- Real embedding provider integration.
- Real pgvector retrieval against stored KB chunks.
- Real reranker provider integration.
- KB ingest from PDFs/web sources.
- Admin trace persistence pages backed by database rows.
- Human review queue persistence.
- LLM-as-judge evaluation.
- EfficientNet / ConvNeXt / YOLO / CLIP local models.

## Local Setup

Prerequisites:

- Docker Desktop or Docker daemon.
- Python 3.12+ for local non-Docker development.

Start with Docker:

```bash
cp .env.example .env
docker compose up --build -d
```

Open the admin UI:

[http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin)

Stop services:

```bash
docker compose down
```

## Services

Docker Compose starts:

| Service | Purpose | Port |
|---|---|---:|
| `api` | FastAPI app and admin UI | `8000` |
| `db` | PostgreSQL with pgvector | `5432` |

## Environment

Copy `.env.example` to `.env` for local overrides.

Important variables:

| Variable | Purpose |
|---|---|
| `APP_NAME` | FastAPI app title |
| `APP_ENV` | Runtime environment label |
| `DATABASE_URL` | Async SQLAlchemy database URL |
| `SYNC_DATABASE_URL` | Sync database URL for Alembic |
| `EMBEDDING_DIMENSION` | Vector dimension for pgvector schema |
| `LLM_PROVIDER` | Stub for future LLM provider selection |
| `VISION_PROVIDER` | Stub for future vision provider selection |
| `EMBEDDINGS_PROVIDER` | Stub for future embedding provider selection |
| `RERANKER_PROVIDER` | Stub for future reranker provider selection |

## API Endpoints

### Health

```text
GET /healthz
GET /readyz
GET /v1/system
```

Examples:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

### Pipeline Query

Run one pipeline:

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Пшеница желтеет, нижние листья сохнут, что делать?",
    "pipeline": "B_HYBRID_RAG",
    "context": {
      "crop": "пшеница",
      "region": "Узбекистан",
      "language": "ru"
    }
  }'
```

Compare all default pipelines:

```bash
curl -X POST http://127.0.0.1:8000/v1/query/compare \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Пшеница желтеет, нижние листья сохнут, что делать?"
  }'
```

Pipeline IDs:

```text
A_PURE_LLM
B_HYBRID_RAG
C_HYBRID_RAG_RERANK
```

### RAG Debug Retrieval

```bash
curl -X POST http://127.0.0.1:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Пшеница желтеет, нижние листья сохнут",
    "top_k": 3
  }'
```

This currently uses stub candidates but exercises:

- language/script detection;
- RU/UZ normalization;
- entity extraction;
- lexical candidate list;
- vector candidate list;
- RRF merge.

### Image Quality Check

```bash
curl -X POST http://127.0.0.1:8000/v1/image/quality-check \
  -F "image=@/path/to/photo.jpg"
```

The endpoint returns:

- original dimensions;
- normalized dimensions;
- blur score;
- exposure score;
- quality score;
- status: `pass`, `warn`, or `fail`;
- issues and retake guidance.

### Evaluation

```bash
curl -X POST http://127.0.0.1:8000/v1/evals/run \
  -H "Content-Type: application/json" \
  -d '{"limit": 1}'
```

The current evaluator runs built-in stub golden cases through A/B/C and returns:

- per-pipeline stub score;
- latency;
- citation presence;
- safety flags;
- summary metrics.

## Admin UI

Open:

```text
http://127.0.0.1:8000/admin
```

Current admin features:

- query playground;
- crop/region/language input;
- A/B/C comparison submit;
- side-by-side answers;
- confidence/cost/latency display;
- actions, warnings, citations;
- JSON trace summary.

Planned admin features:

- persisted trace detail page;
- RAG chunk browser;
- eval dashboard;
- human agronomist review queue;
- KB document/chunk inspection.

## Database

The first Alembic migration creates:

```text
kb_sources
kb_documents
kb_chunks
kb_entities
kb_chunk_entities
images
conversations
messages
pipeline_runs
retrieval_hits
traces
golden_items
eval_runs
eval_results
llm_judgments
feedback
human_reviews
prompt_versions
```

Postgres extensions:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
```

Run migrations inside the API container:

```bash
docker compose exec api alembic upgrade head
```

Current app endpoints do not require migrations yet because the first slices use in-memory/stub behavior. Migrations are ready for the next persistence slices.

## Local Python Development

Create a local virtualenv:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

Run tests:

```bash
.venv/bin/pytest -q
```

Run lint:

```bash
.venv/bin/ruff check app tests alembic
```

Run locally without Docker:

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Note: `/readyz` requires Postgres. Other stub endpoints work without the database.

## Verification Checklist

After `docker compose up --build -d`:

```text
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -o /tmp/phytena-admin.html -w "%{http_code}" http://127.0.0.1:8000/admin
curl -X POST http://127.0.0.1:8000/v1/query/compare -H "Content-Type: application/json" -d '{"question":"Пшеница желтеет"}'
curl -X POST http://127.0.0.1:8000/v1/retrieve -H "Content-Type: application/json" -d '{"query":"Пшеница желтеет","top_k":2}'
curl -X POST http://127.0.0.1:8000/v1/evals/run -H "Content-Type: application/json" -d '{"limit":1}'
```

Expected:

- `healthz` returns `{"status":"ok"}`.
- `readyz` returns status `ready`.
- `/admin` returns HTTP `200`.
- query comparison returns A/B/C results.
- retrieve returns ranked stub candidates.
- eval returns a summary for A/B/C.

## Implementation Roadmap

Next recommended slices:

1. Run Alembic migration automatically or document migration workflow in Compose.
2. Persist pipeline runs, traces, and retrieval hits.
3. Add KB ingest for manual Russian documents.
4. Replace stub retrieval with real Postgres FTS + pgvector retrieval.
5. Add embedding provider client.
6. Add external LLM client with strict JSON output parsing.
7. Add Vision API provider behind `image_analysis.v1`.
8. Add persisted admin trace detail pages.
9. Add LLM-as-judge and human review persistence.

## Safety Position

The prototype intentionally avoids real pesticide or dosage recommendations.

Production rules should remain:

- no invented pesticide names;
- no invented dosage;
- no chemical recommendation without cited, locale-appropriate source;
- low confidence should ask for clarification or agronomist review;
- farmer feedback is not treated as a ground-truth label without human review.
