# AI Agronomy Assistant Prototype

FastAPI prototype for comparing AI agronomy assistant architectures:

- Pipeline A: pure LLM / Vision LLM baseline.
- Pipeline B: hybrid RAG with Postgres full-text search + pgvector.
- Pipeline C: hybrid RAG + reranker.

The architecture plan is in [docs/ai-agronomy-assistant-architecture-plan.md](docs/ai-agronomy-assistant-architecture-plan.md).

## Local Setup

```bash
cp .env.example .env
docker compose up --build
```

Health endpoints:

```text
GET http://localhost:8000/healthz
GET http://localhost:8000/readyz
GET http://localhost:8000/admin
```
