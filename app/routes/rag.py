from fastapi import APIRouter

from app.rag.retrieve import retrieve_stub
from app.schemas import RetrieveRequest, RetrieveResponse

router = APIRouter(prefix="/v1", tags=["rag"])


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    return retrieve_stub(request.query, language=request.language, top_k=request.top_k)
