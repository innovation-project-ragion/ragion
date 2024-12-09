## src/api/v1/__init__.py

from fastapi import APIRouter
from .documents import router as documents_router
from .queries import router as queries_router
api_router = APIRouter()

api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(
    queries_router, 
    prefix="/query",
    tags=["queries"]
)
