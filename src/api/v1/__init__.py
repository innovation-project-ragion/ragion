from fastapi import APIRouter
from .documents import router as documents_router
# from .queries import router as queries_router
# from src.api.v1.pipline import router as pipeline_router
# from .neo4j import router as neo4j_router
# from .milvus import router as milvus_router
api_router = APIRouter()

api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
# api_router.include_router(queries_router, prefix="/query", tags=["queries"])
# api_router.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
# api_router.include_router(neo4j_router, prefix="/neo4j", tags=["neo4j"])
# api_router.include_router(milvus.router, prefix="/milvus", tags=["milvus"])