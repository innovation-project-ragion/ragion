from fastapi import APIRouter, HTTPException, Depends
from src.services.embedding_manager import EmbeddingManager as EnhancedEmbeddingManager
from src.db.milvus import MilvusClient
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

async def get_embedding_manager():
    return EnhancedEmbeddingManager()

async def get_milvus_client():
    return MilvusClient()

@router.post("/reload")
async def reload_pipeline(
    milvus_client: MilvusClient = Depends(get_milvus_client)
):
    """
    Reload the Milvus collection to refresh in-memory data.
    """
    try:
        milvus_client._reload_collection()
        return {"message": "Pipeline reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading pipeline: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/cache/clear")
async def clear_cache(
    embedding_manager: EnhancedEmbeddingManager = Depends(get_embedding_manager)
):
    """
    Clear the embedding cache.
    """
    try:
        embedding_manager.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )