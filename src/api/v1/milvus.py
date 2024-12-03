from fastapi import APIRouter, HTTPException
from src.db.milvus import MilvusClient
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/milvus/create-collection")
async def create_milvus_collection():
    try:
        milvus_client = MilvusClient()
        milvus_client._reload_collection()
        return {"message": "Milvus collection created or already exists."}
    except Exception as e:
        logger.error(f"Error creating Milvus collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
