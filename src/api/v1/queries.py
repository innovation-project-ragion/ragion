## src/api/v1/queries.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Optional
from src.models.query import QueryRequest, QueryResponse
from src.services.query_service import QueryService
from src.db.milvus import MilvusClient
from src.db.neo4j import Neo4jClient
import logging
import asyncio
router = APIRouter()
logger = logging.getLogger(__name__)

async def get_query_service():
    return QueryService()

async def get_milvus_client():
    """Get Milvus client instance."""
    client = MilvusClient()
    try:
        yield client
    finally:
        client.close()

async def get_neo4j_client():
    """Get Neo4j client instance."""
    client = Neo4jClient()
    try:
        yield client
    finally:
        client.close()

@router.post("/query", response_model=QueryResponse)
async def process_query(
    query: QueryRequest,
    background_tasks: BackgroundTasks,
    query_service: QueryService = Depends(get_query_service),
    milvus_client: MilvusClient = Depends(get_milvus_client),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """Process a query using RAG with Milvus and Neo4j."""
    try:
        # Submit query for processing
        result = await query_service.process_query(
            query=query.text,
            milvus_client=milvus_client,
            neo4j_client=neo4j_client,
            max_tokens=query.max_tokens
        )
        
        # Add background task to monitor LLM job
        background_tasks.add_task(
            monitor_query_completion,
            job_id=result["job_id"],
            query_service=query_service
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query/{job_id}")
async def get_query_status(
    job_id: str,
    query_service: QueryService = Depends(get_query_service)
) -> Dict:
    """Get status of a query processing job."""
    try:
        result = await query_service.check_query_status(job_id)
        return result
    except Exception as e:
        logger.error(f"Error checking query status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def monitor_query_completion(job_id: str, query_service: QueryService):
    """Monitor LLM job completion."""
    try:
        while True:
            result = await query_service.check_query_status(job_id)
            
            if result["status"] == "COMPLETED":
                logger.info(f"Query {job_id} completed successfully")
                break
                
            elif result["status"] == "FAILED":
                logger.error(f"Query {job_id} failed")
                break
                
            await asyncio.sleep(10)  # Check every 10 seconds
            
    except Exception as e:
        logger.error(f"Error monitoring query {job_id}: {str(e)}")