## src/api/v1/queries.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Optional
from src.models.query import QueryRequest, QueryResponse, ErrorResponse, QueryStatusResponse
from src.services.query_service import QueryService
from src.db.milvus import MilvusClient
from src.db.neo4j import Neo4jClient
import logging
import asyncio
from src.services.job_manager import PuhtiJobManager
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
    """Submit a query for processing."""
    try:
        result = await query_service.process_query(
            query=query.text,
            milvus_client=milvus_client,
            neo4j_client=neo4j_client,
            max_tokens=query.max_tokens
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return ErrorResponse(
            message="Failed to process query",
            error=str(e)
        )

@router.get("/query/{job_id}", response_model=QueryStatusResponse)
async def get_query_status(
    job_id: str,
    query_service: QueryService = Depends(get_query_service)
):
    """Get status of a query processing job."""
    try:
        result = await query_service.check_query_status(job_id)
        return QueryStatusResponse(
            status=result["status"],
            job_id=job_id,
            result=result.get("response"),
            message=result.get("message"),
            error=result.get("error")
        )
    except Exception as e:
        return QueryStatusResponse(
            status="error",
            job_id=job_id,
            message="Failed to check query status",
            error=str(e)
        )

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

# In your API endpoint or test script
@router.get("/test")
async def test_llm():
    job_manager = PuhtiJobManager()
    test_result = await job_manager.test_llm_job_with_context()
    return {
        "success": test_result,
        "message": "LLM test job completed successfully" if test_result else "LLM test job failed"
    }