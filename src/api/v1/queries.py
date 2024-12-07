## src/api/v1/queries.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Optional
from src.models.query import QueryRequest, QueryResponse, ErrorResponse, QueryStatusResponse, CompletedQueryResponse
from src.services.query_service import QueryService
from src.db.milvus import MilvusClient
from src.db.neo4j import Neo4jClient
import logging
import asyncio
import sys
import os
from datetime import datetime
from src.services.job_manager import PuhtiJobManager
from src.services.service_factory import ServiceFactory

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()
logger = logging.getLogger(__name__)

async def get_query_service():
    return await ServiceFactory.get_query_service()

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

@router.post("/", response_model=QueryResponse)
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
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/{job_id}/", response_model=QueryStatusResponse)
async def get_query_status(
    job_id: str,
    query_service: QueryService = Depends(get_query_service)
):
    """Get status of a query processing job."""
    try:
        result = await query_service.check_query_status(job_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No job found with ID {job_id}"
            )
        
        # Handle different status cases appropriately
        if result["status"] == "COMPLETED":
            if not isinstance(result.get("result"), dict):
                logger.error(f"Invalid result format for job {job_id}: {result}")
                raise HTTPException(
                    status_code=500,
                    detail="Invalid result format from query service"
                )
            
            return QueryStatusResponse(
                status="completed",
                job_id=job_id,
                result=result["result"],
                message="Query completed successfully"
            )
        elif result["status"] == "FAILED":
            return QueryStatusResponse(
                status="failed",
                job_id=job_id,
                message=result.get("message", "Query processing failed"),
                error=result.get("error")
            )
        else:
            return QueryStatusResponse(
                status="processing",
                job_id=job_id,
                message="Query is still being processed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking status for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
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

@router.get("/healthcheck")
async def healthcheck():
    """Check if the API is responsive."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }
# In your API endpoint or test script
@router.get("/test")
async def test_llm():
    job_manager = PuhtiJobManager()
    test_result = await job_manager.test_llm_job_with_context()
    return {
        "success": test_result,
        "message": "LLM test job completed successfully" if test_result else "LLM test job failed"
    }