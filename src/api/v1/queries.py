from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from src.models.query import Query, QueryResponse, QueryStatus
from src.services.job_manager import job_manager
from typing import Dict
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/submit", response_model=Dict[str, str])
async def submit_query(query: Query):
    """Submit a query for processing on Puhti."""
    try:
        query_id = await job_manager.submit_job(
            query=query.text,
            params={
                "max_tokens": 300,
                "temperature": 0.1,
                "top_p": 0.95
            }
        )
        
        return {"query_id": query_id}
        
    except Exception as e:
        logger.error(f"Error submitting query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting query: {str(e)}"
        )

@router.get("/status/{query_id}", response_model=QueryStatus)
async def check_query_status(query_id: str):
    """Check the status of a submitted query."""
    try:
        status = await job_manager.check_job_status(query_id)
        return QueryStatus(**status)
        
    except Exception as e:
        logger.error(f"Error checking query status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking query status: {str(e)}"
        )

@router.get("/results/{query_id}", response_model=QueryResponse)
async def get_query_results(query_id: str):
    """Get the results of a completed query."""
    try:
        status = await job_manager.check_job_status(query_id)
        
        if status["status"] != "COMPLETED":
            raise HTTPException(
                status_code=404,
                detail=f"Results not yet available. Current status: {status['status']}"
            )
        
        results = status["results"]
        
        return QueryResponse(
            answer=results["generated_text"],
            sources=results.get("sources", []),
            confidence=results.get("confidence", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving query results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving query results: {str(e)}"
        )