## src/frontend/utils/api_client.py
import httpx
from typing import Dict, Optional, AsyncGenerator
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

class RAGApiClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=600.0,
            follow_redirects=True,
            headers={"Content-Type": "application/json"}
        )
        self.status_check_client = httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={"Content-Type": "application/json"}
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def submit_query(self, query: str, max_tokens: int = 300) -> Dict:
        """Submit a query to the RAG service with extended retry logic."""
        retry_count = 0
        max_retries = 3
        base_wait = 3

        while retry_count < max_retries:
            try:
                logger.debug(f"Submitting query to {self.base_url}/query/")
                logger.debug(f"Request data: {{'text': {query}, 'max_tokens': {max_tokens}}}")

                response = await self.client.post(
                    f"{self.base_url}/query/",
                    json={"text": query, "max_tokens": max_tokens}
                )
                
                response.raise_for_status()
                data = response.json()
                
                if not isinstance(data, dict) or "status" not in data:
                    raise ValueError(f"Invalid response format: {data}")

                return data

            except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                retry_count += 1
                wait_time = base_wait * (2 ** retry_count)
                
                if retry_count < max_retries:
                    logger.warning(f"Timeout occurred, retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception("Backend initialization timeout - model loading and embedding generation may take longer than expected")

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"Failed to submit query: {str(e)}")
                
                wait_time = base_wait * (2 ** retry_count)
                logger.warning(f"Retry {retry_count}/{max_retries}: {str(e)}")
                await asyncio.sleep(wait_time)

    async def check_query_status(self, job_id: str) -> Dict:
        """Check the status of a submitted query with enhanced response handling."""
        try:
            logger.debug(f"Checking status for job {job_id}")
            response = await self.status_check_client.get(
                f"{self.base_url}/query/{job_id}/"
            )
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Status check response: {data}")

            if data.get("status") == "completed":
                complete_response = {
                    "status": "completed",
                    "query": data.get("query", ""),
                    "response": data.get("response", ""),
                    "sources": data.get("sources", []),
                    "person_contexts": data.get("person_contexts", []),
                    "params": data.get("params", {}),
                    "result": data.get("result", {}) 
                }
                return complete_response
            
            return data

        except Exception as e:
            logger.error(f"Error checking status: {str(e)}")
            raise Exception(f"Failed to check query status: {str(e)}")

    async def stream_response(self, job_id: str, poll_interval: float = 2.0) -> AsyncGenerator[str, None]:
        """Stream the response with improved logging and error handling."""
        retries = 0
        max_retries = 5
        initialization_phase = True
        start_time = asyncio.get_event_loop().time()
        timeout = 300  # 5 minutes total timeout
        
        while True:
            try:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise TimeoutError("Query processing timeout")

                result = await self.check_query_status(job_id)
                logger.debug(f"Status check response: {result}")
                
                if result["status"] == "completed":
                    if "result" in result:
                        logger.debug("Yielding complete response with result structure")
                        yield json.dumps(result)
                        break
                elif result["status"] == "failed":
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"Job failed: {error_msg}")
                    yield f"Error: {error_msg}"
                    break
                elif result["status"] == "processing":
                    if initialization_phase:
                        await asyncio.sleep(poll_interval * 2)
                        initialization_phase = False
                    else:
                        await asyncio.sleep(poll_interval)
                else:
                    logger.warning(f"Unexpected status: {result['status']}")
                    await asyncio.sleep(poll_interval)
                
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Max retries reached: {str(e)}")
                    yield f"Error: Connection failed after {max_retries} attempts"
                    break
                    
                wait_time = poll_interval * (2 ** retries)
                logger.warning(f"Retry {retries}/{max_retries}: {str(e)}")
                await asyncio.sleep(wait_time)

    async def close(self):
        """Close the HTTP clients."""
        await self.client.aclose()
        await self.status_check_client.aclose()