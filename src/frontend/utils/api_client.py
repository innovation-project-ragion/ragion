# src/frontend/utils/api_client.py
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
            timeout=30.0,
            follow_redirects=True,
            headers={"Content-Type": "application/json"}
        )

    async def submit_query(self, query: str, max_tokens: int = 300) -> Dict:
        """Submit a query to the RAG service."""
        try:
            # Log the request details
            logger.debug(f"Submitting query to {self.base_url}/query/")
            logger.debug(f"Request data: {{'text': {query}, 'max_tokens': {max_tokens}}}")

            response = await self.client.post(
                f"{self.base_url}/query/",
                json={"text": query, "max_tokens": max_tokens}
            )
            
            # Log the response
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            logger.debug(f"Response content: {response.text}")

            response.raise_for_status()
            data = response.json()
            
            # Check if we got a valid response with status
            if not isinstance(data, dict) or "status" not in data:
                raise ValueError(f"Invalid response format: {data}")

            return data

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.text}")
            raise Exception(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response JSON: {e}")
            raise Exception(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise Exception(f"Failed to submit query: {str(e)}")

    async def check_query_status(self, job_id: str) -> Dict:
        """Check the status of a submitted query."""
        try:
            logger.debug(f"Checking status for job {job_id}")
            response = await self.client.get(
                f"{self.base_url}/query/{job_id}/"
            )
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Status check response: {data}")
            return data

        except Exception as e:
            logger.error(f"Error checking status: {str(e)}")
            raise Exception(f"Failed to check query status: {str(e)}")

    async def stream_response(self, job_id: str, poll_interval: float = 1.0) -> AsyncGenerator[str, None]:
        """Stream the response as it becomes available."""
        retries = 0
        max_retries = 3
        
        while True:
            try:
                result = await self.check_query_status(job_id)
                
                if result["status"] == "completed" and result.get("result"):
                    yield json.dumps(result["result"])
                    break
                elif result["status"] == "failed":
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"Job failed: {error_msg}")
                    yield f"Error: {error_msg}"
                    break
                elif result["status"] == "processing":
                    # Reset retry counter on successful processing status
                    retries = 0
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
                    
                logger.warning(f"Retry {retries}/{max_retries}: {str(e)}")
                await asyncio.sleep(poll_interval * retries)  # Exponential backoff

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()