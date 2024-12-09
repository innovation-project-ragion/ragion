## src/frontend/utils/document_api.py
import httpx
import asyncio
from typing import BinaryIO, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=1200.0,
            follow_redirects=True
        )
        # Separate client for status checks with shorter timeout
        self.status_client = httpx.AsyncClient(
            timeout=30.0,  # 30 seconds for status checks
            follow_redirects=True
        )


    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def upload_document(self, file: BinaryIO) -> Dict:
        """Upload a document to the backend service."""
        try:
            files = {"file": (file.name, file, "application/octet-stream")}
            response = await self.client.post(
                f"{self.base_url}/documents/upload",
                files=files
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise

    async def get_document_status(self, job_id: str) -> Dict:
        """Check the status of a document processing job."""
        try:
            response = await self.status_client.get(
                f"{self.base_url}/documents/status/{job_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error checking document status: {str(e)}")
            raise

    async def get_document_list(self) -> Dict:
        """Get list of processed documents."""
        try:
            response = await self.client.get(
                f"{self.base_url}/documents/list"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting document list: {str(e)}")
            raise

    async def close(self):
        """Close the HTTP clients."""
        await self.client.aclose()
        await self.status_client.aclose()