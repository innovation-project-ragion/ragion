# Document endpoints
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from src.models.document import DocumentResponse, ProcessedChunk
from src.services.document_processor import DocumentProcessor
from src.services.embedding_manager import EmbeddingManager as EnhancedEmbeddingManager
from src.db.milvus import MilvusClient
from src.db.neo4j import Neo4jClient
from typing import List
import logging
#from src.services.job_manager import submit_job_to_puhti
#from src.services.job_script_generator import generate_batch_script
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)


async def get_document_processor():
    return DocumentProcessor()


async def get_embedding_manager():
    return EnhancedEmbeddingManager()


async def get_milvus_client():
    return MilvusClient()


async def get_neo4j_client():
    return Neo4jClient()


from subprocess import run, CalledProcessError
import os


# @router.post("/upload")
# async def upload_and_process_document(file: UploadFile = File(...)):
#     """
#     Upload a document, generate embeddings, and store in Milvus.
#     """
#     logger.info(f"Received file: {file.filename}")
#     try:
#         # Replace spaces in filename with underscores
#         sanitized_filename = file.filename.replace(" ", "_")
#         local_path = Path(f"/tmp/{sanitized_filename}")
        
#         # Save the uploaded file locally
#         with local_path.open("wb") as f:
#             f.write(await file.read())
#         logger.info(f"File saved locally at: {local_path}")

#         # Generate batch script
#         script_path = generate_batch_script(
#             job_name=f"embed_{sanitized_filename}",
#             input_path=f"/scratch/project_2011638/{sanitized_filename}"
#         )
#         logger.info(f"Generated batch script at: {script_path}")

#         # Submit the job
#         job_id = submit_job_to_puhti(script_path)
#         logger.info(f"Job submitted successfully. Job ID: {job_id}")
#         return {"message": f"Job submitted successfully", "job_id": job_id}

#     except Exception as e:
#         logger.error(f"Error in upload_and_process_document: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/", response_model=List[DocumentResponse])
# async def list_documents(milvus_client: MilvusClient = Depends(get_milvus_client)):
#     """
#     List all processed documents.
#     """
#     try:
#         collection = milvus_client.collection
#         results = collection.query(
#             expr="", output_fields=["document_id", "person_name", "chunk_index"]
#         )

#         # Group by document_id
#         documents = {}
#         for result in results:
#             doc_id = result.get("document_id")
#             if doc_id not in documents:
#                 documents[doc_id] = {
#                     "id": doc_id,
#                     "person_name": result.get("person_name"),
#                     "chunk_count": 0,
#                 }
#             documents[doc_id]["chunk_count"] += 1

#         return list(documents.values())
#     except Exception as e:
#         logger.error(f"Error listing documents: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.delete("/{document_id}")
# async def delete_document(
#     document_id: str,
#     milvus_client: MilvusClient = Depends(get_milvus_client),
#     neo4j_client: Neo4jClient = Depends(get_neo4j_client),
# ):
#     """
#     Delete a document and its chunks from both Milvus and Neo4j.
#     """
#     try:
#         # Delete from Milvus
#         expr = f'document_id == "{document_id}"'
#         milvus_client.collection.delete(expr)

#         # Delete from Neo4j
#         await neo4j_client.delete_document(document_id)

#         return {"message": f"Document {document_id} deleted successfully"}
#     except Exception as e:
#         logger.error(f"Error deleting document: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from typing import Dict
from src.services.job_manager import PuhtiJobManager
router = APIRouter()
logger = logging.getLogger(__name__)
import asyncio
# Dependency injection functions
async def get_job_manager() -> PuhtiJobManager:
    job_manager = PuhtiJobManager()
    try:
        yield job_manager
    finally:
        job_manager.cleanup()

@router.post("/upload")
async def upload_and_process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_manager: PuhtiJobManager = Depends(get_job_manager),
    milvus_client: MilvusClient = Depends(get_milvus_client)
):
    """Upload document and start processing on Puhti."""
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        with temp_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        
        # Submit embedding job to Puhti
        job_id, metadata = await job_manager.submit_embedding_job(temp_path)
        
        # Clean up temporary file
        temp_path.unlink()
        
        # Add background task to check job completion
        background_tasks.add_task(
            monitor_job_completion,
            job_id=job_id,
            job_manager=job_manager,
            milvus_client=milvus_client
        )
        
        return {
            "message": "Document processing started",
            "job_id": job_id,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def monitor_job_completion(job_id: str, job_manager: PuhtiJobManager, milvus_client: MilvusClient):
    """Monitor job completion and store results in Milvus."""
    try:
        while True:
            job_info = await job_manager.check_embedding_job(job_id)
            
            if job_info["status"] == "COMPLETED":
                # Prepare data for Milvus
                embeddings = job_info["embeddings"]
                texts = job_info["texts"]
                metadata = job_info["metadata"]
                
                # Create entities for Milvus
                entities = []
                for i, (embedding, text) in enumerate(zip(embeddings, texts)):
                    entities.append({
                        "text": text,
                        "embedding": embedding.tolist(),
                        "person_name": metadata["person_name"],
                        "person_age": metadata["person_age"],
                        "document_id": metadata["document_id"],
                        "chunk_index": i
                    })
                
                # Insert into Milvus
                milvus_client.collection.insert(entities)
                milvus_client.collection.flush()
                logger.info(f"Successfully stored embeddings for job {job_id}")
                break
                
            elif job_info["status"] == "FAILED":
                logger.error(f"Job {job_id} failed")
                break
                
            await asyncio.sleep(30)  # Check every 30 seconds
            
    except Exception as e:
        logger.error(f"Error monitoring job {job_id}: {str(e)}")

@router.get("/job/{job_id}")
async def get_job_status(
    job_id: str,
    job_manager: PuhtiJobManager = Depends(get_job_manager)
) -> Dict:
    """Get status of a processing job."""
    try:
        job_info = await job_manager.check_embedding_job(job_id)
        return {
            "status": job_info["status"],
            "metadata": job_info.get("metadata", {}),
            "submitted_at": job_info["submitted_at"]
        }
    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))