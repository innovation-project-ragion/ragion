from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from src.models.document import DocumentResponse, ProcessedChunk
from src.services.document_processor import DocumentProcessor
from src.services.embedding_manager import EmbeddingManager as EnhancedEmbeddingManager
from src.db.milvus import MilvusClient
from src.db.neo4j import Neo4jClient
from typing import List
import logging
import torch
import numpy as np
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


async def get_job_manager():
    return PuhtiJobManager()


from subprocess import run, CalledProcessError
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from typing import Dict
from src.services.job_manager import PuhtiJobManager
from src.core.config import settings
router = APIRouter()
logger = logging.getLogger(__name__)
import asyncio


async def monitor_job_completion(
    job_id: str, 
    job_manager: PuhtiJobManager, 
    milvus_client: MilvusClient,
    neo4j_client: Neo4jClient
):
    """Monitor job completion and store results in both Milvus and Neo4j."""
    try:
        while True:
            job_info = await job_manager.check_embedding_job(job_id)
            
            if job_info["status"] == "COMPLETED":
                embeddings = job_info["embeddings"]
                texts = job_info["texts"]
                metadata = job_info["metadata"]
                if isinstance(embeddings, np.ndarray):
                    embedding_dim = embeddings.shape[1]
                else:
                    embedding_dim = embeddings[0].shape[0]
                
                logger.info(f"Received embeddings with dimension: {embedding_dim}")
                
                if embedding_dim != settings.EMBEDDING_DIM:
                    logger.error(f"Dimension mismatch: expected {settings.EMBEDDING_DIM}, got {embedding_dim}")
                    raise ValueError(f"Embedding dimension mismatch")
                
                # 1. Store in Milvus
                entities = []
                for i, (embedding, text) in enumerate(zip(embeddings, texts)):
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.numpy()
                    entities.append({
                        "text": text,
                        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                        "person_name": metadata["person_name"],
                        "person_age": metadata["person_age"],
                        "document_id": metadata["document_id"],
                        "chunk_index": i
                    })
                
                milvus_client.collection.insert(entities)
                milvus_client.collection.flush()
                logger.info(f"Stored {len(entities)} embeddings in Milvus for job {job_id}")

                # 2. Store in Neo4j
                try:
                    # Create person node
                    await neo4j_client.create_person(
                        name=metadata["person_name"],
                        age=metadata["person_age"]
                    )

                    # Create document nodes and relationships
                    for i, text in enumerate(texts):
                        await neo4j_client.create_document_chunk(
                            document_id=f"{metadata['document_id']}_{i}",
                            text=text,
                            person_name=metadata["person_name"],
                            chunk_index=i
                        )
                    
                    logger.info(f"Stored document in Neo4j for job {job_id}")
                except Exception as e:
                    logger.error(f"Error storing in Neo4j: {str(e)}")
                
                break
                
            elif job_info["status"] == "FAILED":
                logger.error(f"Job {job_id} failed")
                break
                
            await asyncio.sleep(30)
            
    except Exception as e:
        logger.error(f"Error monitoring job {job_id}: {str(e)}")

@router.post("/upload")
async def upload_and_process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_manager: PuhtiJobManager = Depends(get_job_manager),
    milvus_client: MilvusClient = Depends(get_milvus_client),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """Upload document and process it on Puhti, then store in both Milvus and Neo4j."""
    try:
        temp_path = Path(f"/tmp/{file.filename}")
        with temp_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        
        # Submit embedding job to Puhti
        job_id, metadata = await job_manager.submit_embedding_job(temp_path)
        
        # Clean up temporary file
        temp_path.unlink()
        
        # Add background task with both Milvus and Neo4j clients
        background_tasks.add_task(
            monitor_job_completion,
            job_id=job_id,
            job_manager=job_manager,
            milvus_client=milvus_client,
            neo4j_client=neo4j_client
        )
        
        return {
            "message": "Document processing started",
            "job_id": job_id,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
