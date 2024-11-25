# Document endpoints
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from src.models.document import DocumentResponse, ProcessedChunk
from src.services.document_processor import DocumentProcessor
from src.services.embedding_manager import EmbeddingManager as EnhancedEmbeddingManager
from src.db.milvus import MilvusClient
from src.db.neo4j import Neo4jClient
from typing import List
import logging

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

@router.post("/upload", response_model=List[DocumentResponse])
async def upload_document(
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_document_processor),
    embedding_manager: EnhancedEmbeddingManager = Depends(get_embedding_manager),
    milvus_client: MilvusClient = Depends(get_milvus_client),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Upload and process a document, storing it in both Milvus and Neo4j.
    """
    try:
        # Process document into chunks
        chunks = await processor.process_file(file)
        
        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_manager.generate(texts)
        
        # Prepare entities for Milvus
        entities = []
        for chunk, embedding in zip(chunks, embeddings):
            entity = {
                "text": chunk.text,
                "embedding": embedding.tolist(),
                "person_name": chunk.metadata.get("person_name"),
                "person_age": chunk.metadata.get("person_age"),
                "document_id": chunk.metadata.get("document_id"),
                "chunk_index": chunk.metadata.get("chunk_index")
            }
            entities.append(entity)
        
        # Insert into Milvus
        milvus_client.collection.insert(entities)
        milvus_client.collection.flush()
        
        # Create graph structure in Neo4j
        await neo4j_client.create_document_graph(chunks)
        
        return chunks
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    milvus_client: MilvusClient = Depends(get_milvus_client)
):
    """
    List all processed documents.
    """
    try:
        collection = milvus_client.collection
        results = collection.query(
            expr="",
            output_fields=["document_id", "person_name", "chunk_index"]
        )
        
        # Group by document_id
        documents = {}
        for result in results:
            doc_id = result.get("document_id")
            if doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "person_name": result.get("person_name"),
                    "chunk_count": 0
                }
            documents[doc_id]["chunk_count"] += 1
        
        return list(documents.values())
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    milvus_client: MilvusClient = Depends(get_milvus_client),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Delete a document and its chunks from both Milvus and Neo4j.
    """
    try:
        # Delete from Milvus
        expr = f'document_id == "{document_id}"'
        milvus_client.collection.delete(expr)
        
        # Delete from Neo4j
        await neo4j_client.delete_document(document_id)
        
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )