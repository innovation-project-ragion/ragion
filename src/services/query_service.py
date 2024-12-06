# query_service.py
from typing import Dict, List
import logging
from pathlib import Path
import json
from .job_manager import PuhtiJobManager
from ..db.milvus import MilvusClient
from ..db.neo4j import Neo4jClient
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.puhti_job_manager = PuhtiJobManager()
        self.query_path = Path("/scratch/project_2011638/rag_queries")

    async def process_query(
        self, 
        query: str,
        milvus_client: MilvusClient,
        neo4j_client: Neo4jClient,
        max_tokens: int = 300
    ) -> Dict:
        """Process a query through the RAG pipeline."""
        milvus_results = []
        person_contexts = []
        mentioned_persons = []
        
        try:
            logger.info(f"Processing query: {query}")
            
            # 1. Generate query embedding locally
            try:
                query_embedding = await self.embedding_service.generate_embedding(query)
                logger.info("Generated query embedding")
            except Exception as e:
                return {
                    "status": "error",
                    "message": "Failed to generate query embedding",
                    "error": str(e)
                }
            
            # 2. Search Milvus
            try:
                milvus_results = await milvus_client.search(
                    query_embedding=query_embedding,
                    limit=5
                )
                logger.info(f"Found {len(milvus_results)} relevant passages in Milvus")
            except Exception as e:
                return {
                    "status": "error",
                    "message": "Failed to search knowledge base",
                    "error": str(e),
                    "data": {
                        "milvus_hits": 0,
                        "person_contexts": 0
                    }
                }
            
            # 3. Get Neo4j context
            try:
                mentioned_persons = await neo4j_client.find_mentioned_persons(query)
                if mentioned_persons:
                    for person in mentioned_persons:
                        context = await neo4j_client.get_person_context(person)
                        if context:
                            person_contexts.append(context)
                logger.info(f"Found {len(person_contexts)} person contexts in Neo4j")
            except Exception as e:
                logger.error(f"Neo4j error: {str(e)}")
                # Continue without Neo4j context
                pass
            
            # 4. Prepare context for LLM
            try:
                context = self._prepare_context(milvus_results, person_contexts)
            except Exception as e:
                return {
                    "status": "error",
                    "message": "Failed to prepare context",
                    "error": str(e),
                    "data": {
                        "milvus_hits": len(milvus_results),
                        "person_contexts": len(person_contexts)
                    }
                }
            
            # 5. Submit LLM job to Puhti
            try:
                input_data = {
                    "query": query,
                    "context": context,
                    "params": {
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                        "top_p": 0.95
                    }
                }
                
                job_id = await self.puhti_job_manager.submit_llm_job(input_data)
                
                if not job_id:
                    return {
                        "status": "error",
                        "message": "Failed to get job ID from Puhti",
                        "data": {
                            "milvus_hits": len(milvus_results),
                            "person_contexts": len(person_contexts),
                            "mentioned_persons": mentioned_persons
                        }
                    }
                    
                logger.info(f"Submitted LLM job: {job_id}")
                
                # 6. Return successful response
                return {
                    "status": "processing",
                    "message": "Query is being processed",
                    "job_id": job_id,  # Ensure job_id is included
                    "data": {
                        "milvus_hits": len(milvus_results),
                        "person_contexts": len(person_contexts),
                        "mentioned_persons": mentioned_persons,
                        "context_length": len(context.split())
                    }
                }
                
            except Exception as e:
                logger.error(f"Error submitting LLM job: {str(e)}")
                return {
                    "status": "error",
                    "message": "Failed to submit processing job",
                    "error": str(e),
                    "data": {
                        "milvus_hits": len(milvus_results),
                        "person_contexts": len(person_contexts),
                        "mentioned_persons": mentioned_persons
                    }
                }
                
        except Exception as e:
            logger.error(f"Unexpected error in process_query: {str(e)}")
            return {
                "status": "error",
                "message": "An unexpected error occurred",
                "error": str(e),
                "data": {
                    "milvus_hits": len(milvus_results),
                    "person_contexts": len(person_contexts),
                    "mentioned_persons": mentioned_persons
                }
            }

    def _prepare_context(self, milvus_results: List[Dict], person_contexts: List[Dict]) -> str:
        """Prepare context for LLM."""
        try:
            context_parts = []
            
            # Add vector search results
            context_parts.append("# Relevantti konteksti:")
            for i, result in enumerate(milvus_results, 1):
                context_parts.append(f"\nDokumentti {i} ({result['document_id']}):")
                context_parts.append(result["text"])
                
            # Add person contexts
            if person_contexts:
                context_parts.append("\n# Henkilötiedot:")
                for context in person_contexts:
                    context_parts.append(f"\n{context['name']}:")
                    context_parts.append(f"- Ikä: {context['age']}")
                    context_parts.append(f"- Suhteet: {', '.join(context['relationships'])}")
            
            # Create the final context
            final_context = "\n".join(context_parts)
            
            # Log the context
            logger.info("Prepared context for LLM:")
            logger.info("-" * 80)
            logger.info(final_context)
            logger.info("-" * 80)
            logger.info(f"Context length: {len(final_context.split())} words")
            
            return final_context
        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            raise

    async def _submit_llm_job(self, query: str, context: str, max_tokens: int) -> str:
        """Submit LLM job to Puhti."""
        try:
            # Log the input data
            logger.info(f"Preparing LLM job with query: {query}")
            logger.info(f"Max tokens: {max_tokens}")
            
            # Create input data
            input_data = {
                "query": query,
                "context": context,
                "params": {
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "top_p": 0.95
                }
            }
            
            # Log input data size
            input_data_str = json.dumps(input_data, ensure_ascii=False)
            logger.info(f"Input data size: {len(input_data_str)} bytes")
            logger.info("Input data structure:")
            logger.info(json.dumps(input_data, ensure_ascii=False, indent=2)[:1000] + "..." if len(input_data_str) > 1000 else input_data_str)
            
            # Submit to Puhti
            job_id = await self.puhti_job_manager.submit_llm_job(input_data)
            
            if job_id:
                logger.info(f"Successfully created LLM job with ID: {job_id}")
            else:
                logger.error("Failed to get job ID from Puhti submission")
                
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting LLM job: {str(e)}")
            raise

    async def check_query_status(self, job_id: str) -> Dict:
        """Check status of query processing job."""
        try:
            job_info = await self.puhti_job_manager.check_llm_job(job_id)
            return job_info
        except Exception as e:
            logger.error(f"Error checking query status: {str(e)}")
            raise