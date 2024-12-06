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
        try:
            logger.info(f"Processing query: {query}")
            
            # 1. Generate query embedding locally
            query_embedding = await self.embedding_service.generate_embedding(query)
            logger.info("Generated query embedding")
            
            # 2. Search Milvus
            milvus_results = await milvus_client.search(
                query_embedding=query_embedding,
                limit=5
            )
            logger.info(f"Found {len(milvus_results)} relevant passages in Milvus")
            
            # 3. Get Neo4j context
            mentioned_persons = await neo4j_client.find_mentioned_persons(query)
            person_contexts = []
            if mentioned_persons:
                for person in mentioned_persons:
                    context = await neo4j_client.get_person_context(person)
                    if context:
                        person_contexts.append(context)
            logger.info(f"Found {len(person_contexts)} person contexts in Neo4j")
            
            # 4. Prepare context for LLM
            context = self._prepare_context(milvus_results, person_contexts)
            
            # 5. Submit LLM job to Puhti
            job_id = await self._submit_llm_job(query, context, max_tokens)
            logger.info(f"Submitted LLM job: {job_id}")
            
            # 6. Return initial response with job ID
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Query is being processed",
                "initial_results": {
                    "milvus_hits": len(milvus_results),
                    "person_contexts": len(person_contexts)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _prepare_context(self, milvus_results: List[Dict], person_contexts: List[Dict]) -> str:
        """Prepare context for LLM."""
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
        
        return "\n".join(context_parts)

    async def _submit_llm_job(self, query: str, context: str, max_tokens: int) -> str:
        """Submit LLM job to Puhti."""
        try:
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
            
            # Submit to Puhti
            job_id = await self.puhti_job_manager.submit_llm_job(input_data)
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