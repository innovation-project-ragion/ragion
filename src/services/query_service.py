# src/services/query_service.py
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import aiofiles
from .job_manager import PuhtiJobManager
from ..db.milvus import MilvusClient
from ..db.neo4j import Neo4jClient
from .embedding_service import EmbeddingService
from src.models.query import CompletedQueryResponse, PersonContext, QueryResponse, QuerySource, InitialQueryResponse

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.puhti_job_manager = PuhtiJobManager()
        self.query_path = Path("/scratch/project_2011638/rag_queries")
        self.local_results_path = Path("./results")
        self.local_results_path.mkdir(exist_ok=True)
        self.jobs = {}
    async def _ensure_query_dirs(self):
        """Ensure query directories exist on both Puhti and locally."""
        try:
            # Create Puhti directories through SSH
            await self.puhti_job_manager.connect()
            sftp = self.puhti_job_manager.ssh_client.open_sftp()
            
            # Create base directory first
            try:
                sftp.stat(str(self.query_path))
            except FileNotFoundError:
                logger.info(f"Creating base directory: {self.query_path}")
                sftp.mkdir(str(self.query_path))
            
            # Create subdirectories
            for subdir in ['inputs', 'outputs']:
                dir_path = self.query_path / subdir
                try:
                    sftp.stat(str(dir_path))
                except FileNotFoundError:
                    logger.info(f"Creating subdirectory: {dir_path}")
                    sftp.mkdir(str(dir_path))
            
            # Create local results directory
            self.local_results_path.mkdir(parents=True, exist_ok=True)
            (self.local_results_path / "cache").mkdir(exist_ok=True)
            
            logger.info("All required directories created successfully")
            
        except Exception as e:
            logger.error(f"Error ensuring query directories: {str(e)}")
            raise
    
    async def _retrieve_answer(self, job_id: str) -> Optional[CompletedQueryResponse]:
        """Retrieve and parse answer from Puhti."""
        try:
            sftp = self.puhti_job_manager.ssh_client.open_sftp()
            remote_path = self.query_path / "outputs" / f"response_{job_id}.json"
            local_cache_path = self.local_results_path / "cache" / f"{job_id}.json"

            try:
                # Download result
                sftp.get(str(remote_path), str(local_cache_path))
                
                # Parse result
                async with aiofiles.open(local_cache_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                
                # Convert to CompletedQueryResponse format
                sources = [
                    QuerySource(
                        text=source["text"],
                        document_id=source["document_id"],
                        score=source.get("score", 0.0)
                    )
                    for source in data.get("sources", [])
                ]
                
                person_contexts = [
                    PersonContext(
                        name=ctx["name"],
                        age=ctx.get("age"),
                        relationships=ctx.get("relationships", []),
                        document_count=ctx.get("document_count", 0)
                    )
                    for ctx in data.get("person_contexts", [])
                ]
                
                result = CompletedQueryResponse(
                    query=data["query"],
                    answer=data["response"],
                    sources=sources,
                    person_contexts=person_contexts,
                    confidence=data.get("confidence", 0.0)
                )
                
                # Clean up remote files
                sftp.remove(str(remote_path))
                input_path = self.query_path / "inputs" / f"query_{job_id}.json"
                try:
                    sftp.remove(str(input_path))
                except FileNotFoundError:
                    pass
                
                return result
                
            except FileNotFoundError:
                logger.warning(f"Results not yet available for job {job_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving answer: {str(e)}")
            raise

    async def process_query(
        self, 
        query: str,
        milvus_client: MilvusClient,
        neo4j_client: Neo4jClient,
        max_tokens: int = 300
    ) -> QueryResponse:
        """Process a query through the RAG pipeline."""
        milvus_results = []
        person_contexts = []
        mentioned_persons = []
        
        try:
            # Ensure query directories exist
            await self._ensure_query_dirs()
            logger.info(f"Processing query: {query}")
            
            # 1. Generate query embedding locally
            try:
                query_embedding = await self.embedding_service.generate_embedding(query)
                logger.info("Generated query embedding")
            except Exception as e:
                logger.error(f"Embedding generation failed: {str(e)}")
                return InitialQueryResponse(
                    status="error",
                    message="Failed to generate query embedding",
                    error=str(e)
                )
            
            # 2. Search Milvus
            try:
                milvus_results = await milvus_client.search(
                    query_embedding=query_embedding,
                    limit=5
                )
                logger.info(f"Found {len(milvus_results)} relevant passages in Milvus")
            except Exception as e:
                logger.error(f"Milvus search failed: {str(e)}")
                return InitialQueryResponse(
                    status="error",
                    message="Failed to search knowledge base",
                    error=str(e),
                    data={
                        "milvus_hits": 0,
                        "person_contexts": 0,
                        "mentioned_persons": [],
                        "context_length": 0
                    }
                )
            
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
                logger.warning(f"Neo4j context retrieval failed: {str(e)}")
                # Continue without Neo4j context
                
            # 4. Prepare context for LLM
            try:
                context = self._prepare_context(milvus_results, person_contexts)
                logger.info(f"Prepared context with {len(context.split())} words")
            except Exception as e:
                logger.error(f"Context preparation failed: {str(e)}")
                return InitialQueryResponse(
                    status="error",
                    message="Failed to prepare context",
                    error=str(e),
                    data={
                        "milvus_hits": len(milvus_results),
                        "person_contexts": len(person_contexts),
                        "mentioned_persons": mentioned_persons,
                        "context_length": 0
                    }
                )
            
            # 5. Submit LLM job to Puhti
            try:
                # Prepare input data with sources for later reference
                input_data = {
                    "query": query,
                    "context": context,
                    "sources": [
                        {
                            "text": result["text"],
                            "document_id": result["document_id"],
                            "score": result.get("score", 0.0)
                        }
                        for result in milvus_results
                    ],
                    "person_contexts": [
                        {
                            "name": ctx["name"],
                            "age": ctx.get("age"),
                            "relationships": ctx.get("relationships", []),
                            "document_count": ctx.get("document_count", 0)
                        }
                        for ctx in person_contexts
                    ],
                    "params": {
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                        "top_p": 0.95
                    }
                }
                
                # Submit to Puhti through job manager
                job_id = await self.puhti_job_manager.submit_llm_job(
                    input_data=input_data,
                    input_dir=self.query_path / "inputs",
                    output_dir=self.query_path / "outputs"
                )
                
                if not job_id:
                    return InitialQueryResponse(
                        status="error",
                        message="Failed to submit LLM job",
                        data={
                            "milvus_hits": len(milvus_results),
                            "person_contexts": len(person_contexts),
                            "mentioned_persons": mentioned_persons,
                            "context_length": len(context.split())
                        }
                    )
                
                logger.info(f"Submitted LLM job: {job_id}")
                
                # Return initial response
                return InitialQueryResponse(
                    status="processing",
                    message="Query is being processed",
                    job_id=job_id,
                    data={
                        "milvus_hits": len(milvus_results),
                        "person_contexts": len(person_contexts),
                        "mentioned_persons": mentioned_persons,
                        "context_length": len(context.split())
                    }
                )
                
            except Exception as e:
                logger.error(f"Error submitting LLM job: {str(e)}")
                return InitialQueryResponse(
                    status="error",
                    message="Failed to submit LLM job",
                    error=str(e),
                    data={
                        "milvus_hits": len(milvus_results),
                        "person_contexts": len(person_contexts),
                        "mentioned_persons": mentioned_persons,
                        "context_length": len(context.split())
                    }
                )
                
        except Exception as e:
            logger.error(f"Unexpected error in process_query: {str(e)}")
            return InitialQueryResponse(
                status="error",
                message="An unexpected error occurred",
                error=str(e),
                data={
                    "milvus_hits": len(milvus_results),
                    "person_contexts": len(person_contexts),
                    "mentioned_persons": mentioned_persons,
                    "context_length": 0
                }
            )

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
        """Check status and retrieve results if complete."""
        try:
            sftp = self.puhti_job_manager.ssh_client.open_sftp()
            remote_path = self.query_path / "outputs" / f"response_query_{job_id}.json"
            local_cache_path = self.local_results_path / "cache" / f"{job_id}.json"

            try:
                # Download result
                sftp.get(str(remote_path), str(local_cache_path))
                
                # Parse result
                async with aiofiles.open(local_cache_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                
                # Convert to CompletedQueryResponse format
                sources = [
                    QuerySource(
                        text=source["text"],
                        document_id=source["document_id"],
                        score=source.get("score", 0.0)
                    )
                    for source in data.get("sources", [])
                ]
                
                person_contexts = [
                    PersonContext(
                        name=ctx["name"],
                        age=ctx.get("age"),
                        relationships=ctx.get("relationships", []),
                        document_count=ctx.get("document_count", 0)
                    )
                    for ctx in data.get("person_contexts", [])
                ]
                
                result = {
                    "status": "COMPLETED",
                    "result": {
                        "query": data["query"],
                        "answer": data["response"],
                        "sources": sources,
                        "person_contexts": person_contexts,
                        "confidence": data.get("confidence", 0.0)
                    },
                    "message": "Query completed successfully"
                }
                
                # Clean up remote files
                sftp.remove(str(remote_path))
                input_path = self.query_path / "inputs" / f"query_{job_id}.json"
                try:
                    sftp.remove(str(input_path))
                except FileNotFoundError:
                    pass
                
                return result
                    
            except FileNotFoundError:
                logger.warning(f"Results not yet available for job {job_id}")
                # Check job status with PuhtiJobManager
                job_status = await self.puhti_job_manager.check_llm_job(job_id)
                return {
                    "status": job_status["status"],
                    "message": "Job is still processing" if job_status["status"] == "RUNNING" else "Job failed",
                    "error": job_status.get("error")
                }
                    
        except Exception as e:
            logger.error(f"Error retrieving answer: {str(e)}")
            return {
                "status": "FAILED",
                "message": "Error retrieving results",
                "error": str(e)
            }