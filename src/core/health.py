from src.db.milvus import MilvusClient
from src.db.neo4j import Neo4jClient
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

async def check_milvus_health() -> bool:
    """Check if Milvus is healthy."""
    try:
        client = MilvusClient()
        # Perform a simple operation
        client.collection.flush()
        return True
    except Exception as e:
        logger.error(f"Milvus health check failed: {str(e)}")
        return False

async def check_neo4j_health() -> bool:
    """Check if Neo4j is healthy."""
    try:
        client = Neo4jClient()
        with client.driver.session() as session:
            result = session.run("RETURN 1 as test")
            return result.single()["test"] == 1
    except Exception as e:
        logger.error(f"Neo4j health check failed: {str(e)}")
        return False

async def check_gpu_availability() -> bool:
    """Check if GPU is available and working."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception as e:
        logger.error(f"GPU check failed: {str(e)}")
        return False