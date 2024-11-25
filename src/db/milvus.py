from pymilvus import connections, Collection, utility
from src.core.config import settings
import logging
import time


logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self):
        self._collection = None
        self._last_reload_time = 0
        self.reload_interval = 300  # 5 minutes
        self.connect()

    def connect(self):
        """Establish connection to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            logger.info(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    @property
    def collection(self):
        """Get the current collection, reloading if necessary."""
        if not self._collection or self._needs_reload():
            self._reload_collection()
        return self._collection

    def _needs_reload(self) -> bool:
        """Check if collection needs to be reloaded."""
        import time
        return time.time() - self._last_reload_time > self.reload_interval

    def _reload_collection(self):
        """Reload the collection."""
        try:
            if not utility.has_collection("document_embeddings"):
                raise Exception("Collection does not exist")
            
            self._collection = Collection("document_embeddings")
            self._collection.load()
            self._last_reload_time = time.time()
            logger.info("Successfully reloaded Milvus collection")
        except Exception as e:
            logger.error(f"Error reloading collection: {str(e)}")
            raise

    async def search(self, query_embedding, limit: int = 5):
        """Search for similar vectors."""
        try:
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["text", "document_id", "person_name", "chunk_index"]
            )
            
            return self._process_results(results)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    def _process_results(self, results):
        """Process search results into a standardized format."""
        processed_results = []
        for hits in results:
            for hit in hits:
                processed_results.append({
                    "text": hit.entity.get("text"),
                    "document_id": hit.entity.get("document_id"),
                    "person_name": hit.entity.get("person_name"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "score": hit.score
                })
        return processed_results

    def close(self):
        """Close the Milvus connection."""
        try:
            connections.disconnect("default")
            logger.info("Closed Milvus connection")
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {str(e)}")