from pymilvus import connections, Collection, utility, CollectionSchema, FieldSchema, DataType
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
        self.ensure_collection_exists()

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

    def ensure_collection_exists(self):
        """Ensure the collection exists, creating it if necessary and loading it."""
        try:
            if not utility.has_collection("document_embeddings"):
                logger.info("Collection 'document_embeddings' does not exist. Creating it now.")
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM),
                    FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="person_age", dtype=DataType.INT64),
                    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64),
                ]
                schema = CollectionSchema(fields=fields, description="Document embeddings collection")
                collection = Collection(name="document_embeddings", schema=schema)
                
                # Create an index for the embedding field
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 1024}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                logger.info("Collection and index created successfully.")
            else:
                logger.info("Collection 'document_embeddings' already exists.")

            # Load the collection into memory
            self._collection = Collection("document_embeddings")
            self._collection.load()
            logger.info("Collection 'document_embeddings' loaded into memory and ready for data insertion.")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise


    @property
    def collection(self):
        """Get the current collection, reloading if necessary."""
        if not self._collection or self._needs_reload():
            self._reload_collection()
        return self._collection

    def _needs_reload(self) -> bool:
        """Check if collection needs to be reloaded."""
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
                output_fields=["text", "document_id"]
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
