## Milvus connection and client creation

from pymilvus import connections
from config import config

from milvus.schema import create_document_collection

class MilvusConnector:

    @staticmethod
    def connect():
        """
        Establishes a connection to the Milvus server.
        """
        try:
            connections.connect(
                alias=config.MILVUS_ALIAS,
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT
            )
            print("Connected to Milvus at tcp://milvus-standalone:19530")
        except Exception as e:
            print("Failed to connect to Milvus: {}".format(str(e)))
            raise e
        
    @staticmethod
    def create_document_collection():
        """
        Create the document collection and load it to Milvus.
        """
        try:
            collection = create_document_collection()
            print(f"Collection {collection.name} is created!")
        except Exception as e:
            print(f"Failed to create collection: {str(e)}")
            raise e        
    
    @staticmethod
    def disconnect():
        """
        Disconnects the default connection from Milvus.
        """
        try:
            connections.disconnect(config.MILVUS_ALIAS)
            print("Disconnected from Milvus")
        except Exception as e:
            print("Failed to disconnect from Milvus: {}".format(str(e)))
            raise e
        


       
