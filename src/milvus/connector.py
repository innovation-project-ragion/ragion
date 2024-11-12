# ragion/src/milvus/connector.py

from pymilvus import connections
from src.config import config

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
            print(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        except Exception as e:
            print(f"Failed to connect to Milvus: {str(e)}")
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
            print(f"Failed to disconnect from Milvus: {str(e)}")
            raise e
