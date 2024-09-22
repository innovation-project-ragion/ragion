# src/test_milvus_connection.py
from milvus.connector import MilvusConnector
from pymilvus import Collection
import sys


def main():
    try:
        # Connect to Milvus
        MilvusConnector.connect()

        # Create the collection
        collection = MilvusConnector.create_document_collection()

        print("Collection and index setup completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Disconnect from Milvus
        MilvusConnector.disconnect()


if __name__ == "__main__":
    main()
