from milvus.connector import MilvusConnector
from milvus.schema import create_document_collection
from pymilvus import utility, Collection
from config import config
from milvus.data_loader import insert_data_into_collection


def test_milvus_connection():
    try:
        # Establish connection
        MilvusConnector.connect()

        # Create or get collection
        collection = create_document_collection()

        # Verify collection exists
        if utility.has_collection(collection.name):
            print(f"Collection '{collection.name}' exists.")
        else:
            print(f"Collection '{collection.name}' does not exist.")
            return

        # Test insertion of real data with embeddings
        dummy_data = {
            "doc_id": ["test_doc_1", "test_doc_2", "test_doc_3"],
            "text": [
                "This article is about the online encyclopedia.",
                "Wikipedia is the largest and most-read reference work in history.",
                "Wikipedia was launched on January 15, 2001.",
            ],
        }

        insert_data_into_collection(collection, dummy_data)
        print("Real data inserted successfully with embeddings.")

    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")
    finally:
        MilvusConnector.disconnect()


if __name__ == "__main__":
    test_milvus_connection()
