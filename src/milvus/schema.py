from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import connections, utility
from config import config


def create_document_schema():
    # Define fields
    fields = [
        FieldSchema(
            name="doc_id",
            dtype=DataType.VARCHAR,
            is_primary=False,
            auto_id=False,
            max_length=100,
        ),
        FieldSchema(
            name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=3072,  # We need to adjust this to match our embedding dimension
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length="65535"),
    ]

    # Create a collection schema
    schema = CollectionSchema(
        fields=fields, description="Collection for document embeddings"
    )
    return schema


def create_document_collection():
    schema = create_document_schema()
    collection_name = "document_embeddings"

    # Check if collection already exists
    if not utility.has_collection(collection_name):
        # Create the collection
        collection = Collection(name=collection_name, schema=schema)

        # Define index parameters
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",  # Inner Product for cosine similarity
            "params": {"nlist": 1024},
        }

        # Create index on the embedding field
        collection.create_index(
            field_name="embedding",
            index_params=index_params,
            index_name="embedding_index",
        )

        print(f"Collection '{collection_name}' and index created!")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")

    ## load the collection
    collection.load()

    return collection
