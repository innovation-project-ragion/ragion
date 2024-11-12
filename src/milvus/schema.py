# ragion/src/milvus/schema.py

from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility

def create_document_schema():
    """
    Define the schema for document embeddings.
    """
    fields = [
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="person_age", dtype=DataType.INT64),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=20)
    ]
    schema = CollectionSchema(fields=fields, description="Document embeddings with person metadata")
    return schema

def create_document_collection():
    """
    Create the Milvus collection based on the schema.
    """
    collection_name = "document_embeddings"
    
    # Check if the collection already exists
    if not utility.has_collection(collection_name):
        schema = create_document_schema()
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created!")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")

    return collection
