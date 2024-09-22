# Milvus schema and collection management
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

#Define schema for the example collection
def create_schema():
    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    
    # Create a collection schema
    schema = CollectionSchema(fields=fields, description="Example collection schema for vectors")
    return schema

def create_collection():
    schema = create_schema()
    
    # Create the collection
    collection = Collection(name="example_collection", schema=schema)

    # Define index parameters
    index_params = {
        "index_type": "IVF_FLAT",    # Use IVF_FLAT index type for vector search
        "metric_type": "L2",     # Cosine similarity metric
        "params": {"nlist": 1024}    # Number of clusters (nlist)
    }

    # Create index on the vector field
    collection.create_index(
        field_name="vector",         # Specify the vector field
        index_params=index_params,   # Use defined index params
        index_name="index_example"   # Name the index
    )
    
    print("Collection 'example_collection' and index created!")
    return collection

# Define schema for document embeddings
def create_document_schema():
    # Define fields
    fields = [
        FieldSchema(name="document_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=500)
    ]
    
    # Create a collection schema
    schema = CollectionSchema(fields=fields, description="Document collection schema for embeddings")
    return schema

def create_document_collection():
    schema = create_document_schema()

    # Create the collection
    collection = Collection(name="document_collection", schema=schema)

    # Define index parameters
    index_params = {

        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {
            "nlist": 1024
        }
    }

    # Create index on the content_vector field
    collection.create_index(
        field_name="content_vector",  # Correct field name for vector search
        index_params=index_params,    # Use the correct index params
        index_name="index_document_example"
    )
    
    print("Collection 'document_collection' and index created!")
    return collection