from pymilvus import Collection
from config import config
from ollama.model import OllamaModel


ollama_model = OllamaModel()


def insert_data_into_collection(collection: Collection, data: dict):
    """
    Insert document embeddings and text into Milvus collection.
    """
    try:
        # For each document, generate embedding using Ollama
        embeddings = []
        for text in data["text"]:
            embedding = ollama_model.generate_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"Failed to generate embedding for: {text}")

        # Insert document ID, embeddings, and text into the collection
        entities = [data["doc_id"], embeddings, data["text"]]
        collection.insert(entities)
        print(f"Data inserted into collection: {collection.name}")
    except Exception as e:
        print(f"Error inserting data into collection: {e}")
