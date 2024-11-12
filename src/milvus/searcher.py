from pymilvus import Collection
import numpy as np

def search_similar_documents(query_embedding, collection_name="document_embeddings", top_k=5):
    query_embedding = query_embedding.astype(np.float32)
    collection = Collection(name=collection_name)
    collection.load()

    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    try:
        results = collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "person_name", "person_age", "document_id"]
        )

        if not results or len(results[0]) == 0:
            print("No relevant documents found in Milvus.")
            return None

        relevant_texts = []
        for hits in results:
            for hit in hits:
                document_id = hit.entity.get("document_id")
                text_snippet = hit.entity.get("text")[:300]  # Truncate to 300 characters
                person_name = hit.entity.get("person_name")
                person_age = hit.entity.get("person_age")

                print(f"Score: {hit.score}, Document ID: {document_id}, Name: {person_name}, Age: {person_age}")
                print(f"Text: {text_snippet}...\n")

                relevant_texts.append(text_snippet)

        return relevant_texts
    
    except Exception as e:
        print(f"Error during search: {e}")
        return None
