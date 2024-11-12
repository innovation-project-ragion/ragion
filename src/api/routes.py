# ragion/src/api/routes.py

from fastapi import APIRouter, HTTPException
from src.milvus.searcher import search_similar_documents
from src.ollama.model import query_ollama_with_context
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
router = APIRouter()

# Load embedding model
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model.eval()

# Generate embeddings for the query
def generate_query_embedding(query):
    inputs = embedding_tokenizer(query, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    return query_embedding.cpu().numpy().astype(np.float32)

@router.post("/query")
def query_documents(question: str):
    """
    API endpoint to process user queries, search Milvus, and get answers from Ollama.
    
    Args:
        question (str): The user's question.
    
    Returns:
        dict: The LLM's response based on document context.
    """
    try:
        # Step 1: Generate query embedding
        query_embedding = generate_query_embedding(question)

        # Step 2: Search for relevant documents in Milvus
        top_results = search_similar_documents(query_embedding)

        if not top_results:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        
        # Combine the top result contexts into a single string
        relevant_context = "\n".join(top_results)


        # Step 3: Query Ollama for the final answer based on the context
        ollama_response = query_ollama_with_context(question, relevant_context)

        return {"response": ollama_response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
