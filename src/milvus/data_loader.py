# ragion/src/milvus/data_loader.py

import os
import re
from docx import Document
from pymilvus import Collection
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from src.milvus.connector import MilvusConnector
from src.milvus.schema import create_document_collection
import itertools
# Load the embedding model
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model.eval()

# Function to extract metadata from the filename
def extract_metadata_from_filename(filename):
    title = os.path.splitext(filename)[0]
    match = re.match(r'([A-Za-z]+)\s+(\d{1,3})v\s+([A-Za-z0-9\-]+)', title)
    if match:
        name = match.group(1)
        age = int(match.group(2))
        doc_id = match.group(3)
        return name, age, doc_id
    return None, None, None

# Extract text from docx file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Preprocess text (remove non-word characters, lowercase, etc.)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Chunk text into smaller parts
def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Generate embeddings
def batched(iterable, n):
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch

# Generate embeddings for the text chunks
def generate_embeddings_local(texts, max_batch_size=32):
    embeddings = []
    for batch in batched(texts, max_batch_size):
        inputs = embedding_tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings.cpu())
    embeddings = torch.cat(embeddings)
    return embeddings.numpy().astype(np.float32)

# Insert data into Milvus
def insert_data_with_metadata(collection, doc_ids, embeddings, texts, person_name, person_age, document_id):
    collection.insert([doc_ids, embeddings.tolist(), texts, [person_name]*len(doc_ids), 
                       [person_age]*len(doc_ids), [document_id]*len(doc_ids)])
    collection.flush()

# Load collection and insert data
def process_and_insert_documents(folder_path):
    collection = create_document_collection()
    file_paths = [f for f in os.listdir(folder_path) if f.endswith('.docx')]
    
    for file in file_paths:
        name, age, doc_id = extract_metadata_from_filename(file)
        file_path = os.path.join(folder_path, file)
        extracted_text = extract_text_from_docx(file_path)
        cleaned_text = preprocess_text(extracted_text)
        chunks = chunk_text(cleaned_text)
        embeddings = generate_embeddings_local(chunks)
        doc_ids = [f"{doc_id}_chunk_{i+1}" for i in range(len(chunks))]
        insert_data_with_metadata(collection, doc_ids, embeddings, chunks, name, age, doc_id)
