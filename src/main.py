from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.jina import JinaEmbeddings
from langchain.vectorstores import Milvus
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain import hub
import os

app = FastAPI()

# Global variable to store the vector store after data loading
vector_store = None

# Set a limit on the number of chunks to avoid overusing Jina tokens
MAX_CHUNKS = 100  # Define the maximum number of chunks to process

# Step 1: Load your PDF file and store in Milvus
@app.get("/load_data")
def load_data():
    global vector_store  # Ensure vector_store is accessible globally
    
    # Load data from the provided PDF file
    loader = PyPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf")  # Replace with your actual PDF path
    data = loader.load()

    # Step 2: Split data into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # Limit the number of chunks to process
    limited_splits = all_splits[:MAX_CHUNKS]

    # Step 3: Generate embeddings using Jina AI and store them in Milvus
    embeddings = JinaEmbeddings(jina_api_key=("jina_75c97efc9dba41e4b84f45c246d6c0b6Cs5ErEgcncGSkDTQLTrjjgUINU2q"))

    # Store embeddings in Milvus
    vector_store = Milvus.from_documents(
        documents=limited_splits,
        embedding=embeddings,
        connection_args={"host": os.getenv("MILVUS_HOST"), "port": os.getenv("MILVUS_PORT")}
    )

    return {"message": f"Data loaded and indexed successfully with {len(limited_splits)} chunks."}


# Step 4: Ask questions using the RAG system
@app.get("/ask")
def ask(query: str):
    global vector_store  # Ensure we're using the globally defined vector_store
    
    if vector_store is None:
        return {"error": "Data has not been loaded yet. Please load data using /load_data first."}

    llm = Ollama(
        model="mistral",  # Using the Mistral model
        base_url="http://ollama:11434",  
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|eot_id|>"]
)


    # Pull a prompt from LangChainHub
    prompt = hub.pull("rlm/rag-prompt")

    # Set up the QA chain with Milvus as the retriever
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # Run the query
    result = qa_chain.run(query)

    return {"query": query, "result": result}


# Simple health check
@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}
