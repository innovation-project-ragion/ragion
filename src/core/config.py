from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "rag-backend"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://attu:3000" 
    ]
    
    # Milvus Settings
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "standalone")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    
    # Neo4j Settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "test")
    
    # Model Settings
    MODEL_ID: str = "Finnish-NLP/llama-7b-finnish-instruct-v0.2"
    EMBEDDING_MODEL: str = "TurkuNLP/sbert-cased-finnish-paraphrase"
    EMBEDDING_DIM: int = 768
    MAX_TOKENS: int = 2048
    
    # GPU Settings
    USE_GPU: bool = True
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    
    # Cache Settings
    CACHE_DIR: str = "/src/cache"  # Updated for container path
    
    # Service Health Check Settings
    MILVUS_HEALTH_CHECK_INTERVAL: int = 30
    NEO4J_HEALTH_CHECK_INTERVAL: int = 30

    MILVUS_HOST: str
    MILVUS_PORT: int
    NEO4J_HOST: str
    NEO4J_PORT: int
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    ETCD_ENDPOINTS: str
    MINIO_ADDRESS: str
    NEO4J_AUTH: str
    JINA_AI_API_KEY: str
    SECRET_KEY: str
    FASTAPI_PORT: int
    STREAMLIT_PORT: int
    BACKEND_URL: str
    ENVIRONMENT: str
    PUHTI_USERNAME: str
    PUHTI_PROJECT: str
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()