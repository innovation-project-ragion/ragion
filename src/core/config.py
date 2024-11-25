from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "rag-backend"
    HOST: str
    PORT: int
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://attu:3000" 
    ]
    
    # Milvus Settings
    MILVUS_HOST: str
    MILVUS_PORT: int
    
    # Neo4j Settings
    NEO4J_HOST: str
    NEO4J_PORT: int
    NEO4J_AUTH: str
    
    # Model Settings
    MODEL_ID: str = "Finnish-NLP/llama-7b-finnish-instruct-v0.2"
    EMBEDDING_MODEL: str = "TurkuNLP/sbert-cased-finnish-paraphrase"
    EMBEDDING_DIM: int = 768
    MAX_TOKENS: int = 2048
    
    # GPU Settings
    USE_GPU: bool = True
    CUDA_VISIBLE_DEVICES: str
    
    # Cache Settings
    CACHE_DIR: str = "/src/cache"
    
    # Service Health Check Settings
    MILVUS_HEALTH_CHECK_INTERVAL: int = 30
    NEO4J_HEALTH_CHECK_INTERVAL: int = 30

    # Other Settings
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    ETCD_ENDPOINTS: str
    MINIO_ADDRESS: str
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
