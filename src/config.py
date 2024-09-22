## This file contains the configuration for the application, load environment variables such as URIs, model paths , API keys etc.
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
# Load environment variables from .env file
load_dotenv()

class Config(BaseSettings):
    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_ALIAS: str

    class Config:
        env_fiel = ".env"

config = Config()



