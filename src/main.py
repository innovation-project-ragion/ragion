from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router
from src.milvus.connector import MilvusConnector
app = FastAPI()

@app.on_event("startup")
def startup_event():
    """
    Establish a connection to Milvus when the FastAPI app starts.
    """
    MilvusConnector.connect()

@app.on_event("shutdown")
def shutdown_event():
    """
    Disconnect from Milvus when the FastAPI app shuts down.
    """
    MilvusConnector.disconnect()

# Include the API routes
app.include_router(api_router)


# Simple health check
@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # URL where Open WebUI is running
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)