from fastapi import FastAPI
from streamlit_chat import message
app = FastAPI()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.core.health import check_milvus_health, check_neo4j_health, check_gpu_availability
from src.api.v1 import api_router
import logging
import asyncio
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Simple health check
#@app.get("/")
#def hello_world():
 #   return {"message": "Hello, World!"}

message("My message")
message("Hello bot!", is_user=True)
# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Store background tasks
background_tasks = set()

async def periodic_health_check():
    """Perform periodic health checks of services."""
    while True:
        try:
            milvus_healthy = await check_milvus_health()
            neo4j_healthy = await check_neo4j_health()
            gpu_available = await check_gpu_availability()
            
            logger.info(f"Health Check - Milvus: {milvus_healthy}, "
                       f"Neo4j: {neo4j_healthy}, GPU: {gpu_available}")
            
            await asyncio.sleep(settings.MILVUS_HEALTH_CHECK_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    """Initialize services and start background tasks."""
    logger.info("Starting up RAG backend...")
    
    # Start health check task
    task = asyncio.create_task(periodic_health_check())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down RAG backend...")
    
    # Cancel all background tasks
    for task in background_tasks:
        task.cancel()
    
    # Wait for all tasks to complete
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

@app.get("/health")
async def health_check() -> Dict:
    """Endpoint to check service health."""
    try:
        milvus_healthy = await check_milvus_health()
        neo4j_healthy = await check_neo4j_health()
        gpu_available = await check_gpu_availability()
        
        status = "healthy" if all([milvus_healthy, neo4j_healthy]) else "degraded"
        
        return {
            "status": status,
            "services": {
                "milvus": "healthy" if milvus_healthy else "unhealthy",
                "neo4j": "healthy" if neo4j_healthy else "unhealthy",
                "gpu": "available" if gpu_available else "unavailable"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )