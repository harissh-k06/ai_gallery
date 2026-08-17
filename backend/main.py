import logging
from typing import Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import gallery_router

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_gallery.main")

# Initialize FastAPI Application
app = FastAPI(
    title="AI Gallery API",
    version="1.0.0",
    description=(
        "Production-grade backend API service for AI Gallery mobile clients. "
        "Bridges client telemetry to LLM Strategy Routers and executes two-tier GPU "
        "vector clustering and VLM analysis via Modal remote pipelines."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure Cross-Origin Resource Sharing (CORS) for Flutter & Web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API subrouters under versioned prefix
app.include_router(gallery_router, prefix="/api/v1")


@app.get(
    "/",
    tags=["Root"],
    summary="Service Metadata & Gateway Status",
    description="Returns high-level service metadata, operational status, and interactive API documentation links.",
)
async def root() -> dict[str, Any]:
    """Root endpoint exposing API service metadata and navigation links."""
    return {
        "service": "AI Gallery API",
        "version": "1.0.0",
        "status": "online",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json",
        },
        "endpoints": {
            "health": "/api/v1/gallery/health",
            "recommend_strategy": "/api/v1/gallery/recommend-strategy",
            "process_gallery": "/api/v1/gallery/process",
        },
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting AI Gallery FastAPI server on http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
