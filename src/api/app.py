"""FastAPI application for StreamDiffusion API."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import streams, health
from ..pipeline.manager import StreamManager
from ..streaming.mediamtx import MediaMTXManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("Starting StreamDiffusion API...")

    # Initialize managers
    stream_manager = StreamManager.get_instance()
    mediamtx_manager = MediaMTXManager.get_instance()

    # Start MediaMTX if not already running
    try:
        await mediamtx_manager.ensure_running()
        logger.info("MediaMTX is running")
    except Exception as e:
        logger.warning(f"MediaMTX not available: {e}. Streaming features will be limited.")

    # Pre-warm GPU if available
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            # Small allocation to initialize CUDA context
            _ = torch.zeros(1, device="cuda")
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"GPU initialization warning: {e}")

    logger.info("StreamDiffusion API ready")

    yield

    # Shutdown
    logger.info("Shutting down StreamDiffusion API...")

    # Stop all streams
    await stream_manager.shutdown()

    # Stop MediaMTX if we started it
    await mediamtx_manager.shutdown()

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="StreamDiffusion API",
        description="Daydream-compatible API for real-time video generation using StreamDiffusion",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware for browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router)
    app.include_router(streams.router)

    return app


# Create app instance
app = create_app()


def main():
    """Run the API server."""
    import uvicorn

    # Windows-specific event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
