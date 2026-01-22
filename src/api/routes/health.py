"""Health check routes."""

from fastapi import APIRouter

from ..models.responses import HealthResponse
from ... import __version__

router = APIRouter(tags=["health"])


def get_gpu_info() -> dict:
    """Get GPU information if available."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            return {
                "available": True,
                "name": props.name,
                "total_mb": total_mem / (1024 * 1024),
                "free_mb": free_mem / (1024 * 1024),
            }
    except Exception:
        pass

    return {"available": False, "name": None, "total_mb": None, "free_mb": None}


def check_mediamtx() -> bool:
    """Check if MediaMTX is running and accessible."""
    try:
        import httpx

        response = httpx.get("http://localhost:9997/v3/paths/list", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and system status."""
    gpu_info = get_gpu_info()
    mediamtx_ok = check_mediamtx()

    # Get active stream count
    try:
        from ...pipeline.manager import StreamManager

        manager = StreamManager.get_instance()
        active_count = len(await manager.list_streams())
    except Exception:
        active_count = 0

    return HealthResponse(
        status="ok",
        version=__version__,
        gpu_available=gpu_info["available"],
        gpu_name=gpu_info["name"],
        gpu_memory_total_mb=gpu_info["total_mb"],
        gpu_memory_free_mb=gpu_info["free_mb"],
        active_streams=active_count,
        mediamtx_connected=mediamtx_ok,
    )


@router.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "StreamDiffusion API",
        "version": __version__,
        "docs_url": "/docs",
    }
