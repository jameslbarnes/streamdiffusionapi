"""Stream management routes."""

import secrets
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..models.requests import CreateStreamRequest, UpdateStreamRequest, PartialStreamParams
from ..models.responses import StreamResponse, StreamStatusResponse, StreamStatus, ErrorResponse

router = APIRouter(prefix="/v1/streams", tags=["streams"])


def get_stream_manager():
    """Dependency to get the stream manager instance."""
    from ...pipeline.manager import StreamManager

    return StreamManager.get_instance()


@router.post(
    "",
    response_model=StreamResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        409: {"model": ErrorResponse, "description": "Resource conflict"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def create_stream(
    request: CreateStreamRequest,
    manager: Annotated[object, Depends(get_stream_manager)],
) -> StreamResponse:
    """Create a new stream.

    This initializes a new StreamDiffusion pipeline and returns URLs for
    ingestion (WHIP) and playback (WHEP/HLS).
    """
    try:
        stream = await manager.create_stream(
            name=request.name,
            pipeline_type=request.pipeline,
            params=request.params.model_dump(),
            output_rtmp_url=request.output_rtmp_url,
        )
        return stream
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{stream_id}",
    response_model=StreamResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Stream not found"},
    },
)
async def get_stream(
    stream_id: str,
    manager: Annotated[object, Depends(get_stream_manager)],
) -> StreamResponse:
    """Get stream details by ID."""
    stream = await manager.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    return stream


@router.patch(
    "/{stream_id}",
    response_model=StreamResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "Stream not found"},
    },
)
async def update_stream(
    stream_id: str,
    request: UpdateStreamRequest,
    manager: Annotated[object, Depends(get_stream_manager)],
) -> StreamResponse:
    """Update stream parameters.

    Some parameters can be updated without restarting the pipeline (hot reload):
    - prompt, negative_prompt
    - guidance_scale, delta
    - num_inference_steps, t_index_list
    - seed
    - controlnet conditioning_scale values

    Other parameters require a full pipeline reload (~30 seconds):
    - model_id, width, height
    - Adding/removing ControlNets or LoRAs
    - Changing use_lcm_lora
    """
    stream = await manager.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    try:
        updated = await manager.update_stream(
            stream_id=stream_id,
            pipeline_type=request.pipeline,
            params=request.params.model_dump() if request.params else None,
            output_rtmp_url=request.output_rtmp_url,
        )
        return updated
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch(
    "/{stream_id}/params",
    response_model=StreamResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "Stream not found"},
    },
)
async def update_stream_params(
    stream_id: str,
    params: PartialStreamParams,
    manager: Annotated[object, Depends(get_stream_manager)],
) -> StreamResponse:
    """Hot-reload stream parameters without pipeline restart.

    Only parameters that support hot reload can be updated through this endpoint.
    Use the full PATCH endpoint for parameters requiring restart.
    """
    stream = await manager.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    try:
        # Only update non-None values
        updates = {k: v for k, v in params.model_dump().items() if v is not None}
        updated = await manager.hot_update_params(stream_id, updates)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/{stream_id}/status",
    response_model=StreamStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Stream not found"},
    },
)
async def get_stream_status(
    stream_id: str,
    manager: Annotated[object, Depends(get_stream_manager)],
) -> StreamStatusResponse:
    """Get detailed status of a stream including performance metrics."""
    status_info = await manager.get_stream_status(stream_id)
    if status_info is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    return status_info


@router.delete(
    "/{stream_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Stream not found"},
    },
)
async def delete_stream(
    stream_id: str,
    manager: Annotated[object, Depends(get_stream_manager)],
) -> None:
    """Stop and delete a stream.

    This will:
    1. Stop the WHIP ingestion
    2. Shut down the pipeline
    3. Stop any RTMP output
    4. Clean up resources
    """
    stream = await manager.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    await manager.delete_stream(stream_id)


@router.get(
    "",
    response_model=list[StreamResponse],
)
async def list_streams(
    manager: Annotated[object, Depends(get_stream_manager)],
) -> list[StreamResponse]:
    """List all active streams."""
    return await manager.list_streams()
