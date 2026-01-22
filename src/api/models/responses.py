"""Response models for the StreamDiffusion API."""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class StreamStatus(str, Enum):
    """Possible stream statuses."""

    PENDING = "pending"  # Stream created, not yet started
    STARTING = "starting"  # Pipeline is loading
    RUNNING = "running"  # Actively processing
    STOPPING = "stopping"  # Shutting down
    STOPPED = "stopped"  # Cleanly stopped
    ERROR = "error"  # Error occurred


class StreamResponse(BaseModel):
    """Response when creating or fetching a stream."""

    id: str = Field(..., description="Unique stream identifier")
    name: str = Field(..., description="Human-readable stream name")
    stream_key: str = Field(..., description="Key for WHIP ingestion")
    created_at: datetime = Field(..., description="Creation timestamp")
    status: StreamStatus = Field(..., description="Current stream status")

    # Ingestion URLs
    whip_url: str = Field(..., description="WHIP URL for WebRTC ingestion")

    # Output URLs
    output_playback_id: str = Field(..., description="Playback identifier")
    output_stream_url: str = Field(..., description="WHEP URL for WebRTC playback")
    output_hls_url: str | None = Field(None, description="HLS playlist URL if available")
    output_rtmp_url: str | None = Field(None, description="RTMP destination if configured")

    # Server info
    gateway_host: str = Field(..., description="Host serving this stream")

    # Pipeline info
    pipeline: str = Field(..., description="Pipeline type")
    params: dict[str, Any] = Field(..., description="Current pipeline parameters")


class StreamStatusResponse(BaseModel):
    """Response for stream status check."""

    id: str = Field(..., description="Stream identifier")
    status: StreamStatus = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: datetime | None = Field(None, description="When processing started")
    uptime_seconds: float | None = Field(None, description="Seconds since started")

    # Performance metrics
    fps: float | None = Field(None, description="Current frames per second")
    frames_processed: int = Field(0, description="Total frames processed")
    latency_ms: float | None = Field(None, description="Current end-to-end latency in ms")

    # Resource usage
    gpu_memory_used_mb: float | None = Field(None, description="GPU memory usage in MB")
    gpu_utilization_percent: float | None = Field(None, description="GPU utilization %")

    # Error info
    error_message: str | None = Field(None, description="Error message if status is ERROR")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("ok", description="Service status")
    version: str = Field(..., description="API version")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: str | None = Field(None, description="GPU model name")
    gpu_memory_total_mb: float | None = Field(None, description="Total GPU memory in MB")
    gpu_memory_free_mb: float | None = Field(None, description="Free GPU memory in MB")
    active_streams: int = Field(0, description="Number of active streams")
    mediamtx_connected: bool = Field(False, description="Whether MediaMTX is connected")
