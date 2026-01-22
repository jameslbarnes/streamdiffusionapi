"""Stream manager for handling multiple StreamDiffusion pipelines."""

import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..api.models.responses import StreamResponse, StreamStatusResponse, StreamStatus

logger = logging.getLogger(__name__)

def _get_external_host() -> tuple[str, str, bool]:
    """Detect external hostname for URLs.

    Returns (base_host, pod_id, is_runpod)
    """
    # Check for RunPod environment
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        return f"{pod_id}", pod_id, True

    # Check for explicit override
    external_host = os.environ.get("EXTERNAL_HOST")
    if external_host:
        return external_host, "", False

    # Default to localhost
    return "localhost", "", False


@dataclass
class StreamState:
    """Internal state for a single stream."""

    id: str
    name: str
    stream_key: str
    pipeline_type: str
    params: dict[str, Any]
    output_rtmp_url: str | None
    created_at: datetime
    started_at: datetime | None = None
    status: StreamStatus = StreamStatus.PENDING
    error_message: str | None = None

    # Pipeline instance (set when started)
    pipeline: Any = None

    # Performance metrics
    frames_processed: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    last_frame_time: float = 0.0

    # Processing task
    processing_task: asyncio.Task | None = None


class StreamManager:
    """Manages multiple concurrent streams and their pipelines."""

    _instance: "StreamManager | None" = None

    def __init__(self):
        self._streams: dict[str, StreamState] = {}
        self._lock = asyncio.Lock()

        # Detect external host for URL generation
        base_host, pod_id, is_runpod = _get_external_host()
        self._is_runpod = is_runpod
        self._pod_id = pod_id

        if is_runpod:
            # RunPod proxy URLs
            self._api_url = f"https://{pod_id}-8080.proxy.runpod.net"
            self._whip_url = f"https://{pod_id}-8889.proxy.runpod.net"
            self._whep_url = f"https://{pod_id}-8889.proxy.runpod.net"  # Same port, different path
            self._hls_url = f"https://{pod_id}-8888.proxy.runpod.net"
        else:
            # Local development
            self._api_url = f"http://{base_host}:8080"
            self._whip_url = f"http://{base_host}:8889"
            self._whep_url = f"http://{base_host}:8889"
            self._hls_url = f"http://{base_host}:8888"

    @classmethod
    def get_instance(cls) -> "StreamManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _generate_id(self) -> str:
        """Generate a unique stream ID."""
        return f"stream_{secrets.token_hex(8)}"

    def _generate_stream_key(self) -> str:
        """Generate a unique stream key."""
        return f"sk_{secrets.token_hex(16)}"

    def _generate_playback_id(self) -> str:
        """Generate a unique playback ID."""
        return f"play_{secrets.token_hex(8)}"

    def _build_response(self, state: StreamState) -> StreamResponse:
        """Build API response from stream state."""
        playback_id = self._generate_playback_id()

        return StreamResponse(
            id=state.id,
            name=state.name,
            stream_key=state.stream_key,
            created_at=state.created_at,
            status=state.status,
            whip_url=f"{self._whip_url}/{state.id}/whip",
            output_playback_id=playback_id,
            output_stream_url=f"{self._whep_url}/{state.id}_out/whep",
            output_hls_url=f"{self._hls_url}/{state.id}_out/index.m3u8",
            output_rtmp_url=state.output_rtmp_url,
            gateway_host=self._api_url,
            pipeline=state.pipeline_type,
            params=state.params,
        )

    async def create_stream(
        self,
        name: str,
        pipeline_type: str,
        params: dict[str, Any],
        output_rtmp_url: str | None = None,
    ) -> StreamResponse:
        """Create a new stream."""
        async with self._lock:
            stream_id = self._generate_id()
            stream_key = self._generate_stream_key()

            state = StreamState(
                id=stream_id,
                name=name,
                stream_key=stream_key,
                pipeline_type=pipeline_type,
                params=params,
                output_rtmp_url=output_rtmp_url,
                created_at=datetime.now(timezone.utc),
            )

            self._streams[stream_id] = state
            logger.info(f"Created stream {stream_id} ({name})")

            # Start the pipeline in background
            state.processing_task = asyncio.create_task(
                self._start_pipeline(stream_id)
            )

            return self._build_response(state)

    async def _start_pipeline(self, stream_id: str) -> None:
        """Start the StreamDiffusion pipeline for a stream."""
        state = self._streams.get(stream_id)
        if state is None:
            return

        try:
            state.status = StreamStatus.STARTING
            logger.info(f"Starting pipeline for stream {stream_id}...")

            # Import here to avoid circular imports and delay GPU init
            from .wrapper import StreamDiffusionWrapper

            # Create pipeline wrapper
            wrapper = StreamDiffusionWrapper(
                pipeline_type=state.pipeline_type,
                params=state.params,
            )

            # Initialize the pipeline (loads model, compiles TensorRT, etc.)
            await wrapper.initialize()

            state.pipeline = wrapper
            state.status = StreamStatus.RUNNING
            state.started_at = datetime.now(timezone.utc)
            logger.info(f"Pipeline started for stream {stream_id}")

            # Start processing loop
            await self._process_stream(stream_id)

        except Exception as e:
            logger.error(f"Failed to start pipeline for {stream_id}: {e}")
            state.status = StreamStatus.ERROR
            state.error_message = str(e)

    async def _process_stream(self, stream_id: str) -> None:
        """Main processing loop for a stream."""
        state = self._streams.get(stream_id)
        if state is None or state.pipeline is None:
            return

        logger.info(f"Processing loop started for stream {stream_id}")

        try:
            while state.status == StreamStatus.RUNNING:
                # Get frame from MediaMTX input
                frame = await self._get_input_frame(stream_id)
                if frame is None:
                    await asyncio.sleep(0.01)  # No frame available, short sleep
                    continue

                # Process through StreamDiffusion
                start_time = time.perf_counter()
                output_frame = await state.pipeline.process_frame(frame)
                process_time = time.perf_counter() - start_time

                # Send to output
                await self._send_output_frame(stream_id, output_frame)

                # Update metrics
                state.frames_processed += 1
                state.latency_ms = process_time * 1000
                now = time.perf_counter()
                if state.last_frame_time > 0:
                    state.fps = 1.0 / (now - state.last_frame_time)
                state.last_frame_time = now

        except asyncio.CancelledError:
            logger.info(f"Processing loop cancelled for stream {stream_id}")
        except Exception as e:
            logger.error(f"Processing error for stream {stream_id}: {e}")
            state.status = StreamStatus.ERROR
            state.error_message = str(e)

    async def _get_input_frame(self, stream_id: str):
        """Get input frame from MediaMTX."""
        # This will be implemented by the streaming module
        from ..streaming.frame_bridge import FrameBridge

        bridge = FrameBridge.get_instance()
        return await bridge.get_input_frame(stream_id)

    async def _send_output_frame(self, stream_id: str, frame) -> None:
        """Send output frame to MediaMTX."""
        from ..streaming.frame_bridge import FrameBridge

        bridge = FrameBridge.get_instance()
        await bridge.send_output_frame(stream_id, frame)

    async def get_stream(self, stream_id: str) -> StreamResponse | None:
        """Get stream by ID."""
        state = self._streams.get(stream_id)
        if state is None:
            return None
        return self._build_response(state)

    async def get_stream_status(self, stream_id: str) -> StreamStatusResponse | None:
        """Get detailed stream status."""
        state = self._streams.get(stream_id)
        if state is None:
            return None

        uptime = None
        if state.started_at:
            uptime = (datetime.now(timezone.utc) - state.started_at).total_seconds()

        # Get GPU metrics
        gpu_mem = None
        gpu_util = None
        try:
            import torch

            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                gpu_mem = (total_mem - free_mem) / (1024 * 1024)
        except Exception:
            pass

        return StreamStatusResponse(
            id=state.id,
            status=state.status,
            created_at=state.created_at,
            started_at=state.started_at,
            uptime_seconds=uptime,
            fps=state.fps if state.fps > 0 else None,
            frames_processed=state.frames_processed,
            latency_ms=state.latency_ms if state.latency_ms > 0 else None,
            gpu_memory_used_mb=gpu_mem,
            gpu_utilization_percent=gpu_util,
            error_message=state.error_message,
        )

    async def update_stream(
        self,
        stream_id: str,
        pipeline_type: str | None = None,
        params: dict[str, Any] | None = None,
        output_rtmp_url: str | None = None,
    ) -> StreamResponse:
        """Update stream configuration (may require pipeline restart)."""
        state = self._streams.get(stream_id)
        if state is None:
            raise ValueError(f"Stream {stream_id} not found")

        needs_restart = False

        if pipeline_type and pipeline_type != state.pipeline_type:
            state.pipeline_type = pipeline_type
            needs_restart = True

        if params:
            # Check if any non-hot-reloadable params changed
            hot_params = {
                "prompt", "negative_prompt", "guidance_scale", "delta",
                "num_inference_steps", "t_index_list", "seed"
            }
            old_params = state.params
            for key, value in params.items():
                if key not in hot_params and old_params.get(key) != value:
                    needs_restart = True
                    break

            state.params = params

        if output_rtmp_url is not None:
            state.output_rtmp_url = output_rtmp_url

        if needs_restart and state.pipeline:
            logger.info(f"Restarting pipeline for stream {stream_id} due to config change")
            await self._restart_pipeline(stream_id)
        elif state.pipeline:
            # Apply hot updates
            await self.hot_update_params(stream_id, params or {})

        return self._build_response(state)

    async def hot_update_params(
        self,
        stream_id: str,
        updates: dict[str, Any],
    ) -> StreamResponse:
        """Update parameters without restarting pipeline."""
        state = self._streams.get(stream_id)
        if state is None:
            raise ValueError(f"Stream {stream_id} not found")

        if state.pipeline is None:
            raise ValueError(f"Stream {stream_id} pipeline not initialized")

        # Update state
        for key, value in updates.items():
            if key in state.params:
                state.params[key] = value

        # Apply to pipeline
        await state.pipeline.update_params(updates)

        logger.info(f"Hot-updated params for stream {stream_id}: {list(updates.keys())}")
        return self._build_response(state)

    async def _restart_pipeline(self, stream_id: str) -> None:
        """Restart the pipeline for a stream."""
        state = self._streams.get(stream_id)
        if state is None:
            return

        # Cancel current processing
        if state.processing_task:
            state.processing_task.cancel()
            try:
                await state.processing_task
            except asyncio.CancelledError:
                pass

        # Cleanup old pipeline
        if state.pipeline:
            await state.pipeline.cleanup()
            state.pipeline = None

        # Start new pipeline
        state.processing_task = asyncio.create_task(
            self._start_pipeline(stream_id)
        )

    async def delete_stream(self, stream_id: str) -> None:
        """Stop and delete a stream."""
        async with self._lock:
            state = self._streams.get(stream_id)
            if state is None:
                return

            logger.info(f"Deleting stream {stream_id}")

            # Set status to stopping
            state.status = StreamStatus.STOPPING

            # Cancel processing task
            if state.processing_task:
                state.processing_task.cancel()
                try:
                    await state.processing_task
                except asyncio.CancelledError:
                    pass

            # Cleanup pipeline
            if state.pipeline:
                await state.pipeline.cleanup()

            # Remove from dict
            del self._streams[stream_id]

            logger.info(f"Stream {stream_id} deleted")

    async def list_streams(self) -> list[StreamResponse]:
        """List all streams."""
        return [self._build_response(state) for state in self._streams.values()]

    async def shutdown(self) -> None:
        """Shutdown all streams."""
        logger.info("Shutting down all streams...")
        stream_ids = list(self._streams.keys())
        for stream_id in stream_ids:
            await self.delete_stream(stream_id)
        logger.info("All streams shut down")
