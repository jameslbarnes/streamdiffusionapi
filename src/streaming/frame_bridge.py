"""Frame bridge between MediaMTX and StreamDiffusion pipeline."""

import asyncio
import logging
import subprocess
import threading
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe frame buffer for a single stream."""

    def __init__(self, max_size: int = 30, loop: asyncio.AbstractEventLoop | None = None):
        self._buffer: deque[np.ndarray] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._event = asyncio.Event()
        # Store the event loop reference for cross-thread signaling
        self._loop = loop

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for cross-thread signaling."""
        self._loop = loop

    def put(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append(frame)
        # Signal that a frame is available (asyncio event)
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._event.set)
            except RuntimeError:
                # Loop may be closed
                pass

    def get(self) -> np.ndarray | None:
        """Get the latest frame (non-blocking)."""
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    async def get_async(self, timeout: float = 1.0) -> np.ndarray | None:
        """Get a frame asynchronously with timeout."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            self._event.clear()
            return self.get()
        except asyncio.TimeoutError:
            return None

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()


class FFmpegReader:
    """Read frames from an RTSP/RTMP stream using FFmpeg."""

    def __init__(self, stream_url: str, width: int, height: int):
        self.stream_url = stream_url
        self.width = width
        self.height = height
        self._process: subprocess.Popen | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._buffer: FrameBuffer | None = None

    def start(self, buffer: FrameBuffer) -> None:
        """Start reading frames."""
        if self._running:
            return

        self._buffer = buffer
        self._running = True

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _start_ffmpeg(self) -> subprocess.Popen | None:
        """Start the FFmpeg process with retry-friendly settings."""
        # FFmpeg command to read RTSP stream with reconnection support
        # Note: -timeout flag causes issues with FFmpeg 4.x, use -stimeout instead
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",  # Use TCP for reliability
            "-stimeout", "5000000",  # 5 second socket timeout in microseconds
            "-i", self.stream_url,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", "30",  # Target framerate
            "-an",  # No audio
            "-",
        ]

        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.width * self.height * 3 * 2,
            )
        except Exception as e:
            logger.warning(f"Failed to start FFmpeg reader: {e}")
            return None

    def _read_loop(self) -> None:
        """Read frames in a loop with auto-reconnect."""
        import time

        frame_size = self.width * self.height * 3
        retry_delay = 1.0  # Initial retry delay
        max_retry_delay = 10.0
        consecutive_failures = 0

        while self._running:
            # Try to connect/reconnect
            self._process = self._start_ffmpeg()

            if self._process is None:
                consecutive_failures += 1
                logger.debug(f"Waiting for stream at {self.stream_url}...")
                time.sleep(min(retry_delay * consecutive_failures, max_retry_delay))
                continue

            logger.info(f"Connected to stream {self.stream_url}")
            consecutive_failures = 0

            # Read frames until connection drops
            frames_read = 0
            while self._running and self._process and self._process.poll() is None:
                try:
                    raw_frame = self._process.stdout.read(frame_size)
                    if len(raw_frame) == frame_size:
                        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                            (self.height, self.width, 3)
                        )
                        if self._buffer:
                            self._buffer.put(frame)
                            frames_read += 1
                            if frames_read == 1:
                                logger.info(f"First frame read from {self.stream_url}")
                            elif frames_read % 300 == 0:  # Log every ~10 seconds at 30fps
                                logger.info(f"Read {frames_read} frames from {self.stream_url}")
                    elif len(raw_frame) == 0:
                        # Stream ended
                        logger.warning(f"Stream ended (0 bytes read) from {self.stream_url}")
                        break
                except Exception as e:
                    logger.warning(f"Frame read error: {e}")
                    break

            # Clean up process before retry
            if self._process:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None

            if self._running:
                logger.debug(f"Stream {self.stream_url} disconnected, retrying...")
                time.sleep(retry_delay)

    def stop(self) -> None:
        """Stop reading frames."""
        self._running = False

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None


class FFmpegWriter:
    """Write frames to an RTSP/RTMP stream using FFmpeg."""

    def __init__(
        self,
        stream_url: str,
        width: int,
        height: int,
        fps: int = 30,
        codec: str = "h264",
        preset: str = "medium",
        output_format: str | None = None,
    ):
        self.stream_url = stream_url
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.preset = preset
        self._process: subprocess.Popen | None = None
        self._running = False

        # Auto-detect format from URL if not specified
        if output_format:
            self.output_format = output_format
        elif stream_url.startswith("rtmp://"):
            self.output_format = "flv"
        else:
            self.output_format = "rtsp"

    def start(self) -> None:
        """Start the FFmpeg writer process."""
        if self._running:
            return

        # FFmpeg command to encode and stream
        # Use FLV1 codec which is universally available and works with RTMP
        cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "flv",  # FLV1/Sorenson Spark - always available, works with RTMP
            "-q:v", "5",  # Quality level (1-31, lower is better)
            "-g", str(self.fps),  # Keyframe every second
            "-f", self.output_format,
            self.stream_url,
        ]

        logger.info(f"Starting FFmpeg writer: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.width * self.height * 3 * 2,
            )
            self._running = True
            logger.info(f"FFmpeg writer started: {self.stream_url} (format: {self.output_format})")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg writer: {e}")
            self._running = False

    _write_frame_count: int = 0

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a frame to the stream."""
        self._write_frame_count += 1

        if not self._running or not self._process:
            if self._write_frame_count <= 3:
                logger.warning(f"Writer not running: running={self._running}, process={self._process is not None}")
            return False

        # Check if process is still alive
        poll_result = self._process.poll()
        if poll_result is not None:
            # Process has exited, try to get error output
            try:
                _, stderr = self._process.communicate(timeout=1)
                logger.error(f"FFmpeg writer crashed (exit code {poll_result}): {stderr.decode()[:500]}")
            except Exception:
                logger.error(f"FFmpeg writer process exited unexpectedly (exit code {poll_result})")
            self._running = False
            return False

        try:
            # Handle PyTorch tensors - convert to numpy array
            if hasattr(frame, 'cpu'):  # It's a torch tensor
                import torch
                # Frame is NCHW format: (batch, channels, height, width)
                # Convert to HWC format: (height, width, channels)
                if frame.dim() == 4:
                    frame = frame[0]  # Remove batch dimension -> (C, H, W)
                if frame.dim() == 3 and frame.shape[0] in (1, 3, 4):
                    frame = frame.permute(1, 2, 0)  # CHW -> HWC
                # Convert to numpy and scale to 0-255
                frame = frame.cpu().float().numpy()
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255)
                frame = frame.astype(np.uint8)
            elif frame.dtype != np.uint8:
                # Already numpy but wrong dtype
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255)
                frame = frame.astype(np.uint8)

            # Resize if needed
            if frame.shape[:2] != (self.height, self.width):
                from PIL import Image
                img = Image.fromarray(frame)
                img = img.resize((self.width, self.height))
                frame = np.array(img)

            frame_bytes = frame.tobytes()
            if self._write_frame_count == 1:
                logger.info(f"Writing first frame: shape={frame.shape}, bytes={len(frame_bytes)}, expected={self.width * self.height * 3}")

            self._process.stdin.write(frame_bytes)
            self._process.stdin.flush()

            if self._write_frame_count == 1:
                logger.info(f"First frame written successfully to {self.stream_url}")

            return True
        except Exception as e:
            logger.warning(f"Frame write error (frame #{self._write_frame_count}): {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self) -> None:
        """Stop the writer."""
        self._running = False

        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


class FrameBridge:
    """Bridge between MediaMTX streams and StreamDiffusion pipelines."""

    _instance: "FrameBridge | None" = None

    def __init__(self):
        self._input_buffers: dict[str, FrameBuffer] = {}
        self._output_buffers: dict[str, FrameBuffer] = {}
        self._readers: dict[str, FFmpegReader] = {}
        self._writers: dict[str, FFmpegWriter] = {}
        self._rtmp_writers: dict[str, FFmpegWriter] = {}  # External RTMP outputs
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "FrameBridge":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def setup_stream(
        self,
        stream_id: str,
        width: int,
        height: int,
        rtsp_input_url: str,
        rtsp_output_url: str,
        output_rtmp_url: str | None = None,
    ) -> None:
        """Setup input/output for a stream.

        Args:
            stream_id: Unique stream identifier
            width: Frame width
            height: Frame height
            rtsp_input_url: Local RTSP URL to read input from (from MediaMTX)
            rtsp_output_url: Local RTSP URL to write output to (to MediaMTX)
            output_rtmp_url: Optional external RTMP URL to push output to
        """
        async with self._lock:
            # Get the current event loop for cross-thread signaling
            loop = asyncio.get_running_loop()

            # Create buffers with loop reference for thread-safe signaling
            self._input_buffers[stream_id] = FrameBuffer(loop=loop)
            self._output_buffers[stream_id] = FrameBuffer(loop=loop)

            # Create reader for input
            reader = FFmpegReader(rtsp_input_url, width, height)
            self._readers[stream_id] = reader
            reader.start(self._input_buffers[stream_id])

            # Create writer for local output (MediaMTX)
            # Use RTMP internally as it's more reliable than RTSP for publishing
            rtmp_output_url = rtsp_output_url.replace("rtsp://", "rtmp://").replace(":8554/", ":1935/")
            logger.info(f"Setting up output writer: {rtmp_output_url}")
            writer = FFmpegWriter(rtmp_output_url, width, height)
            self._writers[stream_id] = writer
            try:
                writer.start()
                # Verify the process actually started
                if writer._process is None:
                    logger.error(f"FFmpeg writer process is None after start()")
                elif writer._process.poll() is not None:
                    logger.error(f"FFmpeg writer exited immediately with code: {writer._process.poll()}")
                else:
                    logger.info(f"FFmpeg writer process started with PID: {writer._process.pid}")
            except Exception as e:
                logger.error(f"Exception starting FFmpeg writer: {e}")
                import traceback
                traceback.print_exc()

            # Create writer for external RTMP output if specified
            if output_rtmp_url:
                rtmp_writer = FFmpegWriter(output_rtmp_url, width, height)
                self._rtmp_writers[stream_id] = rtmp_writer
                rtmp_writer.start()
                logger.info(f"RTMP output enabled for stream {stream_id}: {output_rtmp_url}")

            logger.info(f"Frame bridge setup for stream {stream_id}")

    async def get_input_frame(self, stream_id: str) -> np.ndarray | None:
        """Get the next input frame for a stream."""
        buffer = self._input_buffers.get(stream_id)
        if buffer is None:
            return None
        return await buffer.get_async(timeout=0.1)

    _frame_log_counter: dict[str, int] = {}

    async def send_output_frame(self, stream_id: str, frame: np.ndarray) -> None:
        """Send an output frame for a stream.

        Writes to local MediaMTX output and optional external RTMP destination.
        """
        # Log first frame and periodically
        count = self._frame_log_counter.get(stream_id, 0)
        if count == 0:
            logger.info(f"First output frame for {stream_id}: shape={frame.shape}, dtype={frame.dtype}")
        self._frame_log_counter[stream_id] = count + 1

        # Write to local RTSP output (MediaMTX)
        writer = self._writers.get(stream_id)
        if writer:
            success = writer.write_frame(frame)
            if count == 0:
                logger.info(f"First frame write result for {stream_id}: {success}")
        else:
            if count == 0:
                logger.warning(f"No writer found for stream {stream_id}")

        # Write to external RTMP output if configured
        rtmp_writer = self._rtmp_writers.get(stream_id)
        if rtmp_writer:
            rtmp_writer.write_frame(frame)

    async def teardown_stream(self, stream_id: str) -> None:
        """Teardown input/output for a stream."""
        async with self._lock:
            # Stop reader
            reader = self._readers.pop(stream_id, None)
            if reader:
                reader.stop()

            # Stop local RTSP writer
            writer = self._writers.pop(stream_id, None)
            if writer:
                writer.stop()

            # Stop external RTMP writer
            rtmp_writer = self._rtmp_writers.pop(stream_id, None)
            if rtmp_writer:
                rtmp_writer.stop()
                logger.info(f"RTMP output stopped for stream {stream_id}")

            # Clear buffers
            self._input_buffers.pop(stream_id, None)
            self._output_buffers.pop(stream_id, None)

            logger.info(f"Frame bridge teardown for stream {stream_id}")

    async def shutdown(self) -> None:
        """Shutdown all streams."""
        stream_ids = list(self._readers.keys())
        for stream_id in stream_ids:
            await self.teardown_stream(stream_id)
