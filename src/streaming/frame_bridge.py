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

    def __init__(self, max_size: int = 30):
        self._buffer: deque[np.ndarray] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._event = asyncio.Event()

    def put(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append(frame)
        # Signal that a frame is available (asyncio event)
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self._event.set)
        except RuntimeError:
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
            while self._running and self._process and self._process.poll() is None:
                try:
                    raw_frame = self._process.stdout.read(frame_size)
                    if len(raw_frame) == frame_size:
                        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                            (self.height, self.width, 3)
                        )
                        if self._buffer:
                            self._buffer.put(frame)
                    elif len(raw_frame) == 0:
                        # Stream ended
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
        cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", "zerolatency",
            "-profile:v", "high",
            "-bf", "0",  # No B-frames for lower latency
            "-g", str(self.fps),  # Keyframe every second
            "-f", self.output_format,
            self.stream_url,
        ]

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=self.width * self.height * 3 * 2,
        )
        self._running = True
        logger.info(f"FFmpeg writer started: {self.stream_url} (format: {self.output_format})")

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a frame to the stream."""
        if not self._running or not self._process:
            return False

        try:
            # Ensure frame is the right format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Resize if needed
            if frame.shape[:2] != (self.height, self.width):
                from PIL import Image

                img = Image.fromarray(frame)
                img = img.resize((self.width, self.height))
                frame = np.array(img)

            self._process.stdin.write(frame.tobytes())
            self._process.stdin.flush()
            return True
        except Exception as e:
            logger.warning(f"Frame write error: {e}")
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
            # Create buffers
            self._input_buffers[stream_id] = FrameBuffer()
            self._output_buffers[stream_id] = FrameBuffer()

            # Create reader for input
            reader = FFmpegReader(rtsp_input_url, width, height)
            self._readers[stream_id] = reader
            reader.start(self._input_buffers[stream_id])

            # Create writer for local output (MediaMTX)
            writer = FFmpegWriter(rtsp_output_url, width, height)
            self._writers[stream_id] = writer
            writer.start()

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

    async def send_output_frame(self, stream_id: str, frame: np.ndarray) -> None:
        """Send an output frame for a stream.

        Writes to local MediaMTX output and optional external RTMP destination.
        """
        # Write to local RTSP output (MediaMTX)
        writer = self._writers.get(stream_id)
        if writer:
            writer.write_frame(frame)

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
