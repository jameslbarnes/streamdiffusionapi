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

        # FFmpeg command to read stream and output raw frames
        cmd = [
            "ffmpeg",
            "-i", self.stream_url,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", "30",  # Target framerate
            "-an",  # No audio
            "-",
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.width * self.height * 3 * 2,  # Buffer 2 frames
        )

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        """Read frames in a loop."""
        frame_size = self.width * self.height * 3

        while self._running and self._process and self._process.poll() is None:
            try:
                raw_frame = self._process.stdout.read(frame_size)
                if len(raw_frame) == frame_size:
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    )
                    if self._buffer:
                        self._buffer.put(frame)
            except Exception as e:
                logger.warning(f"Frame read error: {e}")
                break

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
    ):
        self.stream_url = stream_url
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.preset = preset
        self._process: subprocess.Popen | None = None
        self._running = False

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
            "-f", "rtsp",
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
    ) -> None:
        """Setup input/output for a stream."""
        async with self._lock:
            # Create buffers
            self._input_buffers[stream_id] = FrameBuffer()
            self._output_buffers[stream_id] = FrameBuffer()

            # Create reader for input
            reader = FFmpegReader(rtsp_input_url, width, height)
            self._readers[stream_id] = reader
            reader.start(self._input_buffers[stream_id])

            # Create writer for output
            writer = FFmpegWriter(rtsp_output_url, width, height)
            self._writers[stream_id] = writer
            writer.start()

            logger.info(f"Frame bridge setup for stream {stream_id}")

    async def get_input_frame(self, stream_id: str) -> np.ndarray | None:
        """Get the next input frame for a stream."""
        buffer = self._input_buffers.get(stream_id)
        if buffer is None:
            return None
        return await buffer.get_async(timeout=0.1)

    async def send_output_frame(self, stream_id: str, frame: np.ndarray) -> None:
        """Send an output frame for a stream."""
        writer = self._writers.get(stream_id)
        if writer:
            writer.write_frame(frame)

    async def teardown_stream(self, stream_id: str) -> None:
        """Teardown input/output for a stream."""
        async with self._lock:
            # Stop reader
            reader = self._readers.pop(stream_id, None)
            if reader:
                reader.stop()

            # Stop writer
            writer = self._writers.pop(stream_id, None)
            if writer:
                writer.stop()

            # Clear buffers
            self._input_buffers.pop(stream_id, None)
            self._output_buffers.pop(stream_id, None)

            logger.info(f"Frame bridge teardown for stream {stream_id}")

    async def shutdown(self) -> None:
        """Shutdown all streams."""
        stream_ids = list(self._readers.keys())
        for stream_id in stream_ids:
            await self.teardown_stream(stream_id)
