"""MediaMTX integration for WHIP/WHEP/RTMP streaming."""

import asyncio
import logging
import os
import platform
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class MediaMTXManager:
    """Manages MediaMTX media server for streaming."""

    _instance: "MediaMTXManager | None" = None

    # Default ports
    RTSP_PORT = 8554
    RTMP_PORT = 1935
    HLS_PORT = 8888
    WEBRTC_PORT = 8889
    API_PORT = 9997

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._config_path: Path | None = None
        self._binary_path: Path | None = None

    @classmethod
    def get_instance(cls) -> "MediaMTXManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _find_binary(self) -> Path | None:
        """Find MediaMTX binary."""
        # Check common locations
        search_paths = [
            Path("mediamtx"),
            Path("./mediamtx"),
            Path("./bin/mediamtx"),
            Path("/usr/local/bin/mediamtx"),
            Path("/usr/bin/mediamtx"),
        ]

        # Add .exe for Windows
        if platform.system() == "Windows":
            search_paths = [
                p.with_suffix(".exe") if not str(p).endswith(".exe") else p
                for p in search_paths
            ] + [
                Path("mediamtx.exe"),
                Path("./mediamtx.exe"),
                Path("./bin/mediamtx.exe"),
            ]

        for path in search_paths:
            if path.exists():
                return path.absolute()

        # Check PATH
        import shutil

        binary_name = "mediamtx.exe" if platform.system() == "Windows" else "mediamtx"
        path_binary = shutil.which(binary_name)
        if path_binary:
            return Path(path_binary)

        return None

    def _generate_config(self, streams: list[str] | None = None) -> str:
        """Generate MediaMTX configuration."""
        config = f"""
# MediaMTX configuration for StreamDiffusion API
# Auto-generated - do not edit manually

logLevel: info
logDestinations: [stdout]

# API settings
api: yes
apiAddress: :{self.API_PORT}

# RTSP settings
rtsp: yes
rtspAddress: :{self.RTSP_PORT}
protocols: [tcp, udp]

# RTMP settings
rtmp: yes
rtmpAddress: :{self.RTMP_PORT}

# HLS settings
hls: yes
hlsAddress: :{self.HLS_PORT}
hlsAlwaysRemux: yes
hlsSegmentCount: 3
hlsSegmentDuration: 1s

# WebRTC settings
webrtc: yes
webrtcAddress: :{self.WEBRTC_PORT}
webrtcICEServers2: []

# Path defaults
pathDefaults:
  source: publisher
  sourceOnDemand: no
  record: no

# Paths (streams) - dynamically configured
paths:
  all_others:
"""
        return config

    async def ensure_running(self) -> None:
        """Ensure MediaMTX is running."""
        # Check if already running
        if await self._is_running():
            logger.info("MediaMTX already running")
            return

        # Find binary
        self._binary_path = self._find_binary()
        if self._binary_path is None:
            raise RuntimeError(
                "MediaMTX not found. Download from: "
                "https://github.com/bluenviron/mediamtx/releases"
            )

        # Generate config
        config_content = self._generate_config()
        self._config_path = Path("mediamtx.yml")
        self._config_path.write_text(config_content)

        # Start MediaMTX
        logger.info(f"Starting MediaMTX from {self._binary_path}")

        try:
            self._process = subprocess.Popen(
                [str(self._binary_path), str(self._config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for startup
            for _ in range(30):  # 3 second timeout
                await asyncio.sleep(0.1)
                if await self._is_running():
                    logger.info("MediaMTX started successfully")
                    return

            raise RuntimeError("MediaMTX failed to start")

        except Exception as e:
            logger.error(f"Failed to start MediaMTX: {e}")
            raise

    async def _is_running(self) -> bool:
        """Check if MediaMTX API is responding."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{self.API_PORT}/v3/paths/list",
                    timeout=2.0,
                )
                return response.status_code == 200
        except Exception:
            return False

    async def create_path(self, path_name: str, source_type: str = "publisher") -> None:
        """Create a new path (stream endpoint) in MediaMTX."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:{self.API_PORT}/v3/config/paths/add/{path_name}",
                    json={
                        "source": source_type,
                    },
                    timeout=5.0,
                )

                if response.status_code not in (200, 201):
                    logger.warning(f"Failed to create path {path_name}: {response.text}")
        except Exception as e:
            logger.warning(f"Failed to create MediaMTX path: {e}")

    async def delete_path(self, path_name: str) -> None:
        """Delete a path from MediaMTX."""
        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"http://localhost:{self.API_PORT}/v3/config/paths/delete/{path_name}",
                    timeout=5.0,
                )
        except Exception as e:
            logger.warning(f"Failed to delete MediaMTX path: {e}")

    async def get_path_info(self, path_name: str) -> dict | None:
        """Get information about a path."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{self.API_PORT}/v3/paths/get/{path_name}",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        return None

    async def list_paths(self) -> list[dict]:
        """List all paths."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{self.API_PORT}/v3/paths/list",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("items", [])
        except Exception as e:
            logger.warning(f"Failed to list paths: {e}")
        return []

    async def shutdown(self) -> None:
        """Shutdown MediaMTX."""
        if self._process:
            logger.info("Stopping MediaMTX...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        # Cleanup config file
        if self._config_path and self._config_path.exists():
            try:
                self._config_path.unlink()
            except Exception:
                pass

        logger.info("MediaMTX stopped")
