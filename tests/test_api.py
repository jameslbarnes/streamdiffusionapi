"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.models.requests import CreateStreamRequest, StreamParams
from src.api.models.responses import StreamStatus


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "gpu_available" in data
        assert "version" in data


class TestStreamModels:
    """Test Pydantic models."""

    def test_stream_params_defaults(self):
        """Test StreamParams default values."""
        params = StreamParams()
        assert params.model_id == "stabilityai/sdxl-turbo"
        assert params.width == 768
        assert params.height == 448
        assert params.guidance_scale == 1.0

    def test_stream_params_custom(self):
        """Test StreamParams with custom values."""
        params = StreamParams(
            model_id="runwayml/stable-diffusion-v1-5",
            prompt="test prompt",
            width=512,
            height=512,
            guidance_scale=7.5,
            controlnets=[],
        )
        assert params.model_id == "runwayml/stable-diffusion-v1-5"
        assert params.prompt == "test prompt"
        assert params.width == 512

    def test_create_stream_request(self):
        """Test CreateStreamRequest model."""
        request = CreateStreamRequest(
            pipeline="streamdiffusion-sdxl",
            name="Test Stream",
            params=StreamParams(prompt="test"),
        )
        assert request.pipeline == "streamdiffusion-sdxl"
        assert request.name == "Test Stream"
        assert request.params.prompt == "test"

    def test_create_stream_request_with_rtmp(self):
        """Test CreateStreamRequest with RTMP output."""
        request = CreateStreamRequest(
            pipeline="streamdiffusion",
            name="RTMP Test",
            params=StreamParams(),
            output_rtmp_url="rtmp://example.com/live",
        )
        assert request.output_rtmp_url == "rtmp://example.com/live"


class TestStreamEndpoints:
    """Test stream CRUD endpoints."""

    def test_list_streams_empty(self, client):
        """Test listing streams when none exist."""
        response = client.get("/v1/streams")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.skip(reason="Requires GPU and StreamDiffusion installation")
    def test_create_stream(self, client):
        """Test creating a stream."""
        response = client.post(
            "/v1/streams",
            json={
                "pipeline": "streamdiffusion-sdxl",
                "name": "Test Stream",
                "params": {
                    "model_id": "stabilityai/sdxl-turbo",
                    "prompt": "test",
                    "width": 512,
                    "height": 512,
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Stream"
        assert "id" in data
        assert "whip_url" in data
        assert "output_stream_url" in data

    def test_get_nonexistent_stream(self, client):
        """Test getting a stream that doesn't exist."""
        response = client.get("/v1/streams/nonexistent")
        assert response.status_code == 404

    def test_delete_nonexistent_stream(self, client):
        """Test deleting a stream that doesn't exist."""
        response = client.delete("/v1/streams/nonexistent")
        assert response.status_code == 404
