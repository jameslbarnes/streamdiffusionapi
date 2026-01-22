# StreamDiffusion API - Claude Code Instructions

## Project Overview

This is a self-hosted, Daydream-compatible API for real-time video generation using StreamDiffusion. It provides REST endpoints for creating and managing video streams that process input video through Stable Diffusion models in real-time.

## Architecture

```
WHIP Input → MediaMTX → Frame Extraction → StreamDiffusion → Frame Encoding → MediaMTX → WHEP/HLS/RTMP Output
```

- **FastAPI** serves the REST API on port 8080
- **MediaMTX** handles all streaming protocols (WHIP/WHEP/HLS/RTMP)
- **StreamDiffusion** (from Daydream's fork) does the actual AI processing
- **FFmpeg** bridges frames between MediaMTX and the pipeline

## RunPod Setup Instructions

When setting up on RunPod, run these commands in order:

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/streamdiffusionapi.git
cd streamdiffusionapi

# 2. Install PyTorch with CUDA (RunPod typically has CUDA 11.8 or 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install StreamDiffusion with all extras
pip install "git+https://github.com/daydreamlive/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt,controlnet,ipadapter]"

# 4. Install TensorRT acceleration tools
python -m streamdiffusion.tools.install-tensorrt

# 5. Install API dependencies
pip install -r requirements.txt

# 6. Install MediaMTX
wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.1/mediamtx_v1.5.1_linux_amd64.tar.gz
tar -xzf mediamtx_v1.5.1_linux_amd64.tar.gz
sudo mv mediamtx /usr/local/bin/ || mv mediamtx ./

# 7. Create directories for caching
mkdir -p engines models

# 8. Start the API
python -m src.api.app
```

## Key Files

- `src/api/app.py` - FastAPI application entry point
- `src/api/routes/streams.py` - Stream CRUD endpoints
- `src/api/models/requests.py` - Pydantic request models (matches Daydream API)
- `src/pipeline/wrapper.py` - StreamDiffusion integration
- `src/pipeline/manager.py` - Stream lifecycle management
- `src/streaming/mediamtx.py` - MediaMTX server management
- `src/streaming/frame_bridge.py` - FFmpeg frame I/O

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with GPU info |
| POST | `/v1/streams` | Create new stream |
| GET | `/v1/streams` | List all streams |
| GET | `/v1/streams/{id}` | Get stream details |
| PATCH | `/v1/streams/{id}` | Update stream (may restart pipeline) |
| PATCH | `/v1/streams/{id}/params` | Hot-update parameters (no restart) |
| GET | `/v1/streams/{id}/status` | Get stream status and metrics |
| DELETE | `/v1/streams/{id}` | Stop and delete stream |

## Hot-Reloadable Parameters

These can be changed without pipeline restart:
- `prompt`, `negative_prompt`
- `guidance_scale`, `delta`
- `num_inference_steps`, `t_index_list`
- `seed`
- `controlnets[].conditioning_scale`

## Ports

| Port | Service |
|------|---------|
| 8080 | REST API |
| 8554 | RTSP |
| 1935 | RTMP |
| 8888 | HLS |
| 8889 | WHIP (WebRTC input) |
| 8890 | WHEP (WebRTC output) |
| 9997 | MediaMTX API |

## Testing

```bash
# Run tests (no GPU needed for most)
pytest tests/ -v

# Test health endpoint
curl http://localhost:8080/health

# Create a test stream
curl -X POST http://localhost:8080/v1/streams \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "streamdiffusion-sdxl",
    "name": "Test",
    "params": {
      "model_id": "stabilityai/sdxl-turbo",
      "prompt": "cyberpunk city",
      "width": 768,
      "height": 448
    }
  }'
```

## Common Issues

1. **TensorRT compilation slow**: First run compiles engines (~5-10 min). Cached in `engines/`
2. **CUDA OOM**: Reduce resolution or disable ControlNets
3. **MediaMTX not found**: Ensure it's in PATH or current directory
4. **Port conflicts**: Check if ports 8080, 8889, 8890 are available

## Development Notes

- The pipeline wrapper (`src/pipeline/wrapper.py`) is where StreamDiffusion is initialized
- Frame bridge uses FFmpeg subprocesses for frame I/O with MediaMTX
- Models are downloaded to HuggingFace cache on first use
- TensorRT engines are model+resolution specific, cached in `engines/`
