# StreamDiffusion API Implementation Plan

## Goal
Create a self-hosted StreamDiffusion API that is API-compatible with Daydream's API, enabling identical functionality for real-time video-to-video AI generation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        StreamDiffusion API                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  WHIP/WebRTC │    │   FastAPI    │    │  WHEP/HLS/RTMP      │  │
│  │  Ingestion   │───▶│  REST API    │───▶│  Output Delivery    │  │
│  │  Server      │    │              │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      ▲               │
│         ▼                   ▼                      │               │
│  ┌──────────────────────────────────────────────────┐              │
│  │              StreamDiffusion Pipeline            │              │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────────────┐ │              │
│  │  │ Frame   │  │ SD/SDXL  │  │ Post-Processing │ │              │
│  │  │ Decode  │─▶│ Inference│─▶│ + Encode        │─┘              │
│  │  └─────────┘  └──────────┘  └─────────────────┘               │
│  │       ▲                                                        │
│  │       │  ControlNet / IP-Adapter / LoRA                       │
│  │       │  Preprocessing Hooks                                   │
│  └──────────────────────────────────────────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core API Server (Foundation)

### 1.1 Project Structure
```
streamdiffusionapi/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI application
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── streams.py      # POST/PATCH/GET /v1/streams
│   │   │   └── health.py       # Health check endpoints
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py     # Pydantic request models
│   │   │   └── responses.py    # Pydantic response models
│   │   └── middleware/
│   │       ├── __init__.py
│   │       └── auth.py         # Bearer token auth
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── manager.py          # Stream/pipeline lifecycle management
│   │   ├── wrapper.py          # StreamDiffusion wrapper
│   │   └── config.py           # Pipeline configuration
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── whip_server.py      # WHIP ingestion handler
│   │   ├── whep_server.py      # WHEP egress handler
│   │   ├── rtmp_output.py      # RTMP restreaming
│   │   └── hls_output.py       # HLS segmentation
│   └── utils/
│       ├── __init__.py
│       └── frame_utils.py      # Frame conversion utilities
├── configs/
│   └── default.yaml            # Default configuration
├── tests/
│   ├── test_api.py
│   ├── test_pipeline.py
│   └── test_streaming.py
├── docker/
│   ├── Dockerfile              # Main Dockerfile
│   ├── Dockerfile.runpod       # RunPod-specific
│   └── docker-compose.yml      # Local development
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 1.2 API Endpoints (Daydream-Compatible)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/streams` | Create new stream |
| `PATCH` | `/v1/streams/{id}` | Update stream parameters |
| `GET` | `/v1/streams/{id}/status` | Get stream status |
| `DELETE` | `/v1/streams/{id}` | Stop and delete stream |

### 1.3 Request/Response Models

**Create Stream Request:**
```python
{
    "pipeline": "streamdiffusion",
    "name": "my-stream",
    "params": {
        "model_id": "stabilityai/sd-turbo",
        "prompt": "cyberpunk city",
        "negative_prompt": "blurry, ugly",
        "guidance_scale": 1.0,
        "delta": 0.5,
        "num_inference_steps": 1,
        "width": 512,
        "height": 512,
        "seed": 42,
        "t_index_list": [32],
        "use_lcm_lora": false,
        "lora_dict": {},
        "controlnets": [],
        "image_preprocessing": [],
        "image_postprocessing": [],
        "use_safety_checker": false
    },
    "output_rtmp_url": null  # Optional RTMP destination
}
```

**Create Stream Response:**
```python
{
    "id": "stream_abc123",
    "stream_key": "sk_xyz789",
    "created_at": "2024-01-22T10:00:00Z",
    "whip_url": "http://localhost:8080/whip/stream_abc123",
    "output_playback_id": "play_def456",
    "output_stream_url": "http://localhost:8080/whep/play_def456",
    "gateway_host": "localhost:8080",
    "status": "starting"
}
```

---

## Phase 2: StreamDiffusion Pipeline Integration

### 2.1 Pipeline Manager
- Manages multiple concurrent streams
- Handles pipeline lifecycle (create, start, stop, cleanup)
- GPU memory management (one pipeline per GPU for now)
- Hot-reload capable parameters vs full-reload parameters

### 2.2 Hot-Reloadable Parameters (No Pipeline Restart)
- `prompt`, `negative_prompt`
- `guidance_scale`
- `delta`
- `num_inference_steps`
- `t_index_list`
- `seed`
- `controlnets.conditioning_scale`

### 2.3 Full-Reload Parameters
- `model_id`
- `width`, `height`
- Adding/removing ControlNets
- Adding/removing LoRAs
- Changing `use_lcm_lora`

### 2.4 Model Support Matrix
| Model Type | Model ID | Notes |
|------------|----------|-------|
| SD-Turbo | `stabilityai/sd-turbo` | 1-step, fastest |
| LCM | `SimianLuo/LCM_Dreamshaper_v7` | 4-step |
| SDXL-Turbo | `stabilityai/sdxl-turbo` | Higher quality |
| SD 1.5 | `runwayml/stable-diffusion-v1-5` | With LCM-LoRA |

---

## Phase 3: Video Streaming Infrastructure

### 3.1 WHIP Ingestion (Input)
**Option A: aiortc (Pure Python WebRTC)**
- Pros: Pure Python, easy to integrate
- Cons: Slightly higher latency

**Option B: MediaMTX + API Integration**
- Pros: Battle-tested, handles WHIP/WHEP/RTMP natively
- Cons: External dependency, need IPC

**Recommendation: Option B (MediaMTX)**
- Use MediaMTX as media server
- API communicates via MediaMTX's REST API
- Frame extraction via FFmpeg subprocess or direct hook

### 3.2 Frame Pipeline
```
WHIP Input → MediaMTX → Frame Extraction (FFmpeg/PyAV)
    → StreamDiffusion → Frame Encoding
    → MediaMTX → WHEP/HLS/RTMP Output
```

### 3.3 Output Delivery
- **WHEP**: Ultra-low latency WebRTC output via MediaMTX
- **HLS**: Segmented HTTP streaming for broad compatibility
- **RTMP**: Restream to YouTube/Twitch/custom servers

---

## Phase 4: Local Development (Windows)

### 4.1 Prerequisites
- Python 3.10+
- CUDA 11.8 or 12.x
- PyTorch 2.2+
- NVIDIA GPU (RTX 3000+ recommended)

### 4.2 Installation Steps
```bash
# 1. Create environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install StreamDiffusion
pip install "git+https://github.com/daydreamlive/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt,controlnet,ipadapter]"

# 4. Install TensorRT (optional but recommended)
python -m streamdiffusion.tools.install-tensorrt

# 5. Install API dependencies
pip install -r requirements.txt

# 6. Download MediaMTX (Windows)
# Download from: https://github.com/bluenviron/mediamtx/releases
```

### 4.3 Windows-Specific Considerations
- Use `asyncio.WindowsSelectorEventLoopPolicy()` for compatibility
- FFmpeg path configuration
- CUDA driver compatibility check
- TensorRT may need manual Visual C++ setup

---

## Phase 5: RunPod Deployment (Linux)

### 5.1 Dockerfile
```dockerfile
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install StreamDiffusion
RUN pip install "git+https://github.com/daydreamlive/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt,controlnet,ipadapter]"
RUN python -m streamdiffusion.tools.install-tensorrt

# Install MediaMTX
RUN wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.1/mediamtx_v1.5.1_linux_amd64.tar.gz \
    && tar -xzf mediamtx_v1.5.1_linux_amd64.tar.gz \
    && mv mediamtx /usr/local/bin/

# Copy application
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# Expose ports
EXPOSE 8080 8554 8889 8890

CMD ["python", "-m", "src.api.app"]
```

### 5.2 RunPod Configuration
- GPU: RTX 4090 or A100 recommended
- Container Disk: 50GB+ for models
- Network Volume: For model caching across restarts
- Ports: 8080 (API), 8554 (RTSP), 8889 (WHIP), 8890 (WHEP)

---

## Phase 6: Testing Strategy

### 6.1 Unit Tests
- API endpoint validation
- Request/response model serialization
- Pipeline configuration parsing

### 6.2 Integration Tests
- Full stream lifecycle (create → ingest → process → output)
- Parameter hot-reload verification
- Multi-stream concurrent handling

### 6.3 Performance Benchmarks
- Frames per second at various resolutions
- Latency measurement (input to output)
- GPU memory utilization
- API response times

---

## Implementation Order

1. **Week 1: Core API + Basic Pipeline**
   - [ ] Set up project structure
   - [ ] Implement FastAPI routes (create, update, status, delete)
   - [ ] Create Pydantic models matching Daydream API
   - [ ] Integrate StreamDiffusion wrapper (txt2img first)
   - [ ] Basic frame-by-frame processing (no streaming yet)

2. **Week 2: Streaming Infrastructure**
   - [ ] Set up MediaMTX integration
   - [ ] Implement WHIP ingestion handler
   - [ ] Frame extraction from WebRTC stream
   - [ ] WHEP output delivery
   - [ ] RTMP output option

3. **Week 3: Advanced Features**
   - [ ] ControlNet support
   - [ ] LoRA loading/switching
   - [ ] IP-Adapter integration
   - [ ] Hot parameter updates
   - [ ] Multiple concurrent streams

4. **Week 4: Deployment + Polish**
   - [ ] Docker containerization
   - [ ] RunPod deployment scripts
   - [ ] Performance optimization
   - [ ] Documentation
   - [ ] Test suite completion

---

## Key Differences: Local vs RunPod

| Aspect | Windows (Local) | Linux (RunPod) |
|--------|-----------------|----------------|
| CUDA | 11.8 or 12.x | 11.8 (container) |
| TensorRT | Optional manual setup | Pre-installed in Docker |
| MediaMTX | Windows binary | Linux binary |
| Paths | `C:\...` | `/app/...` |
| GPU Access | Direct | `--gpus all` |
| Networking | localhost:8080 | Public IP + ports |

---

## Risk Mitigation

1. **GPU Memory Issues**
   - Implement proper cleanup between streams
   - Use torch.cuda.empty_cache()
   - Consider model offloading for multi-model scenarios

2. **WebRTC Complexity**
   - Start with MediaMTX (proven solution)
   - Fall back to direct FFmpeg if needed
   - Consider GStreamer as alternative

3. **Latency Concerns**
   - Profile each stage of pipeline
   - Use TensorRT acceleration
   - Optimize frame encoding/decoding

4. **API Compatibility**
   - Strict Pydantic validation
   - Integration tests against Daydream client SDKs
   - Document any intentional deviations

---

## Questions Before Starting

1. **Authentication**: Do you want to implement API key auth like Daydream, or is this for private use only?

2. **Concurrent Streams**: How many simultaneous streams do you need to support? (Affects architecture)

3. **Model Preloading**: Should models be loaded at startup or on-demand per stream?

4. **Resolution Support**: What input/output resolutions do you need? (512x512, 768x768, 1024x1024?)

5. **ControlNet Priority**: Which ControlNets are essential? (Canny, Depth, Pose, etc.)
