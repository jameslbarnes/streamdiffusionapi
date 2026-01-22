# StreamDiffusion API

A self-hosted, Daydream-compatible API for real-time video generation using StreamDiffusion.

## Features

- **Daydream API Compatible**: Drop-in replacement for Daydream's API endpoints
- **Real-time Video Processing**: WHIP input → StreamDiffusion → WHEP/HLS/RTMP output
- **Multiple Output Formats**: WebRTC (WHEP), HLS, and RTMP streaming
- **Hot Parameter Updates**: Change prompts, guidance scale, and seeds without pipeline restart
- **ControlNet Support**: Multiple ControlNets with depth, canny, tile, and more
- **IP-Adapter Support**: Style transfer with reference images
- **TensorRT Acceleration**: Optimized inference with TensorRT
- **GPU Memory Efficient**: Tiny VAE and optimized batch processing

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (RTX 3000+ recommended)
- FFmpeg
- MediaMTX (auto-downloaded or manual install)

### Windows Installation

```powershell
# 1. Clone and enter directory
cd streamdiffusionapi

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install StreamDiffusion
pip install "git+https://github.com/daydreamlive/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt,controlnet,ipadapter]"

# 5. Install TensorRT (optional but recommended)
python -m streamdiffusion.tools.install-tensorrt

# 6. Install API dependencies
pip install -r requirements.txt

# 7. Download MediaMTX
# Download from: https://github.com/bluenviron/mediamtx/releases
# Extract mediamtx.exe to the project directory

# 8. Start the API
python -m src.api.app
```

### Linux Installation

```bash
# 1. Clone and enter directory
cd streamdiffusionapi

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install StreamDiffusion
pip install "git+https://github.com/daydreamlive/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt,controlnet,ipadapter]"

# 5. Install TensorRT
python -m streamdiffusion.tools.install-tensorrt

# 6. Install API dependencies
pip install -r requirements.txt

# 7. Install MediaMTX
wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.1/mediamtx_v1.5.1_linux_amd64.tar.gz
tar -xzf mediamtx_v1.5.1_linux_amd64.tar.gz
sudo mv mediamtx /usr/local/bin/

# 8. Start the API
python -m src.api.app
```

### Docker

```bash
# Build
docker build -t streamdiffusion-api -f docker/Dockerfile .

# Run
docker run --gpus all -p 8080:8080 -p 8889:8889 -p 8890:8890 streamdiffusion-api
```

### Docker Compose

```bash
cd docker
docker-compose up
```

## API Usage

### Create a Stream

```bash
curl -X POST http://localhost:8080/v1/streams \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "streamdiffusion-sdxl",
    "name": "My Stream",
    "params": {
      "model_id": "stabilityai/sdxl-turbo",
      "prompt": "beautiful nature, vibrant colors",
      "negative_prompt": "blurry, low quality",
      "width": 768,
      "height": 448,
      "t_index_list": [11],
      "use_lcm_lora": true,
      "acceleration": "tensorrt"
    }
  }'
```

Response:
```json
{
  "id": "stream_abc123",
  "name": "My Stream",
  "stream_key": "sk_xyz789",
  "status": "starting",
  "whip_url": "http://localhost:8889/stream_abc123/whip",
  "output_stream_url": "http://localhost:8890/stream_abc123_out/whep",
  "output_hls_url": "http://localhost:8888/stream_abc123_out/index.m3u8"
}
```

### Update Stream Parameters (Hot Reload)

```bash
curl -X PATCH http://localhost:8080/v1/streams/{id}/params \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cyberpunk city at night",
    "guidance_scale": 1.2,
    "seed": 42
  }'
```

### Get Stream Status

```bash
curl http://localhost:8080/v1/streams/{id}/status
```

### Delete Stream

```bash
curl -X DELETE http://localhost:8080/v1/streams/{id}
```

## Streaming Video

### Input (WHIP)

Use OBS or any WHIP-compatible software:
- **Service**: WHIP
- **Server**: `http://localhost:8889/{stream_id}/whip`

### Output (WHEP)

Play in a WebRTC-compatible player:
- **URL**: `http://localhost:8890/{stream_id}_out/whep`

### Output (HLS)

Play in any HLS player:
- **URL**: `http://localhost:8888/{stream_id}_out/index.m3u8`

### Output (RTMP)

Configure `output_rtmp_url` when creating the stream to restream to YouTube/Twitch.

## Configuration

### Full Stream Config Example

```json
{
  "pipeline": "streamdiffusion-sdxl",
  "name": "Production Stream",
  "params": {
    "model_id": "stabilityai/sdxl-turbo",
    "prompt": "beautiful nature, minimalistic, vibrant",
    "negative_prompt": "blurry, low quality, flat",
    "num_inference_steps": 50,
    "seed": 704475,
    "t_index_list": [11],
    "width": 768,
    "height": 448,
    "acceleration": "tensorrt",
    "use_lcm_lora": true,
    "guidance_scale": 1,
    "delta": 1,
    "controlnets": [
      {
        "enabled": true,
        "model_id": "xinsir/controlnet-depth-sdxl-1.0",
        "preprocessor": "depth_tensorrt",
        "conditioning_scale": 0.5
      }
    ],
    "ip_adapter": {
      "enabled": false,
      "scale": 0.5
    },
    "enable_similar_image_filter": false,
    "video_codec": "h264",
    "video_preset": "medium"
  },
  "output_rtmp_url": "rtmp://your-server.com/live"
}
```

### Hot-Reloadable Parameters

These can be updated without pipeline restart (~instant):
- `prompt`, `negative_prompt`
- `guidance_scale`, `delta`
- `num_inference_steps`, `t_index_list`
- `seed`
- `controlnets[].conditioning_scale`

### Parameters Requiring Restart

These require ~30 seconds pipeline reload:
- `model_id`
- `width`, `height`
- Adding/removing ControlNets
- `use_lcm_lora`
- `acceleration`

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

## RunPod Deployment

### Using the Template

1. Create a new pod with the `runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel` template
2. Mount a network volume to `/runpod-volume` for model caching
3. Build and deploy:

```bash
docker build -t your-registry/streamdiffusion-api -f docker/Dockerfile.runpod .
docker push your-registry/streamdiffusion-api
```

### Network Volume Structure

```
/runpod-volume/
├── engines/     # TensorRT engine cache
└── models/      # HuggingFace model cache
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

## Troubleshooting

### CUDA Out of Memory

- Reduce resolution (`width`, `height`)
- Disable ControlNets
- Use `use_denoising_batch: false`

### TensorRT Compilation Slow

First run compiles TensorRT engines (~5-10 minutes). Engines are cached in `engines/` directory.

### MediaMTX Not Starting

- Check if port 8889 is available
- Download MediaMTX manually and place in project directory
- Check firewall settings

### Low FPS

- Enable TensorRT: `"acceleration": "tensorrt"`
- Use SD-Turbo or SDXL-Turbo models
- Reduce `num_inference_steps`
- Use single `t_index_list` value

## License

MIT License - see LICENSE file.

## Acknowledgments

- [StreamDiffusion](https://github.com/daydreamlive/StreamDiffusion) by Daydream
- [MediaMTX](https://github.com/bluenviron/mediamtx) for streaming infrastructure
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the model pipeline
