# StreamDiffusion API - Daydream Replacement Instructions

This API is a drop-in replacement for the Daydream API. Use these instructions to swap your application from Daydream to this self-hosted StreamDiffusion API.

## Current Deployment

**Base URL:** `https://m7dbsf4fu06e7f-8080.proxy.runpod.net`

Replace your Daydream API base URL with the above.

## API Endpoints

All endpoints match the Daydream API format:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/streams` | Create a new stream |
| GET | `/v1/streams` | List all streams |
| GET | `/v1/streams/{id}` | Get stream details |
| GET | `/v1/streams/{id}/status` | Get stream status with metrics |
| PATCH | `/v1/streams/{id}` | Update stream parameters |
| DELETE | `/v1/streams/{id}` | Delete a stream |
| GET | `/health` | Health check |

## Creating a Stream

```bash
curl -X POST https://m7dbsf4fu06e7f-8080.proxy.runpod.net/v1/streams \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Stream",
    "pipeline": "streamdiffusion-sdxl",
    "params": {
      "model_id": "stabilityai/sdxl-turbo",
      "prompt": "your prompt here",
      "negative_prompt": "",
      "width": 1024,
      "height": 576,
      "guidance_scale": 1.0,
      "num_inference_steps": 50,
      "seed": 42
    },
    "output_rtmp_url": "rtmp://your-server:1935/live/key"
  }'
```

### Response Format

```json
{
  "id": "stream_abc123",
  "name": "My Stream",
  "stream_key": "sk_...",
  "created_at": "2026-01-22T17:00:00Z",
  "status": "pending",
  "whip_url": "https://m7dbsf4fu06e7f-8889.proxy.runpod.net/stream_abc123/whip",
  "output_playback_id": "play_...",
  "output_stream_url": "https://m7dbsf4fu06e7f-8889.proxy.runpod.net/stream_abc123_out/whep",
  "output_hls_url": "https://m7dbsf4fu06e7f-8888.proxy.runpod.net/stream_abc123_out/index.m3u8",
  "output_rtmp_url": "rtmp://your-server:1935/live/key",
  "gateway_host": "https://m7dbsf4fu06e7f-8080.proxy.runpod.net",
  "pipeline": "streamdiffusion-sdxl",
  "params": { ... }
}
```

## Streaming URLs

After creating a stream, use these URLs:

| Purpose | URL Pattern |
|---------|-------------|
| Send video IN (WebRTC) | `{whip_url}` from response |
| Watch output (WebRTC) | `{output_stream_url}` from response |
| Watch output (HLS) | `{output_hls_url}` from response |
| Push to your RTMP server | Set `output_rtmp_url` in create request |

## Stream Status

```bash
curl https://m7dbsf4fu06e7f-8080.proxy.runpod.net/v1/streams/{stream_id}/status
```

Response:
```json
{
  "id": "stream_abc123",
  "status": "running",
  "created_at": "2026-01-22T17:00:00Z",
  "started_at": "2026-01-22T17:00:05Z",
  "uptime_seconds": 120.5,
  "fps": 30.0,
  "frames_processed": 3615,
  "latency_ms": 33.2,
  "gpu_memory_used_mb": 8542.0,
  "error_message": null
}
```

### Status Values

- `pending` - Stream created, pipeline initializing
- `starting` - Pipeline loading model
- `running` - Processing frames
- `stopping` - Shutting down
- `error` - Failed (check `error_message`)

## Hot-Reloadable Parameters

These parameters can be updated without restarting the pipeline:

```bash
curl -X PATCH https://m7dbsf4fu06e7f-8080.proxy.runpod.net/v1/streams/{stream_id} \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "prompt": "new prompt here",
      "negative_prompt": "blurry, ugly",
      "guidance_scale": 1.2,
      "seed": 123
    }
  }'
```

Hot-reloadable: `prompt`, `negative_prompt`, `guidance_scale`, `delta`, `seed`, `t_index_list`

Non-hot-reloadable (requires pipeline restart): `model_id`, `width`, `height`, `controlnets`, `ip_adapter`

## Deleting a Stream

```bash
curl -X DELETE https://m7dbsf4fu06e7f-8080.proxy.runpod.net/v1/streams/{stream_id}
```

## Key Differences from Daydream

1. **No API key required** - This is a private deployment
2. **Single GPU** - One stream at a time recommended for best performance
3. **output_rtmp_url** - Pushes processed output to your RTMP server simultaneously with local playback

## Supported Parameters

```json
{
  "model_id": "stabilityai/sdxl-turbo",
  "acceleration": "tensorrt",
  "prompt": "your prompt",
  "negative_prompt": "",
  "num_inference_steps": 50,
  "t_index_list": [11],
  "guidance_scale": 1.0,
  "delta": 1.0,
  "seed": 42,
  "width": 1024,
  "height": 576,
  "use_lcm_lora": true,
  "controlnets": [
    {
      "enabled": true,
      "model_id": "xinsir/controlnet-depth-sdxl-1.0",
      "conditioning_scale": 0.5
    }
  ],
  "ip_adapter": {
    "enabled": false,
    "scale": 0.5
  }
}
```

## Example: Swap from Daydream

Before (Daydream):
```python
DAYDREAM_API = "https://api.daydream.live"
response = requests.post(f"{DAYDREAM_API}/v1/streams", headers={"Authorization": f"Bearer {API_KEY}"}, json=data)
```

After (This API):
```python
STREAM_API = "https://m7dbsf4fu06e7f-8080.proxy.runpod.net"
response = requests.post(f"{STREAM_API}/v1/streams", json=data)  # No auth needed
```

The response format is identical - just update the base URL and remove authentication.
