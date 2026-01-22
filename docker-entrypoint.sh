#!/bin/bash
set -e

echo "=== Starting StreamDiffusion API ==="

# Start MediaMTX in background
echo "Starting MediaMTX..."
MTX_RTMPADDRESS=:1935 \
MTX_WEBRTCADDRESS=:8889 \
MTX_HLSADDRESS=:8888 \
nohup mediamtx > /tmp/mediamtx.log 2>&1 &

# Wait for MediaMTX to be ready
sleep 3

# Check if MediaMTX started
if ! pgrep -x mediamtx > /dev/null; then
    echo "ERROR: MediaMTX failed to start"
    cat /tmp/mediamtx.log
    exit 1
fi
echo "MediaMTX started"

# Start the API
echo "Starting API server..."
exec python -m src.api.app
