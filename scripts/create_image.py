#!/usr/bin/env python3
"""Create a Docker image from the current RunPod pod state.

This commits the running container to a new image and pushes to Docker Hub,
preserving all models, TensorRT engines, and installed packages.

Usage:
    python scripts/create_image.py <dockerhub_username> [tag]

Example:
    python scripts/create_image.py jameslbarnes streamdiffusion-api:v1

Prerequisites:
    - Docker Hub account
    - Run `docker login` on the pod first (or set DOCKER_USERNAME/DOCKER_PASSWORD env vars)
"""

import os
import sys
import json
import urllib.request
import subprocess
import threading
import time

def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

def get_pod_info():
    """Get current pod info from RunPod API."""
    load_env()
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set")
        sys.exit(1)

    query = """query {
        myself {
            pods {
                id name desiredStatus
                machine { podHostId }
            }
        }
    }"""

    data = json.dumps({"query": query}).encode('utf-8')
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"API Error: {e.code}")
        sys.exit(1)

    pods = result.get('data', {}).get('myself', {}).get('pods', [])
    running = [p for p in pods if p.get('desiredStatus') == 'RUNNING']

    if not running:
        print("No running pods found")
        sys.exit(1)

    return running[0]


def run_ssh_command(ssh_host, command, timeout=300):
    """Run a command on the pod via SSH."""
    proc = subprocess.Popen(
        ["ssh", "-tt", "-o", "ServerAliveInterval=30", "-o", "StrictHostKeyChecking=no", ssh_host],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0
    )

    output_lines = []
    done = threading.Event()

    def read_output():
        try:
            while not done.is_set():
                line = proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode('utf-8', errors='replace')
                output_lines.append(decoded)
                # Print interesting lines
                clean = decoded.strip()
                if clean and not any(x in decoded for x in ['RUNPOD', 'WARNING', 'vulnerable', '[?2004', 'root@', 'Enjoy your Pod']):
                    print(decoded.rstrip())
        except Exception:
            pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    time.sleep(2)

    try:
        proc.stdin.write((command + "\nexit\n").encode())
        proc.stdin.flush()
    except Exception as e:
        print(f"Error sending command: {e}")
        return None

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("Command timed out")
        return None

    done.set()
    reader.join(timeout=2)

    return ''.join(output_lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    docker_username = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "streamdiffusion-api:latest"

    if ':' not in tag:
        tag = f"{tag}:latest"

    full_image = f"{docker_username}/{tag}"

    print(f"=" * 60)
    print(f"Creating Docker image: {full_image}")
    print(f"=" * 60)
    print()

    # Get pod info
    pod = get_pod_info()
    pod_id = pod['id']
    ssh_host = f"{pod['machine']['podHostId']}@ssh.runpod.io"

    print(f"Pod ID: {pod_id}")
    print(f"SSH: {ssh_host}")
    print()

    # Check for Docker credentials
    docker_password = os.environ.get("DOCKER_PASSWORD", "")

    # Build the script to run on the pod
    script = f'''
echo "=== Preparing container for image creation ==="

# Stop API to free memory
pkill -f "python -m src.api.app" 2>/dev/null || true
sleep 2

# Clean up temp files to reduce image size
rm -rf /tmp/* 2>/dev/null || true
rm -rf /root/.cache/pip 2>/dev/null || true
apt-get clean 2>/dev/null || true

# Create entrypoint script
cat > /docker-entrypoint.sh << 'ENTRYPOINT'
#!/bin/bash
set -e
echo "=== Starting StreamDiffusion API ==="

# Start MediaMTX
MTX_RTMPADDRESS=:1935 MTX_WEBRTCADDRESS=:8889 MTX_HLSADDRESS=:8888 nohup mediamtx > /tmp/mediamtx.log 2>&1 &
sleep 3

# Pull latest code (optional - remove if you want frozen code)
cd /root/streamdiffusionapi
git pull 2>/dev/null || true

# Start API
exec python -m src.api.app
ENTRYPOINT
chmod +x /docker-entrypoint.sh

echo "=== Getting container ID ==="
CONTAINER_ID=$(cat /proc/1/cpuset | cut -d/ -f3)
echo "Container ID: $CONTAINER_ID"

# Check if we can access docker socket
if [ -S /var/run/docker.sock ]; then
    echo "=== Docker socket available, committing container ==="

    # Login to Docker Hub
    echo "{docker_password}" | docker login -u {docker_username} --password-stdin 2>/dev/null || echo "Docker login via stdin failed, trying without..."

    # Commit the container
    echo "Committing container to {full_image}..."
    docker commit "$CONTAINER_ID" {full_image}

    # Push to Docker Hub
    echo "Pushing to Docker Hub..."
    docker push {full_image}

    echo ""
    echo "=== SUCCESS ==="
    echo "Image pushed: {full_image}"
    echo ""
    echo "To deploy new pods with this image:"
    echo "  python scripts/runpod_cmd.py deploy 'NVIDIA H100 PCIe' 1024x576 {full_image}"
else
    echo ""
    echo "=== Docker socket not available ==="
    echo ""
    echo "RunPod doesn't expose docker socket by default."
    echo "Alternative options:"
    echo ""
    echo "1. Use the Dockerfile to build locally:"
    echo "   cd streamdiffusionapi"
    echo "   docker build -t {full_image} ."
    echo "   docker push {full_image}"
    echo ""
    echo "2. Use RunPod's network volumes to persist models:"
    echo "   - Create a network volume"
    echo "   - Mount at /root/.cache/huggingface"
    echo "   - Models persist across pod restarts"
fi

# Restart the API
echo ""
echo "Restarting API..."
cd /root/streamdiffusionapi
nohup python -m src.api.app > /tmp/api.log 2>&1 &
echo "API restarted"
'''

    print("Running on pod...")
    print()

    result = run_ssh_command(ssh_host, script, timeout=600)

    if result and "SUCCESS" in result:
        print()
        print("=" * 60)
        print("Image created successfully!")
        print("=" * 60)
        print()
        print(f"Deploy new pods with:")
        print(f"  python scripts/runpod_cmd.py deploy 'NVIDIA H100 PCIe' 1024x576 {full_image}")
        print()
        print("Or set environment variable:")
        print(f"  export STREAMDIFFUSION_IMAGE={full_image}")
        print(f"  python scripts/runpod_cmd.py deploy")


if __name__ == "__main__":
    main()
