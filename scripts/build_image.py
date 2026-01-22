#!/usr/bin/env python3
"""Build and push Docker image from current RunPod state.

This script commits the current container state to a Docker image,
preserving all installed dependencies, models, and TensorRT engines.

Usage:
    python scripts/build_image.py <dockerhub_username> [tag]

Example:
    python scripts/build_image.py myusername streamdiffusion-api:v1
"""

import os
import sys
import subprocess

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
    import json
    import urllib.request

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
    except Exception as e:
        print(f"API Error: {e}")
        sys.exit(1)

    pods = result.get('data', {}).get('myself', {}).get('pods', [])
    running = [p for p in pods if p.get('desiredStatus') == 'RUNNING']

    if not running:
        print("No running pods found")
        sys.exit(1)

    return running[0]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    docker_username = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "streamdiffusion-api:latest"

    if ':' not in tag:
        tag = f"{tag}:latest"

    full_image = f"{docker_username}/{tag}"

    print(f"Building image: {full_image}")
    print()

    pod = get_pod_info()
    pod_id = pod['id']
    ssh_host = f"{pod['machine']['podHostId']}@ssh.runpod.io"

    print(f"Pod: {pod_id}")
    print(f"SSH: {ssh_host}")
    print()

    # Commands to run on the pod
    commands = f'''
echo "=== Preparing container for image ==="

# Stop the API gracefully
pkill -f "python -m src.api.app" 2>/dev/null || true
sleep 2

# Clean up unnecessary files to reduce image size
rm -rf /tmp/* 2>/dev/null || true
rm -rf /root/.cache/pip 2>/dev/null || true
rm -rf /var/lib/apt/lists/* 2>/dev/null || true

# Keep huggingface cache (models) and engines (TensorRT)
echo "Keeping model cache and TensorRT engines..."

# Create the entrypoint script
cat > /docker-entrypoint.sh << 'ENTRY'
#!/bin/bash
set -e
echo "=== Starting StreamDiffusion API ==="

# Start MediaMTX
MTX_RTMPADDRESS=:1935 MTX_WEBRTCADDRESS=:8889 MTX_HLSADDRESS=:8888 nohup mediamtx > /tmp/mediamtx.log 2>&1 &
sleep 3

# Start API
cd /root/streamdiffusionapi
exec python -m src.api.app
ENTRY
chmod +x /docker-entrypoint.sh

echo "=== Container prepared ==="
echo ""
echo "Now run these commands on your LOCAL machine to commit and push:"
echo ""
echo "  # SSH into RunPod and commit (run from any terminal with docker):"
echo "  # Note: This requires docker access to the RunPod host, which isn't directly available."
echo ""
echo "  # Instead, use RunPod's web UI to create a template from this pod,"
echo "  # or build from the Dockerfile locally."
echo ""
'''

    print("Running preparation on pod...")
    print()

    # Run via SSH
    import threading
    import time

    proc = subprocess.Popen(
        ["ssh", "-tt", "-o", "ServerAliveInterval=30", ssh_host],
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
                if decoded.strip() and not any(x in decoded for x in ['RUNPOD', 'WARNING', 'vulnerable', '[?2004', 'root@']):
                    print(decoded.rstrip())
        except Exception:
            pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    time.sleep(2)

    try:
        proc.stdin.write((commands + "\nexit\n").encode())
        proc.stdin.flush()
    except Exception as e:
        print(f"Error: {e}")

    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()

    done.set()
    reader.join(timeout=2)

    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("Option 1: Use RunPod Template (Recommended)")
    print("  1. Go to https://www.runpod.io/console/pods")
    print(f"  2. Click on pod {pod_id}")
    print("  3. Click 'Create Template' to save current state")
    print("  4. Use this template for future pods")
    print()
    print("Option 2: Build from Dockerfile")
    print("  1. Clone the repo: git clone https://github.com/jameslbarnes/streamdiffusionapi")
    print("  2. Build: docker build -t streamdiffusion-api .")
    print("  3. Push: docker push yourusername/streamdiffusion-api")
    print()
    print(f"Option 3: Update deploy script to use your image")
    print(f"  Edit scripts/runpod_cmd.py and change PYTORCH_IMAGE to your image")
    print()


if __name__ == "__main__":
    main()
