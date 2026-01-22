#!/usr/bin/env python3
"""RunPod management commands for StreamDiffusion API"""

import os
import sys
import json
import urllib.request
import urllib.error
import time

API_URL = "https://api.runpod.io/graphql"

# Try to load from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

# Pinned versions
PYTORCH_IMAGE = "pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel"
MEDIAMTX_VERSION = "v1.5.1"
REPO_URL = "https://github.com/jameslbarnes/streamdiffusionapi.git"


def graphql(query):
    """Execute a GraphQL query against RunPod API"""
    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set in .env or environment")
        sys.exit(1)

    data = json.dumps({"query": query}).encode('utf-8')
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.read().decode('utf-8')}")
        sys.exit(1)


def list_gpus():
    """List available GPUs"""
    result = graphql("query { gpuTypes { id displayName memoryInGb secureCloud communityCloud } }")
    gpus = result.get('data', {}).get('gpuTypes', [])

    # Filter for high-end GPUs
    keywords = ['H100', 'H200', 'A100', 'RTX', 'B200', '5090', '6000']
    filtered = [g for g in gpus if any(k.lower() in g['displayName'].lower() for k in keywords)]

    print(f"Found {len(filtered)} relevant GPUs:\n")
    for g in sorted(filtered, key=lambda x: x['memoryInGb'], reverse=True):
        secure = 'available' if g['secureCloud'] else '-'
        community = 'available' if g['communityCloud'] else '-'
        print(f"  {g['id']}")
        print(f"    {g['displayName']} ({g['memoryInGb']}GB)")
        print(f"    Secure: {secure}, Community: {community}")
        print()


def list_pods():
    """List all running pods"""
    result = graphql("""query {
        myself {
            pods {
                id name desiredStatus
                runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } }
                machine { podHostId gpuDisplayName }
            }
        }
    }""")
    pods = result.get('data', {}).get('myself', {}).get('pods', [])

    if not pods:
        print("No pods running")
        return

    for p in pods:
        gpu = p.get('machine', {}).get('gpuDisplayName', 'Unknown')
        host_id = p.get('machine', {}).get('podHostId', '')
        uptime = p.get('runtime', {}).get('uptimeInSeconds', 0) if p.get('runtime') else 0
        hours = uptime // 3600
        mins = (uptime % 3600) // 60
        print(f"{p['id']}: {p['name']} ({p['desiredStatus']})")
        print(f"  GPU: {gpu}")
        print(f"  Uptime: {hours}h {mins}m")
        print(f"  API: https://{p['id']}-8080.proxy.runpod.net")
        print(f"  SSH: ssh {host_id}@ssh.runpod.io")
        print()


def terminate_pod(pod_id):
    """Terminate a pod"""
    result = graphql(f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
    if 'errors' in result:
        print(f"Error: {result['errors']}")
        return False
    print(f"Pod {pod_id} terminated")
    return True


def stop_pod(pod_id):
    """Stop a pod (can be restarted)"""
    result = graphql(f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id }} }}')
    if 'errors' in result:
        print(f"Error: {result['errors']}")
        return False
    print(f"Pod {pod_id} stopped")
    return True


def deploy_pod(gpu_id="NVIDIA H100 PCIe", resolution="1024x576"):
    """Deploy a new StreamDiffusion API pod"""

    width, height = resolution.split('x')

    # Startup script that:
    # 1. Installs ffmpeg and MediaMTX
    # 2. Clones the repo
    # 3. Installs dependencies
    # 4. Starts the API
    startup_script = f'''bash -c "
set -e
echo '=== Installing system dependencies ==='
apt-get update -qq && apt-get install -qq -y ffmpeg git curl > /dev/null 2>&1

echo '=== Installing MediaMTX ==='
curl -sL https://github.com/bluenviron/mediamtx/releases/download/{MEDIAMTX_VERSION}/mediamtx_{MEDIAMTX_VERSION}_linux_amd64.tar.gz | tar xz -C /usr/local/bin

echo '=== Cloning StreamDiffusion API ==='
cd /root
git clone {REPO_URL} streamdiffusionapi || (cd streamdiffusionapi && git pull)
cd streamdiffusionapi

echo '=== Installing PyTorch ==='
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo '=== Installing StreamDiffusion ==='
pip install 'git+https://github.com/daydreamlive/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt,controlnet,ipadapter]'

echo '=== Installing TensorRT ==='
python -m streamdiffusion.tools.install-tensorrt || echo 'TensorRT warning (may be ok)'

echo '=== Installing API dependencies ==='
pip install -r requirements.txt

echo '=== Creating cache directories ==='
mkdir -p engines models

echo '=== Starting MediaMTX ==='
MTX_RTMPADDRESS=:1935 MTX_WEBRTCADDRESS=:8889 MTX_HLSADDRESS=:8888 nohup mediamtx > /tmp/mediamtx.log 2>&1 &
sleep 3

echo '=== Starting StreamDiffusion API ==='
python -m src.api.app
"'''

    # Escape for JSON
    docker_cmd_escaped = startup_script.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    query = f'''mutation {{
        podFindAndDeployOnDemand(input: {{
            imageName: "{PYTORCH_IMAGE}",
            gpuTypeIdList: ["{gpu_id}"],
            gpuCount: 1,
            cloudType: SECURE,
            volumeInGb: 0,
            containerDiskInGb: 100,
            minMemoryInGb: 64,
            name: "streamdiffusion-api",
            ports: "8080/http,8889/http,8890/http,8888/http,1935/tcp",
            dockerArgs: "{docker_cmd_escaped}"
        }}) {{
            id
            machine {{
                podHostId
                gpuDisplayName
            }}
        }}
    }}'''

    print(f"Deploying StreamDiffusion API pod...")
    print(f"  GPU: {gpu_id}")
    print(f"  Resolution: {resolution}")
    print(f"  Image: {PYTORCH_IMAGE}")
    print()

    result = graphql(query)

    if 'errors' in result:
        print(f"Error: {result['errors']}")
        return None

    pod_data = result['data']['podFindAndDeployOnDemand']
    pod_id = pod_data['id']
    gpu = pod_data['machine']['gpuDisplayName']
    host_id = pod_data['machine']['podHostId']

    print(f"Pod deployed successfully!")
    print(f"  ID: {pod_id}")
    print(f"  GPU: {gpu}")
    print()
    print(f"Endpoints (available after startup):")
    print(f"  API:  https://{pod_id}-8080.proxy.runpod.net")
    print(f"  WHIP: https://{pod_id}-8889.proxy.runpod.net/{{stream_id}}/whip")
    print(f"  WHEP: https://{pod_id}-8890.proxy.runpod.net/{{stream_id}}_out/whep")
    print(f"  HLS:  https://{pod_id}-8888.proxy.runpod.net/{{stream_id}}_out/index.m3u8")
    print()
    print(f"SSH: ssh {host_id}@ssh.runpod.io")
    print()
    print("Note: Startup takes ~5-10 minutes for first-time TensorRT compilation")

    return pod_id


def get_rtmp_url(pod_id):
    """Get RTMP ingest URL for a pod"""
    query = f'''query {{
        pod(input: {{ podId: "{pod_id}" }}) {{
            id
            runtime {{
                ports {{ ip isIpPublic privatePort publicPort type }}
            }}
        }}
    }}'''

    result = graphql(query)
    if 'errors' in result:
        print(f"Error: {result['errors']}")
        return None

    pod = result.get('data', {}).get('pod')
    if not pod:
        print("Pod not found")
        return None

    ports = pod.get('runtime', {}).get('ports', [])
    for p in ports:
        if p.get('privatePort') == 1935 and p.get('type') == 'tcp':
            ip = p.get('ip', '')
            public_port = p.get('publicPort', '')
            url = f"rtmp://{ip}:{public_port}/live"
            print(f"RTMP Ingest URL: {url}")
            return url

    print("RTMP port not found. Pod may still be starting.")
    return None


def main():
    if len(sys.argv) < 2:
        print("StreamDiffusion API - RunPod Management")
        print()
        print("Usage:")
        print("  python runpod_cmd.py gpus                      - List available GPUs")
        print("  python runpod_cmd.py pods                      - List running pods")
        print("  python runpod_cmd.py deploy [gpu] [WxH]        - Deploy new pod")
        print("  python runpod_cmd.py stop <pod_id>             - Stop pod")
        print("  python runpod_cmd.py terminate <pod_id>        - Terminate pod")
        print("  python runpod_cmd.py rtmp <pod_id>             - Get RTMP URL")
        print()
        print("Examples:")
        print("  python runpod_cmd.py deploy")
        print("  python runpod_cmd.py deploy 'NVIDIA H100 PCIe' 1024x576")
        print("  python runpod_cmd.py deploy 'NVIDIA H100 NVL' 1024x1024")
        print()
        print("Requires RUNPOD_API_KEY in .env file or environment")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "gpus":
        list_gpus()
    elif cmd == "pods":
        list_pods()
    elif cmd == "deploy":
        gpu = sys.argv[2] if len(sys.argv) > 2 else "NVIDIA H100 PCIe"
        resolution = sys.argv[3] if len(sys.argv) > 3 else "1024x576"
        deploy_pod(gpu, resolution)
    elif cmd == "stop":
        if len(sys.argv) < 3:
            print("Usage: python runpod_cmd.py stop <pod_id>")
            sys.exit(1)
        stop_pod(sys.argv[2])
    elif cmd == "terminate":
        if len(sys.argv) < 3:
            print("Usage: python runpod_cmd.py terminate <pod_id>")
            sys.exit(1)
        terminate_pod(sys.argv[2])
    elif cmd == "rtmp":
        if len(sys.argv) < 3:
            print("Usage: python runpod_cmd.py rtmp <pod_id>")
            sys.exit(1)
        get_rtmp_url(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
