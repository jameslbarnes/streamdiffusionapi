#!/usr/bin/env python3
"""SSH helper for RunPod pods - always uses -tt for proper PTY allocation."""

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

def get_pod_ssh_host():
    """Get SSH host from runpod_cmd.py pods output."""
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
    except urllib.error.HTTPError as e:
        print(f"API Error: {e.code} - check RUNPOD_API_KEY", file=sys.stderr)
        sys.exit(1)

    pods = result.get('data', {}).get('myself', {}).get('pods', [])
    running = [p for p in pods if p.get('desiredStatus') == 'RUNNING']

    if not running:
        print("No running pods found")
        sys.exit(1)

    pod = running[0]
    host_id = pod['machine']['podHostId']
    return f"{host_id}@ssh.runpod.io"


def main():
    if len(sys.argv) < 2:
        print("Usage: python pod_ssh.py <command>")
        print("       python pod_ssh.py pull      - git pull latest code")
        print("       python pod_ssh.py restart   - restart the API")
        print("       python pod_ssh.py logs      - show API logs")
        print("       python pod_ssh.py status    - show git status")
        print("       python pod_ssh.py shell     - interactive shell")
        print("       python pod_ssh.py <cmd>     - run arbitrary command")
        sys.exit(1)

    ssh_host = get_pod_ssh_host()
    cmd = sys.argv[1]

    # Predefined shortcuts
    commands = {
        "pull": "cd /root/streamdiffusionapi && git pull",
        "restart": "cd /root/streamdiffusionapi && pkill -f 'python -m src.api.app'; sleep 2; nohup python -m src.api.app > /tmp/api.log 2>&1 & echo 'API restarted'",
        "logs": "tail -100 /tmp/api.log",
        "status": "cd /root/streamdiffusionapi && git log --oneline -1 && echo '---' && git status -s",
        "shell": None,  # Special case for interactive shell
    }

    if cmd == "shell":
        # Interactive shell
        subprocess.run(["ssh", "-tt", ssh_host])
    elif cmd in commands:
        remote_cmd = commands[cmd]
        run_remote(ssh_host, remote_cmd)
    else:
        # Arbitrary command - join all args
        remote_cmd = " ".join(sys.argv[1:])
        run_remote(ssh_host, remote_cmd)


def run_remote(ssh_host, remote_cmd, timeout_sec=60):
    """Run a command on the remote pod via SSH."""
    import time
    import threading
    import queue

    # Build command that runs and then exits
    full_cmd = f"{remote_cmd}\nexit\n"

    try:
        proc = subprocess.Popen(
            ["ssh", "-tt", "-o", "ServerAliveInterval=30", ssh_host],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0
        )
    except Exception as e:
        print(f"Failed to start SSH: {e}", file=sys.stderr)
        return

    output_lines = []
    done = threading.Event()

    def read_output():
        try:
            while not done.is_set():
                line = proc.stdout.readline()
                if not line:
                    break
                output_lines.append(line.decode('utf-8', errors='replace'))
        except Exception:
            pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    # Wait for banner
    time.sleep(2)

    # Send command
    try:
        if proc.stdin:
            proc.stdin.write(full_cmd.encode())
            proc.stdin.flush()
    except (OSError, BrokenPipeError) as e:
        print(f"Failed to send command: {e}", file=sys.stderr)
        done.set()
        proc.kill()
        return

    # Wait for completion with timeout
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    done.set()
    reader.join(timeout=2)

    # Filter and print output
    for line in output_lines:
        line = line.rstrip()
        # Skip RunPod banner and prompt lines
        if 'RUNPOD.IO' in line or 'Enjoy your Pod' in line:
            continue
        if line.startswith('root@') or '[?2004' in line:
            continue
        if 'WARNING' in line or 'vulnerable' in line or 'upgraded' in line:
            continue
        if 'Connection to' in line and 'closed' in line:
            continue
        if line.strip():
            print(line)


if __name__ == "__main__":
    main()
