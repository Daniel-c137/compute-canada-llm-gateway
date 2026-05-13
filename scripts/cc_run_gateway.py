#!/usr/bin/env python3
"""
On a VM (or laptop): open an SSH local forward to the compute node's vLLM port, then run
the FastAPI gateway on 0.0.0.0:proxy_port with token auth.

Only the gateway port should be exposed publicly; the forwarded port binds to 127.0.0.1.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import os
import signal
import socket
import stat
import subprocess
import time
from typing import Any

import uvicorn
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_control_socket(cli_value: str | None, cfg: dict[str, Any] | None = None) -> str | None:
    raw = cli_value or (cfg or {}).get("ssh_control_socket") or os.environ.get("CC_SSH_CONTROL_SOCKET")
    if not raw or not str(raw).strip():
        return None
    return os.path.abspath(os.path.expanduser(str(raw).strip()))


def validate_control_socket_path(path: str) -> None:
    if not os.path.exists(path):
        print(
            f"error: SSH control socket does not exist: {path}\n\n"
            "Start (or restart) the multiplex master, then retry. Example:\n"
            "  ssh -M -S <same-path> -o ServerAliveInterval=60 -fN <user>@<login-host>\n\n"
            "See README section «SSH and Duo (MFA)».\n",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        st_mode = os.stat(path).st_mode
    except OSError as e:
        print(f"error: cannot access SSH control socket {path}: {e}", file=sys.stderr)
        sys.exit(2)
    if not stat.S_ISSOCK(st_mode):
        print(
            f"error: SSH control socket path is not a Unix socket: {path}\n"
            "Use the same path you passed to `ssh -S` when starting the master.\n",
            file=sys.stderr,
        )
        sys.exit(2)


def require_control_socket(control_socket: str | None, *, allow_without: bool) -> str | None:
    if control_socket:
        validate_control_socket_path(control_socket)
        return control_socket
    if allow_without:
        return None
    print(
        "error: no SSH control socket configured.\n\n"
        "Alliance logins use Duo/MFA; this script runs non-interactive SSH and cannot answer prompts.\n"
        "Open a ControlMaster session first, then use one of:\n"
        "  --ssh-control-socket PATH\n"
        "  export CC_SSH_CONTROL_SOCKET=PATH\n\n"
        "If your login is key-only with no MFA, pass --allow-no-control-socket.\n"
        "See README «SSH and Duo (MFA)».\n",
        file=sys.stderr,
    )
    sys.exit(2)


def ssh_tunnel_cmd(
    login_host: str,
    user: str,
    identity: str | None,
    control_socket: str | None,
    local_forward_spec: str,
) -> list[str]:
    """ssh … -L spec -N user@host (ControlMaster -S when socket set)."""
    cmd = ["ssh"]
    if identity:
        cmd.extend(["-i", identity])
    if control_socket:
        cmd.extend(
            [
                "-S",
                control_socket,
                "-o",
                "ControlMaster=no",
                "-o",
                "BatchMode=yes",
            ]
        )
    else:
        cmd.extend(["-o", "BatchMode=yes"])
    cmd.extend(
        [
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            "-L",
            local_forward_spec,
            "-N",
            f"{user}@{login_host}",
        ]
    )
    return cmd


def run_ssh_capture(
    host: str, user: str, identity: str | None, control_socket: str | None, remote_cmd: str
) -> str:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if control_socket:
        cmd.extend(
            [
                "-S",
                control_socket,
                "-o",
                "ControlMaster=no",
                "-o",
                "BatchMode=yes",
            ]
        )
    else:
        cmd.extend(["-o", "BatchMode=yes"])
    if identity:
        cmd.extend(["-i", identity])
    cmd.extend([f"{user}@{host}", remote_cmd])
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        print(p.stderr or p.stdout, file=sys.stderr)
        if not control_socket and p.stderr and (
            "Permission denied" in p.stderr
            or "keyboard-interactive" in p.stderr
            or "multifactor" in p.stderr.lower()
            or "multifacteur" in p.stderr.lower()
        ):
            print(
                "\nDuo/MFA: start a ControlMaster SSH session first, then set CC_SSH_CONTROL_SOCKET "
                "or pass --ssh-control-socket (see README «SSH and Duo (MFA)»).\n",
                file=sys.stderr,
            )
        raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, p.stderr)
    return p.stdout


def fetch_squeue_rows(
    host: str, user: str, identity: str | None, control_socket: str | None
) -> list[dict[str, Any]]:
    fmt = "%i|%j|%N|%T"
    out = run_ssh_capture(
        host,
        user,
        identity,
        control_socket,
        f"squeue -u \"$USER\" -h -o '{fmt}' 2>/dev/null || squeue -u $(whoami) -h -o '{fmt}'",
    )
    rows: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        rows.append(
            {
                "job_id": parts[0].strip(),
                "name": parts[1].strip(),
                "nodelist": parts[2].strip(),
                "state": parts[3].strip(),
            }
        )
    return rows


def pick_job_interactive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        print("No jobs returned from squeue.", file=sys.stderr)
        sys.exit(1)
    print("Select a SLURM job:")
    for i, r in enumerate(rows):
        print(
            f"  [{i}] id={r['job_id']} name={r['name']} nodes={r['nodelist']} state={r['state']}"
        )
    choice = input("Enter number [0]: ").strip() or "0"
    idx = int(choice)
    return rows[idx]


def wait_for_tcp(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as e:
            last_err = e
            time.sleep(0.2)
    raise TimeoutError(f"Could not connect to {host}:{port} within {timeout_s}s: {last_err}")


def parse_compute_host(nodelist: str) -> str:
    """Take first hostname from SLURM nodelist (best-effort; may need manual override)."""
    n = nodelist.strip()
    if not n:
        return n
    if "[" in n:
        prefix, rest = n.split("[", 1)
        inner = rest.split("]", 1)[0]
        first = inner.split(",")[0].split("-")[0]
        return f"{prefix}{first}"
    return n.split(",")[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="SSH tunnel + LLM gateway on this machine.")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML with ssh_host, ssh_user, ssh_control_socket (same keys as cc_submit_vllm). CLI overrides YAML.",
    )
    ap.add_argument("--login-host", default=None, help="Login node hostname (or set in --config as ssh_host)")
    ap.add_argument("--user", default=None, help="Cluster username (or set in --config as ssh_user)")
    ap.add_argument("--compute-host", default=None, help="Compute hostname reachable from login (for -L)")
    ap.add_argument("--server-port", type=int, required=True, help="vLLM port on compute node")
    ap.add_argument(
        "--forwarded-port",
        type=int,
        required=True,
        help="Local bind on 127.0.0.1 on this machine (not public)",
    )
    ap.add_argument(
        "--proxy-port",
        type=int,
        required=True,
        help="Gateway listen port on 0.0.0.0 (intentionally public on the VM)",
    )
    ap.add_argument("--ssh-identity", default=None)
    ap.add_argument(
        "--ssh-control-socket",
        default=None,
        help="SSH multiplex socket (same as cc_submit_vllm / CC_SSH_CONTROL_SOCKET). Required unless --allow-no-control-socket.",
    )
    ap.add_argument(
        "--allow-no-control-socket",
        action="store_true",
        help="Allow tunnel SSH without a ControlMaster socket (only if login does not require MFA).",
    )
    ap.add_argument(
        "--pick-job",
        action="store_true",
        help="SSH to login, list your squeue jobs, and pick one to derive compute host",
    )
    ap.add_argument(
        "--wait-ready",
        type=float,
        default=300.0,
        help="Seconds to wait for local forwarded TCP port to accept connections",
    )
    args = ap.parse_args()

    cfg: dict[str, Any] = {}
    if args.config is not None:
        if not args.config.exists():
            print(f"error: config not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        cfg = load_config(args.config)

    login_host = args.login_host or cfg.get("ssh_host")
    user = args.user or cfg.get("ssh_user")
    if not login_host or not user:
        print(
            "error: need --login-host and --user (or both ssh_host and ssh_user in --config YAML).\n",
            file=sys.stderr,
        )
        sys.exit(1)

    control_socket = resolve_control_socket(args.ssh_control_socket, cfg)
    control_socket = require_control_socket(
        control_socket,
        allow_without=args.allow_no_control_socket,
    )

    compute_host = args.compute_host
    if args.pick_job:
        rows = fetch_squeue_rows(login_host, user, args.ssh_identity, control_socket)
        running = [r for r in rows if r["state"] == "RUNNING"]
        use_rows = running if running else rows
        job = pick_job_interactive(use_rows)
        compute_host = parse_compute_host(job["nodelist"])
        print(f"Using compute host: {compute_host} (from job {job['job_id']})", file=sys.stderr)
    if not compute_host:
        print("Provide --compute-host or use --pick-job with a RUNNING job.", file=sys.stderr)
        sys.exit(1)

    local_spec = f"127.0.0.1:{args.forwarded_port}:{compute_host}:{args.server_port}"
    ssh_cmd = ssh_tunnel_cmd(
        login_host, user, args.ssh_identity, control_socket, local_spec
    )

    ssh_proc = subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def cleanup(*_: Any) -> None:
        if ssh_proc.poll() is None:
            ssh_proc.terminate()
            try:
                ssh_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ssh_proc.kill()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    time.sleep(1.0)
    if ssh_proc.poll() is not None:
        print(
            "SSH tunnel exited early (check host, keys, ExitOnForwardFailure, compute hostname).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        wait_for_tcp("127.0.0.1", args.forwarded_port, args.wait_ready)
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        cleanup()
        sys.exit(1)

    os.environ["UPSTREAM_BASE_URL"] = f"http://127.0.0.1:{args.forwarded_port}"
    if not os.environ.get("GATEWAY_TOKEN"):
        print(
            "GATEWAY_TOKEN not set; the gateway will generate one and print it on startup.",
            file=sys.stderr,
        )

    try:
        uvicorn.run(
            "cc_llm_gateway.main:app",
            host="0.0.0.0",
            port=args.proxy_port,
            log_level="info",
            timeout_keep_alive=120,
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
