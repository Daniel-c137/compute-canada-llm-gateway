#!/usr/bin/env python3
"""
Submit a vLLM OpenAI API server job on Alliance / Compute Canada via SSH to the login node.

Copies requirements and generated sbatch script, bootstraps a venv on the login node,
optionally pre-downloads Hugging Face weights, runs sbatch, and prints the job ID.

Mac and Linux compatible (uses stdlib + project dependencies).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import itertools
import json
import os
import re
import shlex
import stat
import subprocess
import threading
import uuid
from typing import Any

import yaml
from cc_llm_gateway.alliance_paths import hf_shared_weights_dir
from jinja2 import Environment, FileSystemLoader


DEFAULT_CONFIG = REPO_ROOT / "config" / "nodes.yaml"
TEMPLATE_DIR = REPO_ROOT / "templates"
REQ_LOGIN = REPO_ROOT / "requirements-login.txt"
REQ_VLLM = REPO_ROOT / "requirements-vllm.txt"
# Local metadata distribution so pip does not fetch Alliance `opencv-noinstall` for vLLM.
OPENCV_HEADLESS_STUB = REPO_ROOT / "packaging" / "opencv_python_headless_stub"


class TtyWaitSpinner:
    """Rotating indicator on stderr while a blocking step (SSH) runs."""

    def __init__(self, message: str, *, interval_s: float = 0.12) -> None:
        self._message = message.rstrip()
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> TtyWaitSpinner:
        self._thread = threading.Thread(target=self._spin, name="cc-submit-wait", daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        frames = itertools.cycle("|/-\\")
        prefix = f"{self._message} "
        while not self._stop.wait(self._interval_s):
            print(f"\r{prefix}{next(frames)} ", end="", file=sys.stderr, flush=True)

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        pad = max(len(self._message) + 4, 12)
        print("\r" + " " * pad + "\r", end="", file=sys.stderr, flush=True)


def resolve_control_socket(cli_value: str | None, cfg: dict[str, Any] | None = None) -> str | None:
    """Path to an existing SSH ControlMaster socket (Duo/MFA: authenticate once, then reuse)."""
    raw = cli_value or (cfg or {}).get("ssh_control_socket") or os.environ.get("CC_SSH_CONTROL_SOCKET")
    if not raw or not str(raw).strip():
        return None
    return os.path.abspath(os.path.expanduser(str(raw).strip()))


def validate_control_socket_path(path: str) -> None:
    """Exit with a clear message if the socket path is unusable."""
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
    """Ensure a valid socket is configured unless allow_without is True."""
    if control_socket:
        validate_control_socket_path(control_socket)
        return control_socket
    if allow_without:
        return None
    print(
        "error: no SSH control socket configured.\n\n"
        "Alliance logins use Duo/MFA; this script runs non-interactive ssh/scp and cannot answer prompts.\n"
        "Open a ControlMaster session first, then use one of:\n"
        "  --ssh-control-socket PATH\n"
        "  export CC_SSH_CONTROL_SOCKET=PATH\n"
        "  ssh_control_socket: PATH   # in config/nodes.yaml (this submit script only)\n\n"
        "If your login is key-only with no MFA, pass --allow-no-control-socket (not typical on Alliance).\n"
        "See README «SSH and Duo (MFA)».\n",
        file=sys.stderr,
    )
    sys.exit(2)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(
            f"Config not found: {path}\nCopy {REPO_ROOT / 'config' / 'nodes.example.yaml'} to {path} and edit.",
            file=sys.stderr,
        )
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ssh_base(host: str, user: str, identity: str | None, control_socket: str | None) -> list[str]:
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
    cmd.append(f"{user}@{host}")
    return cmd


def run_remote(
    host: str,
    user: str,
    identity: str | None,
    control_socket: str | None,
    remote_bash_script: str,
) -> str:
    """Execute bash -lc on remote; returns stdout."""
    base = ssh_base(host, user, identity, control_socket)
    cmd = base + ["bash", "-lc", remote_bash_script]
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
                "\nThis cluster likely requires Duo/MFA. Scripted SSH cannot answer prompts.\n"
                "Open a ControlMaster connection in another terminal (Duo once), then rerun with:\n"
                "  --ssh-control-socket ~/.ssh/cm-rorqual   (or set CC_SSH_CONTROL_SOCKET)\n"
                "See README: «SSH and Duo (MFA)».\n",
                file=sys.stderr,
            )
        raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, p.stderr)
    return p.stdout


def scp_to_remote(
    host: str,
    user: str,
    identity: str | None,
    control_socket: str | None,
    local_path: Path,
    remote_path: str,
) -> None:
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
    if control_socket:
        cmd.extend(
            [
                "-o",
                f"ControlPath={control_socket}",
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
    cmd.extend([str(local_path), f"{user}@{host}:{remote_path}"])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def scp_dir_to_remote(
    host: str,
    user: str,
    identity: str | None,
    control_socket: str | None,
    local_dir: Path,
    remote_parent_rel: str,
) -> None:
    """Copy ``local_dir`` to ``remote_parent_rel/<basename(local_dir)>/`` (parent is home-relative)."""
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
    if control_socket:
        cmd.extend(
            [
                "-o",
                f"ControlPath={control_socket}",
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
    parent = remote_parent_rel.rstrip("/") + "/"
    cmd.extend(["-r", str(local_dir.resolve()), f"{user}@{host}:{parent}"])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def pick_preset(presets: list[dict[str, Any]], preset_id: str | None) -> dict[str, Any]:
    if preset_id:
        for p in presets:
            if p.get("id") == preset_id:
                return p
        print(f"Unknown preset id: {preset_id}", file=sys.stderr)
        sys.exit(1)
    print("Select a node preset:")
    for i, p in enumerate(presets):
        label = p.get("label", p.get("id", str(i)))
        print(f"  [{i}] {p.get('id')} — {label}")
    choice = input("Enter number [0]: ").strip() or "0"
    idx = int(choice)
    return presets[idx]


def prompt(default: str, label: str) -> str:
    v = input(f"{label} [{default}]: ").strip()
    return v or default


def merge_str_list(cfg: dict[str, Any], preset: dict[str, Any], key: str) -> list[str]:
    raw = preset.get(key) if preset.get(key) is not None else cfg.get(key)
    if not raw:
        return []
    return [str(x) for x in raw if str(x).strip()]


def sh_module_load(spec: str) -> str:
    """One `module load` shell fragment. Multiple names separated by spaces share one `module load` invocation."""
    parts = spec.strip().split()
    if not parts:
        return ""
    return "module load " + " ".join(shlex.quote(p) for p in parts)


def append_login_modules(parts: list[str], login_modules: list[str]) -> None:
    for m in login_modules:
        line = sh_module_load(m)
        if line:
            parts.append(line)


def opencv_module_specs(login_modules: list[str]) -> list[str]:
    """Module lines that reference OpenCV (Alliance cluster module, not pip)."""
    out: list[str] = []
    for m in login_modules:
        spec = str(m).strip()
        if spec and "opencv" in spec.lower():
            out.append(spec)
    return out


def login_modules_use_alliance_opencv(login_modules: list[str]) -> bool:
    return bool(opencv_module_specs(login_modules))


# Quoted venv interpreter for remote `bash -lc` (cwd is the run directory).
VENV_PY_SH = '"${PWD}/venv/bin/python"'


def remote_venv_pip(pip_cli_args: str, *, alliance_opencv: bool) -> str:
    """Run pip inside the staged venv. Alliance OpenCV wiki: load opencv, then *activate* venv, then pip."""
    if alliance_opencv:
        return f"source venv/bin/activate && pip {pip_cli_args}"
    return f"{VENV_PY_SH} -m pip {pip_cli_args}"


def remote_speculators_preflight(cd_home_then_run: str, hf_weights_dir: str | None) -> str:
    """SSH fragment: fail before sbatch if config.json uses speculators with a Hub-only verifier (offline compute)."""
    if hf_weights_dir:
        path_init = f"cfg_path = pathlib.Path({json.dumps(hf_weights_dir)}) / 'config.json'"
    else:
        path_init = "cfg_path = pathlib.Path('hf_model_cache') / 'config.json'"
    py = f"""import json
import pathlib
import re
import sys

HF_REPO = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
{path_init}
if not cfg_path.is_file():
    sys.exit(0)
try:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
    print(f"warning: could not parse {{cfg_path}}: {{e}}", file=sys.stderr)
    sys.exit(0)
sc = cfg.get("speculators_config")
if not sc:
    sys.exit(0)
ver = (sc.get("verifier") or {{}}).get("name_or_path", "")
if not isinstance(ver, str) or not ver:
    sys.exit(0)
if HF_REPO.fullmatch(ver):
    print(
        "error: " + str(cfg_path) + " has speculators_config with Hugging Face verifier "
        + repr(ver) + ". vLLM loads that verifier on compute; with HF_HUB_OFFLINE=1 the job will fail.\\n"
        "Remove speculators_config from config.json (if you do not need speculators), or snapshot_download "
        "the verifier to a local directory and set verifier.name_or_path to that path.\\n"
        "See README «vLLM logs a different Hugging Face repo».\\n",
        file=sys.stderr,
    )
    sys.exit(4)
"""
    return f"{cd_home_then_run} && python3 <<'PY'\n{py}\nPY"


def resolve_hf_token(cfg: dict[str, Any]) -> str | None:
    """Prefer env (not persisted in repo); optional hf_token in local config only."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok and tok.strip():
        return tok.strip()
    cfg_tok = cfg.get("hf_token")
    if isinstance(cfg_tok, str) and cfg_tok.strip():
        return cfg_tok.strip()
    return None


def render_sbatch(
    cfg: dict[str, Any],
    preset: dict[str, Any],
    walltime: str,
    server_port: int,
    model: str,
    hf_weights_dir: str | None = None,
) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)
    tpl = env.get_template("vllm_sbatch.sh.j2")
    extra = preset.get("extra_sbatch") or []
    extra_compute_modules = merge_str_list(cfg, preset, "extra_compute_modules")
    extra_vllm_cli = merge_str_list(cfg, preset, "extra_vllm_cli")
    return tpl.render(
        slurm_account=cfg["slurm_account"],
        partition=preset["partition"],
        walltime=walltime,
        cpus_per_task=int(preset.get("cpus_per_task", 4)),
        mem=str(preset.get("mem", "32G")),
        gpus_per_node=str(preset.get("gpus_per_node", "1")),
        extra_sbatch=extra,
        extra_compute_modules=extra_compute_modules,
        extra_vllm_cli=extra_vllm_cli,
        python_module=cfg.get("python_module", "python/3.11"),
        server_port=server_port,
        model=model,
        hf_weights_dir=hf_weights_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage vLLM on CC login node and sbatch.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--preset", type=str, default=None, help="Preset id from config (skip menu when set)")
    ap.add_argument("--port", type=int, default=8000, help="vLLM server port on compute node")
    ap.add_argument("--model", type=str, default=None, help="HF model id or on-cluster path")
    ap.add_argument("--walltime", type=str, default="4:00:00")
    ap.add_argument("--ssh-user", type=str, default=None)
    ap.add_argument("--ssh-host", type=str, default=None)
    ap.add_argument("--ssh-identity", type=str, default=None)
    ap.add_argument(
        "--ssh-control-socket",
        type=str,
        default=None,
        help="SSH multiplex socket path (after: ssh -M -S <path> -fN user@host). Required unless --allow-no-control-socket. "
        "Or set CC_SSH_CONTROL_SOCKET or config key ssh_control_socket.",
    )
    ap.add_argument(
        "--allow-no-control-socket",
        action="store_true",
        help="Allow ssh/scp without a ControlMaster socket (only if login does not require MFA).",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip huggingface_hub snapshot_download. Weights must already exist (per-run hf_model_cache/ or shared "
        "/project/<slurm_account>/<ssh_user>/<hf_hub_weights_parent>/…).",
    )
    ap.add_argument(
        "--ignore-hf-download-errors",
        action="store_true",
        help="If snapshot_download fails, still run sbatch (default is to exit before sbatch).",
    )
    ap.add_argument(
        "--fresh-venv",
        action="store_true",
        help="Remove existing venv in the run directory before recreating (use after a failed pip install).",
    )
    ap.add_argument(
        "--non-interactive",
        action="store_true",
        help="No prompts; requires --preset and --model",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    control_socket = resolve_control_socket(args.ssh_control_socket, cfg)
    control_socket = require_control_socket(
        control_socket,
        allow_without=args.allow_no_control_socket,
    )
    presets = cfg.get("presets") or []
    if not presets:
        print("No presets in config.", file=sys.stderr)
        sys.exit(1)

    if args.non_interactive:
        if not args.preset:
            print("--preset is required with --non-interactive", file=sys.stderr)
            sys.exit(1)
        preset = pick_preset(presets, args.preset)
    else:
        preset = pick_preset(presets, args.preset)

    host = args.ssh_host or cfg.get("ssh_host")
    user = args.ssh_user or cfg.get("ssh_user")
    if not host or not user:
        print("ssh_host and ssh_user must be set in config or passed via flags.", file=sys.stderr)
        sys.exit(1)

    if not args.non_interactive:
        args.port = int(prompt(str(args.port), "Server port on compute node"))
        model_default = args.model or "meta-llama/Llama-3.2-1B-Instruct"
        args.model = prompt(model_default, "Model (HF id or path on cluster)")
        args.walltime = prompt(args.walltime, "Walltime (SLURM format, e.g. 4:00:00)")
    else:
        if not args.model:
            print("--model is required with --non-interactive", file=sys.stderr)
            sys.exit(1)

    run_id = uuid.uuid4().hex[:10]
    subdir = cfg.get("remote_subdir", "cc-llm-runs").strip("/").strip()
    # Paths relative to home: OpenSSH starts in $HOME, but `bash -lc` can alter env;
    # `cd ~` then relative dirs avoids `mkdir: missing operand` from empty "$HOME/...".
    run_rel = f"{subdir}/{run_id}"
    run_rel_q = shlex.quote(run_rel)
    cd_home_then_run = f"cd ~ && cd {run_rel_q}"

    identity = args.ssh_identity

    mkdir_script = f"cd ~ && mkdir -p {run_rel_q}"
    run_remote(host, user, identity, control_socket, mkdir_script)

    py_mod = shlex.quote(cfg.get("python_module", "python/3.11"))
    login_modules = merge_str_list(cfg, preset, "extra_login_modules")
    alliance_opencv = login_modules_use_alliance_opencv(login_modules)
    opencv_reload = opencv_module_specs(login_modules)
    if alliance_opencv and not OPENCV_HEADLESS_STUB.is_dir():
        print(
            f"error: Alliance OpenCV workaround requires directory {OPENCV_HEADLESS_STUB} "
            "(bundled stub for pip metadata). Clone or copy the full repo and retry.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    hf_weights_dir = hf_shared_weights_dir(cfg, user, args.model)
    if hf_weights_dir:
        print(
            f"HF weights reuse: snapshot and vLLM will use {hf_weights_dir} "
            "(set hf_hub_weights_parent to change the subdirectory name).\n",
            file=sys.stderr,
        )
        weights_parent = str(Path(hf_weights_dir).parent)
        run_remote(
            host,
            user,
            identity,
            control_socket,
            f"mkdir -p {shlex.quote(weights_parent)}",
        )

    sbatch_body = render_sbatch(
        cfg, preset, args.walltime, args.port, args.model, hf_weights_dir=hf_weights_dir
    )
    local_sh = REPO_ROOT / f".vllm_job_{run_id}.sh"
    local_sh.write_text(sbatch_body, encoding="utf-8")
    try:
        with TtyWaitSpinner("Uploading files to the login node — please wait"):
            scp_to_remote(host, user, identity, control_socket, REQ_LOGIN, f"{run_rel}/requirements-login.txt")
            scp_to_remote(host, user, identity, control_socket, REQ_VLLM, f"{run_rel}/requirements-vllm.txt")
            scp_to_remote(host, user, identity, control_socket, local_sh, f"{run_rel}/vllm_job.sh")
            if alliance_opencv:
                scp_dir_to_remote(host, user, identity, control_socket, OPENCV_HEADLESS_STUB, run_rel)
    finally:
        local_sh.unlink(missing_ok=True)

    ssp = " --system-site-packages" if alliance_opencv else ""
    venv_block = (
        f"rm -rf venv && python3 -m venv{ssp} venv"
        if args.fresh_venv
        else f"test -d venv || python3 -m venv{ssp} venv"
    )
    setup_parts: list[str] = [cd_home_then_run, f"module load {py_mod}"]
    append_login_modules(setup_parts, login_modules)
    setup_parts.append(venv_block)
    pip_up = remote_venv_pip("install --no-index --upgrade pip", alliance_opencv=alliance_opencv)
    pip_login = remote_venv_pip(
        "install --no-index -r requirements-login.txt", alliance_opencv=alliance_opencv
    )
    vllm_pip = remote_venv_pip(
        "install --no-index --no-build-isolation -r requirements-vllm.txt"
        if alliance_opencv
        else "install --no-index -r requirements-vllm.txt",
        alliance_opencv=alliance_opencv,
    )
    setup_parts.append(pip_up)
    setup_parts.append(pip_login)
    # Re-apply OpenCV after other pip work in case MODULEPATH / env drifted.
    append_login_modules(setup_parts, opencv_reload)
    if alliance_opencv:
        stub_dir = OPENCV_HEADLESS_STUB.name
        setup_parts.append(
            remote_venv_pip(
                f"install --no-index --no-build-isolation ./{stub_dir}/",
                alliance_opencv=alliance_opencv,
            )
        )
    setup_parts.append(vllm_pip)
    remote_setup = " && ".join(setup_parts)
    with TtyWaitSpinner(
        "Remote setup: modules, venv, and pip install on the login node — can take several minutes; keep this terminal open"
    ):
        run_remote(host, user, identity, control_socket, remote_setup)

    if not args.skip_download:
        hf_tok = resolve_hf_token(cfg)
        if hf_tok is None and args.model.startswith(("meta-llama/", "meta-llama", "facebook/")):
            print(
                "warning: gated HF models usually need HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in your environment "
                "before running this script; download will likely fail without it.\n",
                file=sys.stderr,
            )
        model_env = shlex.quote(args.model)
        hf_dl = f"""{VENV_PY_SH} <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
repo = os.environ["HF_MODEL_ID"]
dest_dir = os.environ.get("HF_WEIGHTS_DIR", "").strip()
if dest_dir:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(dest))
else:
    snapshot_download(repo_id=repo, local_dir="hf_model_cache")
PY"""
        dl_parts: list[str] = [cd_home_then_run, f"module load {py_mod}"]
        append_login_modules(dl_parts, login_modules)
        if hf_weights_dir:
            dl_parts.append(f"export HF_WEIGHTS_DIR={shlex.quote(hf_weights_dir)}")
        if hf_tok is not None:
            dl_parts.append(f"export HF_TOKEN={shlex.quote(hf_tok)}")
        dl_parts.extend(
            [
                "export HF_HOME=${PWD}/hf_home",
                "mkdir -p ${HF_HOME}",
                f"export HF_MODEL_ID={model_env}",
                hf_dl,
            ]
        )
        dl = " && ".join(dl_parts)
        try:
            with TtyWaitSpinner(
                "Downloading model weights on the login node (Hugging Face) — large repos take a while; do not interrupt"
            ):
                run_remote(host, user, identity, control_socket, dl)
        except subprocess.CalledProcessError:
            if args.ignore_hf_download_errors:
                print(
                    "Warning: Hugging Face snapshot_download failed; continuing because "
                    "--ignore-hf-download-errors was set. Weights must exist at the model path used by the job.\n",
                    file=sys.stderr,
                )
            else:
                print(
                    "error: Hugging Face snapshot_download failed (see stderr above). "
                    "For gated models export HF_TOKEN from https://huggingface.co/settings/tokens "
                    "before re-running, or use --skip-download if weights already exist at the job model path, "
                    "or pass --ignore-hf-download-errors to submit anyway.\n",
                    file=sys.stderr,
                )
                sys.exit(3)

    try:
        run_remote(
            host,
            user,
            identity,
            control_socket,
            remote_speculators_preflight(cd_home_then_run, hf_weights_dir),
        )
    except subprocess.CalledProcessError as e:
        if e.returncode == 4:
            sys.exit(4)
        raise

    submit = " && ".join(
        [
            cd_home_then_run,
            "sbatch vllm_job.sh",
        ]
    )
    out = run_remote(host, user, identity, control_socket, submit).strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        print(out, file=sys.stderr)
        print("Could not parse job id from sbatch output.", file=sys.stderr)
        sys.exit(1)
    job_id = m.group(1)
    print(job_id)
    print(f"Remote run directory: ~/{run_rel}", file=sys.stderr)
    print(f"Check logs: ssh {user}@{host} 'ls -la ~/{run_rel}'", file=sys.stderr)


if __name__ == "__main__":
    main()
