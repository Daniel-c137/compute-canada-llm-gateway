"""Alliance / Compute Canada filesystem paths for shared Hugging Face weights."""

from __future__ import annotations

from typing import Any

DEFAULT_HF_WEIGHTS_PARENT = "cc-llm-hf-weights"


def _sanitize_path_component(value: str) -> str:
    return str(value).strip().strip("/").replace("..", "")


def hf_hub_weights_parent_name(cfg: dict[str, Any]) -> str | None:
    """Subdirectory name under the project user folder; ``None`` disables shared weights."""
    if "hf_hub_weights_parent" not in cfg:
        return DEFAULT_HF_WEIGHTS_PARENT
    raw = cfg.get("hf_hub_weights_parent")
    if raw is False or raw is None:
        return None
    name = _sanitize_path_component(str(raw))
    return name or None


def project_user_dir(slurm_account: str, ssh_user: str) -> str:
    """``/project/<slurm_account>/<ssh_user>`` on the cluster."""
    account = _sanitize_path_component(slurm_account)
    user = _sanitize_path_component(ssh_user)
    if not account or not user:
        raise ValueError("slurm_account and ssh_user must be non-empty for project paths")
    return f"/project/{account}/{user}"


def hf_shared_weights_dir(cfg: dict[str, Any], ssh_user: str, model: str) -> str | None:
    """
    Absolute path for a Hugging Face repo id, e.g.
    ``/project/rrg-tayebi/msina/cc-llm-hf-weights/org--model``.
    """
    parent = hf_hub_weights_parent_name(cfg)
    if parent is None:
        return None
    m = model.strip()
    if not m or m.startswith(("/", "~", "$")):
        return None
    if "/" not in m:
        return None
    account = cfg.get("slurm_account")
    if not isinstance(account, str) or not account.strip():
        return None
    if not ssh_user or not str(ssh_user).strip():
        return None
    slug = m.replace("/", "--").replace("..", "_")
    base = project_user_dir(account, ssh_user)
    return f"{base}/{parent}/{slug}"
