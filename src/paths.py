from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


_ENV_LOADED = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv(dotenv_path: Optional[Path] = None) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    dotenv = Path(dotenv_path or project_root() / ".env")
    if not dotenv.exists():
        _ENV_LOADED = True
        return

    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

    _ENV_LOADED = True


def resolve_path(path_value: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    if not path_value:
        return None

    expanded = Path(os.path.expanduser(os.path.expandvars(path_value)))
    if expanded.is_absolute():
        return expanded
    return (base_dir or project_root()) / expanded


def resolve_storage_path(configured_value: str, env_var: str, default_relative: str) -> Path:
    load_dotenv()
    env_override = os.environ.get(env_var)
    if env_override:
        return resolve_path(env_override, project_root())  # type: ignore[return-value]
    resolved = resolve_path(configured_value or default_relative, project_root())
    return resolved or project_root() / default_relative


def storage_roots(configured: Optional[Dict[str, str]] = None) -> Dict[str, Path]:
    cfg = configured or {}
    return {
        "model_root": resolve_storage_path(cfg.get("model_root", "assets/model_store"), "HYPERMOTION_MODEL_ROOT", "assets/model_store"),
        "xpose_root": resolve_storage_path(cfg.get("xpose_root", "assets/xpose"), "HYPERMOTION_XPOSE_ROOT", "assets/xpose"),
        "outputs_root": resolve_storage_path(cfg.get("outputs_root", "outputs"), "HYPERMOTION_OUTPUTS_ROOT", "outputs"),
        "logs_root": resolve_storage_path(cfg.get("logs_root", "logs"), "HYPERMOTION_LOGS_ROOT", "logs"),
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root()))
    except ValueError:
        return str(path.resolve())
