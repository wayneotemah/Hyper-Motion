#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi

  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      version="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "${version}" == "3.10" || "${version}" == "3.11" || "${version}" == "3.12" ]]; then
        echo "${candidate}"
        return
      fi
    fi
  done

  echo ""
}

PYTHON_CANDIDATE="$(find_python)"
if [[ -z "${PYTHON_CANDIDATE}" ]]; then
  echo "Python 3.10, 3.11, or 3.12 is required. Install one and rerun bootstrap." >&2
  exit 1
fi

echo "Using Python interpreter: ${PYTHON_CANDIDATE}"
"${PYTHON_CANDIDATE}" -m venv "${ROOT_DIR}/.venv"

VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
VENV_PIP="${ROOT_DIR}/.venv/bin/pip"

"${VENV_PIP}" install --upgrade pip setuptools wheel
CORE_REQUIREMENTS="$(mktemp)"
grep -v '^mediapipe' "${ROOT_DIR}/requirements-mac.txt" > "${CORE_REQUIREMENTS}"
"${VENV_PIP}" install -r "${CORE_REQUIREMENTS}"
rm -f "${CORE_REQUIREMENTS}"

MEDIAPIPE_STATUS="not_requested"
if grep -q '^mediapipe' "${ROOT_DIR}/requirements-mac.txt"; then
  if "${VENV_PIP}" install "mediapipe>=0.10.14"; then
    MEDIAPIPE_STATUS="installed"
  else
    MEDIAPIPE_STATUS="optional_install_failed"
  fi
fi

FFMPEG_STATUS="missing"
if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_STATUS="available"
fi

"${VENV_PYTHON}" - <<'PY'
import importlib

checks = {}
for module_name in ("cv2", "torch", "yaml", "PIL", "mediapipe"):
    try:
        importlib.import_module(module_name)
        checks[module_name] = "ok"
    except Exception as exc:  # pragma: no cover - bootstrap probe
        checks[module_name] = f"error: {exc}"

try:
    import torch
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
except Exception:
    mps = False

print("Import summary:")
for name, status in checks.items():
    print(f"  {name}: {status}")
print(f"  torch.backends.mps.is_available(): {mps}")
PY

echo "mediapipe: ${MEDIAPIPE_STATUS}"
echo "ffmpeg: ${FFMPEG_STATUS}"
"${VENV_PYTHON}" "${ROOT_DIR}/scripts/verify_env.py"
