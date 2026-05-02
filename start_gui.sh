#!/usr/bin/env bash
# Launcher for the glayout pcell menu (PySide6 GUI).
#
# It picks an interpreter in this order:
#   1. $GLAYOUT_PYTHON if set
#   2. the conda env named "gLayout" (the env shipped with this repo)
#   3. plain `python3` from PATH
#
# All catalogue and parameter info is introspected at runtime by
# `menu/start_gui.py` -- you should not need to edit anything in `menu/`
# when cells or primitives change.
#
# Required python deps (in addition to glayout's): PySide6, gdstk, matplotlib.
# In the bundled gLayout conda env they are already installed; otherwise:
#     pip install PySide6

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

# The mapped PDK modules dereference PDK_ROOT at import time. The menu only
# generates GDS + spice (no DRC/LVS), so a placeholder is enough to let the
# PDK objects construct. Override with your real PDK_ROOT to enable verification.
if [[ -z "${PDK_ROOT:-}" ]]; then
    export PDK_ROOT="${PDK_ROOT_FALLBACK:-/tmp/glayout_pdk_root_stub}"
    mkdir -p "${PDK_ROOT}"
fi

pick_python() {
    if [[ -n "${GLAYOUT_PYTHON:-}" && -x "${GLAYOUT_PYTHON}" ]]; then
        echo "${GLAYOUT_PYTHON}"
        return
    fi
    # Try common conda layouts.
    for cand in \
        "${HOME}/miniconda3/envs/gLayout/bin/python" \
        "${HOME}/anaconda3/envs/gLayout/bin/python" \
        "${HOME}/.conda/envs/gLayout/bin/python" \
        "/opt/conda/envs/gLayout/bin/python"; do
        if [[ -x "${cand}" ]]; then
            echo "${cand}"
            return
        fi
    done
    if command -v conda >/dev/null 2>&1; then
        local prefix
        prefix="$(conda env list 2>/dev/null | awk '$1=="gLayout"{print $NF}' | head -n1 || true)"
        if [[ -n "${prefix}" && -x "${prefix}/bin/python" ]]; then
            echo "${prefix}/bin/python"
            return
        fi
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    return 1
}

PY="$(pick_python)" || {
    echo "error: could not find a python interpreter (set GLAYOUT_PYTHON to override)" >&2
    exit 1
}

mkdir -p "${ROOT}/out"
exec "${PY}" "${HERE}/start_gui.py" "$@"
