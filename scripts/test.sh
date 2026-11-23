#!/usr/bin/env bash
# 运行测试（标准库 unittest，无需 pytest）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "未找到 ${PY}，请先 conda activate g1wbc（或设置 PYTHON 指向解释器）" >&2; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PY" -m unittest discover -s tests -v "$@"
