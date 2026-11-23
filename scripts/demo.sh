#!/usr/bin/env bash
# 规划 + 回放 + 视频/GIF + 指标（默认场景 demo）。
# 用法：./scripts/demo.sh [--no-video] [--out DIR]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "未找到 ${PY}，请先 conda activate g1wbc（或设置 PYTHON 指向解释器）" >&2; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m g1_wbc.demo "$@"
