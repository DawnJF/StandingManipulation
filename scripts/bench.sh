#!/usr/bin/env bash
# 无渲染复算全部场景指标；默认 wbc（原速、无裁剪），--controller balance 为平衡前馈基线。
# 用法：./scripts/bench.sh [--scenario wave] [--controller wbc|balance|kinematic] [--out PATH]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "未找到 ${PY}，请先 conda activate g1wbc（或设置 PYTHON 指向解释器）" >&2; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m g1_wbc.bench "$@"
