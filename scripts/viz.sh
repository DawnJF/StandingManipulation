#!/usr/bin/env bash
# 打开 MuJoCo 交互窗口实时观看某个场景，无需先 pip install。
# 用法：./scripts/viz.sh wave   （也可直接透传参数：./scripts/viz.sh wave --loop）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "未找到 ${PY}，请先 conda activate g1wbc（或设置 PYTHON 指向解释器）" >&2; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 允许省略 --scenario：第一个参数不是以 - 开头时当作场景名
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  set -- --scenario "$@"
fi

exec "$PY" -m g1_wbc.live "$@"
