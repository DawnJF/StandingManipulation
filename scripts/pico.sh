#!/usr/bin/env bash
# PICO 手柄遥操作：配置 adb 端口映射后启动 G1 实时跟踪。
# 用法：./scripts/pico.sh [--monitor] [--controller wbc] [--scale 1.0] [--yaw 180]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "未找到 ${PY}，请先 conda activate g1wbc（或设置 PYTHON 指向解释器）" >&2; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# PICO 通过 USB 连接时，用 adb reverse 把头盔上的 2233 端口转发到本机，
# 不需要头盔连 Wi-Fi。重插 USB 后需要重新执行一次。
if command -v adb >/dev/null 2>&1; then
  if adb get-state >/dev/null 2>&1; then
    adb reverse tcp:2233 tcp:2233 >/dev/null
    adb shell mkdir -p /storage/emulated/0/Android/data/com.picovr.openxr_demos/files >/dev/null 2>&1 || true
    adb shell "echo 127.0.0.1 > /storage/emulated/0/Android/data/com.picovr.openxr_demos/files/ip.txt" >/dev/null 2>&1 || true
  else
    echo "未检测到 adb 设备：请确认 PICO 已用 USB 连接并开启 USB 调试" >&2
  fi
fi

exec "$PY" -m g1_wbc.pico_live "$@"
