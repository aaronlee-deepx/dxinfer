#!/usr/bin/env bash
# =============================================================================
# run.sh - run a demo on the NPU.
#
# Usage:  ./run.sh [blocking|callback] [model.dxnn] [demo args...]
#   default demo   : callback
#   default model  : dx_app/assets/models/yolo11-n_640x640.dxnn (loaded as network "det")
#   default args   : --stats -l 300
# Examples:
#   ./run.sh
#   ./run.sh blocking --exec sync --stats -l 200
#   ./run.sh callback /path/a.dxnn -n cls=/path/b.dxnn --stats -l 500   # two networks
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

DEMO=callback
if [ $# -gt 0 ] && { [ "$1" = blocking ] || [ "$1" = callback ]; }; then DEMO="$1"; shift; fi

DEFAULT_MODEL="$SCRIPT_DIR/../../assets/models/yolo11-n_640x640.dxnn"   # within dx_app
MODEL="${MODEL:-$DEFAULT_MODEL}"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then MODEL="$1"; shift; fi

BIN="$SCRIPT_DIR/build/demo_$DEMO"
if [ ! -x "$BIN" ]; then
    echo "[run] $BIN not built yet -> run ./setup.sh first"; exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "[run] model not found: $MODEL"
    echo "[run] download it from the dx_app root: ./setup_sample_models.sh --models YOLOV11N"
    echo "[run] or pass a path: ./run.sh $DEMO /path/to/model.dxnn"
    exit 1
fi

if [ $# -eq 0 ]; then set -- --stats -l 300; fi
exec "$BIN" -n "det=$MODEL" "$@"