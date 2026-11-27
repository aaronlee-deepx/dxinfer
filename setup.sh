#!/usr/bin/env bash
# =============================================================================
# setup.sh - build the dxinfer library, the two demos and the host unit tests.
# Touches no NPU, so it also works on a build machine without a DX-M1.
#   env: DXRT_PREFIX=/usr/local (dx_rt install prefix)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
BUILD_DIR="$SCRIPT_DIR/build"
DXRT_PREFIX="${DXRT_PREFIX:-/usr/local}"

echo "[setup] 1/4 toolchain"
for tool in cmake c++ make; do
    command -v "$tool" >/dev/null 2>&1 || { echo "[setup] missing '$tool' -> sudo apt install build-essential cmake"; exit 1; }
done
echo "        cmake $(cmake --version | head -1 | awk '{print $3}'), $(c++ --version | head -1)"

echo "[setup] 2/4 dx_rt"
HDR="$DXRT_PREFIX/include/dxrt/dxrt_cxx_api.h"
if [ ! -f "$HDR" ]; then
    if [ -f "$DXRT_PREFIX/include/dxrt/dxrt_api.h" ]; then
        echo "[setup] dx_rt at $DXRT_PREFIX is TOO OLD: dxinfer needs dx_rt >= 3.5 (dxrt_cxx_api.h)"
    else
        echo "[setup] dx_rt not found at $DXRT_PREFIX -> install it (dx-runtime/install.sh) or set DXRT_PREFIX"
    fi
    exit 1
fi
VER="$(sed -n 's/.*DXRT_VERSION *"\(.*\)".*/\1/p' "$DXRT_PREFIX/include/dxrt/gen.h" 2>/dev/null | head -1)"
echo "        dx_rt : $DXRT_PREFIX (${VER:-unknown version})"

echo "[setup] 3/4 configure + build -> $BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DDXRT_INSTALLED_DIR="$DXRT_PREFIX"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "[setup] 4/4 host unit tests (no NPU needed)"
(cd "$BUILD_DIR" && ctest --output-on-failure)

echo "[setup] done: $BUILD_DIR/demo_blocking  $BUILD_DIR/demo_callback  (run ./run.sh on the target)"