#!/usr/bin/env bash
# Run multi_model_demo with its defaults (models/ and images/ next to this script).
# Extra arguments are passed through, e.g. ./run.sh --stats --ort on -n 12 -o /tmp/out
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd "$SCRIPT_DIR"

BIN=""
for c in "$SCRIPT_DIR/../../build/multi_model_demo" "$SCRIPT_DIR/build/multi_model_demo"; do
    [ -x "$c" ] && { BIN="$c"; break; }
done
if [ -z "$BIN" ]; then
    echo "[error] multi_model_demo is not built. From the library folder run:" >&2
    echo "        cmake -S . -B build && cmake --build build -j   (needs OpenCV: sudo apt install libopencv-dev)" >&2
    exit 1
fi

missing=0
for m in models/yolo26-n_640x640.dxnn models/resnet18_224x224.dxnn; do
    case " $* " in *" --det "*|*" --cls "*) break ;; esac   # custom models given: let the demo check them
    if [ ! -f "$m" ]; then echo "[error] missing $SCRIPT_DIR/$m (download it from the DEEPX model zoo, see README.md)" >&2; missing=1; fi
done
[ "$missing" -eq 0 ] || exit 1

if [ ! -d images/image_object_detection ] || [ ! -d images/image_classification ]; then
    echo "[info] images not found, downloading them first"
    ./get_images.sh
fi

exec "$BIN" "$@"