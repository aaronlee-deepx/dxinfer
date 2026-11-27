# multi_model_demo: two networks through dxinfer's CallbackInference

This demo tests dxinfer the way an application uses it. Two networks run at the same time with async callbacks:

| Network | Model (`.dxnn`) | Task | Auto ORT |
|---|---|---|---|
| `det` | `yolo26-n_640x640.dxnn` | object detection (COCO-80) | on: the model compiles to NPU → CPU |
| `cls` | `resnet18_224x224.dxnn` | image classification (ImageNet-1000) | expected off: NPU only |

What happens in a run:
1. Every image is preprocessed before anything is fed.
2. One feeder thread per network pushes all inputs as fast as dxinfer accepts them.
3. Each network's callback runs on its own delivery thread. It decodes the result, draws the overlay and saves it.
4. At the end the demo checks that results arrived in FIFO order, and that none failed or went missing.

## Quick start
```bash
# 1. models: download both from the DEEPX model zoo into ./models/
#    (the names are the dx_app model registry entries "yolo26n" and "resnet18")
mkdir -p models   # models/yolo26-n_640x640.dxnn   models/resnet18_224x224.dxnn

# 2. images: 25 COCO val2017 photos + 25 ImageNet samples (default; -n N for 1..50 of each)
./get_images.sh            # or: ./get_images.sh -n 20      ./get_images.sh --list  (show URLs only)

# 3. build: once, from the library folder two levels up (needs OpenCV: sudo apt install libopencv-dev)
(cd ../.. && cmake -S . -B build && cmake --build build -j)

# 4. run
./run.sh                    # defaults: 10 images per network, output/ next to this README
./run.sh --stats -n 25      # all 25 downloaded images, with dxinfer statistics
./run.sh --ort on -o /tmp/mm_out
```

## What to check by eye
- **`output/det/<image>_overlay.jpg`:** boxes and `class score` labels on the original photo. For example, `000000039769` shows two cats on a couch with remotes, and `000000000285` shows a bear.
- **`output/cls/<image>_overlay.jpg`:** a top-5 panel in the top-left corner. The file name holds the true ImageNet class, e.g. `n01440764_tench` should rank "tench" first.
- **`output/results.txt`:** the same summary printed on screen. It has one line per image and per network, then `FIFO OK`, `failed 0` and a final `RESULT: PASS`.

## Options
| Option | Default |
|---|---|
| `--det MODEL` / `--cls MODEL` | `models/yolo26-n_640x640.dxnn` / `models/resnet18_224x224.dxnn` |
| `--det-dir DIR` / `--cls-dir DIR` | `images/image_object_detection` / `images/image_classification` |
| `-o, --out DIR` | `output` (overlays go in `DIR/det` and `DIR/cls`) |
| `-n, --count N` | 10 images per network (minimum 10) |
| `--score S` | 0.3 detection score threshold |
| `--ort auto\|on\|off` | `auto` for both networks |
| `-s, --stats` | off |

`--ort off` on yolo26-n returns the NPU's raw heads. The demo then decodes them itself with the DFL path, which is the "faster option" the auto-ORT hint describes.

## get_images.sh
| Option | Meaning |
|---|---|
| `-n, --count N` | Images per folder, the same N for detection and classification. Default 25, range 1–50. |
| `--list` | Print the URLs that would be downloaded, and download nothing. |
| `target_dir` | Where `image_object_detection/` and `image_classification/` are created. Default `images/`. |

- **Image choice:** the script picks the N images spread evenly over its list of 50, so even a small N covers varied ImageNet classes.
- **Existing files:** files already in the folders are kept. A folder can therefore hold more than N images, and the demo uses the first `--count` of them in sorted order.

## How it is built
- **`model_factory.hpp`:** one factory per model in dx_app's `IFactory` shape (`createPreprocessor`, `createPostprocessor`, `createVisualizer`, `getModelName`, `getTaskType`).
- **`postprocess.*`:** OpenCV-free, unit-tested by `tests/test_multi_model_post.cpp`. Contains the letterbox math, the yolo26 decode (end-to-end `[1,300,6]` or raw DFL), NMS, mapping boxes back to the original image, and top-k.
- **`vision.*`:** OpenCV. Loads images, letterboxes them (BGR→RGB, pad 114), draws the overlays and saves them.
- **`labels.hpp`:** dx_app's COCO / ImageNet label file, copied unchanged.
- **`multi_model_demo.cpp`:** loading, preprocessing, callbacks, feeder threads and the summary.

**Why it doesn't reuse dx_app's C++ processors directly:** they take `dxrt::TensorPtrs` from `dxrt_api.h`, while dxinfer is built on the header-only `dxrt_cxx_api.h`. Each header defines its own `dxrt::Tensor`, so the two can't be linked into one program safely. The model-specific logic was therefore ported from dx_app's `yolo26-n_640x640` and `resnet` factories, the letterbox preprocessor, and the YOLOv26 and EfficientNet postprocessors.

## Troubleshooting
- **`model not found`:** download the model into `models/`, or pass `--det` / `--cls`.
- **`need at least 10 images`:** run `./get_images.sh`, or point `--det-dir` / `--cls-dir` at your own folders.
- **`get_images.sh` prints a certificate hint:** the network inspects HTTPS with its own root certificate. Install it in Linux (`sudo update-ca-certificates`), or download the images elsewhere and copy them in.
- **`multi_model_demo is not built`:** install OpenCV, then rebuild. CMake prints `multi_model_demo skipped: OpenCV not found` when OpenCV is missing.