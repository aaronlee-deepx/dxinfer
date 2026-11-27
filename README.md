# dxinfer — multi-network inference library (C++14)

A small library for running several `.dxnn` networks on a DEEPX NPU at the same time:
- **Loading** — several networks are loaded at once, each under a unique name. Loading is all-or-nothing.
- **Queries** — each network's input/output sizes and tensor layout can be looked up by name.
- **Feeding** — the library keeps the device's buffers full. You can block while they are full (`feed`) or not block at all (`try_feed`).
- **Results** — results come back **in FIFO order per network**. You get them either by **blocking** (`BlockingInference`) or by **callback** (`CallbackInference`).
- **dx-runtime mode** — how results reach you is independent of how dx-runtime runs underneath (`ExecMode::Sync` = `Run`, `ExecMode::Async` = `RunAsync` + completion callback).
- **Output copies** — dx-runtime writes each result directly into a block from a pool, so there is no output copy.
- **Measurement** — optional performance statistics, switched on or off with a single flag.

The library does no pre- or post-processing. It feeds the bytes you give it and returns the raw output tensors.

## Quick start
```bash
./setup.sh                 # builds the library, 2 demos and 3 host unit tests (no NPU needed)
./run.sh                   # callback demo, network "det" = yolo11-n_640x640.dxnn, --stats -l 300
./run.sh blocking --exec sync --stats -l 200
./run.sh callback /path/a.dxnn -n cls=/path/b.dxnn --stats -l 500     # two networks at once
```

## API in 30 lines
```cpp
#include <dxinfer/dxinfer.hpp>
using namespace dxinfer;

std::vector<NetworkSpec> nets = {
    {"det", "yolo11-n_640x640.dxnn"},
    {"cls", "resnet18.dxnn", NetworkOptions{/*buffer_count*/0, /*devices*/{}, /*bound*/0, /*ort*/OrtMode::Auto}},
};
InferenceConfig cfg;
cfg.exec = ExecMode::Async;     // or ExecMode::Sync
cfg.enable_stats = true;        // optional measurement

// ---- blocking delivery ----
BlockingInference inf(nets, cfg);            // throws InferenceError if ANY network cannot be loaded
inf.print_info();                            // model, tensors, sizes, device slots
std::vector<uint8_t> img(inf.input_size("det"));
uint64_t seq = inf.feed("det", img.data(), img.size(), /*user_arg*/ my_tag);   // from a feeder thread
Result r = inf.get("det");                   // from a consumer thread: next result in FIFO order
if (r.ok()) { const float* p = r.tensor(0).as<float>(); /* ... */ }
else        { std::cerr << r.status().message(); }               // one failed inference
inf.print_stats("det");

// ---- callback delivery ----
CallbackInference cb(nets, cfg);
cb.set_callback("det", [](Result& r) { /* FIFO order, on det's delivery thread */ });
cb.feed("det", img.data(), img.size());
cb.wait_idle();                              // everything delivered
```

| Call | Meaning |
|---|---|
| `network_names()`, `has_network(n)`, `info(n)` | What was loaded |
| `input_size(n)`, `output_size(n)` | The bytes `feed()` expects, and the size of one result block |
| `input_tensors(n)`, `output_tensors(n)` | Name, shape, dtype, bytes and offset of each tensor |
| `feed(n, ptr, bytes, user_arg)` | Returns a sequence number. Blocks only while all device slots are busy. The input can be reused as soon as it returns. |
| `try_feed(...)` | Never blocks. Returns `false` if no slot is free right now. |
| `get(n)`, `try_get(n, r)`, `get_for(n, timeout, r)` | `BlockingInference`: return the next result in FIFO order |
| `set_callback(n, fn)`, `set_callback(fn)`, `wait_idle()` | `CallbackInference` |
| `wait_completed()`, `pending(n)` | Every fed job has finished on the device; and the number of jobs fed but not yet consumed |
| `stats(n)`, `print_stats()`, `reset_stats()` | Statistics. These are only populated when `enable_stats` is set. |

Multi-input models: put every input tensor back to back in one buffer. The order and offsets are listed by `input_tensors(n)`.

## How it works
```
feed(name, in) ─► Network[name]
                   1. take a job slot   (at most bufferCount × devices on the device)
                   2. take a result block from the pool
                   3. seq = next_seq++
                   4. Sync : Run(in, out=block)              on the feeding thread
                      Async: RunAsync(in, job, out=block)    → dx-runtime callback thread
                   5. complete(job): Result{seq, tensors → block}; free the slot; push into the reorder buffer
                                                     │
          ReorderBuffer releases only seq == next ───┤
                                                     ▼
          BlockingInference::get()        or        CallbackInference delivery thread → your callback
```
- **FIFO order:** with several jobs in flight on several NPU cores, jobs can finish in any order. The reorder buffer holds a finished job until every earlier one has been released.
- **Why async uses the dx-runtime callback, not `Wait`:** the callback frees a job slot at the moment the job really finishes. That keeps the in-flight count exact and the device full even when jobs finish out of order. `Wait` called in feed order would hold the slot until every earlier job is done.
- **Dynamic feeding:** there are as many job slots as dx-runtime has buffers (`bufferCount × devices`). When a job completes, a `feed()` that is waiting for a slot continues immediately. There is no batch-then-wait pattern.
- **Result blocks:** the pool holds `max_in_flight + extra_result_buffers` blocks (by default the extra is the same as max-in-flight). Results you are still holding use blocks. If you hold every block, `feed()` waits until you release some.
- **Errors (hybrid):**
  - Problems with setup or misuse throw `InferenceError`. The error carries a `code()` and `network()`. Codes include `LoadFailed`, `OutOfDeviceMemory`, `DuplicateName`, `UnknownNetwork`, `BadInputSize` and `NoCallback`.
  - A single failed inference does not throw. It arrives as a `Result` whose `status()` holds the error, in its correct FIFO position, so delivery never stalls.
- **All-or-nothing loading:**
  - Names are checked before the device is touched.
  - Networks are then loaded in order. If one fails, every network loaded before it is released and the error lists them.
  - An NPU memory failure (dx-runtime `INSUFFICIENT_MEMORY`, or a dx-runtime message that mentions memory) becomes `OutOfDeviceMemory`, and the message includes each device's free memory.
- **Shutdown:**
  - The destructor stops new feeds and wakes any blocked feeders, which then throw `ShuttingDown`. It then waits for in-flight jobs to finish.
  - `CallbackInference` delivers every remaining result before its destructor returns.
  - A `Result` you keep stays valid even after the inference object is destroyed.

## ONNX Runtime (CPU tasks): `NetworkOptions::ort`
A compiled `.dxnn` can contain CPU tasks next to its NPU task(s). For example, `npu_0 -> cpu_0` is a model whose last reshape/concat/TopK runs on the CPU. The catch is what dx-runtime does when ORT is off:
- **With `useORT=false`, dx-runtime loads only the first NPU task.** CPU tasks are skipped, and later NPU tasks are never loaded.
- **There is no way to resume** the rest of the model from your side.

So before loading, dxinfer reads the model's task graph from the `.dxnn` (`src/model_graph.cpp`), then decides:

| Model graph | `OrtMode::Auto` (default) | `On` | `Off` |
|---|---|---|---|
| NPU only (1 task) | off, silent | on | off |
| 1 NPU + CPU head/tail | **on**, plus a hint that `Off` is faster if you implement the CPU part (it names the raw NPU tensors you would get) | on | off, plus a note listing the CPU tasks you now own |
| 2+ NPU tasks (`npu->cpu->npu`, `npu->npu`) | on (required) | on | **`InvalidArgument`**: the results would be the first NPU segment only |
| graph unreadable | on, plus **WARNING** | on, plus WARNING | off, plus WARNING |

- **The WARNING:** it names the file and the reason, e.g. `unsupported .dxnn format v10 (reader supports v7-v9)`.
  - It means the `.dxnn` format moved on, so the reader in `src/model_graph.cpp` needs updating. Nothing breaks in the meantime: Auto simply enables ORT.
  - The reason is also stored in `info(n).graph.note`.
- **After loading:** dxinfer compares the graph with what dx-runtime really loaded (`GetTaskOrder()`, `IsOrtConfigured()`).
  - If a model needs ORT but dx-runtime dropped it (dx_rt built without `USE_ORT`, or a `debug` compile type), loading fails with `LoadFailed`.
  - If the task count disagrees with the graph, you get a "reader may be out of date" WARNING.
- **What `print_info()` shows:** the graph (`npu_0[NPU] -> cpu_0[CPU]`), what was requested, and what is effective.
- **Demos:** they take `--ort auto|on|off`.

## Result memory: pool blocks, `copy_out()` and the low-buffer warning
Each network has its own pool of `max_in_flight + extra_result_buffers` result blocks (by default 2 × max-in-flight; e.g. 12 blocks of `output_size(name)` bytes). They are allocated once, in the constructor. dx-runtime writes each output straight into a block, and the block lives inside the `Result`:
- **While you hold a pool-backed `Result`**, it occupies a block. Blocks are also held by jobs on the device and by results waiting for delivery.
- **What holding many costs:** up to `extra_result_buffers` held results cost nothing. Beyond that, the device gets fewer jobs. With every block held, `feed()` waits and `try_feed()` returns `false`.
- **To keep outputs long:** call `Result::copy_out()`. It returns a `Result` with the same metadata whose bytes are in its own heap memory, so it holds no pool block (`uses_pool() == false`). It costs one copy of the output.
  ```cpp
  cb.set_callback("det", [&](Result& r) { archive.push_back(r.copy_out()); });  // r's block returns right away
  ```
- **The low-buffer warning (stderr only):** when a network's free blocks drop to **1 or fewer**, dxinfer says where the blocks are and names the main cause:
  ```
  [dxinfer] WARNING det: result blocks low (1 of 12 free)
            in flight 6 | waiting for FIFO order 5 | held by your app 0
            Cause: an earlier job is slow, so later results wait for their turn. Usually start-up warm-up.
            If it persists, raise InferenceConfig::extra_result_buffers (now 6).
  ```
  - **"waiting for FIFO order"** means results that finished early are held back behind a slower earlier job. A short episode at start-up (the first ONNX Runtime / NPU runs) is normal.
  - **"held by your app"** means Results you have not dropped yet. Drop them sooner, or keep them with `copy_out()`.
  - When the free count climbs back to **3 or more**, it prints `[dxinfer] det: result blocks recovered (3 of 12 free) after N ms`.
  - **How often:** once per episode, and at most one warning per second per network. Skipped warnings are counted in the next message.
  - **Which thread prints:** the feeding thread prints the warning, and the thread that drops a `Result` prints the recovery.
  - **When it is off:** when `extra_result_buffers < 2`, because then the pool runs near empty by design. It is also off during shutdown.

## Performance measurement
- **Switching it on:** set `cfg.enable_stats = true`. When it is off, the three hooks in the code return immediately: no clock is read and nothing is stored.
- **Where the code lives:** in `stats.hpp` / `stats.cpp`. The inference code only calls `on_feed`, `on_complete` and `on_deliver`.
- **Bounded memory:** the latency histograms have fixed buckets, so memory use does not grow with run length.

| Field | Meaning |
|---|---|
| FPS | Completed jobs ÷ (first feed → last completion) |
| device | Latency from feed to dx-runtime completion |
| reorder wait | Time from completion to release in FIFO order. This shows how out-of-order the jobs finished. |
| end-to-end | Time from feed until the result is handed to you |
| peak / max in flight | Whether the device stayed full |
| dx-runtime counters | Mean NPU time, queue wait, E2E and core latency. These are the same counters dxrun prints, so you can compare directly with `dxrun -m model.dxnn -l N -v`. They accumulate from engine creation, so `reset_stats()` does not reset them. |

Percentiles come from log-scale buckets with about 4% resolution. Count, mean, stddev, min and max are exact.

## Extending it
- **A new delivery style:** subclass `InferenceBase`, pop from `network(name).results()`, then call `on_delivered()` and `on_consumed()`. `BlockingInference` and `CallbackInference` are each about 60 lines.
- **A new execution backend:** add a branch in `detail::Network::submit()`. Completion always goes through `complete()`.
- **A new per-network option:** add a field to `NetworkOptions` and map it in the `Network` constructor.

## Files
| File | Role |
|---|---|
| `examples/multi_model/` | Two-network test app (yolo26-n detection + resnet18 classification) on `CallbackInference`: preprocesses all images first, feeds both networks at once, saves overlays. `get_images.sh` downloads the test images; see its README |
| `docs/auto-ort.html` | One-slide summary of auto ORT for RD: the 3 model cases after compile and what auto mode decides for each |
| `docs/features.html` | Feature workflows for reviewers: auto ORT from the `.dxnn` task graph, multiple networks, automatic FIFO; evidence and open items |
| `docs/architecture.html` | Architecture overview: layers, ownership, the feed → device → FIFO → delivery pipeline, walkthroughs, threads and reading order |
| `include/dxinfer/dxinfer.hpp` | Umbrella header |
| `include/dxinfer/blocking_inference.hpp`, `callback_inference.hpp` | The two public classes |
| `include/dxinfer/inference_base.hpp` | Shared base: loading, queries, feed, stats |
| `include/dxinfer/result.hpp` | `Result`, `TensorView` |
| `include/dxinfer/types.hpp`, `error.hpp` | Config and info types (`OrtMode`, `ModelGraph`); `InferenceError` and `Status` |
| `include/dxinfer/model_graph.hpp`, `src/model_graph.cpp` | `.dxnn` task-graph reader and ORT decision. This is the only place with `.dxnn` format knowledge, and it is unit-tested on the host. |
| `include/dxinfer/buffer_pool.hpp`, `reorder_buffer.hpp`, `stats.hpp`, `pool_warning.hpp` | Building blocks independent of dx-runtime, unit-tested on the host (pool level watch, low-buffer stderr warning) |
| `src/network.hpp/.cpp` | One network: engine, slots, pool, reorder buffer (internal) |
| `src/inference_base.cpp`, `blocking_inference.cpp`, `callback_inference.cpp`, `stats.cpp` | Implementations |
| `examples/demo_blocking.cpp`, `demo_callback.cpp`, `demo_common.hpp` | Demos that feed zero-filled input to 1..N networks and check FIFO order and `user_arg` |
| `tests/*.cpp` | Host unit tests (`ctest`) |
| `CMakeLists.txt`, `setup.sh`, `run.sh`, `session.log` | Build, run, and the real build/verification output |

## Verification
**Done on the build host (no DX-M1), recorded in `session.log`:**
- The unit tests were compiled first, before the headers existed, and failed (RED). They then passed: 27 test cases in 5 suites, including `copy_out` and the low-buffer warning (addendum).
- `setup.sh` exits 0 (dx_rt v3.5.0).
- A forced rebuild shows 0 warnings, and a strict `-Wpedantic -Wshadow -Wnon-virtual-dtor` pass is clean.
- The demos' command-line checks pass. A duplicate name is rejected before the device is touched.
- The binaries link `libdxrt.so`.
- The `run.sh` model guard works.
- A copy of the folder outside the suite builds and passes its tests.

**Not run (no NPU on this host).** On the target:
```bash
./run.sh callback --stats -l 1000 ; dxrun -m ../../assets/models/yolo11-n_640x640.dxnn -l 1000 -v   # compare FPS / latency
./run.sh blocking --exec sync --stats -l 200
./run.sh callback modelA.dxnn -n b=modelB.dxnn -n c=modelC.dxnn --stats   # multi-network; many large models → OutOfDeviceMemory
```