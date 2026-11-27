// dxinfer - multi-network inference on DEEPX NPUs with FIFO-ordered results.
//
//   #include <dxinfer/dxinfer.hpp>
//
//   BlockingInference  : pull results with get()            (blocking delivery)
//   CallbackInference  : results pushed to your callback    (callback delivery)
//   ExecMode::Sync / ExecMode::Async choose how dx-runtime runs underneath.
#pragma once

#include "dxinfer/blocking_inference.hpp"
#include "dxinfer/buffer_pool.hpp"
#include "dxinfer/callback_inference.hpp"
#include "dxinfer/error.hpp"
#include "dxinfer/inference_base.hpp"
#include "dxinfer/model_graph.hpp"
#include "dxinfer/reorder_buffer.hpp"
#include "dxinfer/result.hpp"
#include "dxinfer/stats.hpp"
#include "dxinfer/types.hpp"