# dxinfer - multi-network inference library with FIFO-ordered results.
# Depends only on the DEEPX runtime (dxrt) and pthreads.
cmake_minimum_required(VERSION 3.14)
project(dxinfer CXX)

set(CMAKE_CXX_STANDARD 14)            # dx_app convention: C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(DXINFER_BUILD_EXAMPLES "Build demo_blocking / demo_callback" ON)
option(DXINFER_BUILD_TESTS "Build host unit tests (no NPU needed)" ON)
set(DXRT_INSTALLED_DIR "/usr/local" CACHE PATH "dx_rt install prefix")

add_compile_options(-Wall -Wextra)
find_package(Threads REQUIRED)

# rapidjson (header-only) ships with dx_rt; used by the .dxnn task-graph reader
find_path(DXINFER_RAPIDJSON_DIR rapidjson/document.h
  HINTS ${DXRT_INSTALLED_DIR}/include/dxrt/extern /usr/local/include/dxrt/extern /usr/include)
if(NOT DXINFER_RAPIDJSON_DIR)
  message(FATAL_ERROR "rapidjson not found (expected in ${DXRT_INSTALLED_DIR}/include/dxrt/extern)")
endif()

# dxinfer_core: dx-runtime-free parts (pool, reorder buffer, stats, .dxnn graph reader) - unit-testable anywhere
add_library(dxinfer_core STATIC src/stats.cpp src/pool_warning.cpp src/model_graph.cpp)
target_include_directories(dxinfer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(dxinfer_core SYSTEM PRIVATE ${DXINFER_RAPIDJSON_DIR})
target_link_libraries(dxinfer_core PUBLIC Threads::Threads)

# dxinfer: the inference classes on top of dxrt::InferenceEngine
find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
add_library(dxinfer STATIC
  src/network.cpp
  src/inference_base.cpp
  src/blocking_inference.cpp
  src/callback_inference.cpp)
target_include_directories(dxinfer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dxinfer PUBLIC dxinfer_core dxrt Threads::Threads)

if(DXINFER_BUILD_EXAMPLES)
  foreach(demo demo_blocking demo_callback)
    add_executable(${demo} examples/${demo}.cpp)
    target_link_libraries(${demo} PRIVATE dxinfer)
  endforeach()
endif()

# examples/multi_model: two networks (yolo26-n + resnet18) through CallbackInference, overlays saved.
# Needs OpenCV (core, imgproc, imgcodecs); skipped with a message when it is not installed.
option(DXINFER_BUILD_MULTI_MODEL "Build examples/multi_model/multi_model_demo (needs OpenCV)" ON)
if(DXINFER_BUILD_MULTI_MODEL)
  find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
  if(OpenCV_FOUND)
    add_executable(multi_model_demo
      examples/multi_model/multi_model_demo.cpp
      examples/multi_model/vision.cpp
      examples/multi_model/postprocess.cpp)
    target_include_directories(multi_model_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/multi_model)
    target_include_directories(multi_model_demo SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(multi_model_demo PRIVATE dxinfer ${OpenCV_LIBS})
  else()
    message(STATUS "multi_model_demo skipped: OpenCV not found (sudo apt install libopencv-dev)")
  endif()
endif()

if(DXINFER_BUILD_TESTS)
  enable_testing()
  # multi-model demo: OpenCV-free post-processing (examples/multi_model/postprocess.*)
  add_executable(test_multi_model_post tests/test_multi_model_post.cpp examples/multi_model/postprocess.cpp)
  target_include_directories(test_multi_model_post PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/multi_model)
  add_test(NAME test_multi_model_post COMMAND test_multi_model_post)
  foreach(t test_reorder_buffer test_buffer_pool test_stats test_result test_pool_warning test_model_graph)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE dxinfer_core)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()