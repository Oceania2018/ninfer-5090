cmake_minimum_required(VERSION 3.28)

# One source tree, one CUDA architecture per build. Set and validate it before CUDA compiler
# detection so CMake never injects a generic host-toolchain default or accepts an unsupported target.
#   120a (default): RTX 5090 and RTX 5060 Ti. Small-SM cards (<= 48 SMs) pick their launch shapes
#                   at runtime (src/ops/common/device_class.h).
#   89:             RTX 4090. Defines NINFER_SM86: PDL off, TMA routes on cp.async, int8-activation
#                   prefill instead of FP8/NVFP4, no NVFP4/K8V4 KV or NVFP4 StateImages.
# Startup rejects a device whose compute capability does not match the build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architecture to build: 120a or 89")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(120a|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=120a or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
# Every target sees the sm_89 switch: kernels, model startup, weight policies and tests.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  add_compile_definitions(NINFER_SM86=1)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
