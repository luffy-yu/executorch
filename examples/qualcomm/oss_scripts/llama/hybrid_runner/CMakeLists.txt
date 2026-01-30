# CMakeLists.txt for Hybrid Vulkan + QNN Multimodal Runner
# Links both vulkan_backend (for vision encoder) and qnn_executorch_backend
# (for text decoder/embedding) into a single runner binary.
#
# This is built as a standalone project that links against pre-built
# ExecuTorch libraries from the main Android build.

cmake_minimum_required(VERSION 3.19)
project(hybrid_multimodal_runner)

# Required: path to ExecuTorch source and build directory
if(NOT EXECUTORCH_ROOT)
  message(FATAL_ERROR "EXECUTORCH_ROOT must be set")
endif()

if(NOT EXECUTORCH_BUILD_DIR)
  message(FATAL_ERROR "EXECUTORCH_BUILD_DIR must be set")
endif()

if(NOT QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT must be set (path to Qualcomm AI Engine Direct SDK)")
endif()

# stb_image for image loading
include(FetchContent)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
)
FetchContent_MakeAvailable(stb)
set(_stb_include_directories ${stb_SOURCE_DIR} ${stb_SOURCE_DIR}/deprecated)

# Source files: reuse qnn_multimodal_runner.cpp and all runner infrastructure
set(HYBRID_RUNNER_SRCS
  # Main entry point (same as QNN runner - it's backend-agnostic)
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/qnn_multimodal_runner.cpp
  # Multimodal runner infrastructure
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_runner.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_processor.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_lhd_token_generator.cpp
  # Core runner infrastructure (shared with qnn_llama_runner)
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/token_generator.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/lhd_token_generator.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.cpp
  ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/kv_manager.cpp
  # Custom ops fallback
  ${EXECUTORCH_ROOT}/extension/llm/custom_ops/op_fallback.cpp
)

add_executable(hybrid_multimodal_runner ${HYBRID_RUNNER_SRCS})

# Include directories
target_include_directories(hybrid_multimodal_runner PRIVATE
  ${EXECUTORCH_ROOT}/..
  ${EXECUTORCH_ROOT}
  ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
  ${_stb_include_directories}
  ${EXECUTORCH_BUILD_DIR}
  ${EXECUTORCH_BUILD_DIR}/third-party/gflags/include
  # Vulkan backend includes
  ${EXECUTORCH_ROOT}/backends/vulkan/third-party/Vulkan-Headers/include
  ${EXECUTORCH_ROOT}/backends/vulkan/third-party/volk
  # QNN backend includes
  ${QNN_SDK_ROOT}/include/QNN
  ${EXECUTORCH_ROOT}/backends/qualcomm/runtime
  ${EXECUTORCH_ROOT}/backends/qualcomm
  # Core includes
  ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
  ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/re2
  ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/abseil-cpp
  ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/sentencepiece/src
  ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/json/single_include
  ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/pcre2/src
)

target_compile_definitions(hybrid_multimodal_runner PRIVATE
  C10_USING_CUSTOM_GENERATED_MACROS
)

# Use file(GLOB_RECURSE) to find all .a and .so files, then match by name.
file(GLOB_RECURSE ALL_ET_LIBS "${EXECUTORCH_BUILD_DIR}/*.a" "${EXECUTORCH_BUILD_DIR}/*.so")

macro(find_et_lib VAR LIBNAME)
  set(${VAR} "")
  foreach(_lib ${ALL_ET_LIBS})
    get_filename_component(_name ${_lib} NAME_WE)
    if(_name STREQUAL "lib${LIBNAME}")
      set(${VAR} ${_lib})
      message(STATUS "Found ${LIBNAME}: ${_lib}")
      break()
    endif()
  endforeach()
  if(NOT ${VAR})
    message(WARNING "Could not find library: ${LIBNAME}")
  endif()
endmacro()

# Both backends
find_et_lib(LIB_VULKAN_BACKEND vulkan_backend)
# QNN backend is built as a shared library (.so)
find_et_lib(LIB_QNN_BACKEND qnn_executorch_backend)

# Core ExecuTorch
find_et_lib(LIB_EXECUTORCH executorch)
find_et_lib(LIB_EXECUTORCH_CORE executorch_core)

# Extensions
find_et_lib(LIB_EXTENSION_MODULE extension_module)
find_et_lib(LIB_EXTENSION_TENSOR extension_tensor)
find_et_lib(LIB_EXTENSION_DATA_LOADER extension_data_loader)
find_et_lib(LIB_EXTENSION_FLAT_TENSOR extension_flat_tensor)
find_et_lib(LIB_EXTENSION_LLM_RUNNER extension_llm_runner)
find_et_lib(LIB_EXTENSION_NAMED_DATA_MAP extension_named_data_map)
find_et_lib(LIB_EXTENSION_EVALUE_UTIL extension_evalue_util)
find_et_lib(LIB_EXTENSION_RUNNER_UTIL extension_runner_util)
find_et_lib(LIB_EXTENSION_THREADPOOL extension_threadpool)

# Ops and kernels
find_et_lib(LIB_PORTABLE_OPS portable_ops_lib)
find_et_lib(LIB_PORTABLE_KERNELS portable_kernels)
find_et_lib(LIB_OPT_PORTABLE_KERNELS optimized_portable_kernels)
find_et_lib(LIB_OPTIMIZED_OPS optimized_native_cpu_ops_lib)
find_et_lib(LIB_OPTIMIZED_KERNELS optimized_kernels)
find_et_lib(LIB_QUANTIZED_OPS quantized_ops_lib)
find_et_lib(LIB_QUANTIZED_KERNELS quantized_kernels)
find_et_lib(LIB_CPUBLAS cpublas)
find_et_lib(LIB_EIGEN_BLAS eigen_blas)
find_et_lib(LIB_KERNELS_UTIL kernels_util_all_deps)
find_et_lib(LIB_CPUINFO cpuinfo)
find_et_lib(LIB_PTHREADPOOL pthreadpool)

# Third-party
# gflags may be built as gflags_nothreads_debug or gflags_nothreads_static
find_et_lib(LIB_GFLAGS gflags_nothreads_debug)
if(NOT LIB_GFLAGS)
  find_et_lib(LIB_GFLAGS gflags_nothreads_static)
endif()
find_et_lib(LIB_TOKENIZERS tokenizers)
find_et_lib(LIB_REGEX_LOOKAHEAD regex_lookahead)
find_et_lib(LIB_SENTENCEPIECE sentencepiece)
find_et_lib(LIB_RE2 re2)
find_et_lib(LIB_PCRE2 pcre2-8)

# Collect all link libraries
set(RUNNER_LINK_LIBS "")

# Whole-archive link for backend registration and op registration
if(LIB_VULKAN_BACKEND)
  list(APPEND RUNNER_LINK_LIBS -Wl,--whole-archive ${LIB_VULKAN_BACKEND} -Wl,--no-whole-archive)
endif()
# QNN backend is a shared library (.so) - link directly (no whole-archive needed)
if(LIB_QNN_BACKEND)
  list(APPEND RUNNER_LINK_LIBS ${LIB_QNN_BACKEND})
endif()
# Optimized ops (includes portable fallbacks)
if(LIB_OPTIMIZED_OPS)
  list(APPEND RUNNER_LINK_LIBS -Wl,--whole-archive ${LIB_OPTIMIZED_OPS} -Wl,--no-whole-archive)
elseif(LIB_PORTABLE_OPS)
  list(APPEND RUNNER_LINK_LIBS -Wl,--whole-archive ${LIB_PORTABLE_OPS} -Wl,--no-whole-archive)
endif()
# Quantized ops for QNN models
if(LIB_QUANTIZED_OPS)
  list(APPEND RUNNER_LINK_LIBS -Wl,--whole-archive ${LIB_QUANTIZED_OPS} -Wl,--no-whole-archive)
endif()
# regex_lookahead needs whole-archive to trigger its static initializer
if(LIB_REGEX_LOOKAHEAD)
  list(APPEND RUNNER_LINK_LIBS -Wl,--whole-archive ${LIB_REGEX_LOOKAHEAD} -Wl,--no-whole-archive)
endif()

# Regular link for everything else
foreach(LIB
  LIB_EXECUTORCH
  LIB_EXECUTORCH_CORE
  LIB_EXTENSION_MODULE
  LIB_EXTENSION_TENSOR
  LIB_EXTENSION_DATA_LOADER
  LIB_EXTENSION_FLAT_TENSOR
  LIB_EXTENSION_LLM_RUNNER
  LIB_EXTENSION_NAMED_DATA_MAP
  LIB_EXTENSION_EVALUE_UTIL
  LIB_EXTENSION_RUNNER_UTIL
  LIB_EXTENSION_THREADPOOL
  LIB_PORTABLE_KERNELS
  LIB_OPT_PORTABLE_KERNELS
  LIB_OPTIMIZED_KERNELS
  LIB_QUANTIZED_KERNELS
  LIB_CPUBLAS
  LIB_EIGEN_BLAS
  LIB_KERNELS_UTIL
  LIB_CPUINFO
  LIB_PTHREADPOOL
  LIB_GFLAGS
  LIB_TOKENIZERS
  LIB_SENTENCEPIECE
  LIB_RE2
  LIB_PCRE2
)
  if(${LIB})
    list(APPEND RUNNER_LINK_LIBS ${${LIB}})
  endif()
endforeach()

# Abseil libraries (needed by sentencepiece/re2)
file(GLOB_RECURSE ABSL_LIBS "${EXECUTORCH_BUILD_DIR}/extension/llm/tokenizers/third-party/abseil-cpp/absl/*.a")
if(ABSL_LIBS)
  list(APPEND RUNNER_LINK_LIBS ${ABSL_LIBS})
endif()

# System libraries (Android)
if(ANDROID)
  list(APPEND RUNNER_LINK_LIBS log android)
endif()

target_link_libraries(hybrid_multimodal_runner ${RUNNER_LINK_LIBS})

set_target_properties(
  hybrid_multimodal_runner PROPERTIES
  LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
