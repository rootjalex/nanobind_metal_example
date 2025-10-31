cmake_minimum_required(VERSION 3.15...3.26)

project(nanobind_gpu_example LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# ------------------------------------------------------------------------------
# 1. Detect CUDA and Apple Silicon
# ------------------------------------------------------------------------------

# Try to detect CUDA support first
include(CheckLanguage)
check_language(CUDA)

set(HAS_CUDA FALSE)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HAS_CUDA TRUE)
endif()

# Detect Apple Silicon
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  set(HAS_METAL TRUE)
endif()

if(HAS_CUDA)
  message(STATUS "Building with CUDA backend.")
  set(BACKEND "cuda")
elseif(HAS_METAL)
  message(STATUS "Building with Metal backend.")
  set(BACKEND "metal")
else()
  message(FATAL_ERROR "No supported GPU backend found (need CUDA or Metal).")
endif()

# ------------------------------------------------------------------------------
# Common Sources (always included)
# ------------------------------------------------------------------------------
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SRC_FILES
    ${SRC_DIR}/nanobind_gpu_example_ext.cpp
)

# ------------------------------------------------------------------------------
# CUDA Configuration
# ------------------------------------------------------------------------------
if(HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})

  # Collect CUDA-specific sources
  file(GLOB CUDA_SOURCES "${SRC_DIR}/cuda/*.cu" "${SRC_DIR}/cuda/*.cpp")
  list(APPEND SRC_FILES ${CUDA_SOURCES})

  # Query available GPU architectures
  execute_process(
      COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
      OUTPUT_VARIABLE GPU_ARCHS_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REPLACE "\n" ";" GPU_ARCH_LIST "${GPU_ARCHS_RAW}")
  list(REVERSE GPU_ARCH_LIST)
  list(GET GPU_ARCH_LIST 0 GPU_ARCH)
  message(STATUS "Detected GPU architecture: ${GPU_ARCH}")

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=${GPU_ARCH} -dc")
  add_compile_definitions(USE_CUDA)

# ------------------------------------------------------------------------------
# Metal Configuration
# ------------------------------------------------------------------------------
elseif(HAS_METAL)
  find_library(FWK_FOUNDATION Foundation REQUIRED)
  find_library(FWK_QUARTZ_CORE QuartzCore REQUIRED)
  find_library(FWK_METAL Metal REQUIRED)

  set(METAL_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp")
  if(NOT EXISTS "${METAL_CPP_DIR}")
      message(FATAL_ERROR "metal-cpp not found at ${METAL_CPP_DIR}.")
  endif()

  # Collect Metal-specific sources
  file(GLOB METAL_SOURCES "${SRC_DIR}/metal/*.cpp")
  list(APPEND SRC_FILES ${METAL_SOURCES})

  add_compile_definitions(USE_METAL)
endif()

# ------------------------------------------------------------------------------
# Build unified extension
# ------------------------------------------------------------------------------
nanobind_add_module(
  # Name of the extension
  nanobind_gpu_example_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  ${SRC_FILES}
)

# ------------------------------------------------------------------------------
# Link appropriate libraries
# ------------------------------------------------------------------------------
if(HAS_CUDA)
  target_link_libraries(nanobind_gpu_example_ext PRIVATE CUDA::cudart)
elseif(HAS_METAL)
  target_include_directories(nanobind_gpu_example_ext PRIVATE ${METAL_CPP_DIR})
  target_include_directories(nanobind_gpu_example_ext PRIVATE "${METAL_CPP_DIR}")

  target_link_libraries(nanobind_gpu_example_ext PRIVATE
    ${FWK_FOUNDATION}
    ${FWK_QUARTZ_CORE}
    ${FWK_METAL}
  )
endif()

# Install directive for scikit-build-core
install(TARGETS nanobind_gpu_example_ext LIBRARY DESTINATION nanobind_gpu_example)
