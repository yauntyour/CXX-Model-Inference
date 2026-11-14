cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CXXModelInference
    VERSION 0.1.0
    DESCRIPTION "C++17/CUDA model inference on TensorN"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 5060 Ti (sm120); 可覆盖为其它架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# ---- TensorN 子目录选项 ----
set(TENSORN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TENSORN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(TENSORN_ENABLE_OPENBLAS OFF CACHE BOOL "" FORCE)
set(TENSORN_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(TENSORN_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)

add_subdirectory(TensorN)

# ---- 主程序 ----
add_executable(qwen35_chat
    src/main.cpp
    src/kernels.cu
    src/fast_ops.cu
    src/qwen_model.cpp
    src/vision.cpp
    src/image.cpp
)

target_link_libraries(qwen35_chat PRIVATE TensorN CUDA::cudart CUDA::cublas)
target_include_directories(qwen35_chat PRIVATE src)

source_group("src" FILES src/main.cpp src/kernels.cu src/fast_ops.cu src/qwen_model.cpp src/vision.cpp src/image.cpp)

# ---- AstrAI / 通用 Dense Transformer 文本模型 ----
add_executable(astrai_chat
    src/astrai_main.cpp
    src/kernels.cu
    src/fast_ops.cu
    src/dense_model.cpp
)

target_link_libraries(astrai_chat PRIVATE TensorN CUDA::cudart CUDA::cublas)
target_include_directories(astrai_chat PRIVATE src)

source_group("src" FILES src/astrai_main.cpp src/kernels.cu src/fast_ops.cu src/dense_model.cpp)
