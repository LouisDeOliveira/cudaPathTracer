cmake_minimum_required(VERSION 3.8)

project(cudapt LANGUAGES CUDA CXX)

add_library(cudalib kernel.cu kernel.cuh math_utils.cuh)

target_compile_features(cudalib PUBLIC cxx_std_11)

set_target_properties(cudalib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cudamain main.cpp)

set_target_properties(cudamain PROPERTIES CUDA_ARCHITECTURES 70)

set_property(TARGET cudamain PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cudamain cudalib sfml-graphics sfml-window sfml-system)