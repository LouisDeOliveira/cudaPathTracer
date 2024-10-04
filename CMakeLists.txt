include(FetchContent)
cmake_minimum_required(VERSION 3.8)

project(cudapt LANGUAGES CUDA CXX)

set(SFML_VERSION 2.6.1)
set(IMGUI_VERSION 1.89)

FetchContent_Declare(
  imgui
  URL "https://github.com/ocornut/imgui/archive/v${IMGUI_VERSION}.zip"
)

FetchContent_MakeAvailable(imgui)

# ImGui-SFML
set(IMGUI_DIR ${imgui_SOURCE_DIR})
option(IMGUI_SFML_FIND_SFML "Use find_package to find SFML" OFF)
option(IMGUI_SFML_IMGUI_DEMO "Build imgui_demo.cpp" ON)


FetchContent_Declare(
  imgui-sfml
  GIT_REPOSITORY https://github.com/SFML/imgui-sfml.git
  GIT_TAG        2.6.x
  # GIT_COMMIT 5f54b69b6aee18db846c81633f86f78c2586dded
  # ^ or like this - sometimes it's better because FetchContent won't look
  # into remote to see if branch head was updated or not - good for stable
  # tags like 'vX.X' corresponding to releases
)

# ImGui-SFML
set(IMGUI_DIR ${imgui_SOURCE_DIR})
option(IMGUI_SFML_IMGUI_DEMO "Build imgui_demo.cpp" ON)

FetchContent_MakeAvailable(imgui-sfml)


add_library(cudalib kernel.cu kernel.cuh)

target_compile_features(cudalib PUBLIC cxx_std_11)

set_target_properties(cudalib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cudamain main.cpp)

set_target_properties(cudamain PROPERTIES CUDA_ARCHITECTURES 70)

set_property(TARGET cudamain PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cudamain cudalib sfml-graphics sfml-window sfml-system ImGui-SFML::ImGui-SFML)