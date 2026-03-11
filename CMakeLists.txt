# Copyright 2023 The Vulkan Shader Profiler authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.24)
project (vulkan-shader-profiler)

set(CMAKE_CXX_STANDARD 17)

if (NOT SPIRV_TOOLS_SOURCE_PATH OR NOT SPIRV_TOOLS_BUILD_PATH)
  message(STATUS "SPIRV_TOOLS_SOURCE_PATH or SPIRV_TOOLS_BUILD_PATH not set, fetching SPIRV-Tools...")
  include(FetchContent)

  set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)

  FetchContent_Declare(
    spirv-headers
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(spirv-headers)

  FetchContent_Declare(
    spirv-tools
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(spirv-tools)

  set(SPIRV_TOOLS_SOURCE_PATH "${spirv-tools_SOURCE_DIR}" CACHE PATH "" FORCE)
  set(SPIRV_TOOLS_BUILD_PATH "${spirv-tools_BINARY_DIR}" CACHE PATH "" FORCE)
  set(SPIRV_HEADERS_INCLUDE_PATH "${spirv-headers_SOURCE_DIR}/include" CACHE PATH "" FORCE)
  set(SPIRV-Tools-opt_LIBRARIES SPIRV-Tools-opt)
  set(SPIRV-Tools-opt_FOUND TRUE)
else()
  find_package(SPIRV-Tools-opt)
  message(STATUS "SPIRV-Tools-opt_LIBRARIES = '${SPIRV-Tools-opt_LIBRARIES}'")
  message(STATUS "SPIRV-Tools-opt_INCLUDE_DIRS = '${SPIRV-Tools-opt_INCLUDE_DIRS}'")
endif()

find_package(Vulkan)
message(STATUS "Vulkan_INCLUDE_DIRS = '${Vulkan_INCLUDE_DIRS}'")
message(STATUS "Vulkan_LIBRARIES = '${Vulkan_LIBRARIES}'")

if (NOT PERFETTO_TRACE_PROCESSOR_LIB AND NOT VKSP_SKIP_EXTRACTOR)
  message(STATUS "PERFETTO_TRACE_PROCESSOR_LIB not set, skipping extractor")
  set(VKSP_SKIP_EXTRACTOR 1)
endif()

if (NOT VKSP_SKIP_LAYER)
  add_subdirectory(layer)
endif()
if (NOT VKSP_SKIP_RUNNER)
  add_subdirectory(runner)
endif()
if (NOT VKSP_SKIP_EXTRACTOR)
  add_subdirectory(extractor)
endif()
if (NOT VKSP_SKIP_MERGE_BUFFERS)
  add_subdirectory(merge-buffers)
endif()
