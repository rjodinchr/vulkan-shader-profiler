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

if (NOT SPIRV_TOOLS_SOURCE_PATH)
  message(FATAL_ERROR "SPIRV_TOOLS_SOURCE_PATH not defined")
endif()
if (NOT SPIRV_TOOLS_BUILD_PATH)
  message(FATAL_ERROR "SPIRV_TOOLS_BUILD_PATH not defined")
endif()

find_package(SPIRV-Tools-opt)
message(STATUS "SPIRV-Tools-opt_LIBRARIES = '${SPIRV-Tools-opt_LIBRARIES}'")
message(STATUS "SPIRV-Tools-opt_INCLUDE_DIRS = '${SPIRV-Tools-opt_INCLUDE_DIRS}'")

find_package(Vulkan)
message(STATUS "Vulkan_INCLUDE_DIRS = '${Vulkan_INCLUDE_DIRS}'")
message(STATUS "Vulkan_LIBRARIES = '${Vulkan_LIBRARIES}'")

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
