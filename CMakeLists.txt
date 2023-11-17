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

cmake_minimum_required (VERSION 3.1)
project (vulkan-shader-profiler)

set(CMAKE_CXX_STANDARD 17)

if (NOT PERFETTO_SDK_PATH)
  message(FATAL_ERROR "PERFETTO_SDK_PATH not defined")
endif ()

find_package(Vulkan)
message(STATUS "Vulkan_INCLUDE_DIRS = '${Vulkan_INCLUDE_DIRS}'")

set (SOURCES src/vulkan-shader-profiler.cpp ${PERFETTO_SDK_PATH}/perfetto.cc)
include_directories(${Vulkan_INCLUDE_DIRS} ${PERFETTO_SDK_PATH})

add_library (vulkan-shader-profiler MODULE ${SOURCES})
target_link_libraries (vulkan-shader-profiler SPIRV-Tools)

set(BACKEND "InProcess"
  CACHE STRING "Select perfetto backend")
set(BACKEND_OPTIONS
  System
  InProcess
)
set_property(CACHE BACKEND PROPERTY STRINGS ${BACKEND_OPTIONS})
if (${BACKEND} STREQUAL InProcess)
  target_compile_definitions(vulkan-shader-profiler PUBLIC BACKEND_INPROCESS)
endif()

set(TRACE_MAX_SIZE 1024 CACHE STRING "Perfetto trace maximum size")
target_compile_options(vulkan-shader-profiler PUBLIC "-DTRACE_MAX_SIZE=${TRACE_MAX_SIZE}")

set(TRACE_DEST "vulkan-shader-profiler.trace" CACHE STRING "Perfetto trace filename")
target_compile_options(vulkan-shader-profiler PUBLIC "-DTRACE_DEST=\"${TRACE_DEST}\"")
