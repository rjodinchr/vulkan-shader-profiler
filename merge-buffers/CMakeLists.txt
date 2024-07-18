# Copyright 2024 The Vulkan Shader Profiler authors.
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

add_executable(vulkan-shader-profiler-merge-buffers merge-buffers.cpp)
target_include_directories(vulkan-shader-profiler-merge-buffers PUBLIC
  ${SPIRV-Tools-opt_INCLUDE_DIRS}
  ${SPIRV_TOOLS_SOURCE_PATH}
  ${SPIRV_TOOLS_BUILD_PATH}
  ${CMAKE_SOURCE_DIR}
)

if (SPIRV_HEADERS_INCLUDE_PATH)
  target_include_directories(vulkan-shader-profiler-merge-buffers PUBLIC ${SPIRV_HEADERS_INCLUDE_PATH})
endif()

target_link_libraries(vulkan-shader-profiler-merge-buffers ${SPIRV-Tools-opt_LIBRARIES})
