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

add_executable(vulkan-shader-profiler-runner runner.cpp)
target_include_directories(vulkan-shader-profiler-runner PUBLIC ${Vulkan_INCLUDE_DIRS})
if (SPIRV_TOOLS_INCLUDE_PATH)
  target_include_directories(vulkan-shader-profiler-runner PUBLIC ${SPIRV_TOOLS_INCLUDE_PATH})
endif()
target_link_libraries(vulkan-shader-profiler-runner ${Vulkan_LIBRARIES} ${SPIRV_TOOLS_OPT_LIBRARY} ${SPIRV_TOOLS_LIBRARY})

