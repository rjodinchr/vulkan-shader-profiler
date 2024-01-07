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

add_library(vksp-spirv STATIC spirv.cpp)
target_include_directories(vksp-spirv PUBLIC ${SPIRV-Tools-opt_INCLUDE_DIRS})
target_link_libraries(vksp-spirv ${SPIRV-Tools-opt_LIBRARIES})

add_executable(vulkan-shader-profiler-extractor extractor.cpp)
target_link_libraries(vulkan-shader-profiler-extractor ${PERFETTO_TRACE_PROCESSOR_LIB} vksp-spirv sqlite3)
target_include_directories(vulkan-shader-profiler-extractor PUBLIC ${Vulkan_INCLUDE_DIRS} ${PERFETTO_INTERNAL_INCLUDE_PATH})
if (PERFETTO_GEN_INCLUDE_PATH)
  target_include_directories(vulkan-shader-profiler-extractor PUBLIC ${PERFETTO_GEN_INCLUDE_PATH})
endif()
if (PERFETTO_CXX_CONFIG_INCLUDE_PATH)
  target_include_directories(vulkan-shader-profiler-extractor PUBLIC ${PERFETTO_CXX_CONFIG_INCLUDE_PATH})
endif()
if (SPIRV_TOOLS_INCLUDE_PATH)
  target_include_directories(vulkan-shader-profiler-extractor PUBLIC ${SPIRV_TOOLS_INCLUDE_PATH})
endif()
if (PERFETTO_CXX_SYSTEM_INCLUDE_PATH)
  target_include_directories(vulkan-shader-profiler-extractor SYSTEM PUBLIC ${PERFETTO_CXX_SYSTEM_INCLUDE_PATH})
endif()
if (EXTRACTOR_NOSTDINCXX)
  target_compile_options(vulkan-shader-profiler-extractor PUBLIC -nostdinc++)
endif()
