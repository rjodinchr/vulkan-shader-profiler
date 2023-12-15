// Copyright 2023 The Vulkan Shader Profiler authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "spirv-tools/optimizer.hpp"
#include <string>

extern "C" bool store_shader_in_output(std::string *shader, std::vector<spvtools::vksp_push_constant> *pc,
    std::vector<spvtools::vksp_descriptor_set> *ds, std::vector<spvtools::vksp_specialization_map_entry> *me,
    spvtools::vksp_configuration *config, const char *output_filename, bool binary_output);
