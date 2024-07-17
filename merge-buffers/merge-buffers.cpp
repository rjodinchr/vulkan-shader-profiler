// Copyright 2024 The Vulkan Shader Profiler authors.
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

#include <spirv-tools/libspirv.h>

static bool gVerbose = false;

#include "common/buffers_file.hpp"
#include "common/common.hpp"
#include "common/spirv-extract.hpp"

#include <filesystem>
#include <iostream>
#include <stdlib.h>
#include <vector>

static void help() { printf("USAGE: vulkan-shader-profiler-merge-buffers <prefix>\n"); }

int main(int argc, char **argv)
{
    if (argc != 2) {
        ERROR("Invalid number of arguments");
        help();
        return -1;
    }

    std::vector<uint32_t> shader;
    std::vector<vksp::vksp_descriptor_set> dsVector;
    std::vector<vksp::vksp_push_constant> pcVector;
    std::vector<vksp::vksp_specialization_map_entry> meVector;
    std::vector<vksp::vksp_counter> counters;
    vksp::vksp_configuration config;
    spv_target_env gSpvTargetEnv = SPV_ENV_VULKAN_1_3;
    if (!extract_from_input(
            argv[1], gSpvTargetEnv, true, false, shader, dsVector, pcVector, meVector, counters, config)) {
        ERROR("Could not extract data from input");
        return -1;
    }

    std::filesystem::path prefix = std::filesystem::absolute(std::filesystem::path(argv[1]));
    auto directory = prefix.parent_path();
    vksp::BuffersFile buffers_file(config.dispatchId);
    std::vector<void *> buffers;

    std::string prefix_str = prefix.string() + ".buffers";
    for (const auto &file : std::filesystem::directory_iterator(directory)) {
        std::string file_str = file.path().string();
        if (file_str.rfind(prefix, 0) != 0 || file_str.size() <= prefix_str.size()) {
            continue;
        }
        auto prefix_end_pos = prefix_str.size();
        auto first_dot = file_str.find(".", prefix_end_pos - 1);
        if (first_dot == std::string::npos) {
            ERROR("Could not figure out file name format, skipping '%s'", file_str.c_str());
            continue;
        }
        auto second_dot = file_str.find(".", first_dot + 1);
        if (second_dot == std::string::npos) {
            ERROR("Could not figure out file name format2, skipping '%s'", file_str.c_str());
            continue;
        }
        auto ds_str = file_str.substr(first_dot + 1, second_dot - first_dot - 1);
        auto binding_str = file_str.substr(second_dot + 1);

        FILE *f_handle = fopen(file_str.c_str(), "r");
        if (f_handle == nullptr) {
            ERROR("Could not open file, skipping '%s'", file_str.c_str());
            continue;
        }

        fseek(f_handle, 0, SEEK_END);
        auto size = ftell(f_handle);
        fseek(f_handle, 0, SEEK_SET);

        void *buffer = malloc(size);
        if (buffer == nullptr) {
            ERROR("Could not malloc buffer, skipping '%s'", file_str.c_str());
            continue;
        }
        buffers.push_back(buffer);

        size_t byte_read = 0;
        while (byte_read != size) {
            byte_read += fread(&(((char *)buffer)[byte_read]), sizeof(char), size - byte_read, f_handle);
        }

        fclose(f_handle);

        buffers_file.AddBuffer(std::stoi(ds_str), std::stoi(binding_str), size, buffer);
    }

    std::string filename = prefix_str;
    if (!buffers_file.WriteToFile(filename.c_str())) {
        ERROR("Could not write buffers to file");
        return -1;
    }

    for (auto buffer : buffers) {
        free(buffer);
    }

    return 0;
}
