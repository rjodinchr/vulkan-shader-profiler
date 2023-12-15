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

#include "spirv-tools/libspirv.h"
#include "spirv-tools/optimizer.hpp"

#include "spirv.hpp"
#include "utils.hpp"

bool create_binary(spv_context context, std::string *shader, std::vector<spvtools::vksp_push_constant> *pc,
    std::vector<spvtools::vksp_descriptor_set> *ds, std::vector<spvtools::vksp_specialization_map_entry> *me,
    spvtools::vksp_configuration *config, std::vector<uint32_t> &binary)
{
    spv_diagnostic diagnostic;
    spv_binary tmp_binary;
    auto status = spvTextToBinary(context, shader->data(), shader->size(), &tmp_binary, &diagnostic);
    if (status != SPV_SUCCESS) {
        ERROR("Error while converting shader from text to binary: %s", diagnostic->error);
        spvDiagnosticDestroy(diagnostic);
        return false;
    }

    spvtools::Optimizer opt(SPV_ENV_VULKAN_1_3);
    opt.RegisterPass(spvtools::CreateInsertVkspReflectInfoPass(pc, ds, me, config));
    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    if (!opt.Run(tmp_binary->code, tmp_binary->wordCount, &binary, options)) {
        ERROR("Error while running 'CreateVkspReflectInfoPass'");
        return false;
    }

    spvBinaryDestroy(tmp_binary);
    return true;
}

bool binary_to_text(spv_context context, std::vector<uint32_t> &binary, spv_text &text)
{
    spv_diagnostic diag;
    const uint32_t *code = binary.data();
    const size_t code_size = binary.size();
    spv_result_t spv_result = spvBinaryToText(context, code, code_size,
        SPV_BINARY_TO_TEXT_OPTION_INDENT | SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_COMMENT,
        &text, &diag);
    if (spv_result != SPV_SUCCESS) {
        ERROR("Error while converting shader from binary to text: %s", diag->error);
        spvDiagnosticDestroy(diag);
        return false;
    }
    return true;
}

extern "C" bool store_shader_in_output(std::string *shader, std::vector<spvtools::vksp_push_constant> *pc,
    std::vector<spvtools::vksp_descriptor_set> *ds, std::vector<spvtools::vksp_specialization_map_entry> *me,
    spvtools::vksp_configuration *config, const char *output_filename, bool binary_output)
{
    spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_3);
    std::vector<uint32_t> binary;
    if (!create_binary(context, shader, pc, ds, me, config, binary)) {
        ERROR("Could not create SPIR-V binary from trace");
        return false;
    }

    FILE *output = fopen(output_filename, "w");
    if (binary_output) {
        fwrite(binary.data(), sizeof(uint32_t), binary.size(), output);
    } else {
        spv_text text;
        if (!binary_to_text(context, binary, text)) {
            ERROR("Could not convert shader from binary to text");
            return false;
        }
        fprintf(output, "%s", text->str);
        spvTextDestroy(text);
    }

    fclose(output);

    spvContextDestroy(context);
    return true;
}
