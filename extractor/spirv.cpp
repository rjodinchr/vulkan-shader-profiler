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

#include "common/common.hpp"
#include "source/opt/pass.h"
#include "spirv/unified1/NonSemanticVkspReflection.h"

#include <filesystem>

namespace vksp {

class InsertVkspReflectInfoPass : public spvtools::opt::Pass {
private:
    uint32_t GetStringId(const char *string)
    {
        auto string_vector = spvtools::utils::MakeVector(string);
        uint32_t id = context()->TakeNextId();
        auto inst = new spvtools::opt::Instruction(
            context(), spv::Op::OpString, 0u, id, { { SPV_OPERAND_TYPE_LITERAL_STRING, string_vector } });
        context()->module()->AddGlobalValue(std::unique_ptr<spvtools::opt::Instruction>(inst));
        return id;
    }

public:
    InsertVkspReflectInfoPass(std::vector<vksp_push_constant> *pc, std::vector<vksp_descriptor_set> *ds,
        std::vector<vksp_specialization_map_entry> *me, vksp_configuration *config)
        : pc_(pc)
        , ds_(ds)
        , me_(me)
        , config_(config)
    {
    }
    const char *name() const override { return "insert-vksp-reflect-info"; }
    Status Process() override
    {
        auto module = context()->module();

        std::vector<uint32_t> ext_words = spvtools::utils::MakeVector(VKSP_EXTINST_STR);
        auto ExtInstId = context()->TakeNextId();
        auto ExtInst = new spvtools::opt::Instruction(
            context(), spv::Op::OpExtInstImport, 0u, ExtInstId, { { SPV_OPERAND_TYPE_LITERAL_STRING, ext_words } });
        module->AddExtInstImport(std::unique_ptr<spvtools::opt::Instruction>(ExtInst));

        uint32_t void_ty_id = context()->get_type_mgr()->GetVoidTypeId();
        auto *cst_mgr = context()->get_constant_mgr();

        auto ConfigId = context()->TakeNextId();
        auto ConfigInst = new spvtools::opt::Instruction(context(), spv::Op::OpExtInst, void_ty_id, ConfigId,
            {
                { SPV_OPERAND_TYPE_ID, { ExtInstId } },
                { SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER, { NonSemanticVkspReflectionConfiguration } },
                { SPV_OPERAND_TYPE_ID, { GetStringId(config_->enabledExtensionNames) } },
                { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(config_->specializationInfoDataSize) } },
                { SPV_OPERAND_TYPE_ID, { GetStringId(config_->specializationInfoData) } },
                { SPV_OPERAND_TYPE_ID, { GetStringId(config_->shaderName) } },
                { SPV_OPERAND_TYPE_ID, { GetStringId(config_->entryPoint) } },
                { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(config_->groupCountX) } },
                { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(config_->groupCountY) } },
                { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(config_->groupCountZ) } },
                { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(config_->dispatchId) } },
            });
        module->AddExtInstDebugInfo(std::unique_ptr<spvtools::opt::Instruction>(ConfigInst));

        for (auto &pc : *pc_) {
            auto PcInstId = context()->TakeNextId();
            auto PcInst = new spvtools::opt::Instruction(context(), spv::Op::OpExtInst, void_ty_id, PcInstId,
                {
                    { SPV_OPERAND_TYPE_ID, { ExtInstId } },
                    { SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER, { NonSemanticVkspReflectionPushConstants } },
                    { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(pc.offset) } },
                    { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(pc.size) } },
                    { SPV_OPERAND_TYPE_ID, { GetStringId(pc.pValues) } },
                    { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(pc.stageFlags) } },
                });
            module->AddExtInstDebugInfo(std::unique_ptr<spvtools::opt::Instruction>(PcInst));
        }

        for (auto &ds : *ds_) {
            switch (ds.type) {
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
                auto DsInstId = context()->TakeNextId();
                auto DstInst = new spvtools::opt::Instruction(context(), spv::Op::OpExtInst, void_ty_id, DsInstId,
                    {
                        { SPV_OPERAND_TYPE_ID, { ExtInstId } },
                        { SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
                            { NonSemanticVkspReflectionDescriptorSetBuffer } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.ds) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.binding) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.type) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.flags) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.queueFamilyIndexCount) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.sharingMode) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.size) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.usage) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.range) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.offset) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.memorySize) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.memoryType) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.bindOffset) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.viewFlags) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.buffer.viewFormat) } },
                    });
                module->AddExtInstDebugInfo(std::unique_ptr<spvtools::opt::Instruction>(DstInst));
            } break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
                auto DsInstId = context()->TakeNextId();
                auto DstInst = new spvtools::opt::Instruction(context(), spv::Op::OpExtInst, void_ty_id, DsInstId,
                    {
                        { SPV_OPERAND_TYPE_ID, { ExtInstId } },
                        { SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
                            { NonSemanticVkspReflectionDescriptorSetImage } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.ds) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.binding) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.type) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.imageLayout) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.imageFlags) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.imageType) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.format) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.width) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.height) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.depth) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.mipLevels) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.arrayLayers) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.samples) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.tiling) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.usage) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.sharingMode) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.queueFamilyIndexCount) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.initialLayout) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.aspectMask) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.baseMipLevel) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.levelCount) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.baseArrayLayer) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.layerCount) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.viewFlags) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.viewType) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.viewFormat) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.component_a) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.component_b) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.component_g) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.component_r) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.memorySize) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.memoryType) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.image.bindOffset) } },
                    });
                module->AddExtInstDebugInfo(std::unique_ptr<spvtools::opt::Instruction>(DstInst));
            } break;
            case VK_DESCRIPTOR_TYPE_SAMPLER: {
                auto DsInstId = context()->TakeNextId();
                auto DstInst = new spvtools::opt::Instruction(context(), spv::Op::OpExtInst, void_ty_id, DsInstId,
                    {
                        { SPV_OPERAND_TYPE_ID, { ExtInstId } },
                        { SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
                            { NonSemanticVkspReflectionDescriptorSetSampler } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.ds) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.binding) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.type) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.flags) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.magFilter) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.minFilter) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.mipmapMode) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.addressModeU) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.addressModeV) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.addressModeW) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetFloatConstId(ds.sampler.uMipLodBias) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.anisotropyEnable) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetFloatConstId(ds.sampler.uMaxAnisotropy) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.compareEnable) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.compareOp) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetFloatConstId(ds.sampler.uMinLod) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetFloatConstId(ds.sampler.uMaxLod) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.borderColor) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(ds.sampler.unnormalizedCoordinates) } },
                    });
                module->AddExtInstDebugInfo(std::unique_ptr<spvtools::opt::Instruction>(DstInst));
            } break;
            default:
                break;
            }
        }

        for (auto &me : *me_) {
            auto MapEntryId = context()->TakeNextId();
            auto MapEntryInst = new spvtools::opt::Instruction(context(), spv::Op::OpExtInst, void_ty_id, MapEntryId,
                {
                    { SPV_OPERAND_TYPE_ID, { ExtInstId } },
                    { SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
                        { NonSemanticVkspReflectionSpecializationMapEntry } },
                    { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(me.constantID) } },
                    { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(me.offset) } },
                    { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(me.size) } },
                });
            module->AddExtInstDebugInfo(std::unique_ptr<spvtools::opt::Instruction>(MapEntryInst));
        }

        return Status::SuccessWithChange;
    };

private:
    std::vector<vksp_push_constant> *pc_;
    std::vector<vksp_descriptor_set> *ds_;
    std::vector<vksp_specialization_map_entry> *me_;
    vksp_configuration *config_;
};
}

bool text_to_binary(spv_context context, std::string *shader, spv_binary &binary)
{
    spv_diagnostic diagnostic;
    auto status = spvTextToBinary(context, shader->data(), shader->size(), &binary, &diagnostic);
    if (status != SPV_SUCCESS) {
        ERROR("Error while converting shader from text to binary: %s", diagnostic->error);
        spvDiagnosticDestroy(diagnostic);
        return false;
    }
    return true;
}

bool create_binary(spv_context context, spv_binary input_binary, std::vector<vksp::vksp_push_constant> *pc,
    std::vector<vksp::vksp_descriptor_set> *ds, std::vector<vksp::vksp_specialization_map_entry> *me,
    vksp::vksp_configuration *config, std::vector<uint32_t> &output_binary)
{
    spvtools::Optimizer opt(SPV_ENV_VULKAN_1_3);
    opt.RegisterPass(
        spvtools::Optimizer::PassToken(std::make_unique<vksp::InsertVkspReflectInfoPass>(pc, ds, me, config)));
    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    if (!opt.Run(input_binary->code, input_binary->wordCount, &output_binary, options)) {
        ERROR("Error while running 'CreateVkspReflectInfoPass'");
        return false;
    }
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

bool store_shader_binary_in_output(spv_context context, spv_binary input_binary,
    std::vector<vksp::vksp_push_constant> *pc, std::vector<vksp::vksp_descriptor_set> *ds,
    std::vector<vksp::vksp_specialization_map_entry> *me, vksp::vksp_configuration *config, const char *output_filename,
    bool binary_output)
{
    std::vector<uint32_t> binary;
    if (!create_binary(context, input_binary, pc, ds, me, config, binary)) {
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

extern "C" bool store_shader_in_output(std::string *shader, std::vector<vksp::vksp_push_constant> *pc,
    std::vector<vksp::vksp_descriptor_set> *ds, std::vector<vksp::vksp_specialization_map_entry> *me,
    vksp::vksp_configuration *config, const char *output_filename, bool binary_output)
{
    spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_3);
    spv_binary binary;
    if (!text_to_binary(context, shader, binary)) {
        ERROR("Could not convert shader to binary");
        return false;
    }

    auto ret = store_shader_binary_in_output(context, binary, pc, ds, me, config, output_filename, binary_output);

    spvBinaryDestroy(binary);
    return ret;
}

extern "C" bool store_shader_buffer_in_output(std::vector<char> *shader_buffer,
    std::vector<vksp::vksp_push_constant> *pc, std::vector<vksp::vksp_descriptor_set> *ds,
    std::vector<vksp::vksp_specialization_map_entry> *me, vksp::vksp_configuration *config, const char *output_filename,
    bool binary_output)
{
    spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_3);
    spv_binary_t binary
        = { .code = (uint32_t *)shader_buffer->data(), .wordCount = shader_buffer->size() / sizeof(uint32_t) };

    return store_shader_binary_in_output(context, &binary, pc, ds, me, config, output_filename, binary_output);
}

extern "C" bool read_shader_buffer(std::string *gShaderFile, std::vector<char> *shader_buffer)
{
    if (!std::filesystem::exists(*gShaderFile)) {
        return false;
    }
    FILE *file = fopen(gShaderFile->c_str(), "r");
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    shader_buffer->resize(file_size);
    size_t size_read = 0;
    do {
        size_read += fread(&(shader_buffer->data()[size_read]), 1, file_size - size_read, file);
    } while (size_read != file_size);
    fclose(file);
    return true;
}
