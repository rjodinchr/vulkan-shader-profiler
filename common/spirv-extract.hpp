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

#pragma once

#include "common/common.hpp"
#include "source/opt/pass.h"
#include "spirv/unified1/NonSemanticVkspReflection.h"

#include <spirv-tools/optimizer.hpp>

#define UNDEFINED_ID (UINT32_MAX)

namespace vksp {

class ExtractVkspReflectInfoPass : public spvtools::opt::Pass {
public:
    ExtractVkspReflectInfoPass(std::vector<vksp_push_constant> *pc, std::vector<vksp_descriptor_set> *ds,
        std::vector<vksp_specialization_map_entry> *me, std::vector<vksp_counter> *counters, vksp_configuration *config,
        bool disableCounters)
        : pc_(pc)
        , ds_(ds)
        , me_(me)
        , counters_(counters)
        , config_(config)
        , disableCounters_(disableCounters)
    {
    }
    const char *name() const override { return "extract-vksp-reflect-info"; }
    Status Process() override
    {
        auto module = context()->module();
        uint32_t ext_inst_id = module->GetExtInstImportId(VKSP_EXTINST_STR);
        int32_t descriptor_set_0_max_binding = -1;
        std::map<uint32_t, uint32_t> id_to_descriptor_set;
        std::map<uint32_t, uint32_t> id_to_binding;
        std::vector<spvtools::opt::Instruction *> start_counters;
        std::vector<spvtools::opt::Instruction *> stop_counters;

        module->ForEachInst([this, ext_inst_id, &id_to_descriptor_set, &id_to_binding, &descriptor_set_0_max_binding,
                                &start_counters, &stop_counters](spvtools::opt::Instruction *inst) {
            ParseInstruction(inst, ext_inst_id, id_to_descriptor_set, id_to_binding, descriptor_set_0_max_binding,
                start_counters, stop_counters);
        });

        if (disableCounters_) {
            return Status::SuccessWithoutChange;
        }

        context()->AddExtension("SPV_KHR_shader_clock");
        context()->AddExtension("SPV_KHR_storage_buffer_storage_class");
        context()->AddCapability(spv::Capability::ShaderClockKHR);
        context()->AddCapability(spv::Capability::Int64);
        context()->AddCapability(spv::Capability::Int64Atomics);

        uint32_t global_counters_ds = 0;
        uint32_t global_counters_binding = descriptor_set_0_max_binding + 1;
        auto counters_size
            = (uint32_t)(sizeof(uint64_t) * (2 + start_counters.size())); // 2 for the number of invocations and the
                                                                          // time of the whole entry point
        ds_->push_back(
            { global_counters_ds, global_counters_binding, (uint32_t)VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER,
                { .buffer = { 0, 0, VK_SHARING_MODE_EXCLUSIVE, counters_size,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, counters_size, 0,
                      counters_size, UINT32_MAX, 0 } } });

        auto *cst_mgr = context()->get_constant_mgr();
        auto *type_mgr = context()->get_type_mgr();

        auto u64_ty = type_mgr->GetIntType(64, 0);
        spvtools::opt::analysis::RuntimeArray run_arr(u64_ty);
        auto u64_run_arr_ty = type_mgr->GetRegisteredType(&run_arr);
        spvtools::opt::analysis::Struct st({ u64_run_arr_ty });
        auto u64_run_arr_st_ty = type_mgr->GetRegisteredType(&st);
        spvtools::opt::analysis::Pointer u64_run_arr_st_ty_ptr(u64_run_arr_st_ty, spv::StorageClass::StorageBuffer);
        auto u64_run_arr_st_ptr_ty = type_mgr->GetRegisteredType(&u64_run_arr_st_ty_ptr);

        spvtools::opt::analysis::Pointer u64_ty_ptr(u64_ty, spv::StorageClass::StorageBuffer);
        auto u64_ptr_ty = type_mgr->GetRegisteredType(&u64_ty_ptr);

        auto counters_ty_id = type_mgr->GetId(u64_run_arr_st_ptr_ty);
        auto u64_ty_id = type_mgr->GetId(u64_ty);
        auto u64_ptr_ty_id = type_mgr->GetId(u64_ptr_ty);
        auto u64_arr_ty_id = type_mgr->GetId(u64_run_arr_ty);
        auto u64_arr_st_ty_id = type_mgr->GetId(u64_run_arr_st_ty);

        uint32_t local_counters_ty_id = UNDEFINED_ID;
        uint32_t u64_private_ptr_ty_id = UNDEFINED_ID;

        if (start_counters.size() > 0) {
            spvtools::opt::analysis::Array arr(u64_ty,
                spvtools::opt::analysis::Array::LengthInfo {
                    cst_mgr->GetUIntConstId((uint32_t)start_counters.size()), { 0, (uint32_t)start_counters.size() } });
            auto u64_arr_ty = type_mgr->GetRegisteredType(&arr);
            spvtools::opt::analysis::Pointer u64_arr_ty_ptr(u64_arr_ty, spv::StorageClass::Private);
            auto u64_arr_ptr_ty = type_mgr->GetRegisteredType(&u64_arr_ty_ptr);
            spvtools::opt::analysis::Pointer u64_ty_ptr_private(u64_ty, spv::StorageClass::Private);
            auto u64_private_ptr_ty = type_mgr->GetRegisteredType(&u64_ty_ptr_private);

            local_counters_ty_id = type_mgr->GetId(u64_arr_ptr_ty);
            u64_private_ptr_ty_id = type_mgr->GetId(u64_private_ptr_ty);
        }

        auto subgroup_scope_id = cst_mgr->GetUIntConstId((uint32_t)spv::Scope::Subgroup);
        auto device_scope_id = cst_mgr->GetUIntConstId((uint32_t)spv::Scope::Device);
        auto acq_rel_mem_sem_id = cst_mgr->GetUIntConstId((uint32_t)spv::MemorySemanticsMask::AcquireRelease);

        uint32_t global_counters_id;
        uint32_t local_counters_id;
        CreateVariables(u64_arr_ty_id, u64_arr_st_ty_id, local_counters_ty_id, counters_ty_id, global_counters_ds,
            global_counters_binding, global_counters_id, local_counters_id);

        bool found = false;
        for (auto &entry_point_inst : module->entry_points()) {
            auto function_name = entry_point_inst.GetOperand(2).AsString();
            if (function_name != std::string(config_->entryPoint)) {
                continue;
            }
            found = true;

            uint32_t read_clock_id;
            spvtools::opt::Function *function;
            CreatePrologue(&entry_point_inst, u64_private_ptr_ty_id, u64_ty_id, subgroup_scope_id, global_counters_id,
                local_counters_id, start_counters, function, read_clock_id);

            function->ForEachInst([this, read_clock_id, u64_ty_id, u64_ptr_ty_id, u64_private_ptr_ty_id,
                                      subgroup_scope_id, device_scope_id, acq_rel_mem_sem_id, global_counters_id,
                                      local_counters_id, &start_counters](spvtools::opt::Instruction *inst) {
                if (inst->opcode() != spv::Op::OpReturn) {
                    return;
                }
                CreateEpilogue(inst, read_clock_id, u64_ty_id, u64_ptr_ty_id, u64_private_ptr_ty_id, subgroup_scope_id,
                    device_scope_id, acq_rel_mem_sem_id, global_counters_id, local_counters_id, start_counters);
            });

            break;
        }
        if (!found) {
            return Status::Failure;
        }

        CreateCounters(
            u64_ty_id, u64_private_ptr_ty_id, subgroup_scope_id, start_counters, stop_counters, local_counters_id);

        return Status::SuccessWithChange;
    };

private:
    int32_t UpdateMaxBinding(uint32_t ds, uint32_t binding, int32_t max_binding)
    {
        if (ds != 0) {
            return max_binding;
        } else {
            return std::max(max_binding, (int32_t)binding);
        }
    }

    void ParseInstruction(spvtools::opt::Instruction *inst, uint32_t ext_inst_id,
        std::map<uint32_t, uint32_t> &id_to_descriptor_set, std::map<uint32_t, uint32_t> &id_to_binding,
        int32_t &descriptor_set_0_max_binding, std::vector<spvtools::opt::Instruction *> &start_counters,
        std::vector<spvtools::opt::Instruction *> &stop_counters)
    {
        uint32_t op_id = 2;
        if (inst->opcode() == spv::Op::OpDecorate) {
            spv::Decoration decoration = (spv::Decoration)inst->GetOperand(1).words[0];
            if (decoration == spv::Decoration::DescriptorSet) {
                auto id = inst->GetOperand(0).AsId();
                auto ds = inst->GetOperand(2).words[0];
                id_to_descriptor_set[id] = ds;
                if (ds == 0 && id_to_binding.count(id) > 0) {
                    descriptor_set_0_max_binding
                        = UpdateMaxBinding(ds, id_to_binding[id], descriptor_set_0_max_binding);
                }
            } else if (decoration == spv::Decoration::Binding) {
                auto id = inst->GetOperand(0).AsId();
                auto binding = inst->GetOperand(2).words[0];
                id_to_binding[id] = binding;
                if (id_to_descriptor_set.count(id) > 0) {
                    descriptor_set_0_max_binding
                        = UpdateMaxBinding(id_to_descriptor_set[id], binding, descriptor_set_0_max_binding);
                }
            }
            return;
        } else if (inst->opcode() != spv::Op::OpExtInst || ext_inst_id != inst->GetOperand(op_id++).AsId()) {
            return;
        }

        auto vksp_inst = inst->GetOperand(op_id++).words[0];
        switch (vksp_inst) {
        case NonSemanticVkspReflectionConfiguration:
            config_->enabledExtensionNames = strdup(inst->GetOperand(op_id++).AsString().c_str());
            config_->specializationInfoDataSize = inst->GetOperand(op_id++).words[0];
            config_->specializationInfoData = strdup(inst->GetOperand(op_id++).AsString().c_str());
            config_->shaderName = strdup(inst->GetOperand(op_id++).AsString().c_str());
            config_->entryPoint = strdup(inst->GetOperand(op_id++).AsString().c_str());
            config_->groupCountX = inst->GetOperand(op_id++).words[0];
            config_->groupCountY = inst->GetOperand(op_id++).words[0];
            config_->groupCountZ = inst->GetOperand(op_id++).words[0];
            config_->dispatchId = inst->GetOperand(op_id++).words[0];
            break;
        case NonSemanticVkspReflectionDescriptorSetBuffer: {
            vksp_descriptor_set ds;
            ds.ds = inst->GetOperand(op_id++).words[0];
            ds.binding = inst->GetOperand(op_id++).words[0];
            ds.type = inst->GetOperand(op_id++).words[0];
            ds.buffer.flags = inst->GetOperand(op_id++).words[0];
            ds.buffer.queueFamilyIndexCount = inst->GetOperand(op_id++).words[0];
            ds.buffer.sharingMode = inst->GetOperand(op_id++).words[0];
            ds.buffer.size = inst->GetOperand(op_id++).words[0];
            ds.buffer.usage = inst->GetOperand(op_id++).words[0];
            ds.buffer.range = inst->GetOperand(op_id++).words[0];
            ds.buffer.offset = inst->GetOperand(op_id++).words[0];
            ds.buffer.memorySize = inst->GetOperand(op_id++).words[0];
            ds.buffer.memoryType = inst->GetOperand(op_id++).words[0];
            ds.buffer.bindOffset = inst->GetOperand(op_id++).words[0];
            ds.buffer.viewFlags = inst->GetOperand(op_id++).words[0];
            ds.buffer.viewFormat = inst->GetOperand(op_id++).words[0];
            ds_->push_back(ds);
            descriptor_set_0_max_binding = UpdateMaxBinding(ds.ds, ds.binding, descriptor_set_0_max_binding);
        } break;
        case NonSemanticVkspReflectionDescriptorSetImage: {
            vksp_descriptor_set ds;
            ds.ds = inst->GetOperand(op_id++).words[0];
            ds.binding = inst->GetOperand(op_id++).words[0];
            ds.type = inst->GetOperand(op_id++).words[0];
            ds.image.imageLayout = inst->GetOperand(op_id++).words[0];
            ds.image.imageFlags = inst->GetOperand(op_id++).words[0];
            ds.image.imageType = inst->GetOperand(op_id++).words[0];
            ds.image.format = inst->GetOperand(op_id++).words[0];
            ds.image.width = inst->GetOperand(op_id++).words[0];
            ds.image.height = inst->GetOperand(op_id++).words[0];
            ds.image.depth = inst->GetOperand(op_id++).words[0];
            ds.image.mipLevels = inst->GetOperand(op_id++).words[0];
            ds.image.arrayLayers = inst->GetOperand(op_id++).words[0];
            ds.image.samples = inst->GetOperand(op_id++).words[0];
            ds.image.tiling = inst->GetOperand(op_id++).words[0];
            ds.image.usage = inst->GetOperand(op_id++).words[0];
            ds.image.sharingMode = inst->GetOperand(op_id++).words[0];
            ds.image.queueFamilyIndexCount = inst->GetOperand(op_id++).words[0];
            ds.image.initialLayout = inst->GetOperand(op_id++).words[0];
            ds.image.aspectMask = inst->GetOperand(op_id++).words[0];
            ds.image.baseMipLevel = inst->GetOperand(op_id++).words[0];
            ds.image.levelCount = inst->GetOperand(op_id++).words[0];
            ds.image.baseArrayLayer = inst->GetOperand(op_id++).words[0];
            ds.image.layerCount = inst->GetOperand(op_id++).words[0];
            ds.image.viewFlags = inst->GetOperand(op_id++).words[0];
            ds.image.viewType = inst->GetOperand(op_id++).words[0];
            ds.image.viewFormat = inst->GetOperand(op_id++).words[0];
            ds.image.component_a = inst->GetOperand(op_id++).words[0];
            ds.image.component_b = inst->GetOperand(op_id++).words[0];
            ds.image.component_g = inst->GetOperand(op_id++).words[0];
            ds.image.component_r = inst->GetOperand(op_id++).words[0];
            ds.image.memorySize = inst->GetOperand(op_id++).words[0];
            ds.image.memoryType = inst->GetOperand(op_id++).words[0];
            ds.image.bindOffset = inst->GetOperand(op_id++).words[0];
            ds_->push_back(ds);
            descriptor_set_0_max_binding = UpdateMaxBinding(ds.ds, ds.binding, descriptor_set_0_max_binding);
        } break;
        case NonSemanticVkspReflectionDescriptorSetSampler: {
            vksp_descriptor_set ds;
            ds.ds = inst->GetOperand(op_id++).words[0];
            ds.binding = inst->GetOperand(op_id++).words[0];
            ds.type = inst->GetOperand(op_id++).words[0];
            ds.sampler.flags = inst->GetOperand(op_id++).words[0];
            ds.sampler.magFilter = inst->GetOperand(op_id++).words[0];
            ds.sampler.minFilter = inst->GetOperand(op_id++).words[0];
            ds.sampler.mipmapMode = inst->GetOperand(op_id++).words[0];
            ds.sampler.addressModeU = inst->GetOperand(op_id++).words[0];
            ds.sampler.addressModeV = inst->GetOperand(op_id++).words[0];
            ds.sampler.addressModeW = inst->GetOperand(op_id++).words[0];
            ds.sampler.uMipLodBias = inst->GetOperand(op_id++).words[0];
            ds.sampler.anisotropyEnable = inst->GetOperand(op_id++).words[0];
            ds.sampler.uMaxAnisotropy = inst->GetOperand(op_id++).words[0];
            ds.sampler.compareEnable = inst->GetOperand(op_id++).words[0];
            ds.sampler.compareOp = inst->GetOperand(op_id++).words[0];
            ds.sampler.uMinLod = inst->GetOperand(op_id++).words[0];
            ds.sampler.uMaxLod = inst->GetOperand(op_id++).words[0];
            ds.sampler.borderColor = inst->GetOperand(op_id++).words[0];
            ds.sampler.unnormalizedCoordinates = inst->GetOperand(op_id++).words[0];
            ds_->push_back(ds);
            descriptor_set_0_max_binding = UpdateMaxBinding(ds.ds, ds.binding, descriptor_set_0_max_binding);
        } break;
        case NonSemanticVkspReflectionPushConstants:
            vksp_push_constant pc;
            pc.offset = inst->GetOperand(op_id++).words[0];
            pc.size = inst->GetOperand(op_id++).words[0];
            pc.pValues = strdup(inst->GetOperand(op_id++).AsString().c_str());
            pc.stageFlags = inst->GetOperand(op_id++).words[0];
            pc_->push_back(pc);
            break;
        case NonSemanticVkspReflectionSpecializationMapEntry:
            vksp_specialization_map_entry me;
            me.constantID = inst->GetOperand(op_id++).words[0];
            me.offset = inst->GetOperand(op_id++).words[0];
            me.size = inst->GetOperand(op_id++).words[0];
            me_->push_back(me);
            break;
        case NonSemanticVkspReflectionStartCounter:
            start_counters.push_back(inst);
            break;
        case NonSemanticVkspReflectionStopCounter:
            stop_counters.push_back(inst);
            break;
        default:
            break;
        }
    }

    void CreateVariables(uint32_t u64_arr_ty_id, uint32_t u64_arr_st_ty_id, uint32_t local_counters_ty_id,
        uint32_t counters_ty_id, uint32_t global_counters_ds, uint32_t global_counters_binding,
        uint32_t &global_counters_id, uint32_t &local_counters_id)
    {
        auto module = context()->module();

        auto decorate_arr_inst = new spvtools::opt::Instruction(context(), spv::Op::OpDecorate, 0, 0,
            { { SPV_OPERAND_TYPE_ID, { u64_arr_ty_id } },
                { SPV_OPERAND_TYPE_DECORATION, { (uint32_t)spv::Decoration::ArrayStride } },
                { SPV_OPERAND_TYPE_LITERAL_INTEGER, { 8 } } });
        module->AddAnnotationInst(std::unique_ptr<spvtools::opt::Instruction>(decorate_arr_inst));

        auto decorate_member_offset_inst = new spvtools::opt::Instruction(context(), spv::Op::OpMemberDecorate, 0, 0,
            { { SPV_OPERAND_TYPE_ID, { u64_arr_st_ty_id } }, { SPV_OPERAND_TYPE_LITERAL_INTEGER, { 0 } },
                { SPV_OPERAND_TYPE_DECORATION, { (uint32_t)spv::Decoration::Offset } },
                { SPV_OPERAND_TYPE_LITERAL_INTEGER, { 0 } } });
        module->AddAnnotationInst(std::unique_ptr<spvtools::opt::Instruction>(decorate_member_offset_inst));

        auto decorate_arr_st_inst = new spvtools::opt::Instruction(context(), spv::Op::OpDecorate, 0, 0,
            { { SPV_OPERAND_TYPE_ID, { u64_arr_st_ty_id } },
                { SPV_OPERAND_TYPE_DECORATION, { (uint32_t)spv::Decoration::Block } } });
        module->AddAnnotationInst(std::unique_ptr<spvtools::opt::Instruction>(decorate_arr_st_inst));

        if (local_counters_ty_id != UNDEFINED_ID) {
            local_counters_id = context()->TakeNextId();
            auto local_counters_inst = new spvtools::opt::Instruction(context(), spv::Op::OpVariable,
                local_counters_ty_id, local_counters_id,
                { { SPV_OPERAND_TYPE_LITERAL_INTEGER, { (uint32_t)spv::StorageClass::Private } } });
            module->AddGlobalValue(std::unique_ptr<spvtools::opt::Instruction>(local_counters_inst));
        } else {
            local_counters_id = UNDEFINED_ID;
        }

        global_counters_id = context()->TakeNextId();
        auto global_counters_inst
            = new spvtools::opt::Instruction(context(), spv::Op::OpVariable, counters_ty_id, global_counters_id,
                { { SPV_OPERAND_TYPE_LITERAL_INTEGER, { (uint32_t)spv::StorageClass::StorageBuffer } } });
        module->AddGlobalValue(std::unique_ptr<spvtools::opt::Instruction>(global_counters_inst));

        auto counters_descriptor_set_inst = new spvtools::opt::Instruction(context(), spv::Op::OpDecorate, 0, 0,
            { { SPV_OPERAND_TYPE_ID, { global_counters_inst->result_id() } },
                { SPV_OPERAND_TYPE_DECORATION, { (uint32_t)spv::Decoration::DescriptorSet } },
                { SPV_OPERAND_TYPE_LITERAL_INTEGER, { global_counters_ds } } });
        module->AddAnnotationInst(std::unique_ptr<spvtools::opt::Instruction>(counters_descriptor_set_inst));

        auto counters_binding_inst = new spvtools::opt::Instruction(context(), spv::Op::OpDecorate, 0, 0,
            { { SPV_OPERAND_TYPE_ID, { global_counters_inst->result_id() } },
                { SPV_OPERAND_TYPE_DECORATION, { (uint32_t)spv::Decoration::Binding } },
                { SPV_OPERAND_TYPE_LITERAL_INTEGER, { global_counters_binding } } });
        module->AddAnnotationInst(std::unique_ptr<spvtools::opt::Instruction>(counters_binding_inst));
    }

    void CreatePrologue(spvtools::opt::Instruction *entry_point_inst, uint32_t u64_private_ptr_ty_id,
        uint32_t u64_ty_id, uint32_t subgroup_scope_id, uint32_t global_counters_id, uint32_t local_counters_id,
        std::vector<spvtools::opt::Instruction *> &start_counters, spvtools::opt::Function *&function,
        uint32_t &read_clock_id)
    {
        auto *cst_mgr = context()->get_constant_mgr();
        entry_point_inst->AddOperand({ SPV_OPERAND_TYPE_ID, { global_counters_id } });
        if (local_counters_id != UNDEFINED_ID) {
            entry_point_inst->AddOperand({ SPV_OPERAND_TYPE_ID, { local_counters_id } });
        }

        auto function_id = entry_point_inst->GetOperand(1).AsId();
        function = context()->GetFunction(function_id);

        auto &function_first_inst = *function->entry()->begin();

        auto u64_cst0_id = cst_mgr->GetDefiningInstruction(cst_mgr->GetIntConst(0, 64, 0))->result_id();

        for (unsigned i = 0; i < start_counters.size(); i++) {
            auto get_id = context()->TakeNextId();
            auto gep_inst
                = new spvtools::opt::Instruction(context(), spv::Op::OpAccessChain, u64_private_ptr_ty_id, get_id,
                    { { SPV_OPERAND_TYPE_ID, { local_counters_id } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(i) } } });
            gep_inst->InsertBefore(&function_first_inst);

            auto store_inst = new spvtools::opt::Instruction(context(), spv::Op::OpStore, 0, 0,
                { { SPV_OPERAND_TYPE_ID, { get_id } }, { SPV_OPERAND_TYPE_ID, { u64_cst0_id } } });
            store_inst->InsertAfter(gep_inst);
        }

        read_clock_id = context()->TakeNextId();
        auto read_clock_inst = new spvtools::opt::Instruction(context(), spv::Op::OpReadClockKHR, u64_ty_id,
            read_clock_id, { { SPV_OPERAND_TYPE_SCOPE_ID, { subgroup_scope_id } } });
        read_clock_inst->InsertBefore(&function_first_inst);
    }

    void CreateEpilogue(spvtools::opt::Instruction *return_inst, uint32_t read_clock_id, uint32_t u64_ty_id,
        uint32_t u64_ptr_ty_id, uint32_t u64_private_ptr_ty_id, uint32_t subgroup_scope_id, uint32_t device_scope_id,
        uint32_t acq_rel_mem_sem_id, uint32_t global_counters_id, uint32_t local_counters_id,
        std::vector<spvtools::opt::Instruction *> &start_counters)
    {
        auto *cst_mgr = context()->get_constant_mgr();

        auto read_clock_end_id = context()->TakeNextId();
        auto read_clock_end_inst = new spvtools::opt::Instruction(context(), spv::Op::OpReadClockKHR, u64_ty_id,
            read_clock_end_id, { { SPV_OPERAND_TYPE_SCOPE_ID, { subgroup_scope_id } } });
        read_clock_end_inst->InsertBefore(return_inst);

        auto substraction_id = context()->TakeNextId();
        auto substraction_inst = new spvtools::opt::Instruction(context(), spv::Op::OpISub, u64_ty_id, substraction_id,
            { { SPV_OPERAND_TYPE_ID, { read_clock_end_id } }, { SPV_OPERAND_TYPE_ID, { read_clock_id } } });
        substraction_inst->InsertAfter(read_clock_end_inst);

        auto gep_invocations_id = context()->TakeNextId();
        auto gep_invocations_inst = new spvtools::opt::Instruction(context(), spv::Op::OpAccessChain, u64_ptr_ty_id,
            gep_invocations_id,
            { { SPV_OPERAND_TYPE_ID, { global_counters_id } }, { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(0) } },
                { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(0) } } });
        gep_invocations_inst->InsertAfter(substraction_inst);

        auto atomic_incr_id = context()->TakeNextId();
        auto atomic_incr_inst
            = new spvtools::opt::Instruction(context(), spv::Op::OpAtomicIIncrement, u64_ty_id, atomic_incr_id,
                { { SPV_OPERAND_TYPE_ID, { gep_invocations_id } }, { SPV_OPERAND_TYPE_SCOPE_ID, { device_scope_id } },
                    { SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID, { acq_rel_mem_sem_id } } });
        atomic_incr_inst->InsertAfter(gep_invocations_inst);

        auto gep_entrypoint_counter_id = context()->TakeNextId();
        auto gep_entrypoint_counter_inst = new spvtools::opt::Instruction(context(), spv::Op::OpAccessChain,
            u64_ptr_ty_id, gep_entrypoint_counter_id,
            { { SPV_OPERAND_TYPE_ID, { global_counters_id } }, { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(0) } },
                { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(1) } } });
        gep_entrypoint_counter_inst->InsertAfter(atomic_incr_inst);

        auto atomic_add_id = context()->TakeNextId();
        auto atomic_add_inst
            = new spvtools::opt::Instruction(context(), spv::Op::OpAtomicIAdd, u64_ty_id, atomic_add_id,
                { { SPV_OPERAND_TYPE_ID, { gep_entrypoint_counter_id } },
                    { SPV_OPERAND_TYPE_SCOPE_ID, { device_scope_id } },
                    { SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID, { acq_rel_mem_sem_id } },
                    { SPV_OPERAND_TYPE_ID, { substraction_id } } });
        atomic_add_inst->InsertAfter(gep_entrypoint_counter_inst);

        for (unsigned i = 0; i < start_counters.size(); i++) {
            auto gep_id = context()->TakeNextId();
            auto gep_inst
                = new spvtools::opt::Instruction(context(), spv::Op::OpAccessChain, u64_private_ptr_ty_id, gep_id,
                    { { SPV_OPERAND_TYPE_ID, { local_counters_id } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(i) } } });
            gep_inst->InsertAfter(atomic_add_inst);

            auto load_id = context()->TakeNextId();
            auto load_inst = new spvtools::opt::Instruction(
                context(), spv::Op::OpLoad, u64_ty_id, load_id, { { SPV_OPERAND_TYPE_ID, { gep_id } } });
            load_inst->InsertAfter(gep_inst);

            auto gep_atomic_id = context()->TakeNextId();
            auto gep_atomic_inst
                = new spvtools::opt::Instruction(context(), spv::Op::OpAccessChain, u64_ptr_ty_id, gep_atomic_id,
                    { { SPV_OPERAND_TYPE_ID, { global_counters_id } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(0) } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(2 + i) } } });
            gep_atomic_inst->InsertAfter(load_inst);

            atomic_add_id = context()->TakeNextId();
            atomic_add_inst = new spvtools::opt::Instruction(context(), spv::Op::OpAtomicIAdd, u64_ty_id, atomic_add_id,
                { { SPV_OPERAND_TYPE_ID, { gep_atomic_id } }, { SPV_OPERAND_TYPE_SCOPE_ID, { device_scope_id } },
                    { SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID, { acq_rel_mem_sem_id } },
                    { SPV_OPERAND_TYPE_ID, { load_id } } });
            atomic_add_inst->InsertAfter(gep_atomic_inst);
        }
    }

    void CreateCounters(uint32_t u64_ty_id, uint32_t u64_private_ptr_ty_id, uint32_t subgroup_scope_id,
        std::vector<spvtools::opt::Instruction *> &start_counters,
        std::vector<spvtools::opt::Instruction *> &stop_counters, uint32_t local_counters_id)
    {
        auto *cst_mgr = context()->get_constant_mgr();
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> start_counters_id_map;
        uint32_t next_counter_id = 2;

        for (auto *inst : start_counters) {
            const char *counter_name = strdup(inst->GetOperand(4).AsString().c_str());

            auto read_clock_id = context()->TakeNextId();
            auto read_clock_inst = new spvtools::opt::Instruction(context(), spv::Op::OpReadClockKHR, u64_ty_id,
                read_clock_id, { { SPV_OPERAND_TYPE_SCOPE_ID, { subgroup_scope_id } } });
            read_clock_inst->InsertBefore(inst);

            counters_->push_back({ next_counter_id, counter_name });
            start_counters_id_map[inst->result_id()] = std::make_pair(read_clock_id, next_counter_id);
            next_counter_id++;
        }

        for (auto *inst : stop_counters) {
            auto read_clock_ext_inst_id = inst->GetOperand(4).AsId();
            if (start_counters_id_map.count(read_clock_ext_inst_id) == 0) {
                continue;
            }
            auto pair = start_counters_id_map[read_clock_ext_inst_id];
            auto read_clock_id = pair.first;
            auto counters_var_index = pair.second;

            auto read_clock_end_id = context()->TakeNextId();
            auto read_clock_end_inst = new spvtools::opt::Instruction(context(), spv::Op::OpReadClockKHR, u64_ty_id,
                read_clock_end_id, { { SPV_OPERAND_TYPE_SCOPE_ID, { subgroup_scope_id } } });
            read_clock_end_inst->InsertAfter(inst);

            auto substraction_id = context()->TakeNextId();
            auto substraction_inst
                = new spvtools::opt::Instruction(context(), spv::Op::OpISub, u64_ty_id, substraction_id,
                    { { SPV_OPERAND_TYPE_ID, { read_clock_end_id } }, { SPV_OPERAND_TYPE_ID, { read_clock_id } } });
            substraction_inst->InsertAfter(read_clock_end_inst);

            auto gep_id = context()->TakeNextId();
            auto gep_inst
                = new spvtools::opt::Instruction(context(), spv::Op::OpAccessChain, u64_private_ptr_ty_id, gep_id,
                    { { SPV_OPERAND_TYPE_ID, { local_counters_id } },
                        { SPV_OPERAND_TYPE_ID, { cst_mgr->GetUIntConstId(counters_var_index - 2) } } });
            gep_inst->InsertAfter(substraction_inst);

            auto load_id = context()->TakeNextId();
            auto load_inst = new spvtools::opt::Instruction(
                context(), spv::Op::OpLoad, u64_ty_id, load_id, { { SPV_OPERAND_TYPE_ID, { gep_id } } });
            load_inst->InsertAfter(gep_inst);

            auto add_id = context()->TakeNextId();
            auto add_inst = new spvtools::opt::Instruction(context(), spv::Op::OpIAdd, u64_ty_id, add_id,
                { { SPV_OPERAND_TYPE_ID, { load_id } }, { SPV_OPERAND_TYPE_ID, { substraction_id } } });
            add_inst->InsertAfter(load_inst);

            auto store_inst = new spvtools::opt::Instruction(context(), spv::Op::OpStore, 0, 0,
                { { SPV_OPERAND_TYPE_ID, { gep_id } }, { SPV_OPERAND_TYPE_ID, { add_id } } });
            store_inst->InsertAfter(add_inst);
        }
    }

    std::vector<vksp_push_constant> *pc_;
    std::vector<vksp_descriptor_set> *ds_;
    std::vector<vksp_specialization_map_entry> *me_;
    std::vector<vksp_counter> *counters_;
    vksp_configuration *config_;
    bool disableCounters_;
};

bool extract_from_input(const char *filename, spv_target_env &spv_target_env, bool disable_counters, bool verbose,
    std::vector<uint32_t> &shader, std::vector<vksp::vksp_descriptor_set> &ds,
    std::vector<vksp::vksp_push_constant> &pc, std::vector<vksp::vksp_specialization_map_entry> &me,
    std::vector<vksp::vksp_counter> &counters, vksp::vksp_configuration &config)
{
    FILE *input = fopen(filename, "r");
    fseek(input, 0, SEEK_END);
    size_t input_size = ftell(input);
    fseek(input, 0, SEEK_SET);
    std::vector<char> input_buffer(input_size);
    size_t size_read = 0;
    do {
        size_read += fread(&input_buffer.data()[size_read], 1, input_size - size_read, input);
    } while (size_read != input_size);
    fclose(input);

    const uint32_t spirv_magic = 0x07230203;
    spv_context context = spvContextCreate(spv_target_env);
    uint32_t *binary = (uint32_t *)input_buffer.data();
    size_t size = input_size / sizeof(uint32_t);
    spv_binary tmp_binary;
    if (*(uint32_t *)input_buffer.data() != spirv_magic) {
        spv_diagnostic diagnostic;
        auto status = spvTextToBinary(context, input_buffer.data(), input_size, &tmp_binary, &diagnostic);
        if (status != SPV_SUCCESS) {
            ERROR("Error while converting shader from text to binary: %s", diagnostic->error);
            spvDiagnosticDestroy(diagnostic);
            return false;
        }

        binary = tmp_binary->code;
        size = tmp_binary->wordCount;
    }

    spvtools::Optimizer opt(SPV_ENV_VULKAN_1_3);
    opt.RegisterPass(spvtools::Optimizer::PassToken(
        std::make_unique<vksp::ExtractVkspReflectInfoPass>(&pc, &ds, &me, &counters, &config, disable_counters)));
    opt.RegisterPass(spvtools::CreateStripReflectInfoPass());
    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    if (!opt.Run(binary, size, &shader, options)) {
        ERROR("Error while running 'CreateVkspReflectInfoPass' and 'CreateStripReflectInfoPass'");
        return false;
    }

    if (verbose) {
        spv_text text;
        spv_diagnostic diag;
        spv_result_t spv_result = spvBinaryToText(context, shader.data(), shader.size(),
            SPV_BINARY_TO_TEXT_OPTION_INDENT | SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES
                | SPV_BINARY_TO_TEXT_OPTION_COMMENT,
            &text, &diag);
        if (spv_result == SPV_SUCCESS) {
            PRINT("Shader:\n%s", text->str);
            spvTextDestroy(text);
        } else {
            ERROR("Could not convert shader from binary to text: %s", diag->error);
            spvDiagnosticDestroy(diag);
        }
    }

    spvContextDestroy(context);

    return true;
}
}
