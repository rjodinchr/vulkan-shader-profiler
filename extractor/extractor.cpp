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

#include "perfetto/trace_processor/read_trace.h"
#include "perfetto/trace_processor/trace_processor.h"

#include "spirv-tools/optimizer.hpp"
#include "vulkan/vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <map>
#include <set>
#include <string>

#include "common/common.hpp"
#include "spirv.hpp"

using namespace perfetto::trace_processor;
using namespace spvtools;

#define CHECK(statement, message, ...)                                                                                 \
    do {                                                                                                               \
        if (!(statement)) {                                                                                            \
            ERROR(message, ##__VA_ARGS__);                                                                             \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define EXECUTE_QUERY_NO_CHECK(it, tp, query)                                                                          \
    auto it = tp->ExecuteQuery(query);                                                                                 \
    if (!it.Status().ok()) {                                                                                           \
        ERROR("Error while executing query: '%s' ('%s')", query.c_str(), it.Status().c_message());                     \
        return false;                                                                                                  \
    }

#define EXECUTE_QUERY(it, tp, query)                                                                                   \
    EXECUTE_QUERY_NO_CHECK(it, tp, query);                                                                             \
    if (!it.Next()) {                                                                                                  \
        ERROR("Error: empty query result ('%s')", query.c_str());                                                      \
        return false;                                                                                                  \
    }

#define GET_INT_VALUE(tp, arg_set_id, key, val)                                                                        \
    do {                                                                                                               \
        std::string _query = "SELECT int_value FROM args WHERE args.arg_set_id = " + std::to_string(arg_set_id)        \
            + " AND args.key = '" key "'";                                                                             \
        EXECUTE_QUERY(_it, tp, _query);                                                                                \
        val = _it.Get(0).AsLong();                                                                                     \
        assert(!_it.Next());                                                                                           \
    } while (0)
#define GET_STR_VALUE(tp, arg_set_id, key, val)                                                                        \
    do {                                                                                                               \
        std::string _query = "SELECT string_value FROM args WHERE args.arg_set_id = " + std::to_string(arg_set_id)     \
            + " AND args.key = '" key "'";                                                                             \
        EXECUTE_QUERY(_it, tp, _query);                                                                                \
        val = strdup(_it.Get(0).AsString());                                                                           \
        assert(!_it.Next());                                                                                           \
    } while (0)
#define GET_FLOAT_VALUE(tp, arg_set_id, key, val)                                                                      \
    do {                                                                                                               \
        std::string _query = "SELECT real_value FROM args WHERE args.arg_set_id = " + std::to_string(arg_set_id)       \
            + " AND args.key = '" key "'";                                                                             \
        EXECUTE_QUERY(_it, tp, _query);                                                                                \
        val = _it.Get(0).AsDouble();                                                                                   \
        assert(!_it.Next());                                                                                           \
    } while (0)

static bool gVerbose = false, gBinary = false;
static uint64_t gDispatchId = UINT64_MAX;
static std::string gInput = "", gOutput = "";
static std::string gShaderFile = "";

void progress_callback(uint64_t size)
{
    if (gVerbose) {
        printf("\r%lu kB", size);
    }
    return;
}

std::unique_ptr<TraceProcessor> initialize_database()
{
    Config config;
    config.sorting_mode = SortingMode::kForceFullSort;
    config.ingest_ftrace_in_raw_table = false;

    auto tp = TraceProcessor::CreateInstance(config);

    PRINT("Reading input file ('%s') ", gInput.c_str());
    auto status = ReadTrace(tp.get(), gInput.c_str(), progress_callback);
    if (gVerbose) {
        printf("\n");
    }
    if (!status.ok()) {
        ERROR("Error while reading '%s' ('%s')", gInput.c_str(), status.c_message());
        return nullptr;
    }

    return tp;
}

bool get_dispatch_compute_and_commandBuffer_from_dispatchId(TraceProcessor *tp, uint64_t dispatchId, uint64_t &dispatch,
    uint64_t &compute, uint64_t &commandBuffer, vksp::vksp_configuration &config)
{
    std::string query = "SELECT arg_set_id FROM args WHERE args.key = 'debug.dispatchId' AND args.int_value = "
        + std::to_string(dispatchId);
    EXECUTE_QUERY(it, tp, query);

    dispatch = it.Get(0).AsLong();
    if (!it.Next()) {
        ERROR("Error: only 1 result, 2 expected");
        return false;
    }
    compute = it.Get(0).AsLong();

    GET_INT_VALUE(tp, dispatch, "debug.commandBuffer", commandBuffer);
    GET_INT_VALUE(tp, dispatch, "debug.groupCountX", config.groupCountX);
    GET_INT_VALUE(tp, dispatch, "debug.groupCountY", config.groupCountY);
    GET_INT_VALUE(tp, dispatch, "debug.groupCountZ", config.groupCountZ);

    GET_STR_VALUE(tp, compute, "debug.shader_name", config.entryPoint);

    config.dispatchId = dispatchId;

    assert(!it.Next());
    return true;
}

bool get_max_timestamp(TraceProcessor *tp, uint64_t dispatch, uint64_t &max_timestamp)
{
    std::string query = "SELECT ts FROM slice WHERE slice.arg_set_id = " + std::to_string(dispatch);
    EXECUTE_QUERY(it, tp, query);
    max_timestamp = it.Get(0).AsLong();
    assert(!it.Next());
    return true;
}

bool get_min_timestamp(TraceProcessor *tp, uint64_t commandBUffer, uint64_t max_timestamp, uint64_t &min_timestamp)
{
    std::string query = "SELECT MAX(ts) FROM slice WHERE slice.name = 'vkBeginCommandBuffer' AND slice.ts < "
        + std::to_string(max_timestamp);
    EXECUTE_QUERY(it, tp, query);
    min_timestamp = it.Get(0).AsLong();
    assert(!it.Next());
    return true;
}

bool get_shader_and_device_from_compute(TraceProcessor *tp, uint64_t compute, std::string &shader,
    std::vector<char> &shader_buffer, uint64_t &device, vksp::vksp_configuration &config)
{
    GET_STR_VALUE(tp, compute, "debug.shader", config.shaderName);

    std::string query = "SELECT arg_set_id FROM slice WHERE slice.name = 'vkCreateShaderModule' AND '"
        + std::string(config.shaderName)
        + "' = (SELECT string_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.shader')";
    EXECUTE_QUERY(it, tp, query);
    uint64_t shader_device_arg_set_id = it.Get(0).AsLong();
    assert(!it.Next());

    GET_INT_VALUE(tp, shader_device_arg_set_id, "debug.device", device);

    if (gShaderFile != "") {
        if (read_shader_buffer(&gShaderFile, &shader_buffer)) {
            return true;
        } else {
            ERROR("'%s' does not exist, get shader code from perfetto file", gShaderFile.c_str());
            // reset gShaderFile to make sure the extractor is using the string, not the buffer;
            gShaderFile = "";
        }
    }

    std::string query2 = "SELECT arg_set_id FROM slice WHERE slice.name = 'vkCreateShaderModule-text' AND '"
        + std::string(config.shaderName)
        + "' = (SELECT string_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.shader') "
          "ORDER BY ts ASC";
    EXECUTE_QUERY(it2, tp, query2);

    shader = "";
    do {
        uint64_t shader_text_arg_set_id = it2.Get(0).AsLong();
        char *shaderCStr;
        GET_STR_VALUE(tp, shader_text_arg_set_id, "debug.text", shaderCStr);

        shader += std::string(shaderCStr);
        free(shaderCStr);
    } while (it2.Next());

    return true;
}

bool get_extensions_from_device(TraceProcessor *tp, uint64_t device, const char *&enabledExtensionNames)
{
    std::string query = "SELECT arg_set_id FROM slice WHERE slice.name = 'vkCreateDevice-enabled' AND "
        + std::to_string(device)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.device' )";
    EXECUTE_QUERY(it, tp, query);
    uint64_t arg_set_id = it.Get(0).AsLong();
    assert(!it.Next());

    GET_STR_VALUE(tp, arg_set_id, "debug.ppEnabledExtensionNames", enabledExtensionNames);

    return true;
}

bool get_push_constants(TraceProcessor *tp, uint64_t commandBuffer, uint64_t max_timestamp, uint64_t min_timestamp,
    std::vector<vksp::vksp_push_constant> &push_constants_vector)
{
    std::string query = "SELECT arg_set_id, ts FROM slice WHERE slice.name = 'vkCmdPushConstants' AND slice.ts > "
        + std::to_string(min_timestamp) + " AND slice.ts < " + std::to_string(max_timestamp) + " AND "
        + std::to_string(commandBuffer)
        + " = ( SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = "
          "'debug.commandBuffer') ORDER BY ts DESC";
    EXECUTE_QUERY_NO_CHECK(it, tp, query);

    std::set<uint32_t> offsetWritten;
    while (it.Next()) {
        uint64_t arg_set_id = it.Get(0).AsLong();
        vksp::vksp_push_constant pc;
        GET_INT_VALUE(tp, arg_set_id, "debug.offset", pc.offset);
        GET_INT_VALUE(tp, arg_set_id, "debug.size", pc.size);
        GET_INT_VALUE(tp, arg_set_id, "debug.stageFlags", pc.stageFlags);
        GET_STR_VALUE(tp, arg_set_id, "debug.pValues", pc.pValues);

        bool pcToRegister = false;
        for (uint32_t i = pc.offset; i < pc.offset + pc.size; i++) {
            if (offsetWritten.count(i) == 0) {
                pcToRegister = true;
                offsetWritten.insert(i);
            }
        }
        if (pcToRegister) {
            push_constants_vector.push_back(pc);
        }
    }

    return true;
}

bool get_buffer_descriptor_set(
    TraceProcessor *tp, uint64_t write_arg_set_id, uint64_t write_timestamp, vksp::vksp_descriptor_set &ds)
{
    uint64_t buffer;
    switch (ds.type) {
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: {
        uint64_t buffer_view;
        GET_INT_VALUE(tp, write_arg_set_id, "debug.bufferView", buffer_view);

        std::string query_buffer_view
            = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkCreateBufferView-result' AND slice.ts < "
            + std::to_string(write_timestamp) + " AND " + std::to_string(buffer_view)
            + "= (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.pView')";
        EXECUTE_QUERY(it_buffer_view, tp, query_buffer_view);
        uint64_t buffer_view_id = it_buffer_view.Get(0).AsLong();
        assert(!it_buffer_view.Next());

        GET_INT_VALUE(tp, buffer_view_id, "debug.buffer", buffer);
        GET_INT_VALUE(tp, buffer_view_id, "debug.flags", ds.buffer.viewFlags);
        GET_INT_VALUE(tp, buffer_view_id, "debug.format", ds.buffer.viewFormat);
        GET_INT_VALUE(tp, buffer_view_id, "debug.offset", ds.buffer.offset);
        GET_INT_VALUE(tp, buffer_view_id, "debug.range", ds.buffer.range);
    } break;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
        GET_INT_VALUE(tp, write_arg_set_id, "debug.buffer", buffer);
        GET_INT_VALUE(tp, write_arg_set_id, "debug.range", ds.buffer.range);
        GET_INT_VALUE(tp, write_arg_set_id, "debug.offset", ds.buffer.offset);
    } break;
    default:
        ERROR("Unexpected descriptor set type");
        return false;
    }

    std::string query
        = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkCreateBuffer-result' AND slice.ts < "
        + std::to_string(write_timestamp) + " AND " + std::to_string(buffer)
        + " = ( SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = "
          "'debug.buffer')";
    EXECUTE_QUERY(it, tp, query);
    uint64_t buffer_arg_set_id = it.Get(0).AsLong();
    assert(!it.Next());

    GET_INT_VALUE(tp, buffer_arg_set_id, "debug.size", ds.buffer.size);
    GET_INT_VALUE(tp, buffer_arg_set_id, "debug.flags", ds.buffer.flags);
    GET_INT_VALUE(tp, buffer_arg_set_id, "debug.queueFamilyIndexCount", ds.buffer.queueFamilyIndexCount);
    GET_INT_VALUE(tp, buffer_arg_set_id, "debug.sharingMode", ds.buffer.sharingMode);
    GET_INT_VALUE(tp, buffer_arg_set_id, "debug.usage", ds.buffer.usage);

    std::string query2 = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkBindBufferMemory' AND slice.ts < "
        + std::to_string(write_timestamp) + " AND " + std::to_string(buffer)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.buffer')";
    EXECUTE_QUERY(it2, tp, query2);
    uint64_t bind_arg_set_id = it2.Get(0).AsLong();
    assert(!it2.Next());

    uint64_t memory;
    GET_INT_VALUE(tp, bind_arg_set_id, "debug.memory", memory);
    GET_INT_VALUE(tp, bind_arg_set_id, "debug.offset", ds.buffer.bindOffset);

    std::string query3
        = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkAllocateMemory-mem' AND slice.ts < "
        + std::to_string(write_timestamp) + " AND " + std::to_string(memory)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.memory')";
    EXECUTE_QUERY(it3, tp, query3);
    uint64_t allocate_arg_set_id = it3.Get(0).AsLong();
    assert(!it3.Next());

    GET_INT_VALUE(tp, allocate_arg_set_id, "debug.size", ds.buffer.memorySize);
    GET_INT_VALUE(tp, allocate_arg_set_id, "debug.type", ds.buffer.memoryType);

    return true;
}

bool get_image_descriptor_set(
    TraceProcessor *tp, uint64_t write_arg_set_id, uint64_t write_timestamp, vksp::vksp_descriptor_set &ds)
{
    uint64_t image_view;
    GET_INT_VALUE(tp, write_arg_set_id, "debug.imageView", image_view);
    GET_INT_VALUE(tp, write_arg_set_id, "debug.imageLayout", ds.image.imageLayout);

    std::string query
        = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkCreateImageView-result' AND slice.ts < "
        + std::to_string(write_timestamp) + " AND " + std::to_string(image_view)
        + " = ( SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = "
          "'debug.pView')";
    EXECUTE_QUERY(it, tp, query);
    uint64_t image_view_arg_set_id = it.Get(0).AsLong();
    assert(!it.Next());

    uint64_t image;
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.image", image);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.aspectMask", ds.image.aspectMask);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.baseArrayLayer", ds.image.baseArrayLayer);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.baseMipLevel", ds.image.baseMipLevel);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.components_a", ds.image.component_a);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.components_b", ds.image.component_b);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.components_g", ds.image.component_g);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.components_r", ds.image.component_r);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.flags", ds.image.viewFlags);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.format", ds.image.viewFormat);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.layerCount", ds.image.layerCount);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.levelCount", ds.image.levelCount);
    GET_INT_VALUE(tp, image_view_arg_set_id, "debug.viewType", ds.image.viewType);

    std::string query2
        = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkCreateImage-result' AND slice.ts < "
        + std::to_string(write_timestamp) + " AND " + std::to_string(image)
        + " = ( SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = "
          "'debug.image')";
    EXECUTE_QUERY(it2, tp, query2);
    uint64_t image_arg_set_id = it2.Get(0).AsLong();
    assert(!it2.Next());

    GET_INT_VALUE(tp, image_arg_set_id, "debug.arrayLayers", ds.image.arrayLayers);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.depth", ds.image.depth);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.flags", ds.image.imageFlags);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.format", ds.image.format);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.height", ds.image.height);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.imageType", ds.image.imageType);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.initialLayout", ds.image.initialLayout);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.mipLevels", ds.image.mipLevels);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.queueFamilyIndexCount", ds.image.queueFamilyIndexCount);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.samples", ds.image.samples);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.sharingMode", ds.image.sharingMode);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.tiling", ds.image.tiling);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.usage", ds.image.usage);
    GET_INT_VALUE(tp, image_arg_set_id, "debug.width", ds.image.width);

    std::string query3 = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkBindImageMemory' AND slice.ts < "
        + std::to_string(write_timestamp) + " AND " + std::to_string(image)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.image')";
    EXECUTE_QUERY(it3, tp, query3);
    uint64_t bind_arg_set_id = it3.Get(0).AsLong();
    assert(!it3.Next());

    uint64_t memory;
    GET_INT_VALUE(tp, bind_arg_set_id, "debug.memory", memory);
    GET_INT_VALUE(tp, bind_arg_set_id, "debug.offset", ds.image.bindOffset);

    std::string query4
        = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkAllocateMemory-mem' AND slice.ts < "
        + std::to_string(write_timestamp) + " AND " + std::to_string(memory)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.memory')";
    EXECUTE_QUERY(it4, tp, query4);
    uint64_t allocate_arg_set_id = it4.Get(0).AsLong();
    assert(!it4.Next());

    GET_INT_VALUE(tp, allocate_arg_set_id, "debug.size", ds.image.memorySize);
    GET_INT_VALUE(tp, allocate_arg_set_id, "debug.type", ds.image.memoryType);

    return true;
}

bool get_sampler_descriptor_set(
    TraceProcessor *tp, uint64_t write_arg_set_id, uint64_t write_timestamp, vksp::vksp_descriptor_set &ds)
{
    uint64_t sampler;
    GET_INT_VALUE(tp, write_arg_set_id, "debug.sampler", sampler);

    std::string query
        = "SELECT arg_set_id, MAX(ts) FROM slice WHERE slice.name = 'vkCreateSampler-result' AND slice.ts < "
        + std::to_string(write_timestamp) + " AND " + std::to_string(sampler)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.sampler')";
    EXECUTE_QUERY(it, tp, query);
    uint64_t sampler_arg_set_id = it.Get(0).AsLong();
    assert(!it.Next());

    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.flags", ds.sampler.flags);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.magFilter", ds.sampler.magFilter);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.minFilter", ds.sampler.minFilter);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.mipmapMode", ds.sampler.mipmapMode);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.addressModeU", ds.sampler.addressModeU);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.addressModeV", ds.sampler.addressModeV);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.addressModeW", ds.sampler.addressModeW);
    GET_FLOAT_VALUE(tp, sampler_arg_set_id, "debug.mipLodBias", ds.sampler.fMipLodBias);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.anisotropyEnable", ds.sampler.anisotropyEnable);
    GET_FLOAT_VALUE(tp, sampler_arg_set_id, "debug.maxAnisotropy", ds.sampler.fMaxAnisotropy);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.compareEnable", ds.sampler.compareEnable);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.compareOp", ds.sampler.compareOp);
    GET_FLOAT_VALUE(tp, sampler_arg_set_id, "debug.minLod", ds.sampler.fMinLod);
    GET_FLOAT_VALUE(tp, sampler_arg_set_id, "debug.maxLod", ds.sampler.fMaxLod);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.borderColor", ds.sampler.borderColor);
    GET_INT_VALUE(tp, sampler_arg_set_id, "debug.unnormalizedCoordinates", ds.sampler.unnormalizedCoordinates);

    return true;
}

bool get_descriptor_set(TraceProcessor *tp, uint64_t commandBuffer, uint64_t max_timestamp, uint64_t min_timestamp,
    std::vector<vksp::vksp_descriptor_set> &descriptor_sets_vector)
{
    std::map<uint32_t, std::set<uint32_t>> dsSeen;
    std::string query
        = "SELECT arg_set_id, ts FROM slice WHERE slice.name = 'vkCmdBindDescriptorSets-ds' AND slice.ts > "
        + std::to_string(min_timestamp) + " AND slice.ts < " + std::to_string(max_timestamp) + " AND "
        + std::to_string(commandBuffer)
        + " = ( SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = "
          "'debug.commandBuffer') ORDER BY ts DESC";
    EXECUTE_QUERY(it, tp, query);

    do {
        uint64_t arg_set_id = it.Get(0).AsLong();
        uint64_t bind_timestamp = it.Get(1).AsLong();
        vksp::vksp_descriptor_set ds = { 0 };

        uint64_t dstSet;
        {
            uint64_t firstSet, index;
            GET_INT_VALUE(tp, arg_set_id, "debug.firstSet", firstSet);
            GET_INT_VALUE(tp, arg_set_id, "debug.index", index);
            GET_INT_VALUE(tp, arg_set_id, "debug.dstSet", dstSet);
            ds.ds = firstSet + index;
        }

        std::string query2
            = "SELECT arg_set_id, ts FROM slice WHERE slice.name = 'vkUpdateDescriptorSets-write' AND slice.ts < "
            + std::to_string(bind_timestamp) + " AND " + std::to_string(dstSet)
            + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.dstSet')";
        EXECUTE_QUERY(it2, tp, query2);
        do {
            uint64_t write_arg_set_id = it2.Get(0).AsLong();
            uint64_t write_timestamp = it2.Get(1).AsLong();

            GET_INT_VALUE(tp, write_arg_set_id, "debug.descriptorType", ds.type);
            GET_INT_VALUE(tp, write_arg_set_id, "debug.dstBinding", ds.binding);

            if (dsSeen[ds.ds].count(ds.binding) != 0) {
                continue;
            }
            dsSeen[ds.ds].insert(ds.binding);

            switch (ds.type) {
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
                if (!get_buffer_descriptor_set(tp, write_arg_set_id, write_timestamp, ds)) {
                    ERROR("Could not get buffer info");
                    return false;
                }
            } break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
                if (!get_image_descriptor_set(tp, write_arg_set_id, write_timestamp, ds)) {
                    ERROR("Could not get image info");
                    return false;
                }
            } break;
            case VK_DESCRIPTOR_TYPE_SAMPLER: {
                if (!get_sampler_descriptor_set(tp, write_arg_set_id, write_timestamp, ds)) {
                    ERROR("Could not get sampler info");
                    return false;
                }
            } break;
            default:
                break;
            }

            descriptor_sets_vector.push_back(ds);
        } while (it2.Next());
    } while (it.Next());

    return true;
}

bool get_map_entries_from_cmd_buffer(TraceProcessor *tp, uint64_t commandBuffer, uint64_t max_timestamp,
    std::vector<vksp::vksp_specialization_map_entry> &map_entry_vector, vksp::vksp_configuration &config)
{
    std::string query = "SELECT arg_set_id FROM slice WHERE slice.name = 'vkCmdBindPipeline' AND "
        + std::to_string(commandBuffer)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = "
          "'debug.commandBuffer') AND slice.ts < "
        + std::to_string(max_timestamp) + " ORDER BY ts DESC";
    EXECUTE_QUERY(it, tp, query);
    uint64_t pipeline_arg_set_id = it.Get(0).AsLong();
    uint64_t pipeline;
    GET_INT_VALUE(tp, pipeline_arg_set_id, "debug.pipeline", pipeline);

    std::string query_create_compute_pipelines
        = "SELECT MAX(ts) FROM slice WHERE slice.name = 'vkCreateComputePipelines-specialization' AND "
        + std::to_string(pipeline)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.pipeline') "
          "AND slice.ts < "
        + std::to_string(max_timestamp);
    EXECUTE_QUERY_NO_CHECK(it_create_compute_pipelines, tp, query_create_compute_pipelines);
    uint64_t create_compute_pipelines_timestamp = 0;
    if (it_create_compute_pipelines.Next() && !it_create_compute_pipelines.Get(0).is_null()) {
        create_compute_pipelines_timestamp = it_create_compute_pipelines.Get(0).AsLong();
        assert(!it_create_compute_pipelines.Next());
    }

    std::string query2 = "SELECT arg_set_id FROM slice WHERE slice.name = 'vkCreateComputePipelines-MapEntry' AND "
        + std::to_string(pipeline)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.pipeline') "
          "AND slice.ts < "
        + std::to_string(max_timestamp) + " AND slice.ts > " + std::to_string(create_compute_pipelines_timestamp);
    EXECUTE_QUERY_NO_CHECK(it2, tp, query2);
    while (it2.Next()) {
        uint64_t arg_set_id = it2.Get(0).AsLong();
        vksp::vksp_specialization_map_entry me;
        GET_INT_VALUE(tp, arg_set_id, "debug.constantID", me.constantID);
        GET_INT_VALUE(tp, arg_set_id, "debug.offset", me.offset);
        GET_INT_VALUE(tp, arg_set_id, "debug.size", me.size);
        map_entry_vector.push_back(me);
    }

    std::string query3
        = "SELECT arg_set_id FROM slice WHERE slice.name = 'vkCreateComputePipelines-specialization' AND "
        + std::to_string(pipeline)
        + " = (SELECT int_value FROM args WHERE args.arg_set_id = slice.arg_set_id AND args.key = 'debug.pipeline') "
          "AND slice.ts < "
        + std::to_string(max_timestamp) + " AND slice.ts >= " + std::to_string(create_compute_pipelines_timestamp);
    EXECUTE_QUERY_NO_CHECK(it3, tp, query3);
    if (it3.Next()) {
        uint64_t arg_set_id = it3.Get(0).AsLong();
        GET_INT_VALUE(tp, arg_set_id, "debug.dataSize", config.specializationInfoDataSize);
        GET_STR_VALUE(tp, arg_set_id, "debug.pData", config.specializationInfoData);
        assert(!it3.Next());
    } else {
        config.specializationInfoDataSize = 0;
        config.specializationInfoData = strdup("");
    }

    return true;
}

void help()
{
    printf("USAGE: vulkan-shader-profiler-extractor [OPTIONS] -i <input> -o <output> -d <dispatchId>\n"
           "\n"
           "OPTIONS:\n"
           "\t-b\tOutput in binary instead of text\n"
           "\t-h\tDisplay this help and exit\n"
           "\t-s\tFile to use instead of perfetto to get shader code\n"
           "\t-v\tVerbose mode\n");
}

bool parse_args(int argc, char **argv)
{
    bool bHelp = false;
    int c;
    while ((c = getopt(argc, argv, "hbvd:i:o:s:")) != -1) {
        switch (c) {
        case 'd':
            gDispatchId = atoi(optarg);
            break;
        case 'i':
            gInput = std::string(optarg);
            break;
        case 'o':
            gOutput = std::string(optarg);
            break;
        case 'b':
            gBinary = true;
            break;
        case 's':
            gShaderFile = std::string(optarg);
            break;
        case 'v':
            gVerbose = true;
            break;
        case 'h':
        default:
            bHelp = true;
        }
    }
    if (bHelp || gInput == "" || gOutput == "" || gDispatchId == UINT64_MAX) {
        help();
        return false;
    } else if (access(gInput.c_str(), F_OK)) {
        ERROR("'%s' does not exist", gInput.c_str());
        help();
        return false;
    }

    return true;
}

int main(int argc, char **argv)
{
    if (!parse_args(argc, argv)) {
        return -1;
    }
    PRINT("Arguments parsed: input '%s' output '%s' dispatchId '%lu' binary '%u' verbose '%u'", gInput.c_str(),
        gOutput.c_str(), gDispatchId, gBinary, gVerbose);

    auto tp = initialize_database();
    CHECK(tp != nullptr, "Initialization failed");
    PRINT("%s read with success", gInput.c_str());

    vksp::vksp_configuration config;
    uint64_t dispatch, compute, commandBuffer;
    CHECK(get_dispatch_compute_and_commandBuffer_from_dispatchId(
              tp.get(), gDispatchId, dispatch, compute, commandBuffer, config),
        "Could not get dispatch, compute and commandBuffer from dispatchId");
    PRINT("Dispatch, compute and commandBuffer: %lu, %lu, %lu", dispatch, compute, commandBuffer);
    PRINT("EntryPoint: '%s' - groupCount: %u-%u-%u", config.entryPoint, config.groupCountX, config.groupCountY,
        config.groupCountZ);

    std::string shader;
    uint64_t device;
    std::vector<char> shader_buffer;
    CHECK(get_shader_and_device_from_compute(tp.get(), compute, shader, shader_buffer, device, config),
        "Could not get shader from compute");
    PRINT("Device: %lu", device);
    if (gShaderFile == "") {
        PRINT("Shader from compute (name: '%s'):\n%s", config.shaderName, shader.c_str());
    } else {
        PRINT("Shader from file '%s' (name: '%s')\n", gShaderFile.c_str(), config.shaderName);
    }

    uint64_t max_timestamp;
    CHECK(get_max_timestamp(tp.get(), dispatch, max_timestamp), "Could not get max_timestamp");
    PRINT("Max timestamp: %lu", max_timestamp);

    uint64_t min_timestamp;
    CHECK(get_min_timestamp(tp.get(), commandBuffer, max_timestamp, min_timestamp), "Could not get min_timestamp");
    PRINT("Min timestamp: %lu", min_timestamp);

    std::vector<vksp::vksp_specialization_map_entry> map_entry_vector;
    CHECK(get_map_entries_from_cmd_buffer(tp.get(), commandBuffer, max_timestamp, map_entry_vector, config),
        "Could not get map entries from command buffer");
    PRINT("specialization info data (size %u): '%s'", config.specializationInfoDataSize, config.specializationInfoData);
    for (auto &me : map_entry_vector) {
        PRINT("map_entry: constantID %u offset %u size %u", me.constantID, me.offset, me.size);
    }

    CHECK(get_extensions_from_device(tp.get(), device, config.enabledExtensionNames),
        "Could not get features and extensions names from device");
    PRINT("Extensions: '%s'", config.enabledExtensionNames);

    std::vector<vksp::vksp_push_constant> push_constants_vector;
    CHECK(get_push_constants(tp.get(), commandBuffer, max_timestamp, min_timestamp, push_constants_vector),
        "Could not get push_constants");
    for (auto &pc : push_constants_vector) {
        PRINT("push_constants: offset %u size %u stageFlags %u pValues %s", pc.offset, pc.size, pc.stageFlags,
            pc.pValues);
    }

    std::vector<vksp::vksp_descriptor_set> descriptor_sets_vector;
    CHECK(get_descriptor_set(tp.get(), commandBuffer, max_timestamp, min_timestamp, descriptor_sets_vector),
        "Could not get descriptor_set");
    for (auto &ds : descriptor_sets_vector) {
        switch (ds.type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            PRINT("descriptor_set: ds %u binding %u type %u BUFFER size %u flags %u queueFamilyIndexCount %u "
                  "sharingMode %u usage %u range %u offset %u memorySize %u memoryType %u bindOffset %u viewFlags %u "
                  "viewFormat %u",
                ds.ds, ds.binding, ds.type, ds.buffer.size, ds.buffer.flags, ds.buffer.queueFamilyIndexCount,
                ds.buffer.sharingMode, ds.buffer.usage, ds.buffer.range, ds.buffer.offset, ds.buffer.memorySize,
                ds.buffer.memoryType, ds.buffer.bindOffset, ds.buffer.viewFlags, ds.buffer.viewFormat);
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            PRINT("descriptor_set: ds %u binding %u type %u IMAGE imageLayout %u flags %u type %u format %u width %u "
                  "height %u depth %u mipLevels %u arrayLayers %u samples %u tiling %u usage %u sharingMode %u "
                  "queueFamilyIndexCount %u initialLayout %u aspectMask %u baseMipLevel %u layerCount %u viewFlags %u "
                  "viewType %u viewFormat %u component_a %u component_b %u component_g %u component_r %u memoryType %u "
                  "memorySize %u bindOffset %u",
                ds.ds, ds.binding, ds.type, ds.image.imageLayout, ds.image.imageFlags, ds.image.imageType,
                ds.image.format, ds.image.width, ds.image.height, ds.image.depth, ds.image.mipLevels,
                ds.image.arrayLayers, ds.image.samples, ds.image.tiling, ds.image.usage, ds.image.sharingMode,
                ds.image.queueFamilyIndexCount, ds.image.initialLayout, ds.image.aspectMask, ds.image.baseMipLevel,
                ds.image.layerCount, ds.image.viewFlags, ds.image.viewType, ds.image.viewFormat, ds.image.component_a,
                ds.image.component_b, ds.image.component_g, ds.image.component_r, ds.image.memoryType,
                ds.image.memorySize, ds.image.bindOffset);
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            PRINT("descriptor_set: ds %u binding %u type %u SAMPLER flags %u magFilter %u minFilter %u mipmapMode %u "
                  "addressModeU %u addressModeV %u addressModeW %u mipLodBias %f anisotropyEnable %u maxAnisotropy %f "
                  "compareEnable %u compareOp %u minLod %f maxLod %f borderColor %u unnormalizedCoordinates %u",
                ds.ds, ds.binding, ds.type, ds.sampler.flags, ds.sampler.magFilter, ds.sampler.minFilter,
                ds.sampler.mipmapMode, ds.sampler.addressModeU, ds.sampler.addressModeV, ds.sampler.addressModeW,
                ds.sampler.fMipLodBias, ds.sampler.anisotropyEnable, ds.sampler.fMaxAnisotropy,
                ds.sampler.compareEnable, ds.sampler.compareOp, ds.sampler.fMinLod, ds.sampler.fMaxLod,
                ds.sampler.borderColor, ds.sampler.unnormalizedCoordinates);
            break;
        default:
            PRINT("descriptor_set, ds %u binding %u type %u UNKNWON_TYPE", ds.ds, ds.binding, ds.type);
            break;
        }
    }

    if (gShaderFile == "") {
        CHECK(store_shader_in_output(&shader, &push_constants_vector, &descriptor_sets_vector, &map_entry_vector,
                  &config, gOutput.c_str(), gBinary),
            "Could not store shader in output file");
    } else {
        CHECK(store_shader_buffer_in_output(&shader_buffer, &push_constants_vector, &descriptor_sets_vector,
                  &map_entry_vector, &config, gOutput.c_str(), gBinary),
            "Could not store shader buffer in output file");
    }

    return 0;
}
