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

#include <assert.h>
#include <spirv-tools/libspirv.h>
#include <spirv-tools/optimizer.hpp>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include <chrono>
#include <map>
#include <set>
#include <vector>
#include <vulkan/vulkan_core.h>

#define PRINT_IMPL(file, message, ...)                                                                                 \
    do {                                                                                                               \
        fprintf(file, "[VKSP] %s: " message "\n", __func__, ##__VA_ARGS__);                                            \
    } while (0)

#define ERROR(message, ...) PRINT_IMPL(stderr, message, ##__VA_ARGS__)

#define PRINT(message, ...)                                                                                            \
    do {                                                                                                               \
        if (gVerbose) {                                                                                                \
            PRINT_IMPL(stdout, message, ##__VA_ARGS__);                                                                \
        }                                                                                                              \
    } while (0)

#define CHECK(statement, message, ...)                                                                                 \
    do {                                                                                                               \
        if (!(statement)) {                                                                                            \
            ERROR(message, ##__VA_ARGS__);                                                                             \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

static std::map<char, uint8_t> charToByte
    = { { '0', 0 }, { '1', 1 }, { '2', 2 }, { '3', 3 }, { '4', 4 }, { '5', 5 }, { '6', 6 }, { '7', 7 }, { '8', 8 },
          { '9', 9 }, { 'a', 10 }, { 'b', 11 }, { 'c', 12 }, { 'd', 13 }, { 'e', 14 }, { 'f', 15 } };

static bool gVerbose = false;
static std::string gInput = "";
static uint32_t gColdRun = 0, gHotRun = 1;

int get_device_queue_and_cmd_buffer(VkPhysicalDevice &pDevice, VkDevice &device, VkQueue &queue,
    VkCommandBuffer &cmdBuffer, VkPhysicalDeviceMemoryProperties &memProperties, const char *enabledExtensionNames)
{
    VkResult res;

    VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "vulkan-shader-profiler-runnner", 0,
        "vulkan-shader-profiler-runner", 0, VK_MAKE_VERSION(1, 3, 0) };
    VkInstanceCreateInfo info = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &appInfo,
        0,
        nullptr,
        0,
        nullptr,
    };

    VkInstance instance;
    res = vkCreateInstance(&info, nullptr, &instance);
    CHECK(res == VK_SUCCESS, "Could not create vulkan instance");

    uint32_t nbDevices;
    res = vkEnumeratePhysicalDevices(instance, &nbDevices, nullptr);
    CHECK(res == VK_SUCCESS, "Could not enumerate physical devices");

    std::vector<VkPhysicalDevice> physicalDevices(nbDevices);
    res = vkEnumeratePhysicalDevices(instance, &nbDevices, physicalDevices.data());
    CHECK(res == VK_SUCCESS, "Could not enumerate physical devices (second call)");
    pDevice = physicalDevices.front();

    vkGetPhysicalDeviceMemoryProperties(pDevice, &memProperties);
    VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures zeroInitializeWgMemFeatures
        = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES, nullptr };
    VkPhysicalDeviceShaderAtomicInt64Features shaderAtomicInt64Features
        = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES, &zeroInitializeWgMemFeatures };
    VkPhysicalDeviceShaderClockFeaturesKHR shaderClockFeatures
        = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR, &shaderAtomicInt64Features };
    VkPhysicalDeviceFeatures2 pDeviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &shaderClockFeatures };
    vkGetPhysicalDeviceFeatures2(pDevice, &pDeviceFeatures);
    CHECK(shaderAtomicInt64Features.shaderBufferInt64Atomics != 0, "shaderBufferInt64Atomics not supported");
    CHECK(shaderClockFeatures.shaderSubgroupClock != 0, "shaderClockFeatures.shaderSubgroupClock not supported");

    uint32_t nbFamilies;
    vkGetPhysicalDeviceQueueFamilyProperties(pDevice, &nbFamilies, nullptr);

    std::vector<VkQueueFamilyProperties> families(nbFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(pDevice, &nbFamilies, families.data());

    uint32_t queueFamilyIndex = UINT32_MAX;
    uint32_t nbQueues;
    for (uint32_t i = 0; i < nbFamilies; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamilyIndex = i;
            nbQueues = families[i].queueCount;
            break;
        }
    }
    CHECK(queueFamilyIndex != UINT32_MAX, "Could not find a VK_QUEUE_COMPUTE_BIT queue");

    std::vector<float> queuePriorities(nbQueues, 1.0f);
    VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
        queueFamilyIndex, nbQueues, queuePriorities.data() };

    std::vector<const char *> extensions;
    size_t pos = 0;
    std::string extensionsStr = std::string(enabledExtensionNames);
    extensionsStr.erase(0, 1);
    while ((pos = extensionsStr.find(".")) != std::string::npos) {
        auto extension = extensionsStr.substr(0, pos);
        extensions.push_back(strdup(extension.c_str()));
        extensionsStr.erase(0, pos + 1);
    }
    extensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);

    const VkDeviceCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        &pDeviceFeatures,
        0,
        1,
        &queueCreateInfo,
        0,
        nullptr,
        (uint32_t)extensions.size(),
        extensions.data(),
        nullptr,
    };

    res = vkCreateDevice(pDevice, &createInfo, nullptr, &device);
    CHECK(res == VK_SUCCESS, "Could not create device");

    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VkCommandPool cmdPool;
    const VkCommandPoolCreateInfo pCreateInfo
        = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, queueFamilyIndex };
    res = vkCreateCommandPool(device, &pCreateInfo, nullptr, &cmdPool);
    CHECK(res == VK_SUCCESS, "Could not create command pool");

    const VkCommandBufferAllocateInfo pAllocateInfo
        = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    res = vkAllocateCommandBuffers(device, &pAllocateInfo, &cmdBuffer);
    CHECK(res == VK_SUCCESS, "Could not allocate command buffer");

    const VkCommandBufferBeginInfo pBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    res = vkBeginCommandBuffer(cmdBuffer, &pBeginInfo);
    CHECK(res == VK_SUCCESS, "Could not begin command buffer");

    return 0;
}

bool extract_from_input(std::vector<uint32_t> &shader, std::vector<spvtools::vksp_descriptor_set> &ds,
    std::vector<spvtools::vksp_push_constant> &pc, std::vector<spvtools::vksp_specialization_map_entry> &me,
    spvtools::vksp_configuration &config)
{
    FILE *input = fopen(gInput.c_str(), "r");
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
    spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
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
    opt.RegisterPass(spvtools::CreateExtractVkspReflectInfoPass(&pc, &ds, &me, &config));
    opt.RegisterPass(spvtools::CreateStripReflectInfoPass());
    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    if (!opt.Run(binary, size, &shader, options)) {
        ERROR("Error while running 'CreateVkspReflectInfoPass' and 'CreateStripReflectInfoPass'");
        return false;
    }

    if (gVerbose) {
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

struct vksp_counter {
    VkBuffer buffer;
    VkDeviceMemory memory;
    const char *name;
};

uint32_t handle_descriptor_set_buffer(spvtools::vksp_descriptor_set &ds, VkDevice device, VkCommandBuffer cmdBuffer,
    VkPhysicalDeviceMemoryProperties &memProperties, std::vector<VkDescriptorSet> &descSet,
    std::vector<vksp_counter> &counters)
{
    VkResult res;
    vksp_counter counter;
    bool bCounter = ds.buffer.vksp_counter != nullptr;
    if (bCounter) {
        ds.buffer.size = sizeof(uint64_t) * 2;
        ds.buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ds.buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        counter.name = ds.buffer.vksp_counter;
    }

    VkBuffer buffer;
    const VkBufferCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, ds.buffer.flags,
        ds.buffer.size, ds.buffer.usage, (VkSharingMode)ds.buffer.sharingMode, 0, nullptr };
    res = vkCreateBuffer(device, &pCreateInfo, nullptr, &buffer);
    CHECK(res == VK_SUCCESS, "Could not create buffer");

    if (bCounter) {
        VkMemoryRequirements memreqs;
        vkGetBufferMemoryRequirements(device, buffer, &memreqs);
        bool memoryTypeFound = false;
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            auto dev_properties = memProperties.memoryTypes[i].propertyFlags;
            bool valid = (1ULL << i) & memreqs.memoryTypeBits;
            auto required_properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            bool satisfactory = (dev_properties & required_properties) == required_properties;
            if (satisfactory && valid) {
                ds.buffer.memoryType = i;
                memoryTypeFound = true;
                break;
            }
        }
        CHECK(memoryTypeFound, "Could not find a memoryType for counter");
        ds.buffer.memorySize = memreqs.size;
        counter.buffer = buffer;
    }

    const VkMemoryAllocateInfo pAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        ds.buffer.memorySize,
        ds.buffer.memoryType,
    };

    VkDeviceMemory memory;
    res = vkAllocateMemory(device, &pAllocateInfo, nullptr, &memory);
    CHECK(res == VK_SUCCESS, "Could not allocate memory for buffer");

    res = vkBindBufferMemory(device, buffer, memory, ds.buffer.bindOffset);
    CHECK(res == VK_SUCCESS, "Could not bind buffer and memory");

    VkDeviceSize range = ds.buffer.range;
    if (bCounter) {
        range = VK_WHOLE_SIZE;
        counter.memory = memory;
        counters.push_back(counter);
    }

    const VkDescriptorBufferInfo bufferInfo = { buffer, ds.buffer.offset, range };
    const VkWriteDescriptorSet write = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        descSet[ds.ds],
        ds.binding,
        0,
        1,
        (VkDescriptorType)ds.type,
        nullptr,
        &bufferInfo,
        nullptr,
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    return 0;
}

uint32_t handle_descriptor_set_image(spvtools::vksp_descriptor_set &ds, VkDevice device, VkCommandBuffer cmdBuffer,
    VkPhysicalDeviceMemoryProperties &memProperties, std::vector<VkDescriptorSet> &descSet)
{
    VkResult res;

    VkImage image;
    VkExtent3D extent = { ds.image.width, ds.image.height, ds.image.depth };
    const VkImageCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, ds.image.imageFlags,
        (VkImageType)ds.image.imageType, (VkFormat)ds.image.format, extent, ds.image.mipLevels, ds.image.arrayLayers,
        (VkSampleCountFlagBits)ds.image.samples, (VkImageTiling)ds.image.tiling, ds.image.usage,
        (VkSharingMode)ds.image.sharingMode, ds.image.queueFamilyIndexCount, nullptr,
        (VkImageLayout)ds.image.initialLayout };
    res = vkCreateImage(device, &pCreateInfo, nullptr, &image);
    CHECK(res == VK_SUCCESS, "Could not create image");

    const VkMemoryAllocateInfo pAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        ds.image.memorySize,
        ds.image.memoryType,
    };

    VkDeviceMemory memory;
    res = vkAllocateMemory(device, &pAllocateInfo, nullptr, &memory);
    CHECK(res == VK_SUCCESS, "Could not allocate memory for image");

    res = vkBindImageMemory(device, image, memory, ds.image.bindOffset);
    CHECK(res == VK_SUCCESS, "Could not bind image and memory");

    VkImageView image_view;
    VkComponentMapping components
        = { (VkComponentSwizzle)ds.image.component_r, (VkComponentSwizzle)ds.image.component_g,
              (VkComponentSwizzle)ds.image.component_b, (VkComponentSwizzle)ds.image.component_a };
    VkImageSubresourceRange subresourceRange = { ds.image.aspectMask, ds.image.baseMipLevel, ds.image.levelCount,
        ds.image.baseArrayLayer, ds.image.layerCount };
    const VkImageViewCreateInfo pViewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, ds.image.viewFlags,
        image, (VkImageViewType)ds.image.viewType, (VkFormat)ds.image.viewFormat, components, subresourceRange };
    res = vkCreateImageView(device, &pViewInfo, nullptr, &image_view);
    CHECK(res == VK_SUCCESS, "Could not create image view");

    const VkDescriptorImageInfo imageInfo = { VK_NULL_HANDLE, image_view, (VkImageLayout)ds.image.imageLayout };
    const VkWriteDescriptorSet write = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        descSet[ds.ds],
        ds.binding,
        0,
        1,
        (VkDescriptorType)ds.type,
        &imageInfo,
        nullptr,
        nullptr,
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    return 0;
}

uint32_t handle_descriptor_set_sampler(spvtools::vksp_descriptor_set &ds, VkDevice device, VkCommandBuffer cmdBuffer,
    VkPhysicalDeviceMemoryProperties &memProperties, std::vector<VkDescriptorSet> &descSet)
{
    VkResult res;
    VkSampler sampler;
    const VkSamplerCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr, ds.sampler.flags,
        (VkFilter)ds.sampler.magFilter, (VkFilter)ds.sampler.minFilter, (VkSamplerMipmapMode)ds.sampler.mipmapMode,
        (VkSamplerAddressMode)ds.sampler.addressModeU, (VkSamplerAddressMode)ds.sampler.addressModeV,
        (VkSamplerAddressMode)ds.sampler.addressModeW, ds.sampler.fMipLodBias, ds.sampler.anisotropyEnable,
        ds.sampler.fMaxAnisotropy, ds.sampler.compareEnable, (VkCompareOp)ds.sampler.compareOp, ds.sampler.fMinLod,
        ds.sampler.fMaxLod, (VkBorderColor)ds.sampler.borderColor, ds.sampler.unnormalizedCoordinates };
    res = vkCreateSampler(device, &pCreateInfo, nullptr, &sampler);
    CHECK(res == VK_SUCCESS, "Could not create sampler");

    const VkDescriptorImageInfo imageInfo = { sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    const VkWriteDescriptorSet write = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        descSet[ds.ds],
        ds.binding,
        0,
        1,
        (VkDescriptorType)ds.type,
        &imageInfo,
        nullptr,
        nullptr,
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    return 0;
}

uint32_t allocate_descriptor_set(VkDevice device, std::vector<VkDescriptorSet> &descSet,
    std::vector<spvtools::vksp_descriptor_set> &dsVector, std::vector<VkDescriptorSetLayout> &descSetLayoutVector)
{
    VkResult res;
    std::map<VkDescriptorType, uint32_t> descTypeCount;

    for (auto &ds : dsVector) {
        VkDescriptorType type = (VkDescriptorType)ds.type;
        if (descTypeCount.count(type) == 0) {
            descTypeCount[type] = 1;
        } else {
            descTypeCount[type]++;
        }
    }
    for (unsigned i = 0; i < descSet.size(); i++) {
        std::vector<VkDescriptorSetLayoutBinding> descSetLayoutBindings;
        for (auto &ds : dsVector) {
            if (ds.ds != i) {
                continue;
            }
            VkDescriptorType type = (VkDescriptorType)ds.type;

            VkDescriptorSetLayoutBinding descSetLayoutBinding
                = { ds.binding, type, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            descSetLayoutBindings.push_back(descSetLayoutBinding);
        }
        VkDescriptorSetLayout descSetLayout;
        const VkDescriptorSetLayoutCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr, 0, (uint32_t)descSetLayoutBindings.size(), descSetLayoutBindings.data() };
        res = vkCreateDescriptorSetLayout(device, &pCreateInfo, nullptr, &descSetLayout);
        CHECK(res == VK_SUCCESS, "Count not create descriptor set layout");

        descSetLayoutVector.push_back(descSetLayout);
    }

    std::vector<VkDescriptorPoolSize> descPoolSize;
    for (auto it : descTypeCount) {
        descPoolSize.push_back({ it.first, it.second });
    }

    const VkDescriptorPoolCreateInfo pCreateInfo
        = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
              (uint32_t)descSet.size(), (uint32_t)descPoolSize.size(), descPoolSize.data() };
    VkDescriptorPool descPool;
    res = vkCreateDescriptorPool(device, &pCreateInfo, nullptr, &descPool);
    CHECK(res == VK_SUCCESS, "Could not create descriptor pool");

    const VkDescriptorSetAllocateInfo pAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
        descPool, (uint32_t)descSetLayoutVector.size(), descSetLayoutVector.data() };
    res = vkAllocateDescriptorSets(device, &pAllocateInfo, descSet.data());
    CHECK(res == VK_SUCCESS, "Could not allocate descriptor sets");

    return 0;
}

uint32_t count_descriptor_set(std::vector<spvtools::vksp_descriptor_set> dsVector)
{
    std::set<uint32_t> ds_set;
    for (auto &ds : dsVector) {
        ds_set.insert(ds.ds);
    }
    return ds_set.size();
}

uint32_t handle_push_constant(std::vector<spvtools::vksp_push_constant> &pcVector, VkPushConstantRange &range,
    VkCommandBuffer cmdBuffer, VkPipelineLayout pipelineLayout)
{
    std::vector<uint8_t> pValues(range.size);
    for (auto &pc : pcVector) {
        auto pValuesLen = strlen(pc.pValues);
        CHECK(pValuesLen % 2 == 0, "push constant string length is not even");
        uint32_t pValuesSize = pValuesLen / 2;
        CHECK(pValuesSize == pc.size, "push constant size does not match string");

        for (unsigned i = 0; i < pValuesSize; i++) {
            uint8_t value = 0;
            value |= charToByte[pc.pValues[i * 2 + 1]];
            value |= (charToByte[pc.pValues[i * 2]]) << 4;
            pValues[pc.offset - range.offset + i] = value;
        }
    }

    vkCmdPushConstants(
        cmdBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, range.offset, range.size, pValues.data());
    return 0;
}

uint32_t allocate_pipeline_layout(VkDevice device, std::vector<spvtools::vksp_push_constant> &pcVector,
    std::vector<VkPushConstantRange> &pcRanges, std::vector<VkDescriptorSetLayout> &descSetLayoutVector,
    VkPipelineLayout &pipelineLayout)
{
    VkResult res;

    const VkPipelineLayoutCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
        (uint32_t)descSetLayoutVector.size(), descSetLayoutVector.data(), (uint32_t)pcRanges.size(), pcRanges.data() };

    res = vkCreatePipelineLayout(device, &pCreateInfo, nullptr, &pipelineLayout);
    CHECK(res == VK_SUCCESS, "Could not create pipeline layout");

    return 0;
}

uint32_t allocate_pipeline(std::vector<uint32_t> &shader, VkPipelineLayout pipelineLayout, VkDevice device,
    VkCommandBuffer cmdBuffer, std::vector<spvtools::vksp_specialization_map_entry> &meVector,
    spvtools::vksp_configuration &config, VkPipeline &pipeline)
{
    VkResult res;
    VkShaderModule shaderModule;
    const VkShaderModuleCreateInfo shaderModuleCreateInfo
        = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, shader.size() * sizeof(uint32_t), shader.data() };
    res = vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule);
    CHECK(res == VK_SUCCESS, "Could not create shader module");

    std::vector<VkSpecializationMapEntry> mapEntries;
    for (auto &me : meVector) {
        VkSpecializationMapEntry mapEntry = { me.constantID, me.offset, me.size };
        mapEntries.push_back(mapEntry);
    }

    auto pDataLen = strlen(config.specializationInfoData);
    CHECK(pDataLen % 2 == 0, "specialization info data string length is not even");
    uint32_t pDataSize = pDataLen / 2;
    CHECK(pDataSize == config.specializationInfoDataSize, "specialization info data size does not match string");

    std::vector<uint8_t> pData;
    for (unsigned i = 0; i < pDataSize; i++) {
        uint8_t value = 0;
        value |= charToByte[config.specializationInfoData[i * 2 + 1]];
        value |= (charToByte[config.specializationInfoData[i * 2]]) << 4;
        pData.push_back(value);
    }
    const VkSpecializationInfo specializationInfo = {
        (uint32_t)mapEntries.size(),
        mapEntries.data(),
        config.specializationInfoDataSize,
        pData.data(),
    };

    const VkComputePipelineCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule,
            config.entryPoint, &specializationInfo },
        pipelineLayout, VK_NULL_HANDLE, 0 };
    res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pCreateInfo, nullptr, &pipeline);
    CHECK(res == VK_SUCCESS, "Could not create compute pipeline");

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    return 0;
}

void print_time(uint64_t time, const char *prefix, const char *message, bool newLine = true)
{
    printf("%s %*s: ", prefix, 8, message);
    const uint64_t second = 1000 * 1000 * 1000;
    const uint64_t msecond = 1000 * 1000;
    const uint64_t usecond = 1000;
    uint64_t time_s = 0;
    uint64_t time_ms = 0;
    uint64_t time_us = 0;
    uint64_t time_ns = 0;
    if (time > second) {
        time_s = time / second;
        time -= time_s * second;
    }
    if (time > msecond) {
        time_ms = time / msecond;
        time -= time_ms * msecond;
    }
    if (time > usecond) {
        time_us = time / usecond;
        time -= time_us * usecond;
    }
    time_ns = time;
    if (time_s) {
        printf("%*lu.%.3lu s", 3, time_s, time_ms);
    } else if (time_ms) {
        printf("%*lu.%.3lu ms", 3, time_ms, time_us);
    } else if (time_us) {
        printf("%*lu.%.3lu us", 3, time_us, time_ns);
    } else {
        printf("%*lu ns", 3, time_ns);
    }
    if (newLine) {
        printf("\n");
    }
}

uint32_t execute(VkPhysicalDevice pDevice, VkDevice device, VkCommandBuffer cmdBuffer, VkQueue queue,
    spvtools::vksp_configuration &config, std::vector<vksp_counter> &counters)
{
    VkResult res;

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(pDevice, &properties);
    double ns_per_tick = properties.limits.timestampPeriod;
    PRINT("ns_per_tick: %f", ns_per_tick);

    const uint32_t queryCount = 3;
    const VkQueryPoolCreateInfo pCreateInfo = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        nullptr,
        0,
        VK_QUERY_TYPE_TIMESTAMP,
        queryCount,
        0,
    };
    VkQueryPool queryPool;
    res = vkCreateQueryPool(device, &pCreateInfo, nullptr, &queryPool);
    CHECK(res == VK_SUCCESS, "Could not create query pool");

    vkCmdResetQueryPool(cmdBuffer, queryPool, 0, queryCount);
    vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 0);

    for (unsigned i = 0; i < gColdRun; i++) {
        vkCmdDispatch(cmdBuffer, config.groupCountX, config.groupCountY, config.groupCountZ);
    }

    const uint64_t zero[2] = { 0ULL, 0ULL };
    for (auto &counter : counters) {
        vkCmdUpdateBuffer(cmdBuffer, counter.buffer, 0, sizeof(zero), zero);
    }

    vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1);
    for (unsigned i = 0; i < gHotRun; i++) {
        vkCmdDispatch(cmdBuffer, config.groupCountX, config.groupCountY, config.groupCountZ);
    }
    vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 2);

    res = vkEndCommandBuffer(cmdBuffer);
    CHECK(res == VK_SUCCESS, "Could not end command buffer");

    const VkSubmitInfo pSubmit
        = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdBuffer, 0, nullptr };

    auto timeBeforeSubmit = std::chrono::steady_clock::now();
    res = vkQueueSubmit(queue, 1, &pSubmit, VK_NULL_HANDLE);
    auto timeAfterSubmit = std::chrono::steady_clock::now();
    CHECK(res == VK_SUCCESS, "Could not submit cmdBuffer in queue");

    res = vkQueueWaitIdle(queue);
    auto timeAfterWaitIdle = std::chrono::steady_clock::now();
    CHECK(res == VK_SUCCESS, "Could not wait for queue idle");

    uint64_t timestamps[queryCount];
    res = vkGetQueryPoolResults(device, queryPool, 0, queryCount, sizeof(timestamps), timestamps, sizeof(timestamps[0]),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    CHECK(res == VK_SUCCESS, "Could not get query pool results");

    uint64_t gpu_cold_tick = timestamps[1] - timestamps[0];
    uint64_t gpu_hot_tick = timestamps[2] - timestamps[1];
    uint64_t gpu_total_tick = timestamps[2] - timestamps[0];

    uint64_t gpu_cold_ns = gpu_cold_tick * ns_per_tick;
    uint64_t gpu_hot_ns = gpu_hot_tick * ns_per_tick;
    uint64_t gpu_total_ns = gpu_total_tick * ns_per_tick;
    uint64_t gpu_avg_hot_ns = (gpu_hot_tick * ns_per_tick) / gHotRun;

    print_time(std::chrono::duration_cast<std::chrono::nanoseconds>(timeAfterSubmit - timeBeforeSubmit).count(),
        "[HOST]", "Submit");
    print_time(std::chrono::duration_cast<std::chrono::nanoseconds>(timeAfterWaitIdle - timeAfterSubmit).count(),
        "[HOST]", "WaitIdle");
    print_time(std::chrono::duration_cast<std::chrono::nanoseconds>(timeAfterWaitIdle - timeBeforeSubmit).count(),
        "[HOST]", "Total");

    print_time(gpu_total_ns, "[GPU ]", "Total");
    if (gColdRun != 0) {
        print_time(gpu_cold_ns, "[GPU ]", "Cold");
        print_time(gpu_hot_ns, "[GPU ]", "Hot");
    }
    print_time(gpu_avg_hot_ns, "[GPU ]", "Average");

    for (auto &counter : counters) {
        uint64_t *values;
        res = vkMapMemory(device, counter.memory, 0, sizeof(uint64_t) * 2, 0, (void **)&values);
        CHECK(res == VK_SUCCESS, "Could not map memory for counter '%s'", counter.name);
        uint64_t count = values[0];
        uint64_t total = values[1];
        uint64_t avg_ns = total * ns_per_tick / count;
        print_time(avg_ns, "[GPU ]", "Counter", false);
        printf(" - %.1f%% - '%s'", (double)avg_ns * 100.0 / (double)gpu_avg_hot_ns, counter.name);
        if (gVerbose) {
            printf(" - count: %lu - total: %lu", count, total);
        }
        printf("\n");
        vkUnmapMemory(device, counter.memory);
    }

    return 0;
}

void help()
{
    printf("USAGE: vulkan-shader-profiler-runner [OPTIONS] -i <input>\n"
           "\n"
           "OPTIONS:\n"
           "\t-h\tDisplay this help and exit\n"
           "\t-m\tNumber of cold run\n"
           "\t-n\tNumber of hot run\n"
           "\t-v\tVerbose mode\n");
}

bool parse_args(int argc, char **argv)
{
    bool bHelp = false;
    int c;
    while ((c = getopt(argc, argv, "hvi:n:m:")) != -1) {
        switch (c) {
        case 'n':
            gHotRun = atoi(optarg);
            break;
        case 'm':
            gColdRun = atoi(optarg);
            break;
        case 'i':
            gInput = std::string(optarg);
            break;
        case 'v':
            gVerbose = true;
            break;
        case 'h':
        default:
            bHelp = true;
        }
    }
    if (bHelp || gInput == "") {
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
    PRINT("Arguments parsed: input '%s' verbose '%u'", gInput.c_str(), gVerbose);

    std::vector<uint32_t> shader;
    std::vector<spvtools::vksp_descriptor_set> dsVector;
    std::vector<spvtools::vksp_push_constant> pcVector;
    std::vector<spvtools::vksp_specialization_map_entry> meVector;
    spvtools::vksp_configuration config;
    CHECK(extract_from_input(shader, dsVector, pcVector, meVector, config), "Could not extract data from input");
    PRINT("Shader name: '%s'", config.shaderName);
    PRINT("Entry point: '%s'", config.entryPoint);
    PRINT("groupCount: %u-%u-%u", config.groupCountX, config.groupCountY, config.groupCountZ);
    PRINT("specializationInfo data (size %u): '%s'", config.specializationInfoDataSize, config.specializationInfoData);
    PRINT("Extensions: '%s'", config.enabledExtensionNames);

    VkPhysicalDevice pDevice;
    VkDevice device;
    VkQueue queue;
    VkCommandBuffer cmdBuffer;
    VkPhysicalDeviceMemoryProperties memProperties;
    CHECK(
        get_device_queue_and_cmd_buffer(pDevice, device, queue, cmdBuffer, memProperties, config.enabledExtensionNames)
            == 0,
        "Could not get Vulkan Queue");
    PRINT("Device, queue and command buffer created");

    std::vector<VkDescriptorSet> descSet(count_descriptor_set(dsVector));
    std::vector<VkDescriptorSetLayout> descSetLayoutVector;
    CHECK(allocate_descriptor_set(device, descSet, dsVector, descSetLayoutVector) == 0,
        "Could not allocate descriptor set");
    PRINT("Descriptor set allocated");

    std::vector<vksp_counter> counters;
    for (auto &ds : dsVector) {
        switch (ds.type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            PRINT(
                "descriptor_set: ds %u binding %u type %u BUFFER size %u flags %u queueFamilyIndexCount %u "
                "sharingMode %u usage %u range %u offset %u memorySize %u memoryType %u bindOffset %u counterName '%s'",
                ds.ds, ds.binding, ds.type, ds.buffer.size, ds.buffer.flags, ds.buffer.queueFamilyIndexCount,
                ds.buffer.sharingMode, ds.buffer.usage, ds.buffer.range, ds.buffer.offset, ds.buffer.memorySize,
                ds.buffer.memoryType, ds.buffer.bindOffset, ds.buffer.vksp_counter);
            CHECK(handle_descriptor_set_buffer(ds, device, cmdBuffer, memProperties, descSet, counters) == 0,
                "Could not handle descriptor set buffer");
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
            CHECK(handle_descriptor_set_image(ds, device, cmdBuffer, memProperties, descSet) == 0,
                "Could not handle descriptor set buffer");
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
            CHECK(handle_descriptor_set_sampler(ds, device, cmdBuffer, memProperties, descSet) == 0,
                "Could not handle descriptor set buffer");
            break;
        default:
            PRINT("descriptor_set, ds %u binding %u type %u UNKNWON_TYPE", ds.ds, ds.binding, ds.type);
            break;
        }
    }
    std::vector<VkPushConstantRange> pcRanges;
    std::map<uint32_t, std::vector<spvtools::vksp_push_constant>> pcMap;
    for (auto &pc : pcVector) {
        PRINT("push_constants: offset %u size %u stageFlags %u pValues %s", pc.offset, pc.size, pc.stageFlags,
            pc.pValues);
        pcMap[pc.stageFlags].push_back(pc);
    }
    for (auto &it : pcMap) {
        uint32_t min_offset = UINT32_MAX;
        uint32_t max_offset = 0;
        for (auto &pc : it.second) {
            min_offset = std::min(min_offset, pc.offset);
            max_offset = std::max(max_offset, pc.offset + pc.size);
        }
        uint32_t size = max_offset - min_offset;
        pcRanges.push_back({ it.first, min_offset, size });
    }

    VkPipelineLayout pipelineLayout;
    CHECK(allocate_pipeline_layout(device, pcVector, pcRanges, descSetLayoutVector, pipelineLayout) == 0,
        "Could not allocate pipeline layout");
    PRINT("Pipeline layout allocated");

    for (auto &range : pcRanges) {
        CHECK(handle_push_constant(pcMap[(uint32_t)range.stageFlags], range, cmdBuffer, pipelineLayout) == 0,
            "Could not handle push constant");
    }

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, descSet.size(), descSet.data(), 0, nullptr);

    for (auto &me : meVector) {
        PRINT("map_entry: constantID %u offset %u size %u", me.constantID, me.offset, me.size);
    }

    VkPipeline pipeline;
    CHECK(allocate_pipeline(shader, pipelineLayout, device, cmdBuffer, meVector, config, pipeline) == 0,
        "Could not allocate pipeline");
    PRINT("Compute pipeline allocated");

    CHECK(execute(pDevice, device, cmdBuffer, queue, config, counters) == 0, "Could not execute");
    PRINT("Execution completed");

    return 0;
}
