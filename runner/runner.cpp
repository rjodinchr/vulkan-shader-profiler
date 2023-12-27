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

#include <algorithm>
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

#define CHECK_VK(result, message, ...) CHECK(result == VK_SUCCESS, message " (result: %i)", ##__VA_ARGS__, result)

static std::map<char, uint8_t> charToByte
    = { { '0', 0 }, { '1', 1 }, { '2', 2 }, { '3', 3 }, { '4', 4 }, { '5', 5 }, { '6', 6 }, { '7', 7 }, { '8', 8 },
          { '9', 9 }, { 'a', 10 }, { 'b', 11 }, { 'c', 12 }, { 'd', 13 }, { 'e', 14 }, { 'f', 15 } };

static bool gVerbose = false;
static std::string gInput = "";
static uint32_t gColdRun = 0, gHotRun = 1;
static VkBuffer gCounterBuffer;
static VkDeviceMemory gCounterMemory;

static const uint32_t gNbGpuTimestamps = 3;

static int get_device_queue_and_cmd_buffer(VkPhysicalDevice &pDevice, VkDevice &device, VkQueue &queue,
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
    CHECK_VK(res, "Could not create vulkan instance");

    uint32_t nbDevices;
    res = vkEnumeratePhysicalDevices(instance, &nbDevices, nullptr);
    CHECK_VK(res, "Could not enumerate physical devices");

    std::vector<VkPhysicalDevice> physicalDevices(nbDevices);
    res = vkEnumeratePhysicalDevices(instance, &nbDevices, physicalDevices.data());
    CHECK_VK(res, "Could not enumerate physical devices (second call)");
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
    CHECK_VK(res, "Could not create device");

    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VkCommandPool cmdPool;
    const VkCommandPoolCreateInfo pCreateInfo
        = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, queueFamilyIndex };
    res = vkCreateCommandPool(device, &pCreateInfo, nullptr, &cmdPool);
    CHECK_VK(res, "Could not create command pool");

    const VkCommandBufferAllocateInfo pAllocateInfo
        = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    res = vkAllocateCommandBuffers(device, &pAllocateInfo, &cmdBuffer);
    CHECK_VK(res, "Could not allocate command buffer");

    const VkCommandBufferBeginInfo pBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    res = vkBeginCommandBuffer(cmdBuffer, &pBeginInfo);
    CHECK_VK(res, "Could not begin command buffer");

    return 0;
}

static bool extract_from_input(std::vector<uint32_t> &shader, std::vector<spvtools::vksp_descriptor_set> &ds,
    std::vector<spvtools::vksp_push_constant> &pc, std::vector<spvtools::vksp_specialization_map_entry> &me,
    std::vector<spvtools::vksp_counter> &counters, spvtools::vksp_configuration &config)
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
    opt.RegisterPass(spvtools::CreateExtractVkspReflectInfoPass(&pc, &ds, &me, &counters, &config));
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

static uint32_t handle_descriptor_set_buffer(spvtools::vksp_descriptor_set &ds, VkDevice device,
    VkCommandBuffer cmdBuffer, VkPhysicalDeviceMemoryProperties &memProperties, std::vector<VkDescriptorSet> &descSet)
{
    VkResult res;

    bool bCounter = ds.type == VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER;
    ds.type &= VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER_MASK;

    VkBuffer buffer;
    const VkBufferCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, ds.buffer.flags,
        ds.buffer.size, ds.buffer.usage, (VkSharingMode)ds.buffer.sharingMode, 0, nullptr };
    res = vkCreateBuffer(device, &pCreateInfo, nullptr, &buffer);
    CHECK_VK(res, "Could not create buffer");

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
        CHECK(ds.buffer.memorySize == memreqs.size, "memorySize for counter buffer does not match");
    }

    const VkMemoryAllocateInfo pAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        ds.buffer.memorySize,
        ds.buffer.memoryType,
    };

    VkDeviceMemory memory;
    res = vkAllocateMemory(device, &pAllocateInfo, nullptr, &memory);
    CHECK_VK(res, "Could not allocate memory for buffer");

    res = vkBindBufferMemory(device, buffer, memory, ds.buffer.bindOffset);
    CHECK_VK(res, "Could not bind buffer and memory");

    const VkDescriptorBufferInfo bufferInfo = { buffer, ds.buffer.offset, ds.buffer.range };
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

    if (bCounter) {
        gCounterBuffer = buffer;
        gCounterMemory = memory;
    }

    return 0;
}

static uint32_t handle_descriptor_set_image(spvtools::vksp_descriptor_set &ds, VkDevice device,
    VkCommandBuffer cmdBuffer, VkPhysicalDeviceMemoryProperties &memProperties, std::vector<VkDescriptorSet> &descSet)
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
    CHECK_VK(res, "Could not create image");

    const VkMemoryAllocateInfo pAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        ds.image.memorySize,
        ds.image.memoryType,
    };

    VkDeviceMemory memory;
    res = vkAllocateMemory(device, &pAllocateInfo, nullptr, &memory);
    CHECK_VK(res, "Could not allocate memory for image");

    res = vkBindImageMemory(device, image, memory, ds.image.bindOffset);
    CHECK_VK(res, "Could not bind image and memory");

    VkImageView image_view;
    VkComponentMapping components
        = { (VkComponentSwizzle)ds.image.component_r, (VkComponentSwizzle)ds.image.component_g,
              (VkComponentSwizzle)ds.image.component_b, (VkComponentSwizzle)ds.image.component_a };
    VkImageSubresourceRange subresourceRange = { ds.image.aspectMask, ds.image.baseMipLevel, ds.image.levelCount,
        ds.image.baseArrayLayer, ds.image.layerCount };
    const VkImageViewCreateInfo pViewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, ds.image.viewFlags,
        image, (VkImageViewType)ds.image.viewType, (VkFormat)ds.image.viewFormat, components, subresourceRange };
    res = vkCreateImageView(device, &pViewInfo, nullptr, &image_view);
    CHECK_VK(res, "Could not create image view");

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

static uint32_t handle_descriptor_set_sampler(spvtools::vksp_descriptor_set &ds, VkDevice device,
    VkCommandBuffer cmdBuffer, VkPhysicalDeviceMemoryProperties &memProperties, std::vector<VkDescriptorSet> &descSet)
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
    CHECK_VK(res, "Could not create sampler");

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

static uint32_t allocate_descriptor_set(VkDevice device, std::vector<VkDescriptorSet> &descSet,
    std::vector<spvtools::vksp_descriptor_set> &dsVector, std::vector<VkDescriptorSetLayout> &descSetLayoutVector)
{
    VkResult res;
    std::map<VkDescriptorType, uint32_t> descTypeCount;

    for (auto &ds : dsVector) {
        VkDescriptorType type = (VkDescriptorType)(ds.type & VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER_MASK);
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
            VkDescriptorType type = (VkDescriptorType)(ds.type & VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER_MASK);

            VkDescriptorSetLayoutBinding descSetLayoutBinding
                = { ds.binding, type, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            descSetLayoutBindings.push_back(descSetLayoutBinding);
        }
        VkDescriptorSetLayout descSetLayout;
        const VkDescriptorSetLayoutCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr, 0, (uint32_t)descSetLayoutBindings.size(), descSetLayoutBindings.data() };
        res = vkCreateDescriptorSetLayout(device, &pCreateInfo, nullptr, &descSetLayout);
        CHECK_VK(res, "Count not create descriptor set layout");

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
    CHECK_VK(res, "Could not create descriptor pool");

    const VkDescriptorSetAllocateInfo pAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
        descPool, (uint32_t)descSetLayoutVector.size(), descSetLayoutVector.data() };
    res = vkAllocateDescriptorSets(device, &pAllocateInfo, descSet.data());
    CHECK_VK(res, "Could not allocate descriptor sets");

    return 0;
}

static uint32_t count_descriptor_set(std::vector<spvtools::vksp_descriptor_set> dsVector)
{
    std::set<uint32_t> ds_set;
    for (auto &ds : dsVector) {
        ds_set.insert(ds.ds);
    }
    return ds_set.size();
}

static uint32_t handle_push_constant(std::vector<spvtools::vksp_push_constant> &pcVector, VkPushConstantRange &range,
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

static uint32_t allocate_pipeline_layout(VkDevice device, std::vector<spvtools::vksp_push_constant> &pcVector,
    std::vector<VkPushConstantRange> &pcRanges, std::vector<VkDescriptorSetLayout> &descSetLayoutVector,
    VkPipelineLayout &pipelineLayout)
{
    VkResult res;

    const VkPipelineLayoutCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
        (uint32_t)descSetLayoutVector.size(), descSetLayoutVector.data(), (uint32_t)pcRanges.size(), pcRanges.data() };

    res = vkCreatePipelineLayout(device, &pCreateInfo, nullptr, &pipelineLayout);
    CHECK_VK(res, "Could not create pipeline layout");

    return 0;
}

static uint32_t allocate_pipeline(std::vector<uint32_t> &shader, VkPipelineLayout pipelineLayout, VkDevice device,
    VkCommandBuffer cmdBuffer, std::vector<spvtools::vksp_specialization_map_entry> &meVector,
    spvtools::vksp_configuration &config, VkPipeline &pipeline)
{
    VkResult res;
    VkShaderModule shaderModule;
    const VkShaderModuleCreateInfo shaderModuleCreateInfo
        = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, shader.size() * sizeof(uint32_t), shader.data() };
    res = vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule);
    CHECK_VK(res, "Could not create shader module");

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
    CHECK_VK(res, "Could not create compute pipeline");

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    return 0;
}

static void print_prefix(uint32_t prefix_max_size, const char *prefix, uint32_t message_max_size, const char *message)
{
    printf("[%*s] %*s: ", prefix_max_size, prefix, message_max_size, message);
}

static void print_time(uint64_t time, const char *prefix, uint32_t prefix_max_size, const char *message,
    uint32_t message_max_size, bool newLine = true)
{
    print_prefix(prefix_max_size, prefix, message_max_size, message);
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
        printf("%*lu ns", 7, time_ns);
    }
    if (newLine) {
        printf("\n");
    }
}

static uint32_t counters_size(std::vector<spvtools::vksp_counter> &counters)
{
    return sizeof(uint64_t) * (2 + counters.size());
}

static uint32_t execute(VkDevice device, VkCommandBuffer cmdBuffer, VkQueue queue, spvtools::vksp_configuration &config,
    std::vector<spvtools::vksp_counter> &counters, uint64_t *gpu_timestamps,
    std::chrono::steady_clock::time_point *host_timestamps)
{
    VkResult res;

    const VkQueryPoolCreateInfo pCreateInfo = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        nullptr,
        0,
        VK_QUERY_TYPE_TIMESTAMP,
        gNbGpuTimestamps,
        0,
    };
    VkQueryPool queryPool;
    res = vkCreateQueryPool(device, &pCreateInfo, nullptr, &queryPool);
    CHECK_VK(res, "Could not create query pool");

    vkCmdResetQueryPool(cmdBuffer, queryPool, 0, gNbGpuTimestamps);
    vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 0);

    for (unsigned i = 0; i < gColdRun; i++) {
        vkCmdDispatch(cmdBuffer, config.groupCountX, config.groupCountY, config.groupCountZ);
    }

    auto countersSize = counters_size(counters);
    auto zero = malloc(countersSize);
    memset(zero, 0, countersSize);
    vkCmdUpdateBuffer(cmdBuffer, gCounterBuffer, 0, countersSize, zero);

    vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1);
    for (unsigned i = 0; i < gHotRun; i++) {
        vkCmdDispatch(cmdBuffer, config.groupCountX, config.groupCountY, config.groupCountZ);
    }
    vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 2);

    res = vkEndCommandBuffer(cmdBuffer);
    CHECK_VK(res, "Could not end command buffer");

    const VkSubmitInfo pSubmit
        = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdBuffer, 0, nullptr };

    host_timestamps[0] = std::chrono::steady_clock::now();
    res = vkQueueSubmit(queue, 1, &pSubmit, VK_NULL_HANDLE);
    host_timestamps[1] = std::chrono::steady_clock::now();
    CHECK_VK(res, "Could not submit cmdBuffer in queue");

    res = vkQueueWaitIdle(queue);
    host_timestamps[2] = std::chrono::steady_clock::now();
    CHECK_VK(res, "Could not wait for queue idle");

    res = vkGetQueryPoolResults(device, queryPool, 0, gNbGpuTimestamps, sizeof(gpu_timestamps[0]) * gNbGpuTimestamps,
        gpu_timestamps, sizeof(gpu_timestamps[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    CHECK_VK(res, "Could not get query pool results");

    return 0;
}

static uint32_t print_results(VkPhysicalDevice pDevice, VkDevice device, spvtools::vksp_configuration &config,
    std::vector<spvtools::vksp_counter> &counters, uint64_t *gpu_timestamps,
    std::chrono::steady_clock::time_point *host_timestamps)
{
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(pDevice, &properties);
    double ns_per_tick = properties.limits.timestampPeriod;
    PRINT("ns_per_tick: %f", ns_per_tick);

    uint64_t gpu_cold_tick = gpu_timestamps[1] - gpu_timestamps[0];
    uint64_t gpu_hot_tick = gpu_timestamps[2] - gpu_timestamps[1];
    uint64_t gpu_total_tick = gpu_timestamps[2] - gpu_timestamps[0];

    uint64_t gpu_cold_ns = gpu_cold_tick * ns_per_tick;
    uint64_t gpu_hot_ns = gpu_hot_tick * ns_per_tick;
    uint64_t gpu_total_ns = gpu_total_tick * ns_per_tick;
    uint64_t gpu_avg_hot_ns = gpu_hot_ns / gHotRun;

    auto get_max_size = [](std::vector<const char *> messages) {
        std::vector<uint32_t> host_messages_length;
        std::for_each(messages.begin(), messages.end(),
            [&host_messages_length](const char *msg) { host_messages_length.push_back(strlen(msg)); });
        return *std::max_element(host_messages_length.begin(), host_messages_length.end());
    };

    printf("%s-%s-%u.%u.%u\n", config.shaderName, config.entryPoint, config.groupCountX, config.groupCountY,
        config.groupCountZ);
    const char *HOST_prefix = "HOST";
    const char *GPU_prefix = "GPU";
    const char *SHADER_prefix = "SHADER";
    const uint32_t prefix_max_size = get_max_size({ HOST_prefix, GPU_prefix, SHADER_prefix });

    std::vector<const char *> messages;
    const char *Submit_msg = "Submit";
    const char *WaitIdle_msg = "WaitIdle";
    const char *Total_msg = "Total";
    const char *Cold_msg = "Cold";
    const char *Hot_msg = "Hot";
    const char *Hot_avg_msg = "Hot avg";
    messages.assign({ Submit_msg, WaitIdle_msg, Total_msg, Cold_msg, Hot_msg, Hot_avg_msg });
    for (auto &counter : counters) {
        messages.push_back(counter.name);
    }
    const uint32_t messages_max_size = get_max_size(messages);

    auto print_separator = [messages_max_size, prefix_max_size]() {
        for (unsigned i = 0; i < messages_max_size + prefix_max_size; i++) {
            printf("-");
        }
        printf("---------------\n");
    };

    print_separator();

    print_time(std::chrono::duration_cast<std::chrono::nanoseconds>(host_timestamps[1] - host_timestamps[0]).count(),
        HOST_prefix, prefix_max_size, Submit_msg, messages_max_size);
    print_time(std::chrono::duration_cast<std::chrono::nanoseconds>(host_timestamps[2] - host_timestamps[1]).count(),
        HOST_prefix, prefix_max_size, WaitIdle_msg, messages_max_size);
    print_time(std::chrono::duration_cast<std::chrono::nanoseconds>(host_timestamps[2] - host_timestamps[0]).count(),
        HOST_prefix, prefix_max_size, Total_msg, messages_max_size);

    print_separator();

    print_time(gpu_total_ns, GPU_prefix, prefix_max_size, Total_msg, messages_max_size);
    print_time(gpu_cold_ns, GPU_prefix, prefix_max_size, Cold_msg, messages_max_size);
    print_time(gpu_hot_ns, GPU_prefix, prefix_max_size, Hot_msg, messages_max_size);
    if (gHotRun != 1) {
        print_time(gpu_avg_hot_ns, GPU_prefix, prefix_max_size, Hot_avg_msg, messages_max_size);
    }

    if (counters.size() == 0) {
        return 0;
    }

    print_separator();

    uint64_t *values;
    auto countersSize = counters_size(counters);
    VkResult res = vkMapMemory(device, gCounterMemory, 0, countersSize, 0, (void **)&values);
    CHECK_VK(res, "Could not map memory for counter");

    uint64_t nb_invocations = values[0];
    double entryPoint_tick = values[1];
    PRINT("Number of invocations: %lu", nb_invocations);

    for (auto &counter : counters) {
        print_prefix(prefix_max_size, SHADER_prefix, messages_max_size, counter.name);
        printf("%*.1f%%\n", 5, values[counter.index] * 100.0 / entryPoint_tick);
    }

    vkUnmapMemory(device, gCounterMemory);

    return 0;
}

static void help()
{
    printf("USAGE: vulkan-shader-profiler-runner [OPTIONS] -i <input>\n"
           "\n"
           "OPTIONS:\n"
           "\t-h\tDisplay this help and exit\n"
           "\t-m\tNumber of cold run\n"
           "\t-n\tNumber of hot run\n"
           "\t-v\tVerbose mode\n");
}

static bool parse_args(int argc, char **argv)
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
    std::vector<spvtools::vksp_counter> counters;
    spvtools::vksp_configuration config;
    CHECK(extract_from_input(shader, dsVector, pcVector, meVector, counters, config),
        "Could not extract data from input");
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

    for (auto &ds : dsVector) {
        switch (ds.type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER:
            PRINT("descriptor_set: ds %u binding %u type %u BUFFER size %u flags %u queueFamilyIndexCount %u "
                  "sharingMode %u usage %u range %u offset %u memorySize %u memoryType %u bindOffset %u",
                ds.ds, ds.binding, ds.type, ds.buffer.size, ds.buffer.flags, ds.buffer.queueFamilyIndexCount,
                ds.buffer.sharingMode, ds.buffer.usage, ds.buffer.range, ds.buffer.offset, ds.buffer.memorySize,
                ds.buffer.memoryType, ds.buffer.bindOffset);
            CHECK(handle_descriptor_set_buffer(ds, device, cmdBuffer, memProperties, descSet) == 0,
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

    uint64_t gpu_timestamps[gNbGpuTimestamps];
    std::chrono::steady_clock::time_point host_timestamps[3];
    CHECK(
        execute(device, cmdBuffer, queue, config, counters, gpu_timestamps, host_timestamps) == 0, "Could not execute");
    PRINT("Execution completed");

    CHECK(print_results(pDevice, device, config, counters, gpu_timestamps, host_timestamps) == 0,
        "Could not print all results");

    return 0;
}
