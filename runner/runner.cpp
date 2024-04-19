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

#include <spirv-tools/libspirv.h>
#include <spirv-tools/optimizer.hpp>
#include <vulkan/vulkan.h>

static bool gVerbose = false;

#include "common/buffers_file.hpp"
#include "common/common.hpp"
#include "common/spirv-extract.hpp"

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <map>
#include <set>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <vector>

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

static std::string gInput = "";
static std::string gBuffersInput = "";
static std::unique_ptr<vksp::BuffersFile> gBuffersContents;
static vksp::buffers_map *gBuffersMap = nullptr;
static uint32_t gColdRun = 0, gHotRun = 1;
static VkBuffer gCounterBuffer = VK_NULL_HANDLE;
static VkDeviceMemory gCounterMemory;
static spv_target_env gSpvTargetEnv = SPV_ENV_VULKAN_1_3;
static bool gDisableCounters = false;
static uint32_t gPriority = UINT32_MAX;
static uint32_t gOutputDs = UINT32_MAX;
static uint32_t gOutputBinding = UINT32_MAX;
static VkBuffer gOutputBuffer = VK_NULL_HANDLE;
static VkImage gOutputImage = VK_NULL_HANDLE;
static vksp::vksp_descriptor_set *gOutputDsPtr = nullptr;
static std::string gOutputString;
static VkDeviceMemory gOutputMemory;
static VkDeviceSize gOutputMemorySize;

static const uint32_t gNbGpuTimestamps = 3;

static VkInstance gInstance;
static VkCommandPool gCmdPool;
static std::vector<VkBuffer> gBuffers;
static std::vector<VkDeviceMemory> gMemories;
static std::vector<VkImage> gImages;
static std::vector<VkImageView> gImageViews;
static std::vector<VkSampler> gSamplers;
static VkDescriptorPool gDescPool;
static VkShaderModule gShaderModule;

static std::vector<const char *> split_string(std::string input, const char *delimiter)
{
    std::vector<const char *> vector;
    size_t pos = 0;
    size_t delimiter_size = strlen(delimiter);
    while ((pos = input.find(delimiter)) != std::string::npos) {
        auto extension = input.substr(0, pos);
        vector.push_back(strdup(extension.c_str()));
        input.erase(0, pos + delimiter_size);
    }
    vector.push_back(strdup(input.c_str()));
    return vector;
}

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

    res = vkCreateInstance(&info, nullptr, &gInstance);
    CHECK_VK(res, "Could not create vulkan instance");

    uint32_t nbDevices;
    res = vkEnumeratePhysicalDevices(gInstance, &nbDevices, nullptr);
    CHECK_VK(res, "Could not enumerate physical devices");

    std::vector<VkPhysicalDevice> physicalDevices(nbDevices);
    res = vkEnumeratePhysicalDevices(gInstance, &nbDevices, physicalDevices.data());
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
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                break;
            }
        }
    }
    CHECK(queueFamilyIndex != UINT32_MAX, "Could not find a VK_QUEUE_COMPUTE_BIT queue");

    void *globalPriority = nullptr;

    VkDeviceQueueGlobalPriorityCreateInfoKHR globalPriorityCreateInfo;
    const uint32_t nb_priorities = 4;
    VkQueueGlobalPriorityKHR priorities[nb_priorities] = { VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
        VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR, VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR, VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR };
    if (gPriority != UINT32_MAX) {
        gPriority = std::min(gPriority, nb_priorities - 1);

        globalPriorityCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR;
        globalPriorityCreateInfo.pNext = nullptr;
        globalPriorityCreateInfo.globalPriority = priorities[gPriority];
        globalPriority = &globalPriorityCreateInfo;
    }

    std::vector<float> queuePriorities(nbQueues, 1.0f);
    VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, globalPriority, 0,
        queueFamilyIndex, nbQueues, queuePriorities.data() };

    size_t pos = 0;
    std::string extensionsStr = std::string(enabledExtensionNames);
    extensionsStr.erase(0, 1); // remove first '.'
    std::vector<const char *> extensions = split_string(extensionsStr, ".");
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

    const VkCommandPoolCreateInfo pCreateInfo
        = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, queueFamilyIndex };
    res = vkCreateCommandPool(device, &pCreateInfo, nullptr, &gCmdPool);
    CHECK_VK(res, "Could not create command pool");

    const VkCommandBufferAllocateInfo pAllocateInfo
        = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, gCmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    res = vkAllocateCommandBuffers(device, &pAllocateInfo, &cmdBuffer);
    CHECK_VK(res, "Could not allocate command buffer");

    const VkCommandBufferBeginInfo pBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    res = vkBeginCommandBuffer(cmdBuffer, &pBeginInfo);
    CHECK_VK(res, "Could not begin command buffer");

    return 0;
}

static uint32_t allocate_buffer(VkDevice device, VkPhysicalDeviceMemoryProperties &memProperties,
    vksp::vksp_descriptor_set &ds, VkBuffer &buffer, VkDeviceMemory &memory)
{
    const VkBufferCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, ds.buffer.size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr };

    VkResult res = vkCreateBuffer(device, &pCreateInfo, nullptr, &buffer);
    CHECK_VK(res, "Could not create initialization buffer");
    gBuffers.push_back(buffer);

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
    CHECK(memoryTypeFound, "Could not find memoryType for initialization buffer");

    const VkMemoryAllocateInfo pAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        ds.buffer.memorySize,
        ds.buffer.memoryType,
    };
    res = vkAllocateMemory(device, &pAllocateInfo, nullptr, &memory);
    CHECK_VK(res, "Could not allocate memory for initialization buffer");
    gMemories.push_back(memory);

    res = vkBindBufferMemory(device, buffer, memory, ds.buffer.bindOffset);
    CHECK_VK(res, "Could not bind buffer memory for initialization buffer");

    return 0;
}

static uint32_t initialize_buffer(VkDevice device, VkCommandBuffer cmdBuffer,
    VkPhysicalDeviceMemoryProperties &memProperties, vksp::vksp_descriptor_set &ds, VkBuffer shaderBuffer)
{
    if (gBuffersMap == nullptr) {
        return 0;
    }
    uint32_t dstSet = ds.ds;
    uint32_t dstBinding = ds.binding;
    vksp::buffer_map_key key = std::make_pair(dstSet, dstBinding);
    auto find = gBuffersMap->find(key);
    if (find == gBuffersMap->end()) {
        return 0;
    }
    void *buffer_data = find->second.second;
    CHECK(find->second.first == ds.buffer.memorySize, "mismatch in memorySize during initialization buffer");

    VkBuffer buffer;
    VkDeviceMemory memory;
    CHECK(allocate_buffer(device, memProperties, ds, buffer, memory) == 0, "Could not allocate initialization buffer");

    void *memory_data;
    VkResult res = vkMapMemory(device, memory, 0, ds.buffer.memorySize, 0, &memory_data);
    CHECK_VK(res, "Could not map memory for initialization buffer");
    memcpy(memory_data, buffer_data, ds.buffer.memorySize);
    vkUnmapMemory(device, memory);

    const VkBufferCopy pRegion = { 0, 0, ds.buffer.size };
    vkCmdCopyBuffer(cmdBuffer, buffer, shaderBuffer, 1, &pRegion);

    return 0;
}

static uint32_t handle_descriptor_set_buffer(vksp::vksp_descriptor_set &ds, VkDevice device, VkCommandBuffer cmdBuffer,
    VkPhysicalDeviceMemoryProperties &memProperties, std::vector<VkDescriptorSet> &descSet)
{
    VkResult res;

    bool bCounter = ds.type == VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER;
    ds.type &= VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER_MASK;

    VkBuffer buffer;
    const VkBufferCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, ds.buffer.flags,
        ds.buffer.size, ds.buffer.usage, (VkSharingMode)ds.buffer.sharingMode, 0, nullptr };
    res = vkCreateBuffer(device, &pCreateInfo, nullptr, &buffer);
    CHECK_VK(res, "Could not create buffer");
    gBuffers.push_back(buffer);

    if (gOutputDs == ds.ds && gOutputBinding == ds.binding) {
        gOutputBuffer = buffer;
        gOutputDsPtr = &ds;
    }

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
        CHECK(ds.buffer.memorySize <= memreqs.size, "memorySize for counter buffer does not match");
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
    gMemories.push_back(memory);

    res = vkBindBufferMemory(device, buffer, memory, ds.buffer.bindOffset);
    CHECK_VK(res, "Could not bind buffer and memory");

    CHECK(
        initialize_buffer(device, cmdBuffer, memProperties, ds, buffer) == 0, "Could not initialize memory for buffer");

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

static uint32_t allocate_image(VkDevice device, VkPhysicalDeviceMemoryProperties &memProperties,
    vksp::vksp_descriptor_set &ds, VkImage &image, VkDeviceMemory &memory)
{
    VkExtent3D extent = { ds.image.width, ds.image.height, ds.image.depth };
    const VkImageCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, ds.image.imageFlags,
        (VkImageType)ds.image.imageType, (VkFormat)ds.image.format, extent, ds.image.mipLevels, ds.image.arrayLayers,
        (VkSampleCountFlagBits)ds.image.samples, (VkImageTiling)ds.image.tiling, VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, (VkImageLayout)ds.image.initialLayout };

    VkResult res = vkCreateImage(device, &pCreateInfo, nullptr, &image);
    CHECK_VK(res, "Could not create initialization image");
    gImages.push_back(image);

    VkMemoryRequirements memreqs;
    vkGetImageMemoryRequirements(device, image, &memreqs);
    bool memoryTypeFound = false;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        auto dev_properties = memProperties.memoryTypes[i].propertyFlags;
        bool valid = (1ULL << i) & memreqs.memoryTypeBits;
        auto required_properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        bool satisfactory = (dev_properties & required_properties) == required_properties;
        if (satisfactory && valid) {
            ds.image.memoryType = i;
            memoryTypeFound = true;
            break;
        }
    }
    CHECK(memoryTypeFound, "Could not find memoryType for initialization image");

    const VkMemoryAllocateInfo pAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        ds.image.memorySize,
        ds.image.memoryType,
    };

    res = vkAllocateMemory(device, &pAllocateInfo, nullptr, &memory);
    CHECK_VK(res, "Could not allocate memory for initialization image");
    gMemories.push_back(memory);

    res = vkBindImageMemory(device, image, memory, ds.image.bindOffset);
    CHECK_VK(res, "Could not bind memory for initialization image");

    return 0;
}

static uint32_t initialize_image(VkDevice device, VkCommandBuffer cmdBuffer,
    VkPhysicalDeviceMemoryProperties &memProperties, vksp::vksp_descriptor_set &ds, VkImage shaderImage,
    VkImageSubresourceRange &subresourceRange)
{
    if (gBuffersMap == nullptr) {
        return 0;
    }
    uint32_t dstSet = ds.ds;
    uint32_t dstBinding = ds.binding;
    vksp::buffer_map_key key = std::make_pair(dstSet, dstBinding);
    auto find = gBuffersMap->find(key);
    if (find == gBuffersMap->end()) {
        return 0;
    }
    void *image_data = find->second.second;
    CHECK(find->second.first == ds.image.memorySize, "mismatch in memorySize during initialization buffer");

    VkImage image;
    VkDeviceMemory memory;
    CHECK(allocate_image(device, memProperties, ds, image, memory) == 0, "Could not allocate initialization image");

    void *memory_data;
    VkResult res = vkMapMemory(device, memory, 0, ds.image.memorySize, 0, &memory_data);
    CHECK_VK(res, "Could not map memory for initialization image");
    memcpy(memory_data, image_data, ds.image.memorySize);
    vkUnmapMemory(device, memory);

    VkImageMemoryBarrier imageBarrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        0,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        (VkImageLayout)ds.image.initialLayout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        0,
        0,
        image,
        subresourceRange,
    };
    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
        nullptr, 0, nullptr, 1, &imageBarrier);

    const VkImageCopy pRegion = {
        .srcSubresource
        = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .srcOffset = { 0, 0, 0 },
        .dstSubresource
        = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .dstOffset = { 0, 0, 0 },
        .extent = { ds.image.width, ds.image.height, ds.image.depth },
    };

    vkCmdCopyImage(
        cmdBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shaderImage, VK_IMAGE_LAYOUT_GENERAL, 1, &pRegion);

    return 0;
}

static uint32_t handle_descriptor_set_image(vksp::vksp_descriptor_set &ds, VkDevice device, VkCommandBuffer cmdBuffer,
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
    CHECK_VK(res, "Could not create image");
    gImages.push_back(image);

    if (gOutputDs == ds.ds && gOutputBinding == ds.binding) {
        gOutputImage = image;
        gOutputDsPtr = &ds;
    }

    const VkMemoryAllocateInfo pAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        ds.image.memorySize,
        ds.image.memoryType,
    };

    VkDeviceMemory memory;
    res = vkAllocateMemory(device, &pAllocateInfo, nullptr, &memory);
    CHECK_VK(res, "Could not allocate memory for image");
    gMemories.push_back(memory);

    res = vkBindImageMemory(device, image, memory, ds.image.bindOffset);
    CHECK_VK(res, "Could not bind image and memory");

    VkImageSubresourceRange subresourceRange = { ds.image.aspectMask, ds.image.baseMipLevel, ds.image.levelCount,
        ds.image.baseArrayLayer, ds.image.layerCount };

    VkImageMemoryBarrier imageBarrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        0,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        (VkImageLayout)ds.image.initialLayout,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        0,
        image,
        subresourceRange,
    };
    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
        nullptr, 0, nullptr, 1, &imageBarrier);

    CHECK(initialize_image(device, cmdBuffer, memProperties, ds, image, subresourceRange) == 0,
        "Could not initalize memory for image");

    imageBarrier.srcAccessMask = imageBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
        nullptr, 0, nullptr, 1, &imageBarrier);

    VkImageView image_view;
    VkComponentMapping components
        = { (VkComponentSwizzle)ds.image.component_r, (VkComponentSwizzle)ds.image.component_g,
              (VkComponentSwizzle)ds.image.component_b, (VkComponentSwizzle)ds.image.component_a };
    const VkImageViewCreateInfo pViewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, ds.image.viewFlags,
        image, (VkImageViewType)ds.image.viewType, (VkFormat)ds.image.viewFormat, components, subresourceRange };
    res = vkCreateImageView(device, &pViewInfo, nullptr, &image_view);
    CHECK_VK(res, "Could not create image view");
    gImageViews.push_back(image_view);

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

static uint32_t handle_descriptor_set_sampler(vksp::vksp_descriptor_set &ds, VkDevice device, VkCommandBuffer cmdBuffer,
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
    CHECK_VK(res, "Could not create sampler");
    gSamplers.push_back(sampler);

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
    std::vector<vksp::vksp_descriptor_set> &dsVector, std::vector<VkDescriptorSetLayout> &descSetLayoutVector)
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
        CHECK_VK(res, "Could not create descriptor set layout");

        descSetLayoutVector.push_back(descSetLayout);
    }

    std::vector<VkDescriptorPoolSize> descPoolSize;
    for (auto it : descTypeCount) {
        descPoolSize.push_back({ it.first, it.second });
    }

    const VkDescriptorPoolCreateInfo pCreateInfo
        = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
              (uint32_t)descSet.size(), (uint32_t)descPoolSize.size(), descPoolSize.data() };
    res = vkCreateDescriptorPool(device, &pCreateInfo, nullptr, &gDescPool);
    CHECK_VK(res, "Could not create descriptor pool");

    const VkDescriptorSetAllocateInfo pAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
        gDescPool, (uint32_t)descSetLayoutVector.size(), descSetLayoutVector.data() };
    res = vkAllocateDescriptorSets(device, &pAllocateInfo, descSet.data());
    CHECK_VK(res, "Could not allocate descriptor sets");

    return 0;
}

static uint32_t count_descriptor_set(std::vector<vksp::vksp_descriptor_set> dsVector)
{
    std::set<uint32_t> ds_set;
    for (auto &ds : dsVector) {
        ds_set.insert(ds.ds);
    }
    return ds_set.size();
}

static uint32_t handle_push_constant(std::vector<vksp::vksp_push_constant> &pcVector, VkPushConstantRange &range,
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

static uint32_t allocate_pipeline_layout(VkDevice device, std::vector<vksp::vksp_push_constant> &pcVector,
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
    VkCommandBuffer cmdBuffer, std::vector<vksp::vksp_specialization_map_entry> &meVector,
    vksp::vksp_configuration &config, VkPipeline &pipeline)
{
    VkResult res;
    const VkShaderModuleCreateInfo shaderModuleCreateInfo
        = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, shader.size() * sizeof(uint32_t), shader.data() };
    res = vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &gShaderModule);
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
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, gShaderModule,
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

static uint32_t counters_size(std::vector<vksp::vksp_counter> &counters)
{
    return sizeof(uint64_t) * (2 + counters.size());
}

static uint32_t record_dump_output(
    VkDevice device, VkCommandBuffer cmdBuffer, VkPhysicalDeviceMemoryProperties &memProperties)
{
    switch (gOutputDsPtr->type) {
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
        VkBuffer buffer;
        CHECK(allocate_buffer(device, memProperties, *gOutputDsPtr, buffer, gOutputMemory) == 0,
            "Could not allocate output buffer");
        gOutputMemorySize = gOutputDsPtr->buffer.memorySize;

        const VkBufferCopy pRegion = { 0, 0, gOutputDsPtr->buffer.size };
        vkCmdCopyBuffer(cmdBuffer, gOutputBuffer, buffer, 1, &pRegion);
    } break;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
        VkImage image;
        CHECK(allocate_image(device, memProperties, *gOutputDsPtr, image, gOutputMemory) == 0,
            "Could not allocate output image");
        gOutputMemorySize = gOutputDsPtr->image.memorySize;

        VkImageSubresourceRange subresourceRange = { gOutputDsPtr->image.aspectMask, gOutputDsPtr->image.baseMipLevel,
            gOutputDsPtr->image.levelCount, gOutputDsPtr->image.baseArrayLayer, gOutputDsPtr->image.layerCount };

        VkImageMemoryBarrier imageBarrier = {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0,
            0,
            image,
            subresourceRange,
        };
        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
            nullptr, 0, nullptr, 1, &imageBarrier);

        const VkImageCopy pRegion = {
            .srcSubresource
            = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .srcOffset = { 0, 0, 0 },
            .dstSubresource
            = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .dstOffset = { 0, 0, 0 },
            .extent = { gOutputDsPtr->image.width, gOutputDsPtr->image.height, gOutputDsPtr->image.depth },
        };

        vkCmdCopyImage(
            cmdBuffer, gOutputImage, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &pRegion);
    } break;
    default:
        PRINT("Unsupported descritpor set type");
        return -1;
    }
    return 0;
}

static uint32_t dump_output(VkDevice device)
{
    void *data;
    VkResult res = vkMapMemory(device, gOutputMemory, 0, gOutputMemorySize, 0, &data);
    CHECK_VK(res, "Could not map output memory");

    std::string filename(gInput);
    filename += "." + gOutputString + ".buffer";
    FILE *fd = fopen(filename.c_str(), "w");
    size_t byte_written = 0;
    while (byte_written != gOutputMemorySize) {
        byte_written += fwrite(&(((char *)data)[byte_written]), sizeof(char), gOutputMemorySize - byte_written, fd);
    }
    fclose(fd);

    vkUnmapMemory(device, gOutputMemory);
    return 0;
}

static uint32_t execute(VkDevice device, VkCommandBuffer cmdBuffer, VkQueue queue, vksp::vksp_configuration &config,
    std::vector<vksp::vksp_counter> &counters, uint64_t *gpu_timestamps,
    std::chrono::steady_clock::time_point *host_timestamps, VkPhysicalDeviceMemoryProperties &memProperties)
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

    VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT };

    for (unsigned i = 0; i < gColdRun; i++) {
        vkCmdDispatch(cmdBuffer, config.groupCountX, config.groupCountY, config.groupCountZ);
        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
            &memoryBarrier, 0, nullptr, 0, nullptr);
    }

    if (gCounterBuffer != VK_NULL_HANDLE) {
        auto countersSize = counters_size(counters);
        auto zero = malloc(countersSize);
        memset(zero, 0, countersSize);
        vkCmdUpdateBuffer(cmdBuffer, gCounterBuffer, 0, countersSize, zero);
    }

    vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1);
    for (unsigned i = 0; i < gHotRun; i++) {
        vkCmdDispatch(cmdBuffer, config.groupCountX, config.groupCountY, config.groupCountZ);
        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
            &memoryBarrier, 0, nullptr, 0, nullptr);
    }
    vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 2);

    if (gOutputDsPtr != nullptr) {
        CHECK(record_dump_output(device, cmdBuffer, memProperties) == 0, "Could not record dumping of output");
    }

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

    vkDestroyQueryPool(device, queryPool, nullptr);

    return 0;
}

static uint32_t print_results(VkPhysicalDevice pDevice, VkDevice device, vksp::vksp_configuration &config,
    std::vector<vksp::vksp_counter> &counters, uint64_t *gpu_timestamps,
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

void clean_vk_objects(VkDevice device, VkCommandBuffer cmdBuffer, std::vector<VkDescriptorSet> &descSet,
    std::vector<VkDescriptorSetLayout> &descSetLayoutVector, VkPipelineLayout pipelineLayout, VkPipeline pipeline)
{
    vkDestroyShaderModule(device, gShaderModule, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for (auto sampler : gSamplers) {
        vkDestroySampler(device, sampler, nullptr);
    }
    for (auto imageView : gImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    for (auto image : gImages) {
        vkDestroyImage(device, image, nullptr);
    }
    for (auto buffer : gBuffers) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    for (auto memory : gMemories) {
        vkFreeMemory(device, memory, nullptr);
    }
    vkFreeDescriptorSets(device, gDescPool, descSet.size(), descSet.data());
    vkDestroyDescriptorPool(device, gDescPool, nullptr);
    for (auto descSetLayout : descSetLayoutVector) {
        vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
    }
    vkFreeCommandBuffers(device, gCmdPool, 1, &cmdBuffer);
    vkDestroyCommandPool(device, gCmdPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(gInstance, nullptr);
}

static void help()
{
    printf("USAGE: vulkan-shader-profiler-runner [OPTIONS] -i <input>\n"
           "\n"
           "OPTIONS:\n"
           "\t-b\tBuffers file associated to the input generated when tracing\n"
           "\t-c\tDisable counters\n"
           "\t-e\tspv_target_env to use (default: vulkan1.3)\n"
           "\t-h\tDisplay this help and exit\n"
           "\t-m\tNumber of cold run\n"
           "\t-n\tNumber of hot run\n"
           "\t-o\tDescriptor set index and binding of a buffer to dump after the execution (example: '1.2')\n"
           "\t-p\tGlobal priority (0:low 1:medium 2:high 3:realtime default:unspecified)\n"
           "\t-v\tVerbose mode\n");
}

static bool parse_args(int argc, char **argv)
{
    bool bHelp = false;
    int c;
    while ((c = getopt(argc, argv, "chvi:n:m:e:p:b:o:")) != -1) {
        switch (c) {
        case 'b':
            gBuffersInput = std::string(optarg);
            break;
        case 'c':
            gDisableCounters = true;
            break;
        case 'e': {
            spv_target_env env;
            if (spvParseTargetEnv(optarg, &env)) {
                gSpvTargetEnv = env;
            } else {
                ERROR("Could not parse spv target env, using default: '%s'", spvTargetEnvDescription(gSpvTargetEnv));
            }

        } break;
        case 'n':
            gHotRun = atoi(optarg);
            break;
        case 'm':
            gColdRun = atoi(optarg);
            break;
        case 'o': {
            gOutputString = std::string(optarg);
            auto splits = split_string(gOutputString, ".");
            if (splits.size() == 2) {
                gOutputDs = atoi(splits[0]);
                gOutputBinding = atoi(splits[1]);
            } else {
                ERROR("'%s' does not match output (expected: 'ds.binding')", gOutputString.c_str());
                bHelp = true;
            }
        } break;
        case 'p':
            gPriority = atoi(optarg);
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
    PRINT("Arguments parsed: input '%s' verbose '%u' spv_target_env '%s' hot_runs '%u' cold_runs '%u' buffers '%s' "
          "output_ds '%u' output_binding '%u' counters '%u' priority '%u'",
        gInput.c_str(), gVerbose, spvTargetEnvDescription(gSpvTargetEnv), gHotRun, gColdRun, gBuffersInput.c_str(),
        gOutputDs, gOutputBinding, gDisableCounters, gPriority);

    std::vector<uint32_t> shader;
    std::vector<vksp::vksp_descriptor_set> dsVector;
    std::vector<vksp::vksp_push_constant> pcVector;
    std::vector<vksp::vksp_specialization_map_entry> meVector;
    std::vector<vksp::vksp_counter> counters;
    vksp::vksp_configuration config;
    CHECK(extract_from_input(gInput.c_str(), gSpvTargetEnv, gDisableCounters, gVerbose, shader, dsVector, pcVector,
              meVector, counters, config),
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

    if (gBuffersInput != "") {
        gBuffersContents = std::make_unique<vksp::BuffersFile>(config.dispatchId);
        if (gBuffersContents->ReadFromFile(gBuffersInput.c_str())) {
            gBuffersMap = gBuffersContents->GetBuffers();
        }
    }

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
    std::map<uint32_t, std::vector<vksp::vksp_push_constant>> pcMap;
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
    CHECK(execute(device, cmdBuffer, queue, config, counters, gpu_timestamps, host_timestamps, memProperties) == 0,
        "Could not execute");
    PRINT("Execution completed");

    CHECK(print_results(pDevice, device, config, counters, gpu_timestamps, host_timestamps) == 0,
        "Could not print all results");

    if (gOutputDsPtr != nullptr) {
        CHECK(dump_output(device) == 0, "Could not dump output memory");
    }

    clean_vk_objects(device, cmdBuffer, descSet, descSetLayoutVector, pipelineLayout, pipeline);

    return 0;
}
