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

#include <stdint.h>
#include <vulkan/vulkan.h>

#define VKSP_EXTINST_STR "NonSemantic.VkspReflection.3"

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

namespace vksp {

struct vksp_push_constant {
    uint32_t offset;
    uint32_t size;
    uint32_t stageFlags;
    const char *pValues;
};

#define VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER_BITS (0xf0000000)
#define VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER_MASK (~VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER_BITS)
#define VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER                                                                    \
    (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER | VKSP_DESCRIPTOR_TYPE_STORAGE_BUFFER_COUNTER_BITS)
struct vksp_descriptor_set {
    uint32_t ds;
    uint32_t binding;
    uint32_t type;
    union {
        struct {
            uint32_t flags;
            uint32_t queueFamilyIndexCount;
            uint32_t sharingMode;
            uint32_t size;
            uint32_t usage;
            uint32_t range;
            uint32_t offset;
            uint32_t memorySize;
            uint32_t memoryType;
            uint32_t bindOffset;
            uint32_t viewFlags;
            uint32_t viewFormat;
        } buffer;
        struct {
            uint32_t imageLayout;
            uint32_t imageFlags;
            uint32_t imageType;
            uint32_t format;
            uint32_t width;
            uint32_t height;
            uint32_t depth;
            uint32_t mipLevels;
            uint32_t arrayLayers;
            uint32_t samples;
            uint32_t tiling;
            uint32_t usage;
            uint32_t sharingMode;
            uint32_t queueFamilyIndexCount;
            uint32_t initialLayout;
            uint32_t aspectMask;
            uint32_t baseMipLevel;
            uint32_t levelCount;
            uint32_t baseArrayLayer;
            uint32_t layerCount;
            uint32_t viewFlags;
            uint32_t viewType;
            uint32_t viewFormat;
            uint32_t component_a;
            uint32_t component_b;
            uint32_t component_g;
            uint32_t component_r;
            uint32_t memorySize;
            uint32_t memoryType;
            uint32_t bindOffset;
        } image;
        struct {
            uint32_t flags;
            uint32_t magFilter;
            uint32_t minFilter;
            uint32_t mipmapMode;
            uint32_t addressModeU;
            uint32_t addressModeV;
            uint32_t addressModeW;
            union {
                float fMipLodBias;
                uint32_t uMipLodBias;
            };
            uint32_t anisotropyEnable;
            union {
                float fMaxAnisotropy;
                uint32_t uMaxAnisotropy;
            };
            uint32_t compareEnable;
            uint32_t compareOp;
            union {
                float fMinLod;
                uint32_t uMinLod;
            };
            union {
                float fMaxLod;
                uint32_t uMaxLod;
            };
            uint32_t borderColor;
            uint32_t unnormalizedCoordinates;
        } sampler;
    };
};

struct vksp_configuration {
    const char *enabledExtensionNames;
    uint32_t specializationInfoDataSize;
    const char *specializationInfoData;
    const char *shaderName;
    const char *entryPoint;
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
    uint32_t dispatchId;
};

struct vksp_specialization_map_entry {
    uint32_t constantID;
    uint32_t offset;
    uint32_t size;
};

struct vksp_counter {
    uint32_t index;
    const char *name;
};

}
