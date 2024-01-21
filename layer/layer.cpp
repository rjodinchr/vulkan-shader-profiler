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

#include "vulkan/vk_layer.h"
#include "vulkan/vulkan.h"

#include "spirv-tools/libspirv.h"

#include <condition_variable>
#include <perfetto.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <thread>

/*****************************************************************************/
/* PERFETTO GLOBAL VARIABLES *************************************************/
/*****************************************************************************/

#define VKSP_PERFETTO_CATEGORY "vksp"

PERFETTO_DEFINE_CATEGORIES(perfetto::Category(VKSP_PERFETTO_CATEGORY).SetDescription("Vulkan Shader Profiler Events"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

#ifdef BACKEND_INPROCESS
static std::unique_ptr<perfetto::TracingSession> gTracingSession;
#endif

/*****************************************************************************/
/* MACROS ********************************************************************/
/*****************************************************************************/

#define GET_PROC_ADDR(func)                                                                                            \
    if (!strcmp(pName, "vk" #func))                                                                                    \
        return (PFN_vkVoidFunction) & vksp_##func;

#define SET_DISPATCH_TABLE(table, func, pointer, gpa, str, statement)                                                  \
    table.func = (PFN_vk##func)gpa(*pointer, "vk" #func);                                                              \
    if (dispatchTable.func == nullptr) {                                                                               \
        PRINT("Could not trace a " str " because '" #func "' is missing");                                             \
        statement;                                                                                                     \
    }

#define DISPATCH_TABLE_ELEMENT(func) PFN_vk##func func;

#define PRINT(message, ...)                                                                                            \
    do {                                                                                                               \
        fprintf(stderr, "[VKSP] %s: " message "\n", __func__, ##__VA_ARGS__);                                          \
        TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "PRINT", "message", perfetto::DynamicString(message));             \
    } while (0)

#define DISPATCH(obj) gdispatch[vksp_key(obj)]

/*****************************************************************************/
/* GLOBAL VARIABLES & TYPES **************************************************/
/*****************************************************************************/

// clang-format off
static std::string byteToStr[] = {
    "00", "01", "02", "03", "04", "05", "06", "07",
    "08", "09", "0a", "0b", "0c", "0d", "0e", "0f",
    "10", "11", "12", "13", "14", "15", "16", "17",
    "18", "19", "1a", "1b", "1c", "1d", "1e", "1f",
    "20", "21", "22", "23", "24", "25", "26", "27",
    "28", "29", "2a", "2b", "2c", "2d", "2e", "2f",
    "30", "31", "32", "33", "34", "35", "36", "37",
    "38", "39", "3a", "3b", "3c", "3d", "3e", "3f",
    "40", "41", "42", "43", "44", "45", "46", "47",
    "48", "49", "4a", "4b", "4c", "4d", "4e", "4f",
    "50", "51", "52", "53", "54", "55", "56", "57",
    "58", "59", "5a", "5b", "5c", "5d", "5e", "5f",
    "60", "61", "62", "63", "64", "65", "66", "67",
    "68", "69", "6a", "6b", "6c", "6d", "6e", "6f",
    "70", "71", "72", "73", "74", "75", "76", "77",
    "78", "79", "7a", "7b", "7c", "7d", "7e", "7f",
    "80", "81", "82", "83", "84", "85", "86", "87",
    "88", "89", "8a", "8b", "8c", "8d", "8e", "8f",
    "90", "91", "92", "93", "94", "95", "96", "97",
    "98", "99", "9a", "9b", "9c", "9d", "9e", "9f",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "a8", "a9", "aa", "ab", "ac", "ad", "ae", "af",
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
    "b8", "b9", "ba", "bb", "bc", "bd", "be", "bf",
    "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7",
    "c8", "c9", "ca", "cb", "cc", "cd", "ce", "cf",
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
    "d8", "d9", "da", "db", "dc", "dd", "de", "df",
    "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7",
    "e8", "e9", "ea", "eb", "ec", "ed", "ee", "ef",
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
    "f8", "f9", "fa", "fb", "fc", "fd", "fe", "ff",
};
// clang-format on

static std::mutex glock;

typedef struct DispatchTable_ {
#define FUNC_INS DISPATCH_TABLE_ELEMENT
#define FUNC_INS_INT DISPATCH_TABLE_ELEMENT
#define FUNC_DEV DISPATCH_TABLE_ELEMENT
#define FUNC_DEV_INT DISPATCH_TABLE_ELEMENT
#include "functions.def"
} DispatchTable;

template <typename DispatchableType> void *vksp_key(DispatchableType obj) { return *(void **)obj; }
static std::map<void *, DispatchTable> gdispatch;

static std::set<VkDevice> DeviceNotToTrace;
static std::map<VkQueue, VkDevice> QueueToDevice;
static std::map<VkDevice, VkPhysicalDevice> DeviceToPhysicalDevice;
static std::map<VkDevice, std::vector<std::pair<VkQueue, std::thread>>> QueueThreadPool;
static std::map<VkCommandBuffer, VkDevice> CmdBufferToDevice;
static std::map<VkCommandBuffer, VkPipeline> CmdBufferToPipeline;
static std::map<VkPipeline, VkShaderModule> PipelineToShaderModule;
static std::map<VkPipeline, std::string> PipelineToShaderModuleName;
static std::map<VkShaderModule, std::string> ShaderModuleToString;

struct ThreadDispatch {
    VkQueryPool query_pool;
    VkPipeline pipeline;
    uint64_t dispatchId;
    uint32_t groupCountX, groupCountY, groupCountZ;
};

struct ThreadJob {
    uint64_t timeline_id;
    std::queue<ThreadDispatch> dispatches;
};

struct ThreadInfo {
    ThreadInfo(VkDevice dev, VkQueue q)
    {
        device = dev;
        queue = q;
        stop = false;
        next_timeline_id = 1;
        sync_dev = sync_host = 0ULL;
        VkSemaphoreTypeCreateInfo timelineCreateInfo;
        timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timelineCreateInfo.pNext = NULL;
        timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineCreateInfo.initialValue = 0;

        VkSemaphoreCreateInfo info = {
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            &timelineCreateInfo,
            0,
        };

        auto res = DISPATCH(device).CreateSemaphore(device, &info, nullptr, &semaphore);
        if (res != VK_SUCCESS) {
            PRINT("vkCreateSemaphore failed (%d)", res);
            stop = true;
        }

        VkPhysicalDeviceProperties properties;
        DISPATCH(DeviceToPhysicalDevice[device])
            .GetPhysicalDeviceProperties(DeviceToPhysicalDevice[device], &properties);
        ns_per_tick = properties.limits.timestampPeriod;
    };
    ~ThreadInfo() { DISPATCH(device).DestroySemaphore(device, semaphore, nullptr); }

    VkDevice device;
    VkQueue queue;
    VkSemaphore semaphore;
    uint64_t next_timeline_id;
    bool stop;
    std::condition_variable cv;
    std::mutex lock;
    std::queue<ThreadJob *> jobs;
    uint64_t sync_dev, sync_host;
    double ns_per_tick;
};

static std::map<VkQueue, ThreadInfo *> QueueToThreadInfo;
static std::map<VkCommandBuffer, std::vector<ThreadDispatch>> CmdBufferToThreadDispatch;
static uint32_t shader_number = 0;

/*****************************************************************************/
/* PERFETTO TRACE PARAMETERS *************************************************/
/*****************************************************************************/

#ifdef BACKEND_INPROCESS
static const char *get_trace_dest()
{
    if (auto trace_dest = getenv("VKSP_TRACE_DEST")) {
        return trace_dest;
    }
    return TRACE_DEST;
}

static const uint32_t get_trace_max_size()
{
    if (auto trace_max_size = getenv("VKSP_TRACE_MAX_SIZE")) {
        return atoi(trace_max_size);
    }
    return TRACE_MAX_SIZE;
}
#endif

/*****************************************************************************/
/* QUEUE THREAD **************************************************************/
/*****************************************************************************/

static VkResult WaitSemaphore(ThreadInfo *info, ThreadJob *job)
{
    VkResult result;
    do {
        VkSemaphoreWaitInfo waitInfo;
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.pNext = NULL;
        waitInfo.flags = 0;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &info->semaphore;
        waitInfo.pValues = &job->timeline_id;

        TRACE_EVENT_BEGIN(VKSP_PERFETTO_CATEGORY, "vkWaitSemaphores", "value", job->timeline_id);
        result = DISPATCH(info->device).WaitSemaphores(info->device, &waitInfo, 1000000);
        TRACE_EVENT_END(VKSP_PERFETTO_CATEGORY);

        if (result != VK_TIMEOUT && result != VK_SUCCESS) {
            PRINT("vkWaitSemaphores failed (%d)", result);
            return result;
        }
    } while (result != VK_SUCCESS);
    return result;
}

static uint64_t timestamp_to_ns(ThreadInfo *info, uint64_t timestamp) { return info->ns_per_tick * timestamp; }

static void sync_timestamps(ThreadInfo *info, uint64_t &start, uint64_t &end)
{
    if (info->sync_host > info->sync_dev) {
        uint64_t deviation = info->sync_host - info->sync_dev;
        start += deviation;
        end += deviation;
    } else {
        uint64_t deviation = info->sync_dev - info->sync_host;
        start -= deviation;
        end -= deviation;
    }
}

#define NB_TIMESTAMP 2
static VkResult sync_timestamps_in_host_timeline(uint64_t &start, uint64_t &end, ThreadInfo *info)
{
    if (end < info->sync_dev) {
        sync_timestamps(info, start, end);
        return VK_SUCCESS;
    }
    uint64_t timestamps[NB_TIMESTAMP];
    uint64_t max_deviation;
    VkCalibratedTimestampInfoEXT timestamp_infos[NB_TIMESTAMP]
        = { { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT },
              { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_DEVICE_EXT } };

    auto res = DISPATCH(info->device)
                   .GetCalibratedTimestampsEXT(info->device, NB_TIMESTAMP, timestamp_infos, timestamps, &max_deviation);
    if (res != VK_SUCCESS) {
        PRINT("vkGetCalibratedTimestampsEXT failed (%d)", res);
        return res;
    }
    info->sync_host = timestamps[0];
    info->sync_dev = timestamp_to_ns(info, timestamps[1]);

    sync_timestamps(info, start, end);

    return VK_SUCCESS;
}

static ThreadJob *get_job(ThreadInfo *info)
{
    std::unique_lock lock(info->lock);
    while (info->jobs.empty()) {
        if (info->stop) {
            return (ThreadJob *)nullptr;
        }
        TRACE_EVENT_BEGIN(
            VKSP_PERFETTO_CATEGORY, "vksp_wait", "device", (void *)info->device, "queue", (void *)info->queue);
        info->cv.wait(lock);
        TRACE_EVENT_END(VKSP_PERFETTO_CATEGORY);
    }
    auto job = info->jobs.front();
    info->jobs.pop();
    return job;
}

static void GenerateTrace(ThreadInfo *info, ThreadDispatch &cmd)
{
    // Get timestamps
    uint64_t timestamps[NB_TIMESTAMP];
    VkResult result = DISPATCH(info->device)
                          .GetQueryPoolResults(info->device, cmd.query_pool, 0, NB_TIMESTAMP, sizeof(timestamps),
                              timestamps, sizeof(timestamps[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) {
        PRINT("vkGetQueryPoolResults failed (%d)", result);
        return;
    }
    DISPATCH(info->device).DestroyQueryPool(info->device, cmd.query_pool, nullptr);

    uint64_t start = timestamp_to_ns(info, timestamps[0]);
    uint64_t end = timestamp_to_ns(info, timestamps[1]);
    result = sync_timestamps_in_host_timeline(start, end, info);
    if (result != VK_SUCCESS) {
        return;
    }

    // Create perfetto trace
    std::string name = ShaderModuleToString[PipelineToShaderModule[cmd.pipeline]] + "-"
        + PipelineToShaderModuleName[cmd.pipeline] + "-" + std::to_string(cmd.groupCountX) + "."
        + std::to_string(cmd.groupCountY) + "." + std::to_string(cmd.groupCountZ);
    TRACE_EVENT_BEGIN(VKSP_PERFETTO_CATEGORY, perfetto::DynamicString(name), perfetto::Track((uintptr_t)info->queue),
        (uint64_t)start, "groupCountX", cmd.groupCountX, "groupCountY", cmd.groupCountY, "groupCountZ", cmd.groupCountZ,
        "dispatchId", cmd.dispatchId, "shader", ShaderModuleToString[PipelineToShaderModule[cmd.pipeline]],
        "shader_name", PipelineToShaderModuleName[cmd.pipeline]);
    TRACE_EVENT_END(VKSP_PERFETTO_CATEGORY, perfetto::Track((uintptr_t)info->queue), (uint64_t)end);
}

static void QueueThreadFct(ThreadInfo *info)
{
    pthread_setname_np(pthread_self(), "vksp");
    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY,
        perfetto::DynamicString("vksp-queue_" + std::to_string((uintptr_t)info->queue)),
        perfetto::Track((uintptr_t)info->queue));
    while (true) {
        auto job = get_job(info);
        if (job == nullptr) {
            return;
        }

        // Wait for job completion
        if (WaitSemaphore(info, job) != VK_SUCCESS) {
            delete job;
            continue;
        }

        TRACE_EVENT_BEGIN(VKSP_PERFETTO_CATEGORY, "GenerateTraces");
        while (!job->dispatches.empty()) {
            auto &cmd = job->dispatches.front();
            job->dispatches.pop();
            GenerateTrace(info, cmd);
        }
        TRACE_EVENT_END(VKSP_PERFETTO_CATEGORY);

        delete job;
    }
}

/*****************************************************************************/
/* INTERCEPT FUNCTIONS *******************************************************/
/*****************************************************************************/

void VKAPI_CALL vksp_GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue *pQueue)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkGetDeviceQueue", "device", (void *)device);

    DISPATCH(device).GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);

    auto info = new ThreadInfo(device, *pQueue);
    QueueToDevice[*pQueue] = device;
    QueueToThreadInfo[*pQueue] = info;
    QueueThreadPool[device].emplace_back(std::make_pair(*pQueue, [info] { QueueThreadFct(info); }));
}

VkResult VKAPI_CALL vksp_QueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkQueueSubmit", "queue", (void *)queue, "submitCount", submitCount);

    auto info = QueueToThreadInfo[queue];
    ThreadJob *job = new ThreadJob();
    {
        std::lock_guard<std::mutex> tlock(info->lock);
        job->timeline_id = info->next_timeline_id++;
    }

    VkSubmitInfo mSubmits[submitCount];
    std::vector<std::vector<VkSemaphore>> semaphores { submitCount };
    std::vector<std::vector<uint64_t>> signalValues { submitCount };
    std::vector<VkTimelineSemaphoreSubmitInfo> timelineInfos { submitCount };
    for (unsigned eachSubmit = 0; eachSubmit < submitCount; eachSubmit++) {
        auto &submit = pSubmits[eachSubmit];
        mSubmits[eachSubmit] = submit;

        for (unsigned i = 0; i < submit.signalSemaphoreCount; i++) {
            semaphores[eachSubmit].push_back(submit.pSignalSemaphores[i]);
        }
        semaphores[eachSubmit].push_back(info->semaphore);

        mSubmits[eachSubmit].signalSemaphoreCount = semaphores[eachSubmit].size();
        mSubmits[eachSubmit].pSignalSemaphores = semaphores[eachSubmit].data();

        VkTimelineSemaphoreSubmitInfo *next = (VkTimelineSemaphoreSubmitInfo *)submit.pNext;
        while (next != nullptr && next->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
            next = (VkTimelineSemaphoreSubmitInfo *)next->pNext;
        }

        if (next != nullptr && next->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
            for (unsigned i = 0; i < next->signalSemaphoreValueCount; i++) {
                signalValues[eachSubmit].push_back(next->pSignalSemaphoreValues[i]);
            }
            signalValues[eachSubmit].push_back(job->timeline_id);

            next->signalSemaphoreValueCount = signalValues[eachSubmit].size();
            next->pSignalSemaphoreValues = signalValues[eachSubmit].data();
        } else {
            VkTimelineSemaphoreSubmitInfo timelineInfo;
            timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timelineInfo.pNext = mSubmits[eachSubmit].pNext;
            timelineInfo.waitSemaphoreValueCount = 0;
            timelineInfo.pWaitSemaphoreValues = nullptr;
            timelineInfo.signalSemaphoreValueCount = 1;
            timelineInfo.pSignalSemaphoreValues = &job->timeline_id;

            timelineInfos.push_back(timelineInfo);
            mSubmits[eachSubmit].pNext = &timelineInfos.back();
        }
    }

    VkResult result = DISPATCH(QueueToDevice[queue]).QueueSubmit(queue, submitCount, mSubmits, fence);

    for (unsigned eachSubmit = 0; eachSubmit < submitCount; eachSubmit++) {
        auto &submit = pSubmits[eachSubmit];
        for (unsigned eachCmdBuffer = 0; eachCmdBuffer < submit.commandBufferCount; eachCmdBuffer++) {
            auto &cmd_buffer = submit.pCommandBuffers[eachCmdBuffer];
            for (auto &dispatch : CmdBufferToThreadDispatch[cmd_buffer]) {
                job->dispatches.push(dispatch);
            }
        }
    }

    std::lock_guard<std::mutex> tlock(info->lock);
    info->jobs.push(job);
    info->cv.notify_one();

    return result;
}

VkResult VKAPI_CALL vksp_AllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo, VkCommandBuffer *pCommandBuffers)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkAllocateCommandBuffers", "device", (void *)device);

    VkResult result = DISPATCH(device).AllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
    if (result != VK_SUCCESS) {
        return result;
    }

    for (unsigned i = 0; i < pAllocateInfo->commandBufferCount; i++) {
        CmdBufferToDevice[pCommandBuffers[i]] = device;
    }

    return result;
}

void VKAPI_CALL vksp_FreeCommandBuffers(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount, const VkCommandBuffer *pCommandBuffers)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkFreeCommandBuffers", "device", (void *)device, "commandBufferCount",
        commandBufferCount);

    for (unsigned i = 0; i < commandBufferCount; i++) {
        CmdBufferToDevice.erase(pCommandBuffers[i]);
        CmdBufferToThreadDispatch[pCommandBuffers[i]].clear();
    }

    DISPATCH(device).FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
}

VkResult VKAPI_CALL vksp_BeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkBeginCommandBuffer", "commandBuffer", (void *)commandBuffer);

    CmdBufferToThreadDispatch[commandBuffer].clear();

    return DISPATCH(CmdBufferToDevice[commandBuffer]).BeginCommandBuffer(commandBuffer, pBeginInfo);
}

static uint64_t dispatchId = 0;
void VKAPI_CALL vksp_CmdDispatch(
    VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    std::lock_guard<std::mutex> lock(glock);
    auto device = CmdBufferToDevice[commandBuffer];
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCmdDispatch", "device", (void *)device, "commandBuffer",
        (void *)commandBuffer, "groupCountX", groupCountX, "groupCountY", groupCountY, "groupCountZ", groupCountZ,
        "dispatchId", dispatchId);

    ThreadDispatch dispatch = {
        .pipeline = CmdBufferToPipeline[commandBuffer],
        .dispatchId = dispatchId++,
        .groupCountX = groupCountX,
        .groupCountY = groupCountY,
        .groupCountZ = groupCountZ,
    };

    VkQueryPoolCreateInfo query_pool_create_info = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        nullptr,
        0,
        VK_QUERY_TYPE_TIMESTAMP,
        NB_TIMESTAMP,
        0,
    };
    auto &d = DISPATCH(device);
    auto res
        = d.CreateQueryPool(CmdBufferToDevice[commandBuffer], &query_pool_create_info, nullptr, &dispatch.query_pool);
    if (res != VK_SUCCESS) {
        PRINT("vkCreateQueryPool failed (%d)", res);
        return d.CmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
    }

    d.CmdResetQueryPool(commandBuffer, dispatch.query_pool, 0, NB_TIMESTAMP);
    d.CmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dispatch.query_pool, 0);
    d.CmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
    d.CmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dispatch.query_pool, 1);

    CmdBufferToThreadDispatch[commandBuffer].push_back(dispatch);
}

void VKAPI_CALL vksp_CmdBindPipeline(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCmdBindPipeline", "commandBuffer", (void *)commandBuffer);

    CmdBufferToPipeline[commandBuffer] = pipeline;

    return DISPATCH(CmdBufferToDevice[commandBuffer]).CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
}

VkResult VKAPI_CALL vksp_CreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache,
    uint32_t createInfoCount, const VkComputePipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator,
    VkPipeline *pPipelines)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCreateComputePipelines", "device", (void *)device, "module",
        (void *)pCreateInfos->stage.module, "createInfoCount", createInfoCount);

    auto specializationInfo = pCreateInfos->stage.pSpecializationInfo;
    if (specializationInfo != nullptr) {
        std::string pData = "";
        for (unsigned i = 0; i < specializationInfo->dataSize; i++) {
            pData += byteToStr[((unsigned char *)specializationInfo->pData)[i]];
        }
        TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCreateComputePipelines-specialization", "module",
            (void *)pCreateInfos->stage.module, "mapEntryCount", specializationInfo->mapEntryCount, "dataSize",
            specializationInfo->dataSize, "pData", perfetto::DynamicString(pData));

        for (unsigned i = 0; i < specializationInfo->mapEntryCount; i++) {
            auto &mapEntry = specializationInfo->pMapEntries[i];
            TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCreateComputePipelines-MapEntry", "module",
                (void *)pCreateInfos->stage.module, "constantID", mapEntry.constantID, "offset", mapEntry.offset,
                "size", mapEntry.size);
        }
    }

    VkResult result = DISPATCH(device).CreateComputePipelines(
        device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);

    for (unsigned i = 0; i < createInfoCount; i++) {
        PipelineToShaderModule[pPipelines[i]] = pCreateInfos[i].stage.module;
        PipelineToShaderModuleName[pPipelines[i]] = std::string(pCreateInfos[i].stage.pName);
    }

    return result;
}

VkResult VKAPI_CALL vksp_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule)
{
    std::lock_guard<std::mutex> lock(glock);
    std::string shader_str = std::string("vksp_s") + std::to_string(shader_number++);

    const spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_3);
    spv_text text;
    spv_diagnostic diag;
    const uint32_t *code = pCreateInfo->pCode;
    const size_t code_size = pCreateInfo->codeSize / sizeof(uint32_t);
    spv_result_t spv_result = spvBinaryToText(context, code, code_size,
        SPV_BINARY_TO_TEXT_OPTION_INDENT | SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_COMMENT,
        &text, &diag);
    if (spv_result == SPV_SUCCESS) {
        TRACE_EVENT_BEGIN(VKSP_PERFETTO_CATEGORY, "vkCreateShaderModule", "device", (void *)device, "shader",
            perfetto::DynamicString(shader_str), "flags", pCreateInfo->flags, "code_size", code_size, "spv_result",
            spv_result, "text", perfetto::DynamicString(text->str));
        spvTextDestroy(text);
    } else {
        TRACE_EVENT_BEGIN(VKSP_PERFETTO_CATEGORY, "vkCreateShaderModule", "device", (void *)device, "shader",
            perfetto::DynamicString(shader_str), "flags", pCreateInfo->flags, "code_size", code_size, "spv_result",
            spv_result, "error", perfetto::DynamicString(diag->error));
        spvDiagnosticDestroy(diag);
    }
    spvContextDestroy(context);

    VkResult result = DISPATCH(device).CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);

    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCreateShaderModule-module", "device", (void *)device, "shader",
        perfetto::DynamicString(shader_str), "module", (void *)*pShaderModule);

    ShaderModuleToString[*pShaderModule] = shader_str;

    TRACE_EVENT_END(VKSP_PERFETTO_CATEGORY);
    return result;
}

void VKAPI_CALL vksp_UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
    const VkWriteDescriptorSet *pDescriptorWrites, uint32_t descriptorCopyCount,
    const VkCopyDescriptorSet *pDescriptorCopies)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkUpdateDescriptorSets", "device", (void *)device, "descriptorWriteCount",
        descriptorWriteCount, "descriptorCopyCount", descriptorCopyCount);

    for (unsigned i = 0; i < descriptorWriteCount; i++) {
        switch (pDescriptorWrites[i].descriptorType) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkUpdateDescriptorSets-write", "dstSet",
                (void *)pDescriptorWrites[i].dstSet, "dstBinding", pDescriptorWrites[i].dstBinding, "descriptorCount",
                pDescriptorWrites[i].descriptorCount, "descriptorType", pDescriptorWrites[i].descriptorType, "buffer",
                (void *)pDescriptorWrites[i].pBufferInfo->buffer, "offset", pDescriptorWrites[i].pBufferInfo->offset,
                "range", pDescriptorWrites[i].pBufferInfo->range);
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkUpdateDescriptorSets-write", "dstSet",
                (void *)pDescriptorWrites[i].dstSet, "dstBinding", pDescriptorWrites[i].dstBinding, "descriptorCount",
                pDescriptorWrites[i].descriptorCount, "descriptorType", (void *)pDescriptorWrites[i].descriptorType,
                "imageLayout", pDescriptorWrites[i].pImageInfo->imageLayout, "imageView",
                (void *)pDescriptorWrites[i].pImageInfo->imageView);
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkUpdateDescriptorSets-write", "dstSet",
                (void *)pDescriptorWrites[i].dstSet, "dstBinding", pDescriptorWrites[i].dstBinding, "descriptorCount",
                pDescriptorWrites[i].descriptorCount, "descriptorType", pDescriptorWrites[i].descriptorType, "sampler",
                (void *)pDescriptorWrites[i].pImageInfo->sampler);
            break;
        default:
            TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkUpdateDescriptorSets-write", "device", (void *)device,
                "dstSet", (void *)pDescriptorWrites[i].dstSet, "dstBinding", pDescriptorWrites[i].dstBinding,
                "descriptorCount", pDescriptorWrites[i].descriptorCount, "descriptorType",
                pDescriptorWrites[i].descriptorType);
            break;
        }
    }

    DISPATCH(device).UpdateDescriptorSets(
        device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
}

void VKAPI_CALL vksp_CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet *pDescriptorSets,
    uint32_t dynamicOffsetCount, const uint32_t *pDynamicOffsets)
{
    std::lock_guard<std::mutex> lock(glock);
    auto device = CmdBufferToDevice[commandBuffer];
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCmdBindDescriptorSets", "device", (void *)device, "commandBuffer",
        (void *)commandBuffer, "pipelineBindPoint", pipelineBindPoint, "firstSet", firstSet, "descriptorSetCount",
        descriptorSetCount, "dynamicOffsetCount", dynamicOffsetCount);

    for (unsigned i = 0; i < descriptorSetCount; i++) {
        TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCmdBindDescriptorSets-ds", "commandBuffer",
            (void *)commandBuffer, "pipelineBindPoint", pipelineBindPoint, "firstSet", firstSet, "dstSet",
            (void *)pDescriptorSets[i], "index", i);
    }

    DISPATCH(device).CmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount,
        pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}

VkResult VKAPI_CALL vksp_CreateBuffer(
    VkDevice device, const VkBufferCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCreateBuffer", "device", (void *)device);

    auto result = DISPATCH(device).CreateBuffer(device, pCreateInfo, pAllocator, pBuffer);

    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCreateBuffer-result", "buffer", (void *)*pBuffer, "flags",
        pCreateInfo->flags, "size", pCreateInfo->size, "usage", pCreateInfo->usage, "sharingMode",
        pCreateInfo->sharingMode, "queueFamilyIndexCount", pCreateInfo->queueFamilyIndexCount);

    return result;
}

void VKAPI_CALL vksp_CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
    VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void *pValues)
{
    std::lock_guard<std::mutex> lock(glock);
    auto device = CmdBufferToDevice[commandBuffer];
    std::string pValuesStr = "";
    for (unsigned i = 0; i < size; i++) {
        pValuesStr += byteToStr[((unsigned char *)pValues)[i]];
    }
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCmdPushConstants", "device", (void *)device, "commandBuffer",
        (void *)commandBuffer, "stageFlags", stageFlags, "offset", offset, "size", size, "pValues",
        perfetto::DynamicString(pValuesStr));

    DISPATCH(device).CmdPushConstants(commandBuffer, layout, stageFlags, offset, size, pValues);
}

VkResult VKAPI_CALL vksp_AllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
    const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkAllocateMemory", "device", (void *)device);

    auto result = DISPATCH(device).AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);

    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkAllocateMemory-mem", "memory", (void *)*pMemory, "size",
        pAllocateInfo->allocationSize, "type", pAllocateInfo->memoryTypeIndex);

    return result;
}

VkResult VKAPI_CALL vksp_BindBufferMemory(
    VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkBindBufferMemory", "device", (void *)device, "buffer", (void *)buffer,
        "memory", (void *)memory, "offset", memoryOffset);

    return DISPATCH(device).BindBufferMemory(device, buffer, memory, memoryOffset);
}

VkResult VKAPI_CALL vksp_CreateImageView(VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCreateImageView", "device", (void *)device);

    auto result = DISPATCH(device).CreateImageView(device, pCreateInfo, pAllocator, pView);

    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCreateImageView-result", "pView", (void *)*pView, "image",
        (void *)pCreateInfo->image, "flags", pCreateInfo->flags, "format", pCreateInfo->format, "viewType",
        pCreateInfo->viewType, "components_a", pCreateInfo->components.a, "components_b", pCreateInfo->components.b,
        "components_g", pCreateInfo->components.g, "components_r", pCreateInfo->components.r, "aspectMask",
        pCreateInfo->subresourceRange.aspectMask, "baseMipLevel", pCreateInfo->subresourceRange.baseMipLevel,
        "baseArrayLayer", pCreateInfo->subresourceRange.baseArrayLayer, "levelCount",
        pCreateInfo->subresourceRange.levelCount, "layerCount", pCreateInfo->subresourceRange.layerCount);

    return result;
}

VkResult VKAPI_CALL vksp_BindImageMemory(
    VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkBindImageMemory", "device", (void *)device, "image", (void *)image, "memory",
        (void *)memory, "offset", memoryOffset);

    return DISPATCH(device).BindImageMemory(device, image, memory, memoryOffset);
}

VkResult VKAPI_CALL vksp_CreateImage(
    VkDevice device, const VkImageCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCreateImage", "device", (void *)device);

    auto result = DISPATCH(device).CreateImage(device, pCreateInfo, pAllocator, pImage);

    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCreateImage-result", "image", (void *)*pImage, "flags",
        pCreateInfo->flags, "imageType", pCreateInfo->imageType, "format", pCreateInfo->format, "width",
        pCreateInfo->extent.width, "depth", pCreateInfo->extent.depth, "height", pCreateInfo->extent.height,
        "mipLevels", pCreateInfo->mipLevels, "arrayLayers", pCreateInfo->arrayLayers, "samples", pCreateInfo->samples,
        "tiling", pCreateInfo->tiling, "usage", pCreateInfo->usage, "sharingMode", pCreateInfo->sharingMode,
        "queueFamilyIndexCount", pCreateInfo->queueFamilyIndexCount, "initialLayout", pCreateInfo->initialLayout);

    return result;
}

VkResult VKAPI_CALL vksp_CreateSampler(VkDevice device, const VkSamplerCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkSampler *pSampler)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCreateSampler", "device", (void *)device);

    auto result = DISPATCH(device).CreateSampler(device, pCreateInfo, pAllocator, pSampler);

    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCreateSampler-result", "sampler", (void *)*pSampler, "flags",
        pCreateInfo->flags, "magFilter", pCreateInfo->magFilter, "minFilter", pCreateInfo->minFilter, "mipmapMode",
        pCreateInfo->mipmapMode, "addressModeU", pCreateInfo->addressModeU, "addressModeV", pCreateInfo->addressModeV,
        "addressModeW", pCreateInfo->addressModeW, "mipLodBias", pCreateInfo->mipLodBias, "anisotropyEnable",
        pCreateInfo->anisotropyEnable, "maxAnisotropy", pCreateInfo->maxAnisotropy, "compareEnable",
        pCreateInfo->compareEnable, "compareOp", pCreateInfo->compareOp, "minLod", pCreateInfo->minLod, "maxLod",
        pCreateInfo->maxLod, "borderColor", pCreateInfo->borderColor, "unnormalizedCoordinates",
        pCreateInfo->unnormalizedCoordinates);

    return result;
}

/*****************************************************************************/
/* CREATE INSTANCE & DEVICE **************************************************/
/*****************************************************************************/

VkResult VKAPI_CALL vksp_CreateInstance(
    const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkInstance *pInstance)
{
    std::lock_guard<std::mutex> lock(glock);
    VkLayerInstanceCreateInfo *layerCreateInfo = (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;

    while (layerCreateInfo
        && (layerCreateInfo->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
            || layerCreateInfo->function != VK_LAYER_LINK_INFO)) {
        layerCreateInfo = (VkLayerInstanceCreateInfo *)layerCreateInfo->pNext;
    }

    if (layerCreateInfo == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr gpa = layerCreateInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    layerCreateInfo->u.pLayerInfo = layerCreateInfo->u.pLayerInfo->pNext;

    PFN_vkCreateInstance createFunc = (PFN_vkCreateInstance)gpa(VK_NULL_HANDLE, "vkCreateInstance");

    VkResult ret = createFunc(pCreateInfo, pAllocator, pInstance);
    if (ret != VK_SUCCESS) {
        return ret;
    }

    DispatchTable dispatchTable;
#define FUNC_INS(f) SET_DISPATCH_TABLE(dispatchTable, f, pInstance, gpa, "instance", return VK_SUCCESS);
#define FUNC_INS_INT(f) SET_DISPATCH_TABLE(dispatchTable, f, pInstance, gpa, "instance", return VK_SUCCESS);
#include "functions.def"

    DISPATCH(*pInstance) = dispatchTable;

    perfetto::TracingInitArgs args;
#ifdef BACKEND_INPROCESS
    args.backends |= perfetto::kInProcessBackend;
#else
    args.backends |= perfetto::kSystemBackend;
#endif
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

#ifdef BACKEND_INPROCESS
    perfetto::protos::gen::TrackEventConfig track_event_cfg;
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(get_trace_max_size());
    auto *ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    gTracingSession = perfetto::Tracing::NewTrace();
    gTracingSession->Setup(cfg);
    gTracingSession->StartBlocking();
#endif

    return VK_SUCCESS;
}

void VKAPI_CALL vksp_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator)
{
    std::lock_guard<std::mutex> lock(glock);

#ifdef BACKEND_INPROCESS
    gTracingSession->StopBlocking();
    std::vector<char> trace_data(gTracingSession->ReadTraceBlocking());

    std::ofstream output;
    output.open(get_trace_dest(), std::ios::out | std::ios::binary);
    output.write(&trace_data[0], trace_data.size());
    output.close();
#else
    perfetto::TrackEvent::Flush();
#endif

    auto DestroyInstance = DISPATCH(instance).DestroyInstance;
    gdispatch.erase(vksp_key(instance));
    return DestroyInstance(instance, pAllocator);
}

VkResult VKAPI_CALL vksp_CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
    std::lock_guard<std::mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCreateDevice");

    std::string ppEnabledExtensionNamesConcat;
    for (unsigned i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        ppEnabledExtensionNamesConcat += "." + std::string(pCreateInfo->ppEnabledExtensionNames[i]);
    }

    VkLayerDeviceCreateInfo *layerCreateInfo = (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
    while (layerCreateInfo
        && (layerCreateInfo->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
            || layerCreateInfo->function != VK_LAYER_LINK_INFO)) {
        layerCreateInfo = (VkLayerDeviceCreateInfo *)layerCreateInfo->pNext;
    }

    if (layerCreateInfo == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr gipa = layerCreateInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = layerCreateInfo->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    layerCreateInfo->u.pLayerInfo = layerCreateInfo->u.pLayerInfo->pNext;

    PFN_vkCreateDevice createFunc = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");

    std::vector<const char *> ppEnabledExtensionNames;
    for (unsigned i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        ppEnabledExtensionNames.push_back(pCreateInfo->ppEnabledExtensionNames[i]);
    }
    ppEnabledExtensionNames.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);

    VkDeviceCreateInfo mCreateInfo = *pCreateInfo;
    mCreateInfo.enabledExtensionCount = ppEnabledExtensionNames.size();
    mCreateInfo.ppEnabledExtensionNames = ppEnabledExtensionNames.data();

    VkResult ret = createFunc(physicalDevice, &mCreateInfo, pAllocator, pDevice);
    if (ret != VK_SUCCESS) {
        return ret;
    }

    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "vkCreateDevice-enabled", "device", (void *)*pDevice,
        "ppEnabledExtensionNames", ppEnabledExtensionNamesConcat);

    DispatchTable dispatchTable;
#define FUNC_DEV(f) SET_DISPATCH_TABLE(dispatchTable, f, pDevice, gdpa, "device", DeviceNotToTrace.insert(*pDevice));
#define FUNC_DEV_INT(f)                                                                                                \
    SET_DISPATCH_TABLE(dispatchTable, f, pDevice, gdpa, "device", DeviceNotToTrace.insert(*pDevice));
#include "functions.def"

    DISPATCH(*pDevice) = dispatchTable;
    DeviceToPhysicalDevice[*pDevice] = physicalDevice;

    return VK_SUCCESS;
}

void VKAPI_CALL vksp_DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator)
{
    std::lock_guard<std::mutex> lock(glock);
    for (auto &[queue, thread] : QueueThreadPool[device]) {
        auto info = QueueToThreadInfo[queue];
        {
            std::lock_guard<std::mutex> guard(info->lock);
            info->stop = true;
            info->cv.notify_one();
        }
        thread.join();
        delete info;
    }
    auto DestroyInstance = DISPATCH(device).DestroyDevice;
    gdispatch.erase(vksp_key(device));
    return DestroyInstance(device, pAllocator);
}

/*****************************************************************************/
/* LAYER ENTRY POINTS FUNCTIONS **********************************************/
/*****************************************************************************/

extern "C" {

PFN_vkVoidFunction VKAPI_CALL vksp_GetDeviceProcAddr(VkDevice device, const char *pName)
{
    std::lock_guard<std::mutex> lock(glock);

    if (DeviceNotToTrace.count(device) == 0) {
#define FUNC_DEV GET_PROC_ADDR
#include "functions.def"
    }

    return DISPATCH(device).GetDeviceProcAddr(device, pName);
}

PFN_vkVoidFunction VKAPI_CALL vksp_GetInstanceProcAddr(VkInstance instance, const char *pName)
{
    std::lock_guard<std::mutex> lock(glock);

#define FUNC_INS GET_PROC_ADDR
#include "functions.def"

    return DISPATCH(instance).GetInstanceProcAddr(instance, pName);
}
}
