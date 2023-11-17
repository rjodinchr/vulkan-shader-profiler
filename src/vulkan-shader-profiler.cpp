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

#define SET_DISPATCH_TABLE(table, func, pointer, gpa) table.func = (PFN_vk##func)gpa(*pointer, "vk" #func);

#define DISPATCH_TABLE_ELEMENT(func) PFN_vk##func func;

#define VKSP_LAYER_NAME "VK_LAYER_SHADER_PROFILER"

#define PRINT(message, ...)                                                                                            \
    do {                                                                                                               \
        fprintf(stderr, "[VKSP] %s: " message "\n", __func__, ##__VA_ARGS__);                                          \
        TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY, "PRINT", "message", perfetto::DynamicString(message));             \
    } while (0)

#define DISPATCH(obj) gdispatch[vksp_key(obj)]

#define SPEC_VERSION VK_MAKE_VERSION(1, 3, 0)
#define ENV_VERSION SPV_ENV_VULKAN_1_3

/*****************************************************************************/
/* GLOBAL VARIABLES & TYPES **************************************************/
/*****************************************************************************/

static std::recursive_mutex glock;

typedef struct DispatchTable_ {
#define FUNC DISPATCH_TABLE_ELEMENT
#include "functions.def"
} DispatchTable;

template <typename DispatchableType> void *vksp_key(DispatchableType obj) { return *(void **)obj; }
static std::map<void *, DispatchTable> gdispatch;

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
/* Queue Thread **************************************************************/
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
        if (info->jobs.empty() && info->stop) {
            return (ThreadJob *)nullptr;
        }
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
        "shader", ShaderModuleToString[PipelineToShaderModule[cmd.pipeline]], "shader_name",
        PipelineToShaderModuleName[cmd.pipeline]);
    TRACE_EVENT_END(VKSP_PERFETTO_CATEGORY, perfetto::Track((uintptr_t)info->queue), (uint64_t)end);
}

static void QueueThreadFct(ThreadInfo *info)
{
    TRACE_EVENT_INSTANT(VKSP_PERFETTO_CATEGORY,
        perfetto::DynamicString("vksp-queue_" + std::to_string((uintptr_t)info->queue)),
        perfetto::Track((uintptr_t)info->queue));
    while (true) {
        auto job = get_job(info);
        if (job == nullptr) {
            return;
        }

        // Wait for job completion
        VkResult result = WaitSemaphore(info, job);
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
    std::lock_guard<std::recursive_mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkGetDeviceQueue", "device", (void *)device);

    DISPATCH(device).GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);

    auto info = new ThreadInfo(device, *pQueue);
    QueueToDevice[*pQueue] = device;
    QueueToThreadInfo[*pQueue] = info;
    QueueThreadPool[device].emplace_back(std::make_pair(*pQueue, [info] { QueueThreadFct(info); }));
}

VkResult VKAPI_CALL vksp_QueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
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

        mSubmits[eachSubmit].signalSemaphoreCount++;
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

            next->signalSemaphoreValueCount++;
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
    std::lock_guard<std::recursive_mutex> lock(glock);
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
    std::lock_guard<std::recursive_mutex> lock(glock);
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
    std::lock_guard<std::recursive_mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkBeginCommandBuffer", "commandBuffer", (void *)commandBuffer);

    CmdBufferToThreadDispatch[commandBuffer].clear();

    return DISPATCH(CmdBufferToDevice[commandBuffer]).BeginCommandBuffer(commandBuffer, pBeginInfo);
}

void VKAPI_CALL vksp_CmdDispatch(
    VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCmdDispatch", "commandBuffer", (void *)commandBuffer, "groupCountX",
        groupCountX, "groupCountY", groupCountY, "groupCountZ", groupCountZ);

    ThreadDispatch dispatch = {
        .pipeline = CmdBufferToPipeline[commandBuffer],
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
    auto &d = DISPATCH(CmdBufferToDevice[commandBuffer]);
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
    std::lock_guard<std::recursive_mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCmdBindPipeline", "commandBuffer", (void *)commandBuffer);

    CmdBufferToPipeline[commandBuffer] = pipeline;

    return DISPATCH(CmdBufferToDevice[commandBuffer]).CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
}

VkResult VKAPI_CALL vksp_CreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache,
    uint32_t createInfoCount, const VkComputePipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator,
    VkPipeline *pPipelines)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
    TRACE_EVENT(VKSP_PERFETTO_CATEGORY, "vkCreateComputePipelines", "device", (void *)device, "createInfoCount",
        createInfoCount);

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
    std::lock_guard<std::recursive_mutex> lock(glock);
    std::string shader_str = std::string("vksp_s") + std::to_string(shader_number++);

    const spv_context context = spvContextCreate(ENV_VERSION);
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

    ShaderModuleToString[*pShaderModule] = shader_str;

    TRACE_EVENT_END(VKSP_PERFETTO_CATEGORY);
    return result;
}

/*****************************************************************************/
/* CREATE INSTANCE & DEVICE **************************************************/
/*****************************************************************************/

VkResult VKAPI_CALL vksp_CreateInstance(
    const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkInstance *pInstance)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
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
#define FUNC(f) SET_DISPATCH_TABLE(dispatchTable, f, pInstance, gpa)
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
    std::lock_guard<std::recursive_mutex> lock(glock);

#ifdef BACKEND_INPROCESS
    gTracingSession->StopBlocking();
    std::vector<char> trace_data(gTracingSession->ReadTraceBlocking());

    std::ofstream output;
    output.open(get_trace_dest(), std::ios::out | std::ios::binary);
    output.write(&trace_data[0], trace_data.size());
    output.close();
#endif

    auto DestroyInstance = DISPATCH(instance).DestroyInstance;
    gdispatch.erase(vksp_key(instance));
    return DestroyInstance(instance, pAllocator);
}

VkResult VKAPI_CALL vksp_CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
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

    VkResult ret = createFunc(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (ret != VK_SUCCESS) {
        return ret;
    }

    DispatchTable dispatchTable;
#define FUNC(f) SET_DISPATCH_TABLE(dispatchTable, f, pDevice, gdpa);
#include "functions.def"

    DISPATCH(*pDevice) = dispatchTable;
    DeviceToPhysicalDevice[*pDevice] = physicalDevice;

    return VK_SUCCESS;
}

void VKAPI_CALL vksp_DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
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
/* ENUMERATION FUNCTIONS *****************************************************/
/*****************************************************************************/

VkResult VKAPI_CALL vksp_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount, VkLayerProperties *pProperties)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
    if (pPropertyCount)
        *pPropertyCount = 1;

    if (pProperties) {
        strcpy(pProperties->layerName, VKSP_LAYER_NAME);
        strcpy(pProperties->description, "vulkan-shader-profiler github.com/rjodinchr/vulkan-shader-profiler");
        pProperties->implementationVersion = 1;
        pProperties->specVersion = SPEC_VERSION;
    }

    return VK_SUCCESS;
}

VkResult VKAPI_CALL vksp_EnumerateDeviceLayerProperties(
    VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount, VkLayerProperties *pProperties)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
    return vksp_EnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VkResult VKAPI_CALL vksp_EnumerateInstanceExtensionProperties(
    const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
    if (pLayerName == NULL || strcmp(pLayerName, VKSP_LAYER_NAME)) {
        return VK_ERROR_LAYER_NOT_PRESENT;
    }

    if (pPropertyCount)
        *pPropertyCount = 0;
    return VK_SUCCESS;
}

VkResult VKAPI_CALL vksp_EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName,
    uint32_t *pPropertyCount, VkExtensionProperties *pProperties)
{
    std::lock_guard<std::recursive_mutex> lock(glock);
    if (pLayerName == NULL || strcmp(pLayerName, VKSP_LAYER_NAME)) {
        if (physicalDevice == VK_NULL_HANDLE)
            return VK_SUCCESS;

        return DISPATCH(physicalDevice)
            .EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
    }

    if (pPropertyCount)
        *pPropertyCount = 0;
    return VK_SUCCESS;
}

/*****************************************************************************/
/* LAYER ENTRY POINTS FUNCTIONS **********************************************/
/*****************************************************************************/

extern "C" {

PFN_vkVoidFunction vksp_GetInstanceProcAddr(VkInstance instance, const char *pName);

PFN_vkVoidFunction VKAPI_CALL vksp_GetDeviceProcAddr(VkDevice device, const char *pName)
{
    std::lock_guard<std::recursive_mutex> lock(glock);

#define FUNC GET_PROC_ADDR
#define FUNC_GET(f) // ignore
#include "functions.def"

    return DISPATCH(device).GetDeviceProcAddr(device, pName);
}

PFN_vkVoidFunction VKAPI_CALL vksp_GetInstanceProcAddr(VkInstance instance, const char *pName)
{
    std::lock_guard<std::recursive_mutex> lock(glock);

#define FUNC GET_PROC_ADDR
#define FUNC_GET(f) // ignore
#include "functions.def"

    return DISPATCH(instance).GetInstanceProcAddr(instance, pName);
}
}
