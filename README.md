# Vulkan Shader Profiler

`vulkan-kernel-profiler` is a perfetto-based Vulkan shader profiler using the layering capability of the [Vulkan-Loader](https://github.com/KhronosGroup/Vulkan-Loader)

# Legal

`vulkan-kernel-profiler` is licensed under the terms of the [Apache 2.0 license](LICENSE).

# Dependencies

`vulkan-kernel-profiler` depends on the following:

* [Vulkan-Loader](https://github.com/KhronosGroup/Vulkan-Loader)
* [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers)
* [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers)
* [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)
* [perfetto](https://github.com/google/perfetto)

`vulkan-kernel-profiler` also (obviously) depends on a Vulkan implementation.

# Building

`vulkan-kernel-profiler` uses CMake for its build system.

To compile it, please run:
```
cmake -B <build_dir> -S <path-to-vulkan-kernel-profiler> -DPERFETTO_SDK_PATH<path-to-perfetto-sdk>
cmake --build <build_dir>
```

# Build options

* `PERFETTO_SDK_PATH` (REQUIRED): path to [perfetto](https://github.com/google/perfetto) sdk (`vulkan-kernel-profiler` is looking for `PERFETTO_SDK_PATH/perfetto.cc` and `PERFETTO_SDK_PATH/perfetto.h`).
* `BACKEND`: [perfetto](https://github.com/google/perfetto) backend to use
  * `InProcess` (default): the application will generate the traces ([perfetto documentation](https://perfetto.dev/docs/instrumentation/tracing-sdk#in-process-mode)). Build options and environment variables can be used to control the maximum size of traces and the destination file where the traces will be recorded.
  * `System`: perfetto `traced` daemon will be responsible for generating the traces ([perfetto documentation](https://perfetto.dev/docs/instrumentation/tracing-sdk#system-mode)).
* `TRACE_MAX_SIZE` (only with `InProcess` backend): Maximum size (in KB) of traces that can be recorded. Can be overriden at runtime using the following environment variable: `CLKP_TRACE_MAX_SIZE` (Default: `1024`).
* `TRACE_DEST` (only with `InProcess` backend): File where the traces will be recorded. Can be overriden at runtime using the following environment variable: `CLKP_TRACE_DEST` (Default: `opencl-kernel-profiler.trace`).

# Running with Vulkan Shader Profiler

To run an application with the `vulkan-kernel-profiler`, one need to ensure the following point

* The `Vulkan-Loader` needs to be able to find the manifest in `<vulkan-shader-profiler>/manifest/vulkan-shader-profiler.json`. This can be achieve by using the follow environment variable: `VK_ADD_LAYER_PATH=<path-to-vulkan-shader-profiler-manifest>`.
* The Layer needs to be enabled. Either directly from the application, or using the following environment variable: `VK_LOADER_LAYERS_ENABLE="VK_LAYER_SHADER_PROFILER"`.

## On ChromeOS

Make sure to have emerged and deployed the `vulkan-shader-profiler`.

Then run the application using `vulkan-shader-profiler.sh`. This script will take care of setting all the environment variables needed to run with the `vulkan-shader-profiler`.

# Using the trace

Once traces have been generated, on can view them using the [perfetto trace viewer](https://ui.perfetto.dev).

# How does it work

`vulkan-shader-profiler` intercept to following calls to generate perfetto traces:

* `vkGetDeviceQueue`: Create internal structures to trace everything executed on this queue.
* `vkAllocateCommandBuffers`: Create internal structures for this command buffer.
* `vkFreeCommandBuffers`: Clean internal structures for this command buffer.
* `vkBeginCommandBuffer`: Initialize internal structures for this command buffer.
* `vkQueueSubmit`: Modify the submit information to add a timeline semaphore used to track this submit internally. Also submit a 'job' to get the command buffer submitted traced.
* `vkCmdDispatch`: Initialize internal structures to know what to trace if this command buffer get submitted.
* `vkCmdBindPipeline`: Initialize internal structures to know what pipeline will be executed if this command buffer get submitted.
* `vkCreateComputePipelines`: Initialize internal structure to know what shader will be executed if this pipeline get submitted.
* `vkCreateShaderModule`: Create an perfetto event with the readable version of the Vulkan SPIR-V source code.

Every intercept call also generates a trace for the function.
