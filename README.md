# Vulkan Shader Profiler

`vulkan-kernel-profiler` is a perfetto-based Vulkan shader profiler using the layering capability of the [Vulkan-Loader](https://github.com/KhronosGroup/Vulkan-Loader)

It allows to visualize a vulkan application using perfetto with information about the compute shader to easily identify which shader is taking most of the application time, and what is its Vulkan SPIR-V source code.

Using the `vulkan-shader-profiler-extractor` and `vulkan-shader-profiler-runner`, it is also possible to extract a specific dispatch from the trace (using the `dispatchId` debug information from the trace), and replay it with profiled section with the runner.

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
cmake -B <build_dir> -S <path-to-vulkan-kernel-profiler> -DPERFETTO_SDK_PATH=<path-to-perfetto-sdk> -DPERFETTO_TRACE_PROCESSOR_LIB=<path-to-libtrace_processor.a> -DPERFETTO_INTERNAL_INCLUDE_PATH=<path-to-perfetto-include> -DSPIRV_TOOLS_LIBRARY=<path-to-libSPIRV-Tools.{so|a}> -DSPIRV_TOOLS_OPT_LIBRARY=<path-to-libSPIRV-Tools-opt.{so|a}>
cmake --build <build_dir>
```

# Build options

* `PERFETTO_SDK_PATH` (REQUIRED): path to [perfetto](https://github.com/google/perfetto) sdk (`vulkan-kernel-profiler` is looking for `PERFETTO_SDK_PATH/perfetto.cc` and `PERFETTO_SDK_PATH/perfetto.h`).
* `PERFETTO_TRACE_PROCESSOR_LIB` (REQUIRED): path to `libtrace_processor.a` produces by a perfetto build.
* `PERFETTO_INTERNAL_INCLUDE_PATH` (REQUIRED): path to perfetto internal include directory (`<perfetto>/include`), or where it is installed.
* `SPIRV_TOOLS_LIBRARY` (REQUIRED): path to `libSPIRV-Tools.so` or `libSPIRV-Tools.a`.
* `SPIRV_TOOLS_OPT_LIBRARY` (REQUIRED): path to `libSPIRV-Tools-opt.so` or `libSPIRV-Tools-opt.a`.
* OPTIONAL:
  * `PERFETTO_GEN_INCLUDE_PATH`: path to a a perfetto build (if not installed) `<perfetto>/out/release/gen/build_config`.
  * `PERFETTO_CXX_CONFIG_INCLUDE_PATH`: path to perfetto buildtools config `<perfetto>/buildtools/libcxx_config`.
  * `PERFETTO_CXX_SYSTEM_INCLUDE_PATH`: path to perfetto buildtools include `<perfetto>/buildtools/libcxx/include`.
  * `EXTRACTOR_NOSTDINCXX`: build `vulkan-shader-profiler-extractor` with `-nostdinc++` to be able to link with some `libtrace_processor.a`.
  * `SPIRV_TOOLS_INCLUDE_PATH`: path to SPIRV-Tools include `<SPIRV-Tools>/include`.
  * `BACKEND`: [perfetto](https://github.com/google/perfetto) backend to use
    * `InProcess` (default): the application will generate the traces ([perfetto documentation](https://perfetto.dev/docs/instrumentation/tracing-sdk#in-process-mode)). Build options and environment variables can be used to control the maximum size of traces and the destination file where the traces will be recorded.
    * `System`: perfetto `traced` daemon will be responsible for generating the traces ([perfetto documentation](https://perfetto.dev/docs/instrumentation/tracing-sdk#system-mode)).
  * `TRACE_MAX_SIZE` (only with `InProcess` backend): Maximum size (in KB) of traces that can be recorded. Can be overriden at runtime using the following environment variable: `VKSP_TRACE_MAX_SIZE` (Default: `1024`).
  * `TRACE_DEST` (only with `InProcess` backend): File where the traces will be recorded. Can be overriden at runtime using the following environment variable: `VKSP_TRACE_DEST` (Default: `opencl-kernel-profiler.trace`).

# Running an application with Vulkan Shader Profiler

To run an application with the `vulkan-kernel-profiler`, one need to ensure the following points:

* The `Vulkan-Loader` needs to be able to find the manifest in `<vulkan-shader-profiler>/manifest/vulkan-shader-profiler.json`. This can be achieve by using the follow environment variable: `VK_ADD_LAYER_PATH=<path-to-vulkan-shader-profiler-manifest>`.
* The Layer needs to be enabled. Either directly from the application, or using the following environment variable: `VK_LOADER_LAYERS_ENABLE="VK_LAYER_SHADER_PROFILER"`.

## On ChromeOS

Make sure to have emerged and deployed the `vulkan-shader-profiler`.

Then run the application using `vulkan-shader-profiler.sh`. This script will take care of setting all the environment variables needed to run with the `vulkan-shader-profiler`.

# Using the trace

Once traces have been generated, on can view them using the [perfetto trace viewer](https://ui.perfetto.dev).

# How does the Vulkan Shader Profiler layer work

`vulkan-shader-profiler` intercept the following calls to generate perfetto traces:

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

`vulkan-shader-profiler` also intercept the following calls which are needed to run the `vulkan-shader-profiler-extractor`:

* `vkUpdateDescriptorSets`
* `vkCmdBindDescriptorSets`
* `vkCmdPushConstants`
* `vkAllocateMemory`
* `vkCreateBuffer`
* `vkBindBufferMemory`
* `vkCreateImage`
* `vkCreateImageView`
* `vkBindImageMemory`
* `vkCreateSampler`

Functions used by `vulkan-shader-profiler` internally:

* `CreateSemaphore`, `DestroySemaphore`, `WaitSemaphores`: To know when a workload end to create the event.
* `CreateQueryPool`, `DestroyQueryPool`, `GetQueryPoolResults`, `CmdResetQueryPool`: To have somewhere to store the timestamp during the command buffer execution.
* `CmdWriteTimestamp`: To store the timestamp during the command buffer execution.
* `GetCalibratedTimestampsEXT`: To convert the device timestamp to the host timeline
* `GetPhysicalDeviceProperties`: To convert the number of ticks returned by `CmdWriteTimestamp` to actual time information in nano-seconds.

# Extracting a dispatch from a trace

Once a trace has been generated from an application, it is possible to extract a single dispatch from it using the `dispatchId` debug information from the trace:

```
$ vulkan-shader-profiler-extractor -i <input_trace> -o <output_file> -d <dispatchId>
```
Required options:

* `-i`: the path to the trace generated by the `vulkan-shader-profiler` when running the vulkan application.
* `-o`: the path where the output of the extractor will be stored (the output is a Vulkan SPIR-V readable file by default).
* `-d`: the dispatchId to extract from the trace

Optional options:

* `-b`: output a binary Vulkan SPIRV-V program instead of a readable one (allow to have something smaller).
* `-v`: enable the verbose mode which is mainly use for debug purpose.

# Run a Vulkan SPIR-V program with the runner

Only program extracted from a trace with the `vulkan-shader-profiler-extractor` can be run with the `vulkan-shader-profiler-runner`:

```
$ vulkan-shader-profiler-runner -i <input>
```
Required options:

* `-i`: path to the Vulkan SPIR-V program generated by the extractor

Optional options:

* `-n`: allow to run the program multiple times
* `-m`: allow to run the program multiple times before starting to benchmark it
* `-v`: enable the verbose mode which is mainly use for debug purpose.

Output example:
```
$ vulkan-shader-profiler-runner -i trace.spvasm -m 1000 -n 100
vksp_s24-main_function-96.1.1
----------------------------------
[  HOST]        Submit:  25.368 ms
[  HOST]      WaitIdle: 201.198 ms
[  HOST]         Total: 226.566 ms
----------------------------------
[   GPU]         Total: 200.544 ms
[   GPU]          Cold: 185.171 ms
[   GPU]           Hot:  15.373 ms
[   GPU]       Hot avg: 153.730 us
```

## Using counters inside a Vulkan SPIR-V program

It is possible to profile section of the program by adding non-semantic instructions inside the program.

To start a section add:
```
%<my_counter> = OpExtInst %<void_type> %<vksp_ext_inst_id> StartCounter "<counter_name>"
```

To end a section add:
```
%<unused> = OpExtInst %<void_type> %<vksp_ext_inst_id> StopCounter %<my_counter>
```

Small partial example:
```
         %49 = OpExtInstImport "NonSemantic.VkspReflection.1"
...
       %void = OpTypeVoid
...
         %ct = OpExtInst %void %49 StartCounter "my_section"
...
         %un = OpExtInst %void %49 StopCounter %ct
```

Output example:
```
$ vulkan-shader-profiler-runner -i trace.spvasm -n 10 -m 100
vksp_s0-test_simple-128.1.1
-------------------------------
[  HOST]     Submit:  19.750 us
[  HOST]   WaitIdle:  55.684 ms
[  HOST]      Total:  55.703 ms
-------------------------------
[   GPU]      Total:  54.845 ms
[   GPU]       Cold:  51.312 ms
[   GPU]        Hot:   3.532 ms
[   GPU]    Hot avg: 353.282 us
-------------------------------
[SHADER] my_section:  29.8%
```
