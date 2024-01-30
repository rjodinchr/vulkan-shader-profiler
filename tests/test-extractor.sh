#!/usr/bin/bash

set -xe

SCRIPT_DIR=$(dirname $(realpath "${BASH_SOURCE[0]}"))

VKSP_EXTRACTOR=${VKSP_EXTRACTOR:-"vulkan-shader-profiler-extractor"}
VKSP_RUNNER=${VKSP_RUNNER:-"vulkan-shader-profiler-runner"}

PERFETTO_OUTPUT_TRACE="${SCRIPT_DIR}/trace"
TRACE_SPVASM="${SCRIPT_DIR}/trace.spvasm"
SHADER_FILE="${SCRIPT_DIR}/vksp_s0.spv"
function clean() {
    rm -f ${TRACE_SPVASM} ${SHADER_FILE} ${PERFETTO_OUTPUT_TRACE}
}
trap clean EXIT

PERFETTO_OUTPUT_TRACE=${PERFETTO_OUTPUT_TRACE} \
VKSP_SHADER_TEXT_BLOCK_SIZE=1024 \
VKSP_SHADER_DIR="${SCRIPT_DIR}" \
VK_ADD_LAYER_PATH="${SCRIPT_DIR}/../manifest/" \
VK_LOADER_LAYERS_ENABLE="VK_LAYER_SHADER_PROFILER" \
${SCRIPT_DIR}/perfetto.sh ${VKSP_RUNNER} -i ${SCRIPT_DIR}/example.spvasm

${VKSP_EXTRACTOR} -i ${PERFETTO_OUTPUT_TRACE} -o ${TRACE_SPVASM} -d 0 -v
diff ${TRACE_SPVASM} ${SCRIPT_DIR}/example-expectation.spvasm

${VKSP_EXTRACTOR} -i ${PERFETTO_OUTPUT_TRACE} -o ${TRACE_SPVASM} -d 0 -v -s ${SCRIPT_DIR}/empty
diff ${TRACE_SPVASM} ${SCRIPT_DIR}/example-expectation.spvasm

${VKSP_EXTRACTOR} -i ${PERFETTO_OUTPUT_TRACE} -o ${TRACE_SPVASM} -d 0 -v -s ${SHADER_FILE}
diff ${TRACE_SPVASM} ${SCRIPT_DIR}/example-vksp_s0-expectation.spvasm
