#!/usr/bin/bash

set -xe

SCRIPT_DIR=$(dirname $(realpath "${BASH_SOURCE[0]}"))

VKSP_RUNNER=${VKSP_RUNNER:-"vulkan-shader-profiler-runner"}

PERFETTO_OUTPUT_TRACE="${SCRIPT_DIR}/trace"
CHECKSUM_OUTPUT_BUFFER="${SCRIPT_DIR}/checksum.spvasm.0.1.buffer"
CHECKSUM_INPUT_BUFFERS="${SCRIPT_DIR}/checksum.spvasm.buffers"
function clean() {
    rm -f ${PERFETTO_OUTPUT_TRACE} ${CHECKSUM_OUTPUT_BUFFER} ${CHECKSUM_INPUT_BUFFERS}
}
trap clean EXIT

${VKSP_RUNNER} -i ${SCRIPT_DIR}/checksum.spvasm -b ${SCRIPT_DIR}/checksum.buffers -o 0.1
diff ${CHECKSUM_OUTPUT_BUFFER} ${SCRIPT_DIR}/checksum.expected_output

PERFETTO_OUTPUT_TRACE=${PERFETTO_OUTPUT_TRACE} \
VK_ADD_LAYER_PATH="${SCRIPT_DIR}/../manifest/" \
VK_LOADER_LAYERS_ENABLE="VK_LAYER_SHADER_PROFILER" \
VKSP_EXTRACT_BUFFERS_FROM="${SCRIPT_DIR}/checksum.spvasm" \
${SCRIPT_DIR}/perfetto.sh ${VKSP_RUNNER} -i ${SCRIPT_DIR}/checksum.spvasm -b ${SCRIPT_DIR}/checksum.buffers
diff ${CHECKSUM_INPUT_BUFFERS} ${SCRIPT_DIR}/checksum.buffers
