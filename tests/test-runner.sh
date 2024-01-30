#!/usr/bin/bash

set -xe

SCRIPT_DIR=$(dirname $(realpath "${BASH_SOURCE[0]}"))

VKSP_RUNNER=${VKSP_RUNNER:-"vulkan-shader-profiler-runner"}

${VKSP_RUNNER} -i "${SCRIPT_DIR}/example.spvasm" -v
${VKSP_RUNNER} -i "${SCRIPT_DIR}/example-counter.spvasm" | grep "my_section"
