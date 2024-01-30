#!/usr/bin/bash

set -xe

SCRIPT_DIR=$(dirname $(realpath "${BASH_SOURCE[0]}"))
CONFIG="${SCRIPT_DIR}/perfetto_config"

PERFETTO_OUTPUT_TRACE=${PERFETTO_OUTPUT_TRACE:-"$(basename $(echo $@ | sed 's|^\([^ ]*\).*$|\1|')).$(date +%y%m%d-%H%M%S).trace"}
PERFETTO_TRACED=${PERFETTO_TRACED:-"traced"}
PERFETTO_BINARY=${PERFETTO_BINARY:-"perfetto"}

${PERFETTO_TRACED} &
TRACED_PID=$(echo $!)
PERFETTO_PID=$(${PERFETTO_BINARY} -c ${CONFIG} --txt -o ${PERFETTO_OUTPUT_TRACE} --background-wait)

function clean() {
    kill ${PERFETTO_PID}
    # Cannot use 'wait' because process is not a child of this shell
    while [[ -n $(pgrep -x perfetto) ]]; do sleep 0.2; done

    kill ${TRACED_PID}
}
trap clean EXIT

$@
