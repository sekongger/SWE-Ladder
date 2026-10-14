#!/usr/bin/env bash
# Pure, consistent CANN 8.3.RC1 env, replicated STATICALLY from the live training job's
# own environment (/proc/<ray-worker>/environ). We deliberately do NOT `source
# ascend-toolkit/8.3.RC1/bin/setenv.bash`: that script runs a filesystem-wide `find`
# (scans /proc, NFS, ...) that hammers this already-loaded box and produces 32MB of noise.
# These are exactly the vars the working ray::WorkerDict processes carry, so torch_npu
# device init (GEInitialize/FEOpsKernelInfoStore) succeeds without version mixing.
#
# Usage:  ASCEND_DEV=0 bash npu_env.sh <python> <args...>
set -u
A=/usr/local/Ascend/ascend-toolkit/8.3.RC1

export ASCEND_TOOLKIT_HOME="$A"
export ASCEND_HOME_PATH="$A"
export ASCEND_TOOLKIT_ROOT="$A"
export ASCEND_AICPU_PATH="$A"
export ASCEND_OPP_PATH="$A/opp"
export ASCEND_INSTALL_PATH=/usr/local/Ascend/
export ASCEND_TOOLKIT_VERSION_DIR=8.3.RC1
export TOOLCHAIN_HOME="$A/toolkit"
export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1

export LD_LIBRARY_PATH="\
$ATB_HOME_PATH/lib:\
$A/tools/aml/lib64:$A/tools/aml/lib64/plugin:\
$A/lib64:$A/lib64/plugin/opskernel:$A/lib64/plugin/nnengine:\
$A/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:\
/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:\
$A/runtime/lib64:$A/compiler/lib64:$A/compiler/lib64/plugin/opskernel:$A/compiler/lib64/plugin/nnengine:\
$A/hccl/lib64:$A/toolkit/lib64:$A/opp/lib64:$A/aarch64-linux/lib64:$A/fwkacllib/lib64:$A/atc/lib64:\
/usr/local/Ascend/driver/tools/hccn_tool/:/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export PYTHONPATH="\
$A/python/site-packages:\
$A/opp/built-in/op_impl/ai_core/tbe:\
$A/compiler/python/site-packages:\
$A/hccl/python/site-packages:\
$A/toolkit/python/site-packages:\
$A/pyACL/python/site-packages/acl${PYTHONPATH:+:$PYTHONPATH}"

export PATH="$A/bin:$A/compiler/ccec_compiler/bin:$A/tools/ccec_compiler/bin:/usr/bin:/bin:$PATH"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEV:-0}"
export PYTHONNOUSERSITE=1

exec "$@"
