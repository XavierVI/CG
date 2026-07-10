#!/bin/bash

# read arguments
export FI_CXI_RDZV_THRESHOLD=$1
export FI_CXI_RX_MATCH_MODE=$2
export MPICH_GPU_IPC_ENABLED=$3
export MPICH_ASYNC_PROGRESS=$4
export MAX_ITERS=$5
export MATRIX=$6


export MATRIX_PATH=$PWD/../petsc_matrices/${MATRIX}.petsc

TAG="NODES-${NNODES}_PROCS-${PROCS}_RDZV-${FI_CXI_RDZV_THRESHOLD}_MATCH-${FI_CXI_RX_MATCH_MODE}_IPC-${MPICH_GPU_IPC_ENABLED}_ASYNC-${MPICH_ASYNC_PROGRESS}_ITERATIONS-${MAX_ITERS}_MAT-${MATRIX}"

# caliper configs
export CALI_CONFIG="runtime-profile(output=${TAG}.cali,profile.mpi,mpi.message.count,mpi.message.size)"
export TOPOLOGY_FILE=${TAG}.json

# export CALI_SERVICES_ENABLE=debug
# export CALI_LOG_VERBOSITY=2

echo "Running CG with the following parameters:"
echo "FI_CXI_RDZV_THRESHOLD: $FI_CXI_RDZV_THRESHOLD"
echo "FI_CXI_RX_MATCH_MODE: $FI_CXI_RX_MATCH_MODE"
echo "MPICH_GPU_IPC_ENABLED: $MPICH_GPU_IPC_ENABLED"
echo "MPICH_ASYNC_PROGRESS: $MPICH_ASYNC_PROGRESS"
echo "MAX_ITERS: $MAX_ITERS"
echo "MATRIX: $MATRIX"
echo "MATRIX_PATH: $MATRIX_PATH"

$HOME/workspace/CG/build/hip-cg $MATRIX_PATH $MAX_ITERS