#!/bin/bash

# read arguments
export FI_CXI_RDZV_THRESHOLD=$1
export FI_CXI_RX_MATCH_MODE=$2
export MPICH_GPU_IPC_ENABLED=$3
export MPICH_ASYNC_PROGRESS=$4
export MATRIX=$5
export MAX_ITERS=$6

export MATRIX_PATH=$HOME/workspace/CG/matrices/${MATRIX}.petsc

# caliper configs
export CALI_CONFIG="runtime-profile(output=cg_solver.cali,profile.mpi,mpi.message.count,mpi.message.size)"
# export CALI_SERVICES_ENABLE=debug
# export CALI_LOG_VERBOSITY=2

$HOME/workspace/CG/build/hip-cg $MATRIX_PATH $MAX_ITERS