#!/bin/bash

#flux: --job-name=cg_test
#flux: --output=cg_test.out
#flux: --queue=pdebug
#flux: --nodes=2
#flux: --nslots=4
#flux: --time-limit=5m

export FLUX_BATCH_JOB_ID=$FLUX_JOB_ID
echo "FLUX_JOB_ID: $FLUX_JOB_ID"

module load craype-accel-amd-gfx942

export MPICH_GPU_SUPPORT_ENABLED=1
export MATRIX=Flan_1565
export MATRIX_PATH=$HOME/workspace/CG/matrices/${MATRIX}.petsc
# MATRIX=$PWD/../src/Dubcova2.pm

# caliper configs
export CALI_CONFIG="runtime-profile(output=cg_solver.cali,profile.mpi,mpi.message.count,mpi.message.size)"
# export CALI_SERVICES_ENABLE=debug
# export CALI_LOG_VERBOSITY=2

export FI_CXI_RDZV_THRESHOLD=128
export FI_CXI_RX_MATCH_MODE=1
export MPICH_GPU_IPC_ENABLED=1
export MPICH_ASYNC_PROGRESS=1

flux run -N 1 -n 4 -g 1 ../build/hip-cg $MATRIX_PATH