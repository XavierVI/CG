#!/bin/bash

#flux: --job-name=cg_test
#flux: --output=cg_test.out
#flux: --queue=pdebug
#flux: --nodes=1
#flux: --nslots=2
#flux: --time-limit=5m

module load craype-accel-amd-gfx942

export MPICH_GPU_SUPPORT_ENABLED=1
MATRIX=$HOME/workspace/CG/matrices/Flan_1565.petsc
# MATRIX=$PWD/../src/Dubcova2.pm

# caliper configs
export CALI_CONFIG="runtime-profile(output=cg_solver.cali,profile.mpi,mpi.message.count,mpi.message.size)"

flux run -N 1 -n 2 -g 1 ../build/hip-cg $MATRIX