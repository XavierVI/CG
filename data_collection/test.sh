#!/bin/bash

#flux: --job-name=cg_test
#flux: --output=cg_test.out
#flux: --error=cg_test.err
#flux: --queue=pdebug
#flux: --nodes=2
#flux: --nslots=8
#flux: --time=00:10:00

module load craype-accel-amd-gfx942

export MPICH_GPU_SUPPORT_ENABLED=1
MATRIX=


flux run -N 2 -n 8 -g 1 ../build/cg