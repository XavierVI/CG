#!/bin/bash

# load modules
module load cmake
module load rocm/6.4.3
module load cray-mpich/9.0.1
# needed for GTL
module load craype-accel-amd-gfx942

rm -rf build/
mkdir build
cd build

hipcc -o ./hip-cg ../src/hip/hip_cg.cpp \
  -I${CRAY_MPICH_DIR}/include \
  -L${CRAY_MPICH_DIR}/lib -lmpi \
  -lnuma \
  -lmpi_gtl_hsa \
  -DGPU -DGPU_AWARE \
  -lrocsparse \
  -lrocblas

