#!/bin/bash

# load modules
module load cmake
module load rocm/6.4.3
module load cray-mpich/9.0.1
# needed for GTL
module load craype-accel-amd-gfx942

CALIPER_PATH=$(spack location -i caliper@2.14.0)
ADIAK_PATH=$(spack location -i adiak %cce@20.0.0)

rm -rf build/
mkdir build
cd build

echo "building with $(which hipcc)"
echo "              $(which mpicc)"
echo "              ${CRAY_MPICH_DIR}"

hipcc -o ./hip-cg ../src/hip/hip_cg.cpp \
  -I${CRAY_MPICH_DIR}/include \
  -I${CALIPER_PATH}/include \
  -I${ADIAK_PATH}/include \
  -L${CRAY_MPICH_DIR}/lib -lmpi \
  -L${CALIPER_PATH}/lib64 \
  -L${ADIAK_PATH}/lib \
  -Wl,-rpath,${CALIPER_PATH}/lib64 \
  -Wl,-rpath,${ADIAK_PATH}/lib \
  -lcaliper \
  -ladiak \
  -lnuma \
  -lmpi_gtl_hsa \
  -DGPU -DGPU_AWARE \
  -lrocsparse \
  -lrocblas

