#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include "hip/hip_runtime.h"

// From ROCm HIP-Tests
#define HIP_CHECK(cmd)                                               \
  {                                                                  \
    hipError_t error = cmd;                                          \
    if (error != hipSuccess) {                                       \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                  \
            hipGetErrorString(error), error, __FILE__,__LINE__);     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                \
  }
#define ROCSPARSE_CHECK(cmd)                                         \
  {                                                                  \
    rocsparse_status error = cmd;                                    \
    if (error != rocsparse_status_success) {                         \
      fprintf(stderr, "error: '%d' at %s:%d\n",                      \
            (int)error, __FILE__, __LINE__);                         \
      MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                \
  }                                                                  
#define ROCBLAS_CHECK(cmd)                                           \
  {                                                                  \
    rocblas_status error = cmd;                                      \
    if (error != rocblas_status_success) {                           \
      fprintf(stderr, "error: '%d' at %s:%d\n",                      \
            (int)error, __FILE__, __LINE__);                         \
      MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                \
  }  
