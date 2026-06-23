#include "hip_sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include "hip_utils.hpp"
#include <math.h>
#include <random>


// NOTE: this file is a work in progress for combining hip computation and point to point communication (which exists in two difference branches as no one seperated the code into different files, just over wrote the original code)

__global__ void pack(
    const double* __restrict__ x,
    const int* __restrict__ idx,
    double* __restrict__ packed_buf,
    int n
) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) packed_buf[i] = x[idx[i]];
}


/**
Function call to rocsparse_spmv to perform sparse matrix-vector multiplication on device.
@args
- handle: rocsparse handle
- A: rocsparse sparse matrix descriptor
- alpha: scalar multiplier for A*x
- x: rocsparse dense vector descriptor for input vector
- beta: scalar multiplier for y
- y: rocsparse dense vector descriptor for output vector
- tmp_buffer_size: size of temporary buffer for rocsparse_spmv
- tmp_buffer: pointer to temporary buffer for rocsparse_spmv

*/
void local_spmv(
    rocsparse_handle handle, rocsparse_spmat_descr A,
    double alpha, rocsparse_dnvec_descr x,
    double beta, rocsparse_dnvec_descr y,
    size_t tmp_buffer_size, void* tmp_buffer
) {
    ROCSPARSE_CHECK(
        rocsparse_spmv(
            handle, rocsparse_operation_none,
            &alpha, A, x, &beta, y,
            rocsparse_datatype_f64_r,
            rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_compute,
            &tmp_buffer_size, tmp_buffer
        )
    );
}



/**
Parallel SpMV: b = alpha*A*x + beta*b (point to point version).

@args
- alpha: scalar multiplier for A*x
- A: ParMat object containing the distributed matrix
- x_d: input vector (local portion, device pointer)
- vec_x: rocsparse dense vector descriptor for input vector
- beta: scalar multiplier for b
- b_d: output vector (local portion, device pointer)
- vec_b: rocsparse dense vector descriptor for output vector
- sendbuf: device buffer for sending halo data
- recvbuf: device buffer for receiving halo data
- vec_recv: rocsparse dense vector descriptor for received data
*/
void parallel_spmv(
    double alpha, ParMat& A, double* x_d, rocsparse_dnvec_descr vec_x,
    double beta, double* b_d, rocsparse_dnvec_descr vec_b,
    double* sendbuf, double* recvbuf, rocsparse_dnvec_descr vec_recv
) {
    int proc, start, end;
    int tag = 0;

    // Launch Pack Kernel -- Pack Send Buffer
    if (A.send_comm.size_msgs) {
        dim3 threads(256);
        dim3 blocks((A.send_comm.size_msgs + threads.x - 1) / threads.x);
        pack<<<blocks, threads, 0, 0>>>(
            x_d,
            (const int*)A.send_comm.d_idx,
            sendbuf,
            A.send_comm.size_msgs
        );
        HIP_CHECK(hipStreamSynchronize(0));
    }

    // Initialize Receive Requests
    for (int i = 0; i < A.recv_comm.n_msgs; i++) {
        proc  = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i];
        end   = A.recv_comm.ptr[i + 1];
        MPI_Irecv(
            &(recvbuf[start]),
            (int)(end - start),
            MPI_DOUBLE,
            proc,
            tag,
            MPI_COMM_WORLD,
            &(A.recv_comm.req[i])
        );
    }

    // Initialize Send Requests
    for (int i = 0; i < A.send_comm.n_msgs; i++) {
        proc  = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end   = A.send_comm.ptr[i + 1];
        MPI_Isend(
            &(sendbuf[start]),
            (int)(end - start),
            MPI_DOUBLE,
            proc,
            tag,
            MPI_COMM_WORLD,
            &(A.send_comm.req[i])
        );
    }

    // Perform local computation while communication is in flight
    local_spmv(
        A.sparse_handle,
        A.d_on_proc.descr,
        alpha,
        vec_x,
        beta,
        vec_b,
        A.d_on_proc.buf_size,
        A.d_on_proc.buffer
    );

    // Wait for all communication to complete
    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    if (A.send_comm.n_msgs)
    {
        MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    // Perform off-process computation with received halo data
    local_spmv(
        A.sparse_handle,
        A.d_off_proc.descr,
        alpha,
        vec_recv,
        1.0,
        vec_b,
        A.d_off_proc.buf_size,
        A.d_off_proc.buffer
    );

}


/**
Parallel inner product of two device vectors using rocBLAS ddot + MPI_Allreduce.
@args
- handle: rocblas handle
- n: local number of elements
- a_d: device pointer to first vector
- b_d: device pointer to second vector
*/
double inner_product(rocblas_handle handle, int n, double* a_d, double* b_d)
{
    double local_sum, global_sum;
    rocblas_ddot(handle, n, a_d, 1, b_d, 1, &local_sum);
    HIP_CHECK(hipStreamSynchronize(0));
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}


/**
AXPY on device: x = x + alpha * y using rocBLAS daxpy.
@args
- handle: rocblas handle
- n: number of elements
- alpha: scalar multiplier
- x_d: device pointer to output vector (updated in-place)
- y_d: device pointer to input vector
*/
void axpy(rocblas_handle handle, int n, double alpha, double* x_d, double* y_d)
{
    rocblas_daxpy(handle, n, &alpha, y_d, 1, x_d, 1);
    HIP_CHECK(hipStreamSynchronize(0));
}


/**
Scale a device vector in-place: x = alpha * x using rocBLAS dscal.
@args
- handle: rocblas handle
- n: number of elements
- alpha: scalar multiplier
- x_d: device pointer to vector (updated in-place)
*/
void scale(rocblas_handle handle, int n, double alpha, double* x_d)
{
    rocblas_dscal(handle, n, &alpha, x_d, 1);
    HIP_CHECK(hipStreamSynchronize(0));
}


/**
CG solver. Assumes x is initialized to zero and b is already computed on device.
Returns the number of iterations taken.
@args
- A: distributed ParMat (matrix on both host and device)
- x: device pointer to solution vector (must be zeroed before call)
- vec_x: rocsparse dense vector descriptor for x
- b: device pointer to right-hand side vector
- vec_b: rocsparse dense vector descriptor for b
- sendbuf: device buffer for outgoing halo data
- recvbuf: device buffer for incoming halo data
- vec_recv: rocsparse dense vector descriptor for recvbuf
- final_norm: output pointer to store the final residual 2-norm
*/
int CG(
    ParMat& A, double* x, rocsparse_dnvec_descr vec_x,
    double* b, rocsparse_dnvec_descr vec_b,
    double* sendbuf, double* recvbuf, rocsparse_dnvec_descr vec_recv,
    double* final_norm, int max_iter
) {
    std::vector<double> res;
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initialize CG vectors and descriptors
    double *r, *p, *Ap;
    rocsparse_dnvec_descr vec_r, vec_p, vec_Ap;

    HIP_CHECK(hipMalloc((void**)&r, A.local_rows*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&p, A.local_rows*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&Ap, A.local_rows*sizeof(double)));

    ROCSPARSE_CHECK(
        rocsparse_create_dnvec_descr(
        &vec_r, A.local_rows, r, rocsparse_datatype_f64_r
        )
    );
    ROCSPARSE_CHECK(
        rocsparse_create_dnvec_descr(
        &vec_p, A.local_rows, p, rocsparse_datatype_f64_r
        )
    );
    ROCSPARSE_CHECK(
        rocsparse_create_dnvec_descr(
        &vec_Ap, A.local_rows, Ap, rocsparse_datatype_f64_r
        )
    );

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r, tol = 1e-6;

    // r0 = b - A * x0
    HIP_CHECK(
        hipMemcpyAsync(r, b, A.local_rows*sizeof(double),
            hipMemcpyDeviceToDevice, 0));
    HIP_CHECK(hipStreamSynchronize(0));
    parallel_spmv(-1.0, A, x, vec_x, 1.0, r, vec_r, sendbuf, recvbuf, vec_recv);

    // p0 = r0
    HIP_CHECK(hipMemcpyAsync(p, r, A.local_rows*sizeof(double),
            hipMemcpyDeviceToDevice, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    // Find initial (r, r) and residual
    rr_inner = inner_product(A.blas_handle, A.local_rows, r, r);
    norm_r = sqrt(rr_inner);
    res.push_back(norm_r);

    // Scale tolerance by norm_r
    if (norm_r != 0.0) tol = tol * norm_r;

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter) {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        parallel_spmv(1.0, A, p, vec_p, 0.0, Ap, vec_Ap, sendbuf, recvbuf, vec_recv);
        App_inner = inner_product(A.blas_handle, A.local_rows, Ap, p);
        
        if (App_inner < 0.0) {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        alpha = rr_inner / App_inner;

        // x_{i+1} = x_i + alpha_i * p_i
        axpy(A.blas_handle, A.local_rows, alpha, x, p);

        if ((iter % recompute_r) && iter > 0) {
            // r_{i+1} = r_i - alpha_i * Ap_i
            axpy(A.blas_handle, A.local_rows, -1.0*alpha, r, Ap);
        }
        else {
            // Periodically recompute r from scratch to avoid floating point drift
            HIP_CHECK(hipMemcpyAsync(r, b, A.local_rows*sizeof(double),
                    hipMemcpyDeviceToDevice, 0));
            HIP_CHECK(hipStreamSynchronize(0));
            parallel_spmv(-1.0, A, x, vec_x, 1.0, r, vec_r, sendbuf, recvbuf, vec_recv);
        }

        next_inner = inner_product(A.blas_handle, A.local_rows, r, r);
        beta = next_inner / rr_inner;

        // p_{i+1} = r_{i+1} + beta_i * p_i
        scale(A.blas_handle, A.local_rows, beta, p);
        axpy(A.blas_handle, A.local_rows, 1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);

        res.push_back(norm_r);

        iter++;
    }

    // Cleanup CG vectors and descriptors
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_r));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_p));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_Ap));
    HIP_CHECK(hipFree(r));
    HIP_CHECK(hipFree(p));
    HIP_CHECK(hipFree(Ap));

    *final_norm = norm_r;
    return iter;
}


/* MAIN
    @args - filename: path to matrix file in par_binary format (default: "Dubcova2.pm")
*/
int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    double t0, tfinal;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /*
      Read in the matrix

    */
    const char* filename = "Dubcova2.pm";
    if (argc > 1) filename = argv[1];

    ParMat A;
    MPI_Barrier(MPI_COMM_WORLD);

    t0 = MPI_Wtime();

    readParMatrix(filename, A);

    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) printf("Read matrix: %e\n", tfinal);
    fflush(stdout);

    /*
      Form communication patterns

    */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    form_comm(A); // defined in hip_sparse_mat.hpp

    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) printf("Form comm: %e\n", tfinal);
    fflush(stdout);

    /*
      Copy matrix to device and initialize dense vector buffers

      */
    copy_to_device(A); // defined in hip_sparse_mat.hpp

    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    double *x_d, *b_d, *r_d;
    HIP_CHECK(hipMalloc((void**)&x_d, A.local_cols*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&b_d, A.local_rows*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&r_d, A.local_rows*sizeof(double)));


    /*
      Initialize send and receive buffers

      */
    double* sendbuf = NULL;
    double* recvbuf = NULL;

    if (A.send_comm.size_msgs) {
        HIP_CHECK(hipMalloc((void**)&sendbuf,
                A.send_comm.size_msgs*sizeof(double)));
    }
    if (A.recv_comm.size_msgs) {
        HIP_CHECK(hipMalloc((void**)&recvbuf,
                A.recv_comm.size_msgs*sizeof(double)));
    }

    /*
      Initialize device descriptors for dense vectors x, b, r, and recv buffer

    */
    rocsparse_dnvec_descr vec_x, vec_b, vec_r, vec_recv;
    ROCSPARSE_CHECK(
        rocsparse_create_dnvec_descr(
            &vec_x, A.local_cols, x_d, rocsparse_datatype_f64_r)
    );
    ROCSPARSE_CHECK(
        rocsparse_create_dnvec_descr(&vec_b, A.local_rows, b_d,
            rocsparse_datatype_f64_r)
    );
    ROCSPARSE_CHECK(
        rocsparse_create_dnvec_descr(&vec_r, A.local_rows, r_d,
            rocsparse_datatype_f64_r)
    );
    ROCSPARSE_CHECK(
        rocsparse_create_dnvec_descr(&vec_recv, A.recv_comm.size_msgs,
                recvbuf, rocsparse_datatype_f64_r)
    );


    /*
      Initialize SpMV Buffers

      Two rocsparse_spmv calls with rocsparse_spmv_stage_buffer_size
        (one for d_on_proc, one for d_off_proc × vec_recv)
        to query how much scratch space rocsparse needs,
        then hipMalloc those buffers and store them on the GPUMat structs.

      on_proc = data owned by this process
      off_proc = data owned by another process
    */
    double one = 1.0; // ???
    double zero = 0.0;
    // call rocsparse_spmv to figure out the buffer size needed
    ROCSPARSE_CHECK(
        rocsparse_spmv(
            A.sparse_handle,
            rocsparse_operation_none,
            &one, A.d_on_proc.descr, vec_x, &zero, vec_b,
            rocsparse_datatype_f64_r,
            rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_buffer_size,
            &A.d_on_proc.buf_size, NULL
        )
    );

    // if buffer size is non-zero, allocate device memory for the buffer
    if (A.d_on_proc.buf_size) {
        HIP_CHECK(hipMalloc(&A.d_on_proc.buffer,
            A.d_on_proc.buf_size));
    }
    // repeat for off-proc
    ROCSPARSE_CHECK(
        rocsparse_spmv(
            A.sparse_handle,
            rocsparse_operation_none,
            &one, A.d_off_proc.descr, vec_recv, &zero, vec_b,
            rocsparse_datatype_f64_r,
            rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_buffer_size,
            &A.d_off_proc.buf_size, NULL
        )
    );
    if (A.d_off_proc.buf_size) {
        HIP_CHECK(hipMalloc(&A.d_off_proc.buffer,
                A.d_off_proc.buf_size));
    }

    /*
      Initialize x

      Set x to random values, b = A*x
      Will reset x to 0 before CG

      */
    double alpha = 1.0;
    double beta = 0.0;
    std::mt19937 rng(rank + 12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(rng); });
    HIP_CHECK(
        hipMemcpy(x_d, x.data(), x.size() * sizeof(double), hipMemcpyHostToDevice)
    );
    // compute b = A*x
    parallel_spmv(
        alpha, A, x_d, vec_x,
        beta, b_d, vec_b,
        sendbuf, recvbuf, vec_recv
    );

    /*
      Run CG
      TODO: add code to run multiple iterations and time them
    */
    double norm_r;
    int max_iter = 500;
    MPI_Barrier(MPI_COMM_WORLD);
    HIP_CHECK(hipMemsetAsync(x_d, 0, A.local_cols*sizeof(double), 0));
    HIP_CHECK(hipStreamSynchronize(0));
    
    int conv_iters = CG(A, x_d, vec_x, b_d, vec_b, sendbuf, recvbuf, vec_recv, &norm_r, max_iter);

    if (rank == 0)
    {
        if (conv_iters == max_iter)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iterations required to converge\n", conv_iters);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
    }


    // Finalize
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_b));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_r));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_recv));

    HIP_CHECK(hipFree(x_d));
    HIP_CHECK(hipFree(b_d));
    HIP_CHECK(hipFree(r_d));
    HIP_CHECK(hipFree(sendbuf));
    HIP_CHECK(hipFree(recvbuf));

    MPI_Finalize();
    return 0;
}
