#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include "hip_utils.hpp"

/* ========================================================

           Linear algebra operations

=========================================================*/

/* 
Function call to rocsparse_spmv to perform sparse matrix-vector multiplication on device.
@args - handle: rocsparse handle
        A: rocsparse sparse matrix descriptor
        alpha: scalar multiplier for A*x
        x: rocsparse dense vector descriptor for input vector
        beta: scalar multiplier for y
        y: rocsparse dense vector descriptor for output vector
        tmp_buffer_size: size of temporary buffer for rocsparse_spmv
        tmp_buffer: pointer to temporary buffer for rocsparse_spmv

*/
void roc_spmv(
    rocsparse_handle handle, rocsparse_spmat_descr A,
    double alpha, rocsparse_dnvec_descr x, 
    double beta, rocsparse_dnvec_descr y,
    size_t tmp_buffer_size, void* tmp_buffer
) {
    ROCSPARSE_CHECK(rocsparse_spmv(handle, rocsparse_operation_none,
            &alpha, A, x, &beta, y, 
            rocsparse_datatype_f64_r,
            rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_compute,
            &tmp_buffer_size, tmp_buffer));
}

/* 
Parallel SpMV b = alpha*A*x + beta*b (point to point version).

@args - alpha: scalar multiplier for A*x
        A: ParMat object containing the distributed matrix
        x: input vector (local portion)
        beta: scalar multiplier for b
        b: output vector (local portion)
*/
void ptp_spmv(
    double alpha, ParMat& A,
    double* x_d, rocsparse_dnvec_descr vec_x,
    double beta, double* b_d, rocsparse_dnvec_descr vec_b,
    double* sendbuf, double* recvbuf, rocsparse_dnvec_descr vec_recv)
{
    int proc, start, end;
    int tag = 0;

     // Launch Pack Kernel -- Pack Send Buffer
    if (A.send_comm.size_msgs) {
        dim3 threads(256);
        dim3 blocks((A.send_comm.size_msgs + threads.x - 1) / threads.x);
        pack<<<blocks, threads, 0, 0>>>(x_d, (const int*)A.send_comm.d_idx,
                sendbuf, A.send_comm.size_msgs);
        HIP_CHECK(hipStreamSynchronize(0));
    }

    for (int i = 0; i < A.recv_comm.n_msgs; i++) {
        proc  = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i];
        end   = A.recv_comm.ptr[i + 1];
        MPI_Irecv(&(recvbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.recv_comm.req[i]));
    }

    for (int i = 0; i < A.send_comm.n_msgs; i++) {
        proc  = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end   = A.send_comm.ptr[i + 1];
        for (int j = start; j < end; j++)
            sendbuf[j] = x[A.send_comm.idx[j]];
        MPI_Isend(&(sendbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.send_comm.req[i]));
    }

    roc_spmv(alpha, A.on_proc, x, beta, b);

    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    if (A.send_comm.n_msgs)
    {
        MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    roc_spmv(alpha, A.off_proc, recvbuf, 1.0, b);

}

double inner_product(std::vector<double> a, std::vector<double> b)
{
    double sum, sum_local;

    sum_local = 0;
    for (int i = 0; i < a.size(); i++)
        sum_local += a[i] * b[i];

    MPI_Allreduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return sum;
}

void axpy(double alpha, std::vector<double>& x, std::vector<double>& y)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = x[i] + alpha*y[i];
}

void scale(double alpha, std::vector<double>& x)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = alpha*x[i];
}


/* 
*/
void CG(
    ParMat& A, double* x, rocsparse_dnvec_descr vec_x,
    double* b, rocsparse_dnvec_descr vec_b,
    double* sendbuf, double* recvbuf, rocsparse_dnvec_descr vec_recv
) {
    std::vector<double> res;s
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 1. Initialize CG Variables and Descriptors
    
    double *r, *p, *Ap;
    rocsparse_dnvec_descr vec_r, vec_p, vec_Ap;

    HIP_CHECK(hipMalloc((void**)&r, A_local_rows*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&p, A_local_rows*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&Ap, A_local_rows*sizeof(double)));

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

    // Set b to random values, x to 0
    srand(time(NULL) + rank);
    std::generate(x.begin(), x.end(), 
            [&](){ return (double)(rand()) / RAND_MAX; });
    spmv(1.0, A, x, 0.0, b);
    std::fill(x.begin(), x.end(), 0);
    
    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r, tol = 1e-6;
    // int max_iter = ((int)(1.3*b.size())) + 2;
    int max_iter = 500;

    // r0 = b - A * x0
    HIP_CHECK(
        hipMemcpyAsync(r, b, A.local_rows*sizeof(double),
            hipMemcpyDeviceToDevice, 0));
    HIP_CHECK(hipStreamSynchronize(0));
    ptp_spmv(-1.0, A, x, 1.0, r);

    // p0 = r0
    HIP_CHECK(hipMemcpyAsync(p, r, A.local_rows*sizeof(double),
            hipMemcpyDeviceToDevice, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    // Find initial (r, r) and residual
    rr_inner = inner_product(r, r);
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
        ptp_spmv(1.0, A, p, 0.0, Ap);
        App_inner = inner_product(Ap, p);
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rr_inner / App_inner;

        axpy(alpha, x, p);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            axpy(-1.0*alpha, r, Ap);
        }
        else
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r);
        }

        next_inner = inner_product(r, r);
        beta = next_inner / rr_inner;

        scale(beta, p);
        axpy(1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);

        res.push_back(norm_r);

        iter++;
    }
}


/* MAIN
    @args - filename: path to matrix file in par_binary format??? (default: "Dubcova2.pm")
*/
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 1. Read in the matrix
    const char* filename = "Dubcova2.pm";
    if (argc > 1) filename = argv[1];

    ParMat A;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    readParMatrix(filename, A);
    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Read matrix: %e\n", t0);
    fflush(stdout);

    // 2. Form communication patterns
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    form_comm(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Form comm: %e\n", t0);
    fflush(stdout);

    // 3. Copy matrix to device and initialize buffers
    copy_to_device(A);

    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    double *x_d, *b_d, *r_d;
    HIP_CHECK(hipMalloc((void**)&x_d, A.local_cols*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&b_d, A.local_rows*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&r_d, A.local_rows*sizeof(double)));

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

    // 4. Initialize device descriptors for x, b, r, and recv buffer
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


    // 5. Initialize SpMV Buffers
    double one = 1.0;
    double zero = 0.0;
    ROCSPARSE_CHECK(rocsparse_spmv(A.sparse_handle, 
            rocsparse_operation_none,
            &one, A.d_on_proc.descr, vec_x, &zero, vec_b,
            rocsparse_datatype_f64_r,
            rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_buffer_size,
            &A.d_on_proc.buf_size, NULL));
    if (A.d_on_proc.buf_size)
    {
        HIP_CHECK(hipMalloc(&A.d_on_proc.buffer,
            A.d_on_proc.buf_size));
    }
    ROCSPARSE_CHECK(rocsparse_spmv(A.sparse_handle, 
            rocsparse_operation_none,
            &one, A.d_off_proc.descr, vec_recv, &zero, vec_b,
            rocsparse_datatype_f64_r,
            rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_buffer_size,
            &A.d_off_proc.buf_size, NULL)); 
    if (A.d_off_proc.buf_size)
    {
        HIP_CHECK(hipMalloc(&A.d_off_proc.buffer,
                A.d_off_proc.buf_size));
    }

    // 6. Initialize x
    // Set x to random values, b = A*x
    // Will reset x to 0 before each CG
    double alpha = 1.0;
    double beta = 0.0;
    std::mt19937 rng(rank + 12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::generate(x.begin(), x.end(),
              [&]() { return dist(rng); });
    HIP_CHECK(hipMemcpy(x_d, x.data(), x.size() * sizeof(double),
            hipMemcpyHostToDevice));
    ptp_spmv(
        alpha, A, x_d, vec_x,
        beta, b_d, vec_b,
        sendbuf, recvbuf, vec_recv
    );



   

    if (rank == 0) 
    {
        if (iter == max_iter)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iteration required to converge\n", iter);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
    }

    MPI_Finalize();
}
