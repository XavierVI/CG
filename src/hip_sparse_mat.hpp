#ifndef MPI_SPARSE_MAT_HPP
#define MPI_SPARSE_MAT_HPP

#include <mpi.h>

#include <vector>

#include "hip_utils.hpp"

struct GPUMat
{
    int* rowptr;
    int* col_idx;
    double* data;

    int n_rows;
    int n_cols;
    int nnz;
    rocsparse_spmat_descr descr;
    size_t buf_size;
    void* buffer;
};

struct Mat
{
    std::vector<int> rowptr;
    std::vector<int> col_idx;
    std::vector<double> data;
    int n_rows;
    int n_cols;
    int nnz;
};

struct Comm
{
    int n_msgs;
    int size_msgs;
    std::vector<int> procs;
    std::vector<int> ptr;
    std::vector<int> counts;
    std::vector<int> idx;
    std::vector<MPI_Request> req;

    void* d_idx;
};

struct ParMat
{
    Mat on_proc;
    Mat off_proc;
    GPUMat d_on_proc;
    GPUMat d_off_proc;
    int global_rows;
    int global_cols;
    int local_rows;
    int local_cols;
    int first_row;
    int first_col;
    int off_proc_num_cols;
    std::vector<long> off_proc_columns;
    Comm send_comm;
    Comm recv_comm;
    MPI_Comm dist_graph_comm;

    rocsparse_handle sparse_handle;
    rocblas_handle blas_handle;
};

void form_recv_comm(ParMat& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Gather first col for all processes into list
    std::vector<int> first_cols(num_procs + 1);
    MPI_Allgather(
        &A.first_col, 1, MPI_INT, first_cols.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_cols[num_procs] = A.global_cols;

    // Map Columns to Processes
    int proc      = 0;
    int prev_proc = -1;
    for (int i = 0; i < A.off_proc_num_cols; i++)
    {
        int global_col = A.off_proc_columns[i];
        while (first_cols[proc + 1] <= global_col)
        {
            proc++;
        }
        if (proc != prev_proc)
        {
            A.recv_comm.procs.push_back(proc);
            A.recv_comm.ptr.push_back((i));
            prev_proc = proc;
        }
    }

    // Set Recv Sizes
    A.recv_comm.ptr.push_back((A.off_proc_num_cols));
    A.recv_comm.n_msgs    = A.recv_comm.procs.size();
    A.recv_comm.size_msgs = A.off_proc_num_cols;
    if (A.recv_comm.n_msgs == 0)
    {
        return;
    }

    A.recv_comm.req.resize(A.recv_comm.n_msgs);
    A.recv_comm.counts.resize(A.recv_comm.n_msgs);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        A.recv_comm.counts[i] = A.recv_comm.ptr[i + 1] - A.recv_comm.ptr[i];
    }
}

// Must Form Recv Comm before Send!
void form_send_comm_standard(ParMat& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<long> recv_buf;
    std::vector<int> sizes(num_procs, 0);
    int proc, count, ctr;
    MPI_Status recv_status;

    // Allreduce to find size of data I will receive
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        sizes[A.recv_comm.procs[i]] = A.recv_comm.ptr[i + 1] - A.recv_comm.ptr[i];
    }
    MPI_Allreduce(
        MPI_IN_PLACE, sizes.data(), num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    A.send_comm.size_msgs = sizes[rank];

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int msg_tag = 1234;
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        MPI_Isend(&(A.off_proc_columns[A.recv_comm.ptr[i]]),
                  A.recv_comm.counts[i],
                  MPI_LONG,
                  proc,
                  msg_tag,
                  MPI_COMM_WORLD,
                  &(A.recv_comm.req[i]));
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    if (A.send_comm.size_msgs)
    {
        A.send_comm.idx.resize(A.send_comm.size_msgs);
        recv_buf.resize(A.send_comm.size_msgs);
    }
    ctr = 0;
    A.send_comm.ptr.push_back(0);
    while (ctr < A.send_comm.size_msgs)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, msg_tag, MPI_COMM_WORLD, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        A.send_comm.procs.push_back(proc);
        MPI_Get_count(&recv_status, MPI_LONG, &count);
        A.send_comm.counts.push_back(count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]),
                 count,
                 MPI_LONG,
                 proc,
                 msg_tag,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++)
        {
            A.send_comm.idx[ctr + i] = (recv_buf[ctr + i] - A.first_col);
        }
        ctr += count;
        A.send_comm.ptr.push_back((ctr));
    }

    // Set send sizes
    A.send_comm.n_msgs = A.send_comm.procs.size();

    if (A.send_comm.n_msgs)
    {
        A.send_comm.req.resize(A.send_comm.n_msgs);
    }

    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }
}

void form_comm(ParMat& A)
{
    // Form Recv Side
    form_recv_comm(A);

    // Form Send Side (Algorithm Options Here!)
    form_send_comm_standard(A);
}


void copy_to_device(const Mat& h, GPUMat& d)
{
    d.n_rows = h.n_rows;
    d.n_cols = h.n_cols;
    d.nnz = h.nnz;
    HIP_CHECK(hipMalloc(&d.rowptr, (d.n_rows+1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d.col_idx, d.nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&d.data, d.nnz*sizeof(double)));
    HIP_CHECK(hipMemcpy(d.rowptr, h.rowptr.data(), (d.n_rows+1)*sizeof(int),
            hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d.col_idx, h.col_idx.data(), d.nnz*sizeof(int),
            hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d.data, h.data.data(), d.nnz*sizeof(double),
            hipMemcpyHostToDevice));

    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&d.descr,
            d.n_rows, d.n_cols, d.nnz,
            d.rowptr, d.col_idx, d.data,
            rocsparse_indextype_i32, rocsparse_indextype_i32,
            rocsparse_index_base_zero, rocsparse_datatype_f64_r));
}

void copy_to_device(ParMat& A)
{
    ROCSPARSE_CHECK(rocsparse_create_handle(&A.sparse_handle));
    ROCBLAS_CHECK(rocblas_create_handle(&A.blas_handle));

    copy_to_device(A.on_proc, A.d_on_proc);
    copy_to_device(A.off_proc, A.d_off_proc);

    if (A.send_comm.size_msgs)
    {
        HIP_CHECK(hipMalloc(&A.send_comm.d_idx, A.send_comm.size_msgs * sizeof(int)));
        HIP_CHECK(hipMemcpy(A.send_comm.d_idx, A.send_comm.idx.data(),
                A.send_comm.size_msgs*sizeof(int), hipMemcpyHostToDevice));
    }

}

void free_gpu_mat(GPUMat& d)
{
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(d.descr));
    HIP_CHECK(hipFree(d.rowptr));
    HIP_CHECK(hipFree(d.col_idx));
    HIP_CHECK(hipFree(d.data));
}

void free_mat(ParMat& A)
{
    free_gpu_mat(A.d_on_proc);
    free_gpu_mat(A.d_off_proc);

    if (A.send_comm.size_msgs)
    {
        HIP_CHECK(hipFree(A.send_comm.d_idx));
    }

    ROCSPARSE_CHECK(rocsparse_destroy_handle(A.sparse_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(A.blas_handle));
}

#endif