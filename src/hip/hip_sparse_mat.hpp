#ifndef MPI_SPARSE_MAT_HPP
#define MPI_SPARSE_MAT_HPP

#include <mpi.h>
#include <vector>
#include "hip_utils.hpp"
#include <cstring>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

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


/*
 * gather_hostnames
 * -----------------
 * Inputs:
 *   comm - MPI communicator over which to gather hostnames.
 * Returns:
 *   A vector of size num_procs, where entry i holds the hostname of rank i,
 *   identical on every rank in comm.
 * Purpose:
 *   MPI_Get_processor_name only returns the name of the calling process;
 *   there is no local way to learn another rank's hostname. This function
 *   performs the required collective communication (MPI_Allgather) to
 *   build a full rank -> hostname table on every process.
 */
std::vector<std::string> gather_hostnames(MPI_Comm comm) {
    int num_procs;                                    // number of ranks in comm
    MPI_Comm_size(comm, &num_procs);                   // populate num_procs

    char local_hostname[MPI_MAX_PROCESSOR_NAME];        // buffer for this rank's hostname
    int name_len = 0;                                   // length returned by MPI, unused after the call
    MPI_Get_processor_name(local_hostname, &name_len);  // fill local_hostname with this process's node name

    // Zero-pad the unused tail of the buffer so MPI_Allgather sends a
    // deterministic fixed-size block regardless of hostname length.
    for (int i = name_len; i < MPI_MAX_PROCESSOR_NAME; ++i) {
        local_hostname[i] = '\0';                        // clear trailing bytes
    }

    // Flat receive buffer: num_procs blocks of MPI_MAX_PROCESSOR_NAME chars each.
    std::vector<char> all_hostnames(static_cast<size_t>(num_procs) * MPI_MAX_PROCESSOR_NAME);

    // Collective all-to-all gather: every rank ends up with every hostname.
    MPI_Allgather(local_hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  all_hostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  comm);

    // Slice the flat buffer into one std::string per rank.
    std::vector<std::string> hostnames(num_procs);       // output table, indexed by rank
    for (int i = 0; i < num_procs; ++i) {
        hostnames[i] = std::string(&all_hostnames[static_cast<size_t>(i) * MPI_MAX_PROCESSOR_NAME]); // relies on null padding above
    }

    return hostnames;                                    // identical content on every rank
}

/*
 * assign_node_ids
 * -----------------
 * Inputs:
 *   hostnames - rank-indexed vector of hostnames, identical on every rank
 *               (output of gather_hostnames).
 * Outputs (written by reference):
 *   node_ids    - rank-indexed vector; node_ids[r] identifies which
 *                 physical node rank r ran on. Nodes are numbered in
 *                 order of first appearance by rank.
 *   local_ranks - rank-indexed vector; local_ranks[r] is the rank's index
 *                 among all ranks sharing its node, in rank order
 *                 (equivalent to what MPI_Comm_split_type with
 *                 MPI_COMM_TYPE_SHARED would give as the local rank).
 * Purpose:
 *   Converts a flat hostname list into (node_id, local_rank) pairs so
 *   downstream analysis does not need to depend on hostname strings.
 *   Purely local computation: no communication needed, since hostnames
 *   is already replicated on every rank.
 */
void assign_node_ids(const std::vector<std::string>& hostnames,
                      std::vector<int>& node_ids,
                      std::vector<int>& local_ranks) {
    int num_procs = static_cast<int>(hostnames.size());   // total rank count
    node_ids.assign(num_procs, -1);                         // -1 marks "not yet assigned"
    local_ranks.assign(num_procs, 0);                        // default local rank

    std::vector<std::string> seen_hosts;                     // hostnames in order of first appearance
    std::vector<int> counts_per_node;                         // running count of ranks assigned per node so far

    for (int r = 0; r < num_procs; ++r) {                     // walk ranks in order
        int node_id = -1;                                      // node id for this rank, to be found or created
        for (size_t n = 0; n < seen_hosts.size(); ++n) {        // search already-registered nodes
            if (seen_hosts[n] == hostnames[r]) {                  // hostname matches an existing node
                node_id = static_cast<int>(n);                     // record the match
                break;                                              // stop searching
            }
        }
        if (node_id == -1) {                                    // hostname not seen before -> new node
            node_id = static_cast<int>(seen_hosts.size());         // next available node id
            seen_hosts.push_back(hostnames[r]);                      // register hostname
            counts_per_node.push_back(0);                             // initialize its rank counter
        }
        node_ids[r] = node_id;                                   // record this rank's node
        local_ranks[r] = counts_per_node[node_id];               // this rank's position within the node so far
        counts_per_node[node_id]++;                              // bump the counter for the next rank on this node
    }
}

/*
 * write_topology_json
 * ---------------------
 * Inputs:
 *   filename    - path to write the JSON file to.
 *   hostnames   - rank-indexed hostnames.
 *   node_ids    - rank-indexed node ids (see assign_node_ids).
 *   local_ranks - rank-indexed local ranks (see assign_node_ids).
 * Purpose:
 *   Serializes the rank/hostname/node/local-rank table to a JSON file
 *   with no external JSON library dependency. Intended to be called by
 *   exactly one rank (rank 0) to avoid concurrent writes to the same path.
 */
void write_topology_json(const std::string& filename,
                          const std::vector<std::string>& hostnames,
                          const std::vector<int>& node_ids,
                          const std::vector<int>& local_ranks) {
    int num_procs = static_cast<int>(hostnames.size());    // total rank count

    int num_nodes = 0;                                        // running max node id + 1
    for (int id : node_ids) {                                  // scan all assigned node ids
        if (id + 1 > num_nodes) num_nodes = id + 1;               // track the highest id seen
    }

    std::ofstream out(filename);                              // open output file, truncating if it exists
    if (!out.is_open()) {                                      // guard against an unwritable path
        std::cerr << "save_node_topology: failed to open " << filename << " for writing\n"; // report failure
        return;                                                  // abort write; caller still proceeds (non-fatal)
    }

    out << "{\n";                                               // start top-level JSON object
    out << "  \"num_procs\": " << num_procs << ",\n";           // total rank count, for sanity-checking joins later
    out << "  \"num_nodes\": " << num_nodes << ",\n";           // total distinct nodes used
    out << "  \"ranks\": [\n";                                  // begin array of per-rank records

    for (int r = 0; r < num_procs; ++r) {                       // emit one JSON object per rank
        out << "    {\n";
        out << "      \"rank\": " << r << ",\n";                  // MPI rank; join key against Caliper's mpi.rank
        out << "      \"hostname\": \"" << hostnames[r] << "\",\n"; // node hostname
        out << "      \"node_id\": " << node_ids[r] << ",\n";     // integer node id, ordered by first appearance
        out << "      \"local_rank\": " << local_ranks[r] << "\n"; // rank's position among co-located ranks
        out << "    }" << (r + 1 < num_procs ? "," : "") << "\n"; // comma-separate, no trailing comma on the last entry
    }

    out << "  ]\n";                                             // close ranks array
    out << "}\n";                                                // close top-level object
    out.close();                                                 // flush and close the file
}

/*
 * save_node_topology
 * --------------------
 * Inputs:
 *   comm - MPI communicator whose rank-to-node mapping should be recorded.
 * Purpose:
 *   Top-level entry point: gathers hostnames across comm, derives node ids
 *   and local ranks, and writes the result to a JSON file so it can be
 *   joined against Caliper profiling output by MPI rank during analysis.
 *   The output path defaults to "node_topology.json" and can be overridden
 *   with the NODE_TOPOLOGY_FILE environment variable. Every rank must call
 *   this function, since MPI_Allgather is collective; only rank 0 performs
 *   the actual file write.
 */
void save_node_topology(MPI_Comm comm) {
    int rank;                                                    // this process's rank in comm
    MPI_Comm_rank(comm, &rank);                                   // populate rank

    // Collective: every rank must reach this call; result is identical everywhere.
    std::vector<std::string> hostnames = gather_hostnames(comm);   // rank -> hostname table

    std::vector<int> node_ids;                                    // rank -> node id table, filled below
    std::vector<int> local_ranks;                                 // rank -> local rank table, filled below
    assign_node_ids(hostnames, node_ids, local_ranks);             // local computation, no further communication needed

    // Resolve output filename: environment variable overrides the default.
    std::string filename = "node_topology.json";                  // default path
    const char* env_filename = std::getenv("NODE_TOPOLOGY_FILE");  // check for override
    if (env_filename) {                                             // if the variable is set,
        filename = env_filename;                                     // use it instead of the default
    }

    if (rank == 0) {                                               // only one rank should write the file
        write_topology_json(filename, hostnames, node_ids, local_ranks); // perform the write
    }

    MPI_Barrier(comm);                                             // ensure the file exists on disk before any rank proceeds
}

#endif