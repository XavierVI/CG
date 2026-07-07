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
#include <algorithm>

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
 * gather_variable_ints
 * ----------------------
 * Inputs:
 *   local - this rank's data (arbitrary length, may differ per rank).
 *   comm  - MPI communicator to gather over.
 * Outputs (written by reference, only meaningful on rank 0):
 *   all_data - concatenation of every rank's `local` vector, in rank order.
 *   counts   - all_data[displs[r] .. displs[r]+counts[r]) is rank r's slice.
 *   displs   - offset of rank r's slice into all_data.
 * Purpose:
 *   MPI_Gather requires equal-sized sends, but each rank has a different
 *   number of communication neighbors. This does the two-step
 *   size-then-data gather (MPI_Gather + MPI_Gatherv) needed for that.
 */
void gather_variable_ints(const std::vector<int>& local, MPI_Comm comm,
                           std::vector<int>& all_data,
                           std::vector<int>& counts,
                           std::vector<int>& displs) {
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int local_n = static_cast<int>(local.size());
    counts.resize(num_procs);
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    displs.assign(num_procs, 0);
    int total = 0;
    if (rank == 0) {
        for (int i = 0; i < num_procs; ++i) {
            displs[i] = total;
            total += counts[i];
        }
        all_data.resize(total);
    }

    MPI_Gatherv(local.data(), local_n, MPI_INT,
                rank == 0 ? all_data.data() : nullptr, counts.data(), displs.data(),
                MPI_INT, 0, comm);
}

/*
 * CommStats / compute_comm_stats
 * ---------------------------------
 * Per-rank summary of a ParMat's halo-exchange communication pattern:
 * how many neighbors it sends to / receives from, the total number of
 * doubles exchanged, and the min/max single-message size in each
 * direction. Derived purely from local send_comm/recv_comm state built
 * by form_comm(), no communication needed.
 */
struct CommStats
{
    int nrecipients;   // number of ranks this rank sends to (send_comm.n_msgs)
    int nsenders;      // number of ranks this rank receives from (recv_comm.n_msgs)
    int sendsize;      // total doubles sent, all neighbors (send_comm.size_msgs)
    int recvsize;      // total doubles received, all neighbors (recv_comm.size_msgs)
    int send_msg_min;
    int send_msg_max;
    int recv_msg_min;
    int recv_msg_max;
};

CommStats compute_comm_stats(const ParMat& A) {
    CommStats s;
    s.nrecipients = A.send_comm.n_msgs;
    s.nsenders    = A.recv_comm.n_msgs;
    s.sendsize    = A.send_comm.size_msgs;
    s.recvsize    = A.recv_comm.size_msgs;

    s.send_msg_min = s.nrecipients ? A.send_comm.counts[0] : 0;
    s.send_msg_max = s.nrecipients ? A.send_comm.counts[0] : 0;
    for (int c : A.send_comm.counts) {
        s.send_msg_min = std::min(s.send_msg_min, c);
        s.send_msg_max = std::max(s.send_msg_max, c);
    }

    s.recv_msg_min = s.nsenders ? A.recv_comm.counts[0] : 0;
    s.recv_msg_max = s.nsenders ? A.recv_comm.counts[0] : 0;
    for (int c : A.recv_comm.counts) {
        s.recv_msg_min = std::min(s.recv_msg_min, c);
        s.recv_msg_max = std::max(s.recv_msg_max, c);
    }

    return s;
}

/*
 * write_topology_json
 * ---------------------
 * Inputs:
 *   filename                 - path to write the JSON file to.
 *   hostnames/node_ids/local_ranks - rank-indexed node placement
 *     (see gather_hostnames / assign_node_ids).
 *   all_stats  - flattened rank-indexed CommStats, 8 ints per rank in the
 *     order declared in CommStats (as gathered via plain MPI_Gather).
 *   send_data/send_counts/send_displs - gathered (neighbor_rank, count)
 *     pairs per rank for send_comm, as produced by gather_variable_ints
 *     (counts here are in ints, i.e. 2x the number of neighbor entries).
 *   recv_data/recv_counts/recv_displs - same, for recv_comm.
 * Purpose:
 *   Serializes, per rank, which node it ran on and which ranks it
 *   exchanges halo data with (and how much), to a JSON file with no
 *   external JSON library dependency. Intended to be called by exactly
 *   rank 0.
 */
void write_topology_json(const std::string& filename,
                          const std::vector<std::string>& hostnames,
                          const std::vector<int>& node_ids,
                          const std::vector<int>& local_ranks,
                          const std::vector<int>& all_stats,
                          const std::vector<int>& send_data,
                          const std::vector<int>& send_counts,
                          const std::vector<int>& send_displs,
                          const std::vector<int>& recv_data,
                          const std::vector<int>& recv_counts,
                          const std::vector<int>& recv_displs) {
    int num_procs = static_cast<int>(hostnames.size());

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "save_topology: failed to open " << filename << " for writing\n";
        return;
    }

    // Emits a JSON array of {rank, count, bytes} for one rank's neighbor list.
    auto write_neighbors = [&out](const std::vector<int>& data, int count, int displ) {
        int n_entries = count / 2; // each entry is (neighbor_rank, n_doubles)
        out << "[";
        for (int i = 0; i < n_entries; ++i) {
            int neighbor = data[displ + 2 * i];
            int n_doubles = data[displ + 2 * i + 1];
            out << "{\"rank\": " << neighbor << ", \"count\": " << n_doubles
                << ", \"bytes\": " << (static_cast<long>(n_doubles) * sizeof(double)) << "}";
            if (i + 1 < n_entries) out << ", ";
        }
        out << "]";
    };

    out << "{\n";
    out << "  \"comm_size\": " << num_procs << ",\n";
    out << "  \"element_size_bytes\": " << sizeof(double) << ",\n";
    out << "  \"ranks\": [\n";

    for (int r = 0; r < num_procs; ++r) {
        const int* s          = &all_stats[static_cast<size_t>(r) * 8];
        int nrecipients       = s[0];
        int nsenders          = s[1];
        int sendsize          = s[2];
        int recvsize          = s[3];
        int send_msg_min      = s[4];
        int send_msg_max      = s[5];
        int recv_msg_min      = s[6];
        int recv_msg_max      = s[7];

        out << "    {\n";
        out << "      \"rank\": " << r << ",\n";
        out << "      \"hostname\": \"" << hostnames[r] << "\",\n";
        out << "      \"node_id\": " << node_ids[r] << ",\n";
        out << "      \"local_rank\": " << local_ranks[r] << ",\n";
        out << "      \"nrecipients\": " << nrecipients << ",\n";
        out << "      \"nsenders\": " << nsenders << ",\n";
        out << "      \"sendsize\": " << sendsize << ",\n";
        out << "      \"recvsize\": " << recvsize << ",\n";
        out << "      \"send_bytes\": " << (static_cast<long>(sendsize) * sizeof(double)) << ",\n";
        out << "      \"recv_bytes\": " << (static_cast<long>(recvsize) * sizeof(double)) << ",\n";
        out << "      \"send_msg_min\": " << send_msg_min << ",\n";
        out << "      \"send_msg_max\": " << send_msg_max << ",\n";
        out << "      \"recv_msg_min\": " << recv_msg_min << ",\n";
        out << "      \"recv_msg_max\": " << recv_msg_max << ",\n";
        out << "      \"neighbors_send\": ";
        write_neighbors(send_data, send_counts[r], send_displs[r]);
        out << ",\n";
        out << "      \"neighbors_recv\": ";
        write_neighbors(recv_data, recv_counts[r], recv_displs[r]);
        out << "\n";
        out << "    }" << (r + 1 < num_procs ? "," : "") << "\n";
    }

    out << "  ]\n";
    out << "}\n";
    out.close();
}

/*
 * save_topology
 * ---------------
 * Inputs:
 *   A    - distributed matrix whose send_comm/recv_comm hold this rank's
 *          halo-exchange neighbors (procs) and per-neighbor message sizes
 *          (counts), as built by form_comm().
 *   comm - MPI communicator matching A's distribution.
 * Purpose:
 *   Top-level entry point combining node placement (which physical node
 *   each rank ran on) with communication topology (which ranks it
 *   exchanges halo data with, and how much) into a single JSON file, so
 *   it can be joined against Caliper profiling output by MPI rank during
 *   analysis. The output path defaults to "topology.json" and can be
 *   overridden with the TOPOLOGY_FILE environment variable. Every rank
 *   must call this function, since the gathers are collective; only
 *   rank 0 performs the actual file write.
 */
void save_topology(ParMat& A, MPI_Comm comm) {
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Node placement: hostname -> node_id/local_rank, replicated on every rank.
    std::vector<std::string> hostnames = gather_hostnames(comm);
    std::vector<int> node_ids, local_ranks;
    assign_node_ids(hostnames, node_ids, local_ranks);

    // Per-rank communication summary stats: fixed size, so a plain MPI_Gather suffices.
    CommStats stats = compute_comm_stats(A);
    int local_stats[8] = {
        stats.nrecipients, stats.nsenders, stats.sendsize, stats.recvsize,
        stats.send_msg_min, stats.send_msg_max, stats.recv_msg_min, stats.recv_msg_max
    };
    std::vector<int> all_stats(static_cast<size_t>(num_procs) * 8);
    MPI_Gather(local_stats, 8, MPI_INT, all_stats.data(), 8, MPI_INT, 0, comm);

    // Per-rank neighbor lists: variable size, so Gather (sizes) + Gatherv (data).
    std::vector<int> local_send;
    for (int i = 0; i < A.send_comm.n_msgs; ++i) {
        local_send.push_back(A.send_comm.procs[i]);
        local_send.push_back(A.send_comm.counts[i]);
    }
    std::vector<int> local_recv;
    for (int i = 0; i < A.recv_comm.n_msgs; ++i) {
        local_recv.push_back(A.recv_comm.procs[i]);
        local_recv.push_back(A.recv_comm.counts[i]);
    }

    std::vector<int> send_data, send_counts, send_displs;
    std::vector<int> recv_data, recv_counts, recv_displs;
    gather_variable_ints(local_send, comm, send_data, send_counts, send_displs);
    gather_variable_ints(local_recv, comm, recv_data, recv_counts, recv_displs);

    if (rank == 0) {
        std::string filename = "topology.json";
        const char* env_filename = std::getenv("TOPOLOGY_FILE");
        if (env_filename) filename = env_filename;
        write_topology_json(filename, hostnames, node_ids, local_ranks, all_stats,
                             send_data, send_counts, send_displs,
                             recv_data, recv_counts, recv_displs);
    }

    MPI_Barrier(comm); // ensure the file exists on disk before any rank proceeds
}

#endif