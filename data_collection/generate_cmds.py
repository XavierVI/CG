import argparse
import math
import subprocess
import itertools
from pathlib import Path
# from scipy.stats import qmc

import polars as pl

# matrices_paths = Path("../mat_binary_files")
matrices_paths = Path("../petsc_matrices")
matrices = [
    # matrices_paths / "494_bus.mtx",
    # matrices_paths / "mesh_deform.mtx",
    # matrices_paths / "coPapersDBLP.mtx",
    # matrices_paths / "Flan_1565/Flan_1565.mtx"
    # matrices_paths / "Stranke94/Stranke94.mtx"

    matrices_paths / "Flan_1565",
    matrices_paths / "audikw_1",
    matrices_paths / "Bump_2911",
    matrices_paths / "Cube_Coup_dt6",
    matrices_paths / "Queen_4147",
    matrices_paths / "Serena",
]


def get_mpi_env_params():
    """
    This function is used to get the levels for the MPI runtime parameters we are investigating.

    We are moving to active learning, so this may no longer be used.
    """
    # default is 2048
    # https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us&page=user/libfabric_runtime_configurable_parameters.html&docLocale=en_US#fi_cxi_rdzv_threshold-5
    # but this source also says the default is 16,384
    # https://cpe.ext.hpe.com/docs/24.03/mpt/mpich/intro_mpi.html#libfabric-environment-variables-for-hpe-slingshot-nic-slingshot-11
    # cxi_thresholds = [0, 65_536]
    cxi_thresholds = [2048, 8192, 16_384]
    cxi_rx_match_modes = ["software", "hybrid", "hardware"]
    mpich_gpu_ipc_enabled = [0, 1]
    mpich_async_progresses = [0, 1]

    return cxi_thresholds, cxi_rx_match_modes, mpich_gpu_ipc_enabled, mpich_async_progresses

def fetch_number_of_nodes(node_list):
    # Count the number of nodes in each part of the node list
    node_counts = {
        node: 0 for node in node_list.split(';')
    }
    for node in node_list.split(';'):
        if '-' in node:
            prefix = node.split('[')[0]
            range_part = node.split('[')[1].split(']')[0]
            start, end = map(int, range_part.split('-'))
            
            node_counts[node] += (end - start + 1)
        
        elif ',' in node:
            prefix = node.split('[')[0]
            list_part = node.split('[')[1].split(']')[0]
            nodes = list_part.split(',')
            
            for n in nodes:
                node_counts[node] += 1
        else:
            node_counts[node] += 1

    return node_counts



def write_commands_to_file(commands, filename):
    print(f"Generating {len(commands)} commands for CG solver...")
    
    with open(filename, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('set -e\n')

        for i, cmd in enumerate(commands):
            f.write(cmd + '\n')

            if (i + 1) % 50 == 0:
                # wait every 50 jobs
                f.write("\n")
                f.write("flux job wait --all\n")
                f.write("python data.py --dir=. --db=$HOME/data/cg_results.duckdb\n")
                # clean up the output directory
                f.write("rm *.cali *.json\n")
                f.write("\n")

        # f.write("flux watch --all\n")
        # f.write("flux job wait --all\n")


def generate_cmds(args):
    iterations = [i for i in range(args.min_iterations, args.max_iterations + 1, 100)]
    cxi_thresholds, cxi_rx_match_modes, mpich_gpu_ipc_enabled, mpich_async_progresses = get_mpi_env_params()

    # full factorial analysis
    params = itertools.product(
        cxi_thresholds,
        cxi_rx_match_modes,
        mpich_gpu_ipc_enabled,
        mpich_async_progresses,
        iterations,
        matrices
    )

    # LHS sampling
    # param_levels = [
    #     cxi_thresholds,
    #     cxi_rx_match_modes,
    #     mpich_gpu_ipc_enabled,
    #     mpich_async_progresses,
    #     iterations,
    #     matrices
    # ]

    # samples = qmc.LatinHypercube(d=len(param_levels)).random(n=args.N)
    # params = [tuple(p[int(s * len(p))] for s, p in zip(row, param_levels)) for row in samples]

    # NOTE: for now, we will just keep it at 4 tasks per node
    # to use as many GPUs as possible
    cmd_base = f"flux submit --flags=waitable -n {args.num_procs} -N {args.num_nodes} -x"
    script = "cg_wrapper.sh"

    commands = []
    

    for p in params:
        cxi_threshold, cxi_rx_match_mode, mpich_gpu_ipc_enabled, mpich_async_progress, iterations, matrix = p
        tasks_per_node = max(args.num_procs // args.num_nodes, 1)

        base = f"halo_stats_NODES-{args.num_nodes}_PROCS-{args.num_procs}_RDZV-{cxi_threshold}_MATCH-{cxi_rx_match_mode}_IPC-{mpich_gpu_ipc_enabled}_ASYNC-{mpich_async_progress}_ITERATIONS-{iterations}_MAT-{matrix.name}"
        json_file = f"{base}.json"
        log_file = f"{base}.log"

        cmd = (f"{cmd_base} --output={log_file} --error={log_file} {script} "
            f"{cxi_threshold} {cxi_rx_match_mode} {mpich_gpu_ipc_enabled} {mpich_async_progress} "
            f"{iterations} {matrix} {json_file}"
        )

        commands.append(cmd)

    write_commands_to_file(commands, "cg_commands.sh")


def generate_custom_cmds(args):
    if args.param_file is None:
        raise ValueError("Error: --param_file must be specified for custom command generation.")

    commands = []
    cmd_base = f"flux submit --flags=waitable -x"
    run_script = "cg_wrapper.sh"
    # read parameters CSV using polars
    cols = [
        'procs', 'nodes', 'rdzv', 'match_mode', 'gpu_ipc', 'async', 'max_iters', 'matrix'
    ]
    params = pl.read_csv(args.param_file)
    # use select to ensure ordering
    params = params.select(cols)

    for row in params.iter_rows():
        procs = row[0]
        nodes = row[1]
        rdzv = row[2]
        match_mode = row[3]
        gpu_ipc = row[4]
        async_progress = row[5]
        max_iters = row[6]
        matrix = row[7]

        # matrix_path = matrices_paths / matrix
        # append .petsc to the end
        # matrix_path = matrix_path.with_suffix(".petsc")

        tasks_per_node = max(procs // nodes, 1)
        log_file = (f"halo_stats_NODES-{nodes}_PROCS-{procs}_RDZV-{rdzv}_MATCH-{match_mode}_IPC-{gpu_ipc}_ASYNC-{async_progress}_ITERATIONS-{max_iters}_MAT-{matrix}.log")
        
        cmd = (f"{cmd_base} --output={log_file} -n {procs} -N {nodes} {run_script} "
            f"{rdzv} {match_mode} {gpu_ipc} {async_progress} "
            f"{max_iters} {matrix}"
        )

        commands.append(cmd)

    write_commands_to_file(commands, "cg_commands.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate flux commands for CG solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-nodes", type=int, default=1,
        help="Number of nodes to use for CG solver.",
    )
    parser.add_argument(
        "--num-procs", type=int, default=4,
        help="Number of processes to use for CG solver.",
    )
    parser.add_argument(
        "--min-iterations", type=int, default=100,
        help="Minimum number of iterations for CG solver.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=200,
        help="Maximum number of iterations for CG solver.",
    )
    parser.add_argument(
        '--param_file', type=str, default=None,
        help='Path to a file containing parameter combinations to run (optional).',
    )

    args = parser.parse_args()

    if args.param_file:
        generate_custom_cmds(args)
    else:
        generate_cmds(args)
