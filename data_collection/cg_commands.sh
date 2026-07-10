#!/bin/bash
set -e
flux submit --flags=waitable -x --output=halo_stats_NODES-1_PROCS-2_RDZV-128_MATCH-software_IPC-1_ASYNC-1_ITERATIONS-100_MAT-Flan_1565.log -n 2 -N 1 cg_wrapper.sh 128 software 1 1 100 Flan_1565
flux submit --flags=waitable -x --output=halo_stats_NODES-1_PROCS-4_RDZV-2048_MATCH-hybrid_IPC-0_ASYNC-0_ITERATIONS-200_MAT-Serena.log -n 4 -N 1 cg_wrapper.sh 2048 hybrid 0 0 200 Serena
