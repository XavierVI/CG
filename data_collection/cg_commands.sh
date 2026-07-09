#!/bin/bash
set -e
flux submit --flags=waitable -x -n 8 -N 2 cg_wrapper.sh 128 software 1 1 100 Flan_1565
flux submit --flags=waitable -x -n 16 -N 4 cg_wrapper.sh 2048 hybrid 0 0 200 Serena
