#!/bin/bash

#flux: --job-name=cg_test
#flux: --output=cg_test.out
#flux: --queue=pdebug
#flux: --nodes=2
#flux: --nslots=16
#flux: --time-limit=10m


# SETUP
module load craype-accel-amd-gfx942
source ../.venv/bin/activate

export MPICH_GPU_SUPPORT_ENABLED=1

#NNODES=$(flux resource list -no {nnodes})
NNODES=2
PROCS=$(( NNODES * 4 ))

# set up the output directory
export SUBMISSION_DIR=$(pwd)
export LUSTRE_OUTPUT_DIR=/p/lustre5/$(whoami)/cg_study
mkdir -p "$LUSTRE_OUTPUT_DIR"

cp -r $SUBMISSION_DIR/../petsc_matrices/ $LUSTRE_OUTPUT_DIR/..
cp $SUBMISSION_DIR/cg_wrapper.sh $LUSTRE_OUTPUT_DIR/
cp $SUBMISSION_DIR/*.py $LUSTRE_OUTPUT_DIR/
cp $SUBMISSION_DIR/params.csv $LUSTRE_OUTPUT_DIR/

cd "$LUSTRE_OUTPUT_DIR"

echo "Entered directory: $(pwd)"
echo "Submission directory: $SUBMISSION_DIR"
echo "Python virtual environment: $(which python3)"
echo "Batch Resources:"
echo "  Nodes: $NNODES"
echo "  Slots: $PROCS"

python3 generate_cmds.py --param_file=params.csv

# copy the generated commands to the submission directory
# for later reference
cp cg_commands.sh $SUBMISSION_DIR/

bash cg_commands.sh
    
# wait for all jobs to complete
flux job wait --all

python data.py --dir=. --db=$HOME/data/cg_results.duckdb
# clean up the output directory
rm *.cali *.json
