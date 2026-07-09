#!/bin/bash

#flux: --job-name=cg_test
#flux: --output=cg_test.out
#flux: --queue=pdebug
#flux: --nodes=8
#flux: --nslots=32
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
export LUSTRE_OUTPUT_DIR=/p/lustre5/$(whoami)/acg_study
mkdir -p "$LUSTRE_OUTPUT_DIR"

cp -r $SUBMISSION_DIR/../mat_binary_files/ $LUSTRE_OUTPUT_DIR/..
cp $SUBMISSION_DIR/aCG_wrapper.sh $LUSTRE_OUTPUT_DIR/
cp $SUBMISSION_DIR/*.py $LUSTRE_OUTPUT_DIR/

cd "$LUSTRE_OUTPUT_DIR"

echo "Entered directory: $(pwd)"
echo "Submission directory: $SUBMISSION_DIR"
echo "Performing aCG parameter sweep"
echo "  Nodes: $NNODES"
echo "  Slots: $PROCS"

python3 generate_cmds.py --num-nodes $NNODES --num-procs $PROCS --min-iterations $MIN_ITERATIONS --max-iterations $MAX_ITERATIONS

# copy the generated commands to the submission directory
# for later reference
cp acg_commands.sh $SUBMISSION_DIR/

echo "Replication $i/$REPLICATIONS"
bash acg_commands.sh
    
# wait for all jobs to complete
flux job wait --all

python data.py --logs=./ --db=$HOME/data/acg_results.duckdb --table=acg_runs --append
# clean up the output directory
rm *.cali *.json
