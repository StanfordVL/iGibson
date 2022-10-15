#!/bin/bash
# This job should be run on the SC headnode.
# Usage: sbatch --export=IG_IGIBSON_PATH=path,IG_OUTPUT_PATH=path,IG_ENTRYPOINT_MODULE=path run.sh
#SBATCH --partition=svl --qos=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-gpu=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=8
#SBATCH --job-name="vision-dataset-generation"
#SBATCH --output=logs/%x_%A_%2t.out
#SBATCH --error=logs/%x_%A_%2t.err
#SBATCH --gpu-bind=single:2

######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_LOCALID="$SLURM_LOCALID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Execute the tasks in parallel.
srun task.sh