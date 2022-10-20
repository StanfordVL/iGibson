#!/bin/bash
# This job should be run on the SC headnode.
# Usage: sbatch --export=IG_DATA_PATH=path,IG_OUTPUT_PATH=path,IG_ENTRYPOINT_COMMAND=path run.sh
#SBATCH --partition=svl,viscam --qos=normal
#SBATCH --array=0-14
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name="vision-dataset-generation"
#SBATCH --output=logs/%x_%A_%a_%2t.out
#SBATCH --error=logs/%x_%A_%a_%2t.err

######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

echo "IG_DATA_PATH="$IG_DATA_PATH
echo "IG_OUTPUT_PATH="$IG_OUTPUT_PATH
echo "IG_ENTRYPOINT_COMMAND="$IG_ENTRYPOINT_COMMAND

# Execute the tasks in parallel.
srun task.sh