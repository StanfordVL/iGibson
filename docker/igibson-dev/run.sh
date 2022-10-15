#!/bin/bash
# This job should be run on the SC headnode.
# Usage: sbatch run.sh --export=IG_IGIBSON_PATH=path,IG_OUTPUT_PATH=path,IG_ENTRYPOINT_MODULE=path
#SBATCH --partition=svl --qos=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=2080ti:1
#SBATCH --job-name="vision-dataset-generation"
#SBATCH --output=logs/%x_%A_%4J_%2t.out
#SBATCH --error=logs/%x_%A_%4J_%2t.err

######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_LOCALID="$SLURM_LOCALID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Then, create a container.
enroot create -n igibson /cvgl/group/igibson-docker/igibson-dev.sqsh && {
  # Run the container, mounting iGibson at the right spot
  enroot start -r -w -m ${IG_IGIBSON_PATH}:/igibson -m ${IG_OUTPUT_PATH}:/out -e SLURM_LOCALID=${SLURM_LOCALID} igibson python -m ${IG_ENTRYPOINT_MODULE};
  # Remove the container.
  enroot remove -f igibson;
}