#!/bin/bash
#
#SBATCH --job-name=ss_baseline
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time 192:00:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err

#SBATCH --mail-user=li2053@stanford.edu
#SBATCH --mail-type=ALL


######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

python run.py --exp-config config/audiogoal_continuous.yaml
