#!/bin/bash
#
#SBATCH --job-name=ss_baseline
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time 240:00:00
#SBATCH --output=slurm/slurm_%A.out
#SBATCH --error=slurm/slurm_%A.err

#SBATCH --mail-user=wangzz@stanford.edu
#SBATCH --mail-type=ALL


######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# python pretraining/audiogoal_trainer.py --run-type train --model-dir data/pretraining_1202 --predict-label
python run.py --exp-config config/savi_step1.yaml #--model-dir data/step1_1231



