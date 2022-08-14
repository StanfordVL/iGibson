#!/bin/bash
#
#SBATCH --job-name=audiogoal_occl1
#SBATCH --partition=svl --qos=normal --nodelist=svl15
#SBATCH --nodes=1
#SBATCH --mem=95G
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=40
#SBATCH --time 192:00:00
#SBATCH --output=/viscam/u/li2053/logs/sonic_slurm_%A.out
#SBATCH --error=/viscam/u/li2053/logs/sonic_slurm_%A.err
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

##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
source /sailhome/li2053/.bashrc
conda activate igibson
echo "Virtual Env Activated"

##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
# export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/lib/x86_64-linux-gnu 
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd /viscam/u/li2053/iGibson-dev/igibson/agents/av_nav/
###################
# Run your script #
###################
echo "running command : <RUN_COMMAND>"

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MASTER_PORT=8888

# python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'

set -x
# FREEPORT = $(python find_free_port.py 2>&1)
python -u -m torch.distributed.launch --nproc_per_node 6 run.py --exp-config config/audiopointgoal_single.yaml --free_port $(python find_free_port.py 2>&1)