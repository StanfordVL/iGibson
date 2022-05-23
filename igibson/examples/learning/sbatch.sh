#!/bin/bash
#SBATCH -J Dense_Skill_RL
#SBATCH -N 1
#SBATCH --mem 108G
#SBATCH --gres=gpu:1
#SBATCH -p svl-interactive
srun python stable_baselines3_ppo_skill_example.py
wait
