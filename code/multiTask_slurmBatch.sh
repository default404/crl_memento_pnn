#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=CRL_atari
#SBATCH --output=multiTaskRL%j.%N.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=batch

# Activate everything you need
module load cuda/10.1
pyenv activate ray086
# Run your python code
python main_multiTask.py --debug-ray False --output "/misc/no_backup/d1346/tune_out/multi_task/PNN_pong_soup_alpha_lr" --stop-iters 1000 --stop-timesteps 6000000