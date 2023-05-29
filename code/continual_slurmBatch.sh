#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=continualRL
#SBATCH --output=continualRL%j.%N.log
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
python main_trainable.py --debug-ray False --output "~/no_backup/d1346/tune_out/multi_task/PNN_pong_soup" --stop-iters 1000 --stop-reward 21