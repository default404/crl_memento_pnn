#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=CRL_plotter
#SBATCH --output=CRL_plotter.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --gpus=0
#SBATCH --qos=batch

# Activate everything you need
module load cuda/10.1
pyenv activate ray086
# Run your python code
python common/exp_analysis.py -i "" -o ""