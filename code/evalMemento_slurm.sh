#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=mementoSB
#SBATCH --output=mementoSB%j.%N.log
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
python memento_experiment/create_state_buffer.py --experiment_dir "" --checkpoint "" --env_spec "" --output "" 
