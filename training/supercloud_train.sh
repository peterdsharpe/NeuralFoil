#!/bin/bash

# To set up the environment: run `uv sync` from the repo root (training deps are in the dev dependency group)
# Just run with `sbatch supercloud_train.sh`

# Slurm sbatch options
#SBATCH -o log.log-%j --gres=gpu:volta:1 -c 20

# Loading the required module
module unload anaconda
module load cuda/11.8
module load anaconda/2023a-pytorch

# Run the script
python -u ./train.py