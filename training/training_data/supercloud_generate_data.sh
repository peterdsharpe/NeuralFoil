#!/bin/bash

# To set up the environment: run `uv sync` from the repo root (training deps are in the dev dependency group)
# Just run with `sbatch supercloud_generate_data.sh`

# Slurm sbatch options
#SBATCH -o log.log-%j --gres=gpu:volta:1 -c 20

# Loading the required module
module unload anaconda
module load anaconda/2023a

# Run the script
python -u ./generate_xfoil_data_multiprocessing.py