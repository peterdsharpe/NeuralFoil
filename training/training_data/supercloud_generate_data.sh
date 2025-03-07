#!/bin/bash

#pip install -r ../requirements.txt
# Just run with `sbatch supercloud_train_blind_neural_network.sh`

# Slurm sbatch options
#SBATCH -o log.log-%j --gres=gpu:volta:1 -c 20

# Loading the required module
module unload anaconda
module load anaconda/2023a

# Run the script
python -u ./generate_xfoil_data_multiprocessing.py