#!/usr/bin/env bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load u18/cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh  # Ensure Conda is available
conda activate my_env

python 1_long.py 