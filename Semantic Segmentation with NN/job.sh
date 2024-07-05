#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=02:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
python main.py