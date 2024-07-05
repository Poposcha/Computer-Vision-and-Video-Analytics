#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
python main.py