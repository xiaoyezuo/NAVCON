#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCh --job-name=rxr_extract
#SBATCH --output=task.out

python frame_extraction.py
