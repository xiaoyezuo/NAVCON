#!/bin/bash

#SBATCH --job-name=rxr_extract
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --partition kostas-compute
#SBATCH --qos kd-med
#SBATCH --time=24:00:00
#SBATCH --output=frame_extraction_%a.out
#SBATCH --error=frame_extraction_%a.out
#SBATCH --array=0-9

conda activate habitat
python frame_extract.py --job_idx "$SLURM_ARRAY_TASK_ID" --num_job_splits 10
