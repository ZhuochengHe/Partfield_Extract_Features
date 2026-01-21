#!/bin/bash
#SBATCH --job-name=partfield_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000m
#SBATCH --time=00:20:00
#SBATCH --account=eecs442f25_class

#SBATCH --output=download_partslip.log 
source ~/.bashrc
conda activate partfield
hf download minghua/PartSLIP --repo-type dataset --include "data/*" --local-dir ./dataset