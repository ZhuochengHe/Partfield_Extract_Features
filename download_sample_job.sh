#!/bin/bash
# The interpreter used to execute the script
# "#SBATCH" directives that convey submission options:
#SBATCH--job-name=download_data
#SBATCH--cpus-per-task=1
#SBATCH--nodes=1
#SBATCH--ntasks=1
#SBATCH--mem=16G
#SBATCH--time=00:10:00
#SBATCH--account=eecs442f25_class
#SBATCH--partition=gpu_mig40,gpu,spgpu
#SBATCH--gres=gpu:1
#SBATCH--output=download_data.log
source ~/.bashrc
conda init
conda activate partfield
python download.py --batch 1 --data_dir /scratch/eecs442f25_class_root/eecs442f25_class/jonzhe/data