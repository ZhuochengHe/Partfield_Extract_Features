#!/bin/bash
# The interpreter used to execute the script
# "#SBATCH" directives that convey submission options:
#SBATCH--job-name=infere
#SBATCH--cpus-per-task=1
#SBATCH--nodes=1
#SBATCH--ntasks=1
#SBATCH--mem=16G
#SBATCH--time=00:05:00
#SBATCH--account=eecs442f25_class
#SBATCH--partition=gpu_mig40,gpu,spgpu
#SBATCH--gres=gpu:1
#SBATCH--output=infere.log
source ~/.bashrc
conda init
conda activate partfield
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name exp_results/partfield_features/objaverse dataset.data_path data/objaverse_samples