#!/bin/bash
#SBATCH --job-name=partfield_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --time=00:05:00
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu_mig40,spgpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --output=test.log

source ~/.bashrc
conda activate partfield

python partfield_inference.py \
    -c configs/final/demo.yaml \
    --opts continue_ckpt model/model_objaverse.ckpt \
    result_name test \
    dataset.data_path test \
	is_pc True