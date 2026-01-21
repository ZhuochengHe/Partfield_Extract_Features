#!/bin/bash
#SBATCH --job-name=partfield_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --time=00:30:00
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --gres=gpu:1
#SBATCH --output=test_batch.log

# -------------------------
# 环境
# -------------------------
source ~/.bashrc
conda activate partfield

# -------------------------
# 路径
# -------------------------
SCRATCH_BASE="/scratch/eecs442f25_class_root/eecs442f25_class/jonzhe"
OUTPUT_BASE="$SCRATCH_BASE/output"

DATA_DIR="dataset/data/test/Bottle"

# -------------------------
# 遍历每个子文件夹
# -------------------------
for SUBFOLDER in "$DATA_DIR"/*; do
    if [ -d "$SUBFOLDER" ]; then
        # 获取文件夹名
        SUBNAME=$(basename "$SUBFOLDER")
        
        # 输出文件夹
        OUT_DIR="$OUTPUT_BASE/$SUBNAME"
        mkdir -p "$OUT_DIR"

        echo "📌 Renaming .ply files in $SUBNAME ..."
        for FILE in "$SUBFOLDER"/*.ply; do
            BASENAME=$(basename "$FILE")
            NEWNAME="${SUBNAME}_pc.ply"
            mv "$FILE" "$SUBFOLDER/$NEWNAME"
        done

        echo "📌 Running PartField inference for $SUBNAME ..."
        python partfield_inference.py \
            -c configs/final/demo.yaml \
            --opts continue_ckpt model/model_objaverse.ckpt \
            result_name "$OUT_DIR" \
            dataset.data_path "$SUBFOLDER" \
            is_pc True
    fi
done

echo "🎉 All done!"
