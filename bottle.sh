#!/bin/bash
#SBATCH --job-name=partfield_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --time=00:10:00
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=test_batch.log

source ~/.bashrc
conda activate partfield

SCRATCH_BASE="/scratch/eecs442f25_class_root/eecs442f25_class/jonzhe"
OUTPUT_BASE="$SCRATCH_BASE/output"

DATA_DIR="dataset/data/test/Bottle"
TARGETS=("3380" "3398" "3558" "3854" "3990")

total_infer_time=0
count=0

for SUBFOLDER in "$DATA_DIR"/*; do
    if [ -d "$SUBFOLDER" ]; then
        SUBNAME=$(basename "$SUBFOLDER")
        OUT_DIR="$OUTPUT_BASE/$SUBNAME"
        mkdir -p "$OUT_DIR"

        echo "📌 Renaming .ply files in $SUBNAME ..."
        for FILE in "$SUBFOLDER"/*.ply; do
            BASENAME=$(basename "$FILE")
            NEWNAME="${SUBNAME}_pc.ply"
            mv "$FILE" "$SUBFOLDER/$NEWNAME"
        done

        echo "📌 Running PartField inference for $SUBNAME ..."

        # ------------------------------
        # ONLY measure inference time
        # ------------------------------
        start=$(date +%s)

        python partfield_inference.py \
            -c configs/final/demo.yaml \
            --opts continue_ckpt model/model_objaverse.ckpt \
            result_name "$OUT_DIR" \
            dataset.data_path "$SUBFOLDER" \
            is_pc True

        end=$(date +%s)
        infer_time=$((end - start))
        total_infer_time=$((total_infer_time + infer_time))
        count=$((count + 1))

        echo "⏱️ Inference time for $SUBNAME: ${infer_time}s"
    fi
done

avg_time=$((total_infer_time / count))

echo "🎉 All done!"
echo "🔥 Total inference time: ${total_infer_time}s"
echo "📊 Average per folder: ${avg_time}s"
