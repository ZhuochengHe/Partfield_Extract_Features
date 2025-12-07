#!/bin/bash

OUTPUT_BASE="D:/Umich/EECS 442/project/Partfield_Extract_Features/output"
DATA_DIR="D:\Umich\EECS 442\project\Partfield_Extract_Features\test\Bottle"

# 只处理这些子目录
TARGETS=("3380" "3398" "3558" "3854" "3990")

start_total=$(date +%s)

for SUBNAME in "${TARGETS[@]}"; do
    SUBFOLDER="$DATA_DIR/$SUBNAME"

    if [ -d "$SUBFOLDER" ]; then
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
    else
        echo "⚠️ Warning: folder $SUBNAME does not exist, skipping..."
    fi
done

end_total=$(date +%s)
total_time=$((end_total - start_total))
avg_time=$((total_time / 5))

echo "🎉 All done!"
echo "⏱️ Total runtime: ${total_time}s"
echo "📊 Average per folder: ${avg_time}s"
