#!/bin/bash
#SBATCH --job-name=partfield_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000m
#SBATCH --time=00:45:00
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --gres=gpu:1

#sbatch --output=partfield_pipeline_${BATCH_ID}.log pipeline_infere.sh BATCH_ID
# ======================
# 环境
# ======================
source ~/.bashrc
conda activate partfield

# ======================
# 参数
# ======================
BATCH_ID=$1
SCRATCH_BASE="/scratch/eecs442f25_class_root/eecs442f25_class/jonzhe"

DATA_BASE="$SCRATCH_BASE/data"
GLB_DIR="$DATA_BASE/objaverse_glbs/batch_${BATCH_ID}"

OUTPUT_BASE="$SCRATCH_BASE/output"
FEAT_DIR="$OUTPUT_BASE/partfield_features/batch_${BATCH_ID}"

mkdir -p "$GLB_DIR" "$FEAT_DIR"

echo "Directories:"
echo "  GLB_DIR     = $GLB_DIR"
echo "  FEAT_DIR    = $FEAT_DIR"

# ======================
# 下载 GLB
# ======================
python download.py \
    --batch "$BATCH_ID" \
    --batch_size 200 \
    --data_dir "$DATA_BASE"

# ======================
# PartField 推理
# ======================
echo "📌 Running PartField inference..."
python partfield_inference.py \
    -c configs/final/demo.yaml \
    --opts continue_ckpt model/model_objaverse.ckpt \
    result_name "$FEAT_DIR" \
    dataset.data_path "$GLB_DIR"

cd "$FEAT_DIR"
rm -f *.ply

# ======================
# 压缩结果
# ======================
echo "📦 Compressing results in $FEAT_DIR ..."

TAR_NAME="partfield_batch_${BATCH_ID}.tar.gz"
CHECKSUM_NAME="${TAR_NAME}.sha256"

# 打包
tar -czvf "$TAR_NAME" ./*.npy

# 生成校验文件（强烈推荐）
sha256sum "$TAR_NAME" > "$CHECKSUM_NAME"

echo "📦 Created archive: $TAR_NAME"
echo "🔐 SHA256 saved  : $CHECKSUM_NAME"

# 如需删除原始文件，可打开下面两行
echo "🧹 Cleaning original .npy files..."
rm -f ./*.npy
rm -rf "$DATA_BASE/objaverse_glbs/batch_${BATCH_ID}"

echo "🎉 All done!"
