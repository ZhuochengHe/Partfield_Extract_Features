#!/bin/bash
#SBATCH --job-name=partfield_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000m
#SBATCH --time=00:05:00
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu_mig40,spgpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --output=partfield_render.log

source ~/.bashrc
conda activate partfield

BATCH_ID=$1
SCRATCH_BASE="/scratch/eecs442f25_class_root/eecs442f25_class/jonzhe"

DATA_BASE="$SCRATCH_BASE/data"
GLB_DIR="$DATA_BASE/objaverse_glbs/batch_${BATCH_ID}"

OUTPUT_BASE="$SCRATCH_BASE/output"
FEAT_DIR="$OUTPUT_BASE/partfield_features/batch_${BATCH_ID}"
IMG_DIR="$OUTPUT_BASE/2d_images/batch_${BATCH_ID}"


echo "Directories:"
echo "  GLB_DIR     = $GLB_DIR"
echo "  FEAT_DIR    = $FEAT_DIR"
echo "  IMG_DIR     = $IMG_DIR"

echo "🎨 Rendering 2D projections..."


python mesh_to_2d.py \
    --input-dir "$GLB_DIR" \
    --output-dir "$IMG_DIR" \
    --num-views 6 \
    --resolution 512 512

echo "📦 Compressing PNG files..."
cd "$IMG_DIR"

tar -czf "images_batch_${BATCH_ID}.tar.gz" *.png

echo "🧹 Removing original PNG files..."
rm -f *.png

echo "✨ Done! Compressed images at: $IMG_DIR/images_batch_${BATCH_ID}.tar.gz"
