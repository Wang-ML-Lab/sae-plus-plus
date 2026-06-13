#!/usr/bin/env bash
# End-to-end worked example: CSAE on Qwen3-VL-4B-Instruct x ImageNet.
# Mirrors configs/qwen_imagenet.yaml. Run from the repo root.
#
# Prereqs:
#   pip install -r requirements.txt
#   IMAGENET_DIR = a folder of class subdirectories of images (e.g. ImageNet val).
set -euo pipefail

IMAGENET_DIR="${IMAGENET_DIR:?set IMAGENET_DIR to an ImageNet-style image folder}"
DEVICE="${DEVICE:-cuda:0}"
DATA_DIR="./data"
RUN_DIR="./runs/imagenet"
LAYER="model.visual.blocks.23"
mkdir -p "$DATA_DIR" "$RUN_DIR"

ACT_H5="$DATA_DIR/imagenet_qwen_block23.h5"
DINO_H5="$DATA_DIR/imagenet_dinov3.h5"

# 1) Extract Qwen3-VL vision activations (HDF5 group "X" / layer dataset).
python data_gen/extract_activations.py \
  --image-dir "$IMAGENET_DIR" \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --layers 23 --device "$DEVICE" \
  --out "$ACT_H5"

# 2) DINOv3 reference embeddings for the HMS metric (same image ordering).
python data_gen/gen_dino_reference.py \
  --image-dir "$IMAGENET_DIR" \
  --device "$DEVICE" \
  --out "$DINO_H5"

# 3) Train CSAE.
python train_csae.py \
  --save_dir "$RUN_DIR" \
  --model_name Qwen --dataset_name ImageNet \
  --data_path "$ACT_H5" --layer_name "$LAYER" \
  --device "$DEVICE" \
  --dict_size 20000 --dict2_rule half --k1 6 --k2 1 \
  --lr 1e-4 --seed 0 --num_tokens 500000000 --sae_batch_size 1024 \
  --warmup_steps 500 --sae2_start_step 10000

# 4) Evaluate HMS (point at the produced ae.pt checkpoint).
CKPT=$(find "$RUN_DIR" -name ae.pt | head -n1)
python eval_hms.py \
  --ckpt-path "$CKPT" \
  --data-path "$ACT_H5" --layer-name "$LAYER" \
  --embedding-path "$DINO_H5" \
  --device "$DEVICE"

# 5) (Optional) CSAE concept steering — run separately, e.g.:
#      python steering/run_demo.py --ckpt-path "$CKPT" \
#        --data-path "$ACT_H5" --image-dir "$IMAGENET_DIR" --device "$DEVICE"
echo "Done: extract -> train -> eval. For steering, see steering/run_demo.py + README."
