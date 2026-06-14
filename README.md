# CSAE: Cascaded Sparse Autoencoders for Multi-Level Visual Concepts in MLLMs

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![🤗 Checkpoint](https://img.shields.io/badge/🤗%20Hugging%20Face-Checkpoint-blue)](https://huggingface.co/YusongZhao666/csae-ckpt)

This is the official implementation of the paper:

**SAE++: Learning Multi-Level Visual Concepts from Multimodal LLMs with Cascaded Sparse Autoencoders**

Yusong Zhao, Hengyi Wang, Tanuja Ganu, Akshay Nambi, Hao Wang

## Overview

**CSAE** (Cascaded Sparse Autoencoder) discovers *multi-level* visual concepts in
multimodal LLMs. A Level-1 SAE decomposes the MLLM's vision activations into
atomic concepts — its decoder columns are the concept directions — and a Level-2
SAE is trained **on the Level-1 decoder atoms themselves**, learning higher-order
"concepts of concepts."

- **Hierarchical** — Level-2 units group semantically coherent Level-1 atoms.
- **End-to-end** — both levels are trained jointly.
- **Steerable** — clamping a Level-2 unit causally inserts or suppresses its concept in the MLLM.

Pipeline: get images → (1) generate activations → (2) train CSAE →
(3) evaluate HMS → (4) steer concepts. A CUDA GPU is required.

## Installation

```bash
pip install -r requirements.txt      # or: conda env create -f environment.yml
```

`dictionary_learning.training.trainSAE` is a pip dependency (not vendored). MLLM
backbones (Qwen3-VL) and the DINOv3 encoder download from the Hugging Face Hub on
first use under their own licenses (see `NOTICE`).

## Datasets

Obtain any of the datasets from the paper:

| Dataset | Source |
|---------|--------|
| ImageNet-1k | https://www.image-net.org/download.php |
| iNaturalist 2021 | https://github.com/visipedia/inat_comp/tree/master/2021 |
| MS-COCO | https://cocodataset.org/#download |
| Color | https://github.com/Wang-ML-Lab/interpretable-foundation-models |

Only obtain the **images**; the `.h5` files under `./data/` are produced by the
data-generation scripts in Step 1. Arrange the images as **one subdirectory per class**:

```
<image-dir>/
  class_a/  img001.jpg  img002.jpg  ...
  class_b/  img001.jpg  ...
```

Set the splits and create the output dirs:

```bash
export TRAIN_IMAGES=/path/to/imagenet/train_sample   # images to train the SAE on
export VAL_IMAGES=/path/to/imagenet/val              # images for HMS evaluation
export DEVICE=cuda:0
mkdir -p data runs results
```

## Step 1. Generate activations & embeddings

Extract MLLM vision activations (group `X`, one dataset per layer, plus
`token_offsets`) for the **training** images, and — for evaluation — activations
plus DINOv3 reference embeddings for the **validation** images.

```bash
# training-split activations (used by step 2)
python data_gen/extract_activations.py \
  --image-dir "$TRAIN_IMAGES" --model-path Qwen/Qwen3-VL-4B-Instruct \
  --layers 23 --out ./data/imagenet_train_acts.h5 --device "$DEVICE"

# validation-split activations + DINOv3 embeddings (used by steps 3-4)
python data_gen/extract_activations.py \
  --image-dir "$VAL_IMAGES" --model-path Qwen/Qwen3-VL-4B-Instruct \
  --layers 23 --out ./data/imagenet_val_acts.h5 --device "$DEVICE"

python data_gen/gen_dino_reference.py \
  --image-dir "$VAL_IMAGES" --out ./data/imagenet_val_dino.h5 --device "$DEVICE"
```

## Step 2. Train CSAE

```bash
python train_csae.py \
  --save_dir ./runs/imagenet \
  --model_name Qwen --dataset_name ImageNet \
  --data_path ./data/imagenet_train_acts.h5 \
  --layer_name model.visual.blocks.23 --device "$DEVICE" \
  --dict_size 20000 --k1 6 --k2 1 \
  --lr 1e-4 --seed 0 \
  --num_tokens 500000000 --sae_batch_size 1024 \
  --warmup_steps 500 --sae2_start_step 10000
```

The checkpoint lands at `./runs/imagenet/<submodule>/trainer_0/ae.pt`.
`--sae2_start_step` lets Level-1 stabilize before Level-2 starts; see
`configs/qwen_imagenet.yaml`.

## Step 3. Evaluate HMS

```bash
python eval_hms.py \
  --ckpt-path ./runs/imagenet/*/trainer_0/ae.pt \
  --data-path ./data/imagenet_val_acts.h5 \
  --embedding-path ./data/imagenet_val_dino.h5 \
  --layer-name model.visual.blocks.23 --device "$DEVICE"
```

## Step 4. Steering

A **pretrained checkpoint is available** at
<https://huggingface.co/YusongZhao666/csae-ckpt/tree/main>. Download `ae.pt`
(with its `config.json`) and point `CKPT` at it:

```bash
export CKPT=/path/to/ae.pt
```

**List the alive Level-2 units** (id, #atoms, top concept) and pick one:

```bash
python steering/run_demo.py --ckpt-path "$CKPT" \
  --data-path ./data/imagenet_val_acts.h5 --image-dir "$VAL_IMAGES" --device "$DEVICE"
```

**Bald eagle — unit #7452:**

```bash
python steering/run_demo.py --ckpt-path "$CKPT" \
  --data-path ./data/imagenet_val_acts.h5 --image-dir "$VAL_IMAGES" \
  --device "$DEVICE" --cluster 7452
```

**Schooner — unit #7806:**

```bash
python steering/run_demo.py --ckpt-path "$CKPT" \
  --data-path ./data/imagenet_val_acts.h5 --image-dir "$VAL_IMAGES" \
  --device "$DEVICE" --cluster 7806
```

## Supported Backbones

The released `data_gen/extract_activations.py` targets **Qwen3-VL** (it loads
`Qwen3VLForConditionalGeneration` and hooks `model.visual.blocks.{i}`). The paper
also reports Gemma-3 and LLaVA-1.5; to use those, swap the model class and hook
path in `extract_activations.py`. `<L>` is the layer index (`--layers`):

| Backbone | HF model | hook path (`--layer_name`) |
|----------|----------|----------------------------|
| Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` | `model.visual.blocks.<L>` (<L> is the number of layer) |
| Gemma-3-4B-IT | `google/gemma-3-4b-it` | `model.vision_tower.vision_model.encoder.layers.<L>` |
| LLaVA-1.5-13B | `llava-hf/llava-1.5-13b-hf` | `model.layers.<L>` |

## Project Structure

```
CSAE/
├── csae/
│   └── model.py                  # CSAE: BatchTopKSAE, TwoLevelBatchTopKSAE, TwoLevelBatchTopKTrainer
├── data_gen/
│   ├── extract_activations.py    # MLLM vision activations -> HDF5
│   └── gen_dino_reference.py     # DINOv3 image embeddings (HMS reference space)
├── train_csae.py                 # Train the CSAE end-to-end on streamed HDF5 activations
├── eval_hms.py                   # Hierarchical Mono-Semanticity (HMS) evaluation
├── steering/
│   ├── core.py                   # cluster discovery, per-cluster scale (sigma_A), ClampHook
│   └── run_demo.py               # steer one Level-2 unit; print baseline vs steered captions
├── configs/
│   └── qwen_imagenet.yaml        # worked-example hyperparameters
├── requirements.txt
├── environment.yml
├── LICENSE
└── NOTICE
```

## Citation

```bibtex
@inproceedings{zhao2026csae,
  title   = {SAE++: Learning Multi-Level Visual Concepts from Multimodal LLMs with Cascaded Sparse Autoencoders},
  author  = {Zhao, Yusong and Wang, Hengyi and Ganu, Tanuja and Nambi, Akshay and Wang, Hao},
  year    = {2026},
}
```

## License

MIT (see `LICENSE`). Model backbones and third-party deps retain their own
licenses; see `NOTICE`.
