# SAE++: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2606.16193-b31b1b.svg)](https://arxiv.org/abs/2606.16193)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![🤗 Checkpoint](https://img.shields.io/badge/🤗%20Hugging%20Face-Checkpoint-blue)](https://huggingface.co/YusongZhao666/csae-ckpt)

This is the official implementation of the NeurIPS 2026 paper:

**[SAE++: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs](https://arxiv.org/abs/2606.16193)**

Yusong Zhao*, Hengyi Wang*, Tanuja Ganu, Akshay Nambi, Hao Wang

## Outline for This README
* [What SAE++ Does](#what-sae-does)
* [Multi-Level Concepts](#multi-level-concepts)
* [Concept Steering](#concept-steering)
* [Installation](#installation) and [Datasets](#datasets)
* Pipeline: [Step 1. Generate activations](#step-1-generate-activations--embeddings) →
  [Step 2. Train](#step-2-train-sae) → [Step 3. Evaluate HMS](#step-3-evaluate-hms) →
  [Step 4. Steering](#step-4-steering)
* [Citation](#citation)

## What SAE++ Does
Sparse autoencoders (SAEs) decompose an MLLM's dense activations into sparse,
interpretable features, but they usually learn a **flat** dictionary. SAE++ learns
**multi-level** visual concepts: a Level-1 SAE learns fine-grained concepts from the
MLLM's activations, and a Level-2 SAE is trained **directly on the Level-1 decoder
weights** (each decoder column is one Level-1 concept direction). Level-2 therefore
learns "concepts of concepts", and both levels are trained jointly, end to end.

The figure compares hierarchical SAE designs:
* **(a) Matryoshka SAEs** build the hierarchy from one nested prefix chain, so early
  directions are reused by every later level, and errors in them propagate.
* **(b) Stacked SAEs** re-compress the Level-1 sparse codes through a smaller
  bottleneck, which loses capacity.
* **(c) SAE++** feeds the learned Level-1 weights, not the sparse codes, to Level-2.

<p align="center">
<img src="fig/fig1_overview.png" alt="Hierarchical SAE designs: Matryoshka, stacked, and SAE++" width="95%"/>
</p>

## Multi-Level Concepts
SAE++ groups related Level-1 concepts under a coherent Level-2 concept, e.g.
*Truck* and *Jeep* under *Vehicle*, and *Pinwheels* and *Waterwheels* under
*Rotating Structures*. The matched Matryoshka SAE (MSAE) concepts are less
consistent and often mix unrelated visual patterns. Each row shows the top
activating images of one Level-1 concept.

<p align="center">
<img src="fig/fig2_concepts.png" alt="Level-1 and Level-2 concepts from SAE++ and Matryoshka SAEs" width="95%"/>
</p>

Across Qwen3-VL, Gemma-3 and LLaVA on four datasets, SAE++ achieves the best
Hierarchical Mono-Semanticity (HMS, the coherence of Level-1 concepts under each
Level-2 parent) in every setting. See the paper for the full tables.

## Concept Steering
Because a Level-2 concept groups related Level-1 concepts, clamping all of them
together steers the MLLM's output at the concept level. Below, steering the
SAE++ Level-2 concept *Table* inserts or removes "table" from Qwen3-VL's caption,
while steering a single Level-1 concept or matched MSAE concepts does not.

<p align="center">
<img src="fig/fig3_steering.png" alt="Concept insertion and suppression with SAE++ versus baselines" width="95%"/>
</p>

On 1,000 COCO evaluations judged by four LLMs, SAE++ steering succeeds in 71.0% of
insertions and 40.1% of suppressions on average, versus 59.9% / 34.1% for the best
single Level-1 concept and 40.9% / 23.6% for matched MSAE clusters.

The rest of this README walks through the pipeline: get images → (1) generate
activations → (2) train SAE++ → (3) evaluate HMS → (4) steer concepts. A CUDA GPU
is required.

## Installation

```bash
git clone https://github.com/Wang-ML-Lab/sae-plus-plus.git
cd sae-plus-plus
pip install -r requirements.txt      # or: conda env create -f environment.yml
```

`dictionary_learning.training.trainSAE` is a pip dependency (not vendored). MLLM
backbones (Qwen3-VL) and the DINOv3 encoder download from the Hugging Face Hub on
first use under their own licenses (see `NOTICE`). The DINOv3 encoder is a
**gated** model with about 7B parameters: request access on its
[model page](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m)
and run `huggingface-cli login` before Step 1.

The Python package is named `csae` (cascaded SAE) because `sae++` is not a
valid Python identifier.

## Datasets

Obtain any of the datasets from the paper. The paper samples them as follows
(App. E.1):

| Dataset | Source | Train / eval split in the paper |
|---------|--------|---------------------------------|
| ImageNet-1k | https://www.image-net.org/download.php | 50 random train images per class (50,000) / full val set (50,000) |
| iNaturalist 2021 | https://github.com/visipedia/inat_comp/tree/master/2021 | 5 images per species for each split (10,000 species) |
| MS-COCO | https://cocodataset.org/#download | 250 images per category for each split (80 categories) |
| Color | https://github.com/Wang-ML-Lab/interpretable-foundation-models | mono-semantic subset (1,000 images), 8:2 split |

Only obtain the **images**; the `.h5` files under `./data/` are produced by the
data-generation scripts in Step 1. Arrange the images as **one subdirectory per class**:

```
<image-dir>/
  class_a/  img001.jpg  img002.jpg  ...
  class_b/  img001.jpg  ...
```

Set the splits and create the output dirs:

```bash
export TRAIN_IMAGES=/path/to/imagenet/train_sample   # 50 images/class from the train split
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

## Step 2. Train SAE++

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

These are the settings of the released checkpoint (see below), and they are also
`train_csae.py`'s defaults: dictionary sizes n1 = 20,000 and n2 = 10,000, TopK
sparsities k1 = 6 and k2 = 1, Adam with learning rate 1e-4, and batches of 1,024
activation tokens for about 500M tokens. `--sae2_start_step` lets Level-1
stabilize before Level-2 starts. See `configs/qwen_imagenet.yaml` for the full list.

The checkpoint lands at `./runs/imagenet/<submodule>/trainer_0/ae.pt`.

## Step 3. Evaluate HMS

```bash
python eval_hms.py \
  --ckpt-path ./runs/imagenet/*/trainer_0/ae.pt \
  --data-path ./data/imagenet_val_acts.h5 \
  --embedding-path ./data/imagenet_val_dino.h5 \
  --layer-name model.visual.blocks.23 --device "$DEVICE"
```

This prints HMS_min, HMS_med, HMS_max and HMS_mean over Level-2 parents with at
least two Level-1 children (paper Sec. 5.1, App. E.5). The paper's main tables
report HMS_mean and HMS_med. More details on the HMS metric are in the paper.

## Step 4. Steering

A **pretrained checkpoint is available** at
<https://huggingface.co/YusongZhao666/csae-ckpt/tree/main> (Qwen3-VL-4B x
ImageNet, trained with the Step 2 command). Download `ae.pt` with its
`config.json`, or use your own Step 2 checkpoint, and point `CKPT` at it:

```bash
export CKPT=/path/to/ae.pt
```

This demo steers the ImageNet checkpoint. The paper's quantitative steering
results (Sec. 5, App. E.6) use a separate SAE++ trained on COCO val2014.

**List the alive Level-2 units** (id, #atoms, top concept) and pick one:

```bash
python steering/run_demo.py --ckpt-path "$CKPT" \
  --data-path ./data/imagenet_val_acts.h5 --image-dir "$VAL_IMAGES" --device "$DEVICE" \
  --cluster-cache ./results/imagenet_clusters.pt
```

`--cluster-cache` saves the cluster discovery pass, so the commands below reuse
it instead of rescanning the activations. Unit ids below refer to the released
checkpoint.

**Bald eagle — unit #7452:**

```bash
python steering/run_demo.py --ckpt-path "$CKPT" \
  --data-path ./data/imagenet_val_acts.h5 --image-dir "$VAL_IMAGES" \
  --device "$DEVICE" --cluster-cache ./results/imagenet_clusters.pt --cluster 7452
```

**Schooner — unit #7806:**

```bash
python steering/run_demo.py --ckpt-path "$CKPT" \
  --data-path ./data/imagenet_val_acts.h5 --image-dir "$VAL_IMAGES" \
  --device "$DEVICE" --cluster-cache ./results/imagenet_clusters.pt --cluster 7806
```

## Supported Backbones

The released `data_gen/extract_activations.py` targets **Qwen3-VL** (it loads
`Qwen3VLForConditionalGeneration` and hooks `model.visual.blocks.{i}`). The paper
also reports Gemma-3 and LLaVA-1.5; to use those, swap the model class and hook
path in `extract_activations.py`. `<L>` is the layer index (`--layers`); the
paper's layer is listed for each backbone (hook paths as of transformers 4.57):

| Backbone | HF model | hook path (`--layer_name`) | Paper layer |
|----------|----------|----------------------------|-------------|
| Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` | `model.visual.blocks.<L>` (vision) | 23 |
| Gemma-3-4B-IT | `google/gemma-3-4b-it` | `model.vision_tower.vision_model.encoder.layers.<L>` (vision) | 26 |
| LLaVA-1.5-13B | `llava-hf/llava-1.5-13b-hf` | `model.language_model.layers.<L>` (language backbone) | 39 |

## Project Structure

```
sae-plus-plus/
├── csae/
│   └── model.py                  # SAE++: BatchTopKSAE, TwoLevelBatchTopKSAE, TwoLevelBatchTopKTrainer
├── data_gen/
│   ├── extract_activations.py    # MLLM vision activations -> HDF5
│   └── gen_dino_reference.py     # DINOv3 image embeddings (HMS reference space)
├── train_csae.py                 # Train SAE++ end-to-end on streamed HDF5 activations
├── eval_hms.py                   # Hierarchical Mono-Semanticity (HMS) evaluation
├── steering/
│   ├── core.py                   # cluster discovery, per-cluster scale (sigma_A), ClampHook
│   └── run_demo.py               # steer one Level-2 unit; print baseline vs steered captions
├── fig/                          # figures from the paper used in this README
├── configs/
│   └── qwen_imagenet.yaml        # worked-example hyperparameters
├── requirements.txt
├── environment.yml
├── LICENSE
└── NOTICE
```

## Citation

```bibtex
@inproceedings{zhao2026saepp,
  title     = {{SAE++}: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal {LLMs}},
  author    = {Zhao, Yusong and Wang, Hengyi and Ganu, Tanuja and Nambi, Akshay and Wang, Hao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.16193},
}
```

## License

MIT (see `LICENSE`). Model backbones and third-party deps retain their own
licenses; see `NOTICE`.
