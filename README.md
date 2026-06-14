# CSAE: Cascaded Sparse Autoencoders for Multi-Level Visual Concepts in MLLMs

Reference code for **"SAE++: Learning Multi-Level Visual Concepts from Multimodal
LLMs with Cascaded Sparse Autoencoders."**

## File Structure

- `csae/`
  - `model.py` — CSAE definition

- `data_gen/`
  - `extract_activations.py` — Run an MLLM over images and dump its vision activations to HDF5
  - `gen_dino_reference.py` — Compute DINOv3 image embeddings

- `train_csae.py` — Train the CSAE end-to-end on MLLM activations
  
- `eval_hms.py` — Calculate Hierarchical Mono-Semanticity Score with trained CSAE
  
- `steering/`
  - `core.py` — Cluster discovery, calculate per-cluster scale (`sigma_A`), and intervention
  - `run_demo.py` — Steer one Level-2 unit and print baseline vs. steered captions

- `configs/`
  - `qwen_imagenet.yaml` — Hyperparameters for the Qwen3-VL × ImageNet worked example


## Installation

```bash
pip install -r requirements.txt    
```

## Datasets

Download any of the datasets from the paper:

| Dataset | Download |
|---------|----------|
| ImageNet-1k | https://www.image-net.org/download.php |
| iNaturalist 2021 | https://github.com/visipedia/inat_comp/tree/master/2021 |
| MS-COCO | https://cocodataset.org/#download |
| Color | https://github.com/Wang-ML-Lab/interpretable-foundation-models |


Only download **images**; the `.h5` files under `./data/` are produced by
step data generation scripts. 

Arrange the images as **one subdirectory per class**:
```
<image-dir>/
  class_a/  img001.jpg  img002.jpg  ...
  class_b/  img001.jpg  ...
```

Set both and create the output dirs:

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



## Other backbones

Point `data_gen/extract_activations.py` at another model/layer and retrain:

| Backbone | HF model | `--layer_name` |
|----------|----------|----------------|
| Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` | `model.visual.blocks.23` |
| Gemma-3-4B-IT | `google/gemma-3-4b-it` | `model.vision_tower.vision_model.encoder.layers.26` |
| LLaVA-1.5-13B | `llava-hf/llava-1.5-13b-hf` | language-model backbone layer 39 |

## License

MIT (see `LICENSE`). Model backbones and third-party deps retain their own
licenses; see `NOTICE`.
