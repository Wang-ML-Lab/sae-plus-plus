# CSAE: Cascaded Sparse Autoencoders for Multi-Level Visual Concepts in MLLMs

Reference code for **"SAE++: Learning Multi-Level Visual Concepts from Multimodal
LLMs with Cascaded Sparse Autoencoders."**

**CSAE** is a two-level cascaded SAE trained end-to-end: Level-1 decomposes an
MLLM activation into atomic concepts (its decoder columns are the concept
directions), and Level-2 is trained **on the Level-1 decoder atoms themselves**,
learning "concepts of concepts."

The pipeline is: **get images → (1) generate activations & embeddings → (2) train
CSAE → (3) evaluate HMS → (4) steer concepts.** A CUDA GPU is required.

## Repository layout

```
csae/model.py       CSAE: BatchTopKSAE, TwoLevelBatchTopKSAE, TwoLevelBatchTopKTrainer
train_csae.py       Train CSAE on streamed HDF5 activations
eval_hms.py         Hierarchical Mono-Semanticity (HMS) evaluation
steering/           Concept steering (core.py: ClampHook; run_demo.py: caption demo)
data_gen/           extract_activations.py, gen_dino_reference.py
configs/, scripts/  worked-example config and end-to-end script
```

## Installation

```bash
pip install -r requirements.txt      # or: conda env create -f environment.yml
```

`dictionary_learning.training.trainSAE` is a pip dependency (not vendored). MLLM
backbones (Qwen3-VL) and the DINOv3 encoder download from the Hugging Face Hub on
first use under their own licenses (see `NOTICE`).

## Data

The **only thing you download is a folder of images**; the `.h5` files under
`./data/` are *produced* by step 1, and checkpoints are not committed (train your
own in step 2).

Images must be arranged as **one subdirectory per class**:

```
<image-dir>/
  class_a/  img001.jpg  img002.jpg  ...
  class_b/  img001.jpg  ...
  ...
```

The worked example uses the **ImageNet-1k validation set** (50k images, 1000
class folders) — download it from <https://www.image-net.org/> and lay it out as
above. Any class-subdir image folder (`.jpg/.jpeg/.png`) works for a quick try.

Point the pipeline at it and create the output dirs:

```bash
export IMAGES=/path/to/imagenet/val      # the folder with class subdirs above
export DEVICE=cuda:0
mkdir -p data runs results
```

## 1. Generate activations & embeddings

This reads `$IMAGES` and writes two HDF5 files into `./data/`: the MLLM vision
activations (group `X`, one dataset per layer, plus `token_offsets`) and the
DINOv3 image embeddings used as the HMS reference space. **Both scripts walk the
images in the same order**, so the two files stay aligned.

```bash
python data_gen/extract_activations.py \
  --image-dir "$IMAGES" --model-path Qwen/Qwen3-VL-4B-Instruct \
  --layers 23 --out ./data/imagenet_qwen_block23.h5 --device "$DEVICE"

python data_gen/gen_dino_reference.py \
  --image-dir "$IMAGES" --out ./data/imagenet_dinov3.h5 --device "$DEVICE"
```

## 2. Train

```bash
python train_csae.py \
  --save_dir ./runs/imagenet \
  --model_name Qwen --dataset_name ImageNet \
  --data_path ./data/imagenet_qwen_block23.h5 \
  --layer_name model.visual.blocks.23 --device "$DEVICE" \
  --dict_size 20000 --k1 6 --k2 1 \
  --lr 1e-4 --seed 0 \
  --num_tokens 500000000 --sae_batch_size 1024 \
  --warmup_steps 500 --sae2_start_step 10000
```

The checkpoint lands at `./runs/imagenet/<submodule>/trainer_0/ae.pt`.
`--sae2_start_step` lets Level-1 stabilize before Level-2 starts; see
`configs/qwen_imagenet.yaml`.

## 3. Evaluate (HMS)

```bash
python eval_hms.py \
  --ckpt-path ./runs/imagenet/*/trainer_0/ae.pt \
  --data-path ./data/imagenet_qwen_block23.h5 \
  --embedding-path ./data/imagenet_dinov3.h5 \
  --layer-name model.visual.blocks.23 --device "$DEVICE"
```

Prints `HMS_{min,med,max,mean}` over the discovered Level-2 clusters.

## 4. Steering

Steer a Level-2 unit with `--cluster <unit#>`. **Run it once without `--cluster`
to print the list of alive units** (id, #atoms, top concept), then pick one.
Clamping a unit's atoms inserts its concept into images that lack it and removes
it from images that have it.

**Bald eagle — unit #7452:**

```bash
python steering/run_demo.py --ckpt-path ./runs/imagenet/*/trainer_0/ae.pt \
  --data-path ./data/imagenet_qwen_block23.h5 --image-dir "$IMAGES" \
  --device "$DEVICE" --cluster-cache ./results/clusters.pt --cluster 7452
```

```
INSERT  (+3σ)  fountain image -> "A majestic white-tailed sea eagle soars above a forest, wings spread wide"
INSERT  (+3σ)  isopod image   -> "A large, feathered sea eagle perches on a mossy, rocky surface"
SUPPRESS(-3σ)  eagle on branch -> "black and white striped tiles ..."   (eagle gone)
```

**Schooner — unit #7806:**

```bash
python steering/run_demo.py --ckpt-path ./runs/imagenet/*/trainer_0/ae.pt \
  --data-path ./data/imagenet_qwen_block23.h5 --image-dir "$IMAGES" \
  --device "$DEVICE" --cluster-cache ./results/clusters.pt --cluster 7806
```

```
INSERT  (+3σ)  alligator image -> "A large, traditional sailing ship with multiple masts and a dark hull"
INSERT  (+3σ)  nematode image  -> "A large, ornate sailing ship with multiple masts and sails"
SUPPRESS(-3σ)  tall ship w/ sails -> "a black and white striped object ..."   (ship gone)
```

(Unit numbers are specific to your trained checkpoint — use the list printed by
the no-`--cluster` run.)

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
