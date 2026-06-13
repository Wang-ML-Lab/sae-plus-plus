# CSAE: Cascaded Sparse Autoencoders for Multi-Level Visual Concepts in MLLMs

Reference code for **"SAE++: Learning Multi-Level Visual Concepts from Multimodal
LLMs with Cascaded Sparse Autoencoders."**.


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

`dictionary_learning.training.trainSAE` is a pip dependency (not vendored); MLLM
backbones download from the Hugging Face Hub on first use. Trained checkpoints
are not committed — train your own below.

## 1. Generate data

Extract MLLM vision activations to HDF5 (group `X`, one dataset per layer, plus
`token_offsets`), and DINOv3 image embeddings for the HMS metric (same image
order):

```bash
python data_gen/extract_activations.py \
  --image-dir /path/to/imagenet/val \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --layers 23 --out ./data/imagenet_qwen_block23.h5 --device cuda:0

python data_gen/gen_dino_reference.py \
  --image-dir /path/to/imagenet/val \
  --out ./data/imagenet_dinov3.h5 --device cuda:0
```

## 2. Train

```bash
python train_csae.py \
  --save_dir ./runs/imagenet \
  --model_name Qwen --dataset_name ImageNet \
  --data_path ./data/imagenet_qwen_block23.h5 \
  --layer_name model.visual.blocks.23 --device cuda:0 \
  --dict_size 20000 --k1 6 --k2 1 \
  --lr 1e-4 --seed 0 \
  --num_tokens 500000000 --sae_batch_size 1024 \
  --warmup_steps 500 --sae2_start_step 10000
```

`--sae2_start_step` lets Level-1 stabilize before Level-2 starts. See
`configs/qwen_imagenet.yaml`.

## 3. Evaluate (HMS)

```bash
python eval_hms.py \
  --ckpt-path ./runs/imagenet/.../ae.pt \
  --data-path ./data/imagenet_qwen_block23.h5 \
  --embedding-path ./data/imagenet_dinov3.h5 \
  --layer-name model.visual.blocks.23 --device cuda:0
```

Prints `HMS_{min,med,max,mean}` over the discovered Level-2 clusters.

## 4. Steering

Steer a Level-2 unit with `--cluster <unit#>` (run without it to list units):

```bash
python steering/run_demo.py \
  --ckpt-path ./runs/imagenet/.../ae.pt \
  --data-path ./data/imagenet_qwen_block23.h5 \
  --image-dir /path/to/imagenet/val --device cuda:0 \
  --cluster-cache ./results/clusters.pt \
  --cluster 4198 --n-images 3 --alpha 3.0
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
