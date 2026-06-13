import argparse
import os
import torch
import h5py
import json
import numpy as np
import sys
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def _resolve_modules(model, layer_idxs):
    mods = {}
    for i in layer_idxs:
        name = f"model.visual.blocks.{i}"
        mod = model.get_submodule(name)
        mods[name] = mod
    return mods

def _make_collection_hook(store, key):
    def hook(module, inputs, outputs):
        # outputs is the hidden state (Batch, Seq, Dim)
        store[key].append(outputs.float().detach().cpu())
    return hook

def process_sampled_imagenet_to_hdf5(
    image_dir,
    model,
    seed,
    layers,
    gen_new_tokens,
    out_path,
):
    torch.manual_seed(seed)

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir or ".", exist_ok=True)

    # --- 1. Robust Dataset Discovery ---
    dataset = []
    # Walk the directory to find all sub-folders, regardless of naming
    all_subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    print(f"Discovering images in {image_dir}...")
    for cls in sorted(all_subdirs):
        cls_dir = os.path.join(image_dir, cls)
        # Filter for actual image files
        valid_images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in valid_images:
            dataset.append({
                "rel_path": os.path.join(cls, img_file),
                "class_name": cls
            })

    num_found_classes = len(set([d['class_name'] for d in dataset]))
    print(f"Found {len(dataset)} images across {num_found_classes} classes.")
    
    if num_found_classes < 1000:
        print(f"⚠️ WARNING: Only {num_found_classes}/1000 classes found. Check for empty folders.")

    # --- 2. Setup Model & Hooks ---
    name_to_module = _resolve_modules(model, layers)
    hook_names = list(name_to_module.keys())

    all_class_labels = [] 
    total_n = 0
    token_offsets = []
    total_tokens = 0

    # Image processing settings (standardized for Qwen3-VL SAE work)
    min_pixels = 256 * 256
    max_pixels = 256 * 256
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model.eval()

    h5_file = None
    h5_dsets_x = {}

    # --- 3. Main Processing Loop ---
    try:
        for entry in tqdm(dataset, desc="Processing Images"):
            image_path = os.path.join(image_dir, entry["rel_path"])
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Skip {image_path}: {e}")
                continue

            prompt_text = "Describe the image content accurately."
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            store = {name: [] for name in hook_names}
            handles = [
                name_to_module[name].register_forward_hook(_make_collection_hook(store, name))
                for name in hook_names
            ]

            try:
                with torch.inference_mode():
                    _ = model(**inputs, use_cache=False)
            finally:
                for h in handles:
                    h.remove()

            # --- 4. Store in HDF5 ---
            x_data_cpu = {name: store[name][0] for name in hook_names}
            
            # Safety check: Ensure tensors are 3D [Batch, Seq, Dim]
            for name in hook_names:
                if x_data_cpu[name].dim() == 2:
                    x_data_cpu[name] = x_data_cpu[name].unsqueeze(0)
                elif x_data_cpu[name].dim() != 3:
                    raise RuntimeError(f"Unexpected tensor dim {x_data_cpu[name].dim()} for {name}")

            # Now this unpacking will succeed safely
            _, S, _ = x_data_cpu[hook_names[0]].shape

            if h5_file is None:
                h5_file = h5py.File(out_path, "w")
                h5_group_x = h5_file.create_group("X")
                for name, x_tensor in x_data_cpu.items():
                    D = x_tensor.shape[-1]
                    h5_dsets_x[name] = h5_group_x.create_dataset(
                        name, shape=(0, D), maxshape=(None, D),
                        dtype=x_tensor.numpy().dtype, chunks=(min(1024, D), D)
                    )

            # Flat token insertion
            for name, x_tensor in x_data_cpu.items():
                seq = x_tensor.squeeze(0).numpy() # (S, D)
                dset = h5_dsets_x[name]
                old_T = dset.shape[0]
                dset.resize((old_T + S, x_tensor.shape[-1]))
                dset[old_T : old_T + S] = seq

            token_offsets.append(total_tokens)
            total_tokens += S
            all_class_labels.append(entry["class_name"])
            total_n += 1

        # Add final offset for the last image boundary
        token_offsets.append(total_tokens)
        h5_file.create_dataset("token_offsets", data=np.asarray(token_offsets, dtype=np.int64))

        meta = dict(
            n=total_n, seed=seed, layers=layers,
            total_tokens=total_tokens,
            min_pixels=min_pixels, max_pixels=max_pixels
        )

        h5_file.attrs["meta"] = json.dumps(meta)
        h5_file.attrs["class_names"] = json.dumps(all_class_labels)

        print(f"✅ Finished. Saved {total_n} images.")
        return meta

    finally:
        if h5_file:
            h5_file.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=str, required=True)
    p.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=str, default="5,11,17,23")
    p.add_argument("--gen-new-tokens", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()

if __name__ == "__main__":
    cfg = parse_args()
    layers = [int(x) for x in cfg.layers.split(",") if x.strip()]

    print(f"Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        dtype=torch.float16, # Updated from torch_dtype
        device_map=cfg.device,
    )

    process_sampled_imagenet_to_hdf5(
        image_dir=cfg.image_dir,
        model=model,
        seed=cfg.seed,
        layers=layers,
        gen_new_tokens=cfg.gen_new_tokens,
        out_path=cfg.out,
    )