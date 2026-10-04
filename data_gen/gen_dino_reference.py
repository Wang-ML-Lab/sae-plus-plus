"""
Generate DINOv3 embeddings for ImageNet-mini images.
Image ordering: sorted class dirs, sorted filenames (matches SigLIP/CLIP HDF5).
Output: data/imagenet_mini_dinov3_embeddings.h5 with shape [N_images, D_emb]
"""

import argparse
import os
import torch
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

DINO_MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


def discover_images(image_dir):
    """Discover ImageNet images in sorted order (same as SigLIP/CLIP scripts)."""
    dataset = []
    all_subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    for cls in sorted(all_subdirs):
        cls_dir = os.path.join(image_dir, cls)
        valid_images = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for img_file in valid_images:
            dataset.append(os.path.join(image_dir, cls, img_file))
    return dataset


def main():
    parser = argparse.ArgumentParser("Generate DINOv3 embeddings for ImageNet-mini.")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    image_paths = discover_images(args.image_dir)
    print(f"Found {len(image_paths)} images")

    print(f"Loading DINOv3 model: {DINO_MODEL}")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL, torch_dtype=torch.float16).to(args.device).eval()

    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="DINO embeddings"):
        batch_paths = image_paths[i:i + args.batch_size]
        images = []
        for p in batch_paths:
            # Skip unreadable images, exactly as extract_activations.py does, so the
            # embedding rows stay aligned image-for-image with the activations.
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Skip {p}: {e}")
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].float().cpu()
        all_embeddings.append(cls_emb)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    print(f"Embeddings shape: {embeddings.shape}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=embeddings)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
