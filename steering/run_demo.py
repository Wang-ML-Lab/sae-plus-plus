#!/usr/bin/env python3
"""
CSAE concept steering — qualitative demo (no LLM judge).

Discovers Level-2 clusters from a trained CSAE, then for each cluster:
  - SUPPRESS (clamp cluster atoms to -alpha*sigma_A) on images where the concept
    is present (top-activating) -> does the concept leave the caption?
  - INSERT  (clamp to +alpha*sigma_A) on images where it is absent
    (low-activating) -> does the concept appear?

Prints baseline vs steered captions side by side so you can read the effect
directly. Requires a labelled image folder (class subdirs) whose images line up
with the activation HDF5 (same sorted order as data_gen/extract_activations.py).

Example:
    python steering/run_demo.py \
        --ckpt-path ./runs/imagenet/.../ae.pt \
        --data-path ./data/imagenet_qwen_block23.h5 \
        --image-dir /path/to/imagenet/val \
        --device cuda:0 --n-clusters 3 --n-images 3
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from steering import core as sps


def discover_images(image_dir):
    """Sorted class dirs, sorted files — matches the HDF5 build order."""
    paths, classes = [], []
    for cls in sorted(os.listdir(image_dir)):
        d = os.path.join(image_dir, cls)
        if not os.path.isdir(d):
            continue
        clean = cls.split(",")[0].replace("_", " ").strip().lower()
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(d, fn))
                classes.append(clean)
    return paths, classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--layer-name", default="model.visual.blocks.23")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cluster-cache", default=None)
    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--min-size", type=int, default=2, help="min atoms per cluster to steer")
    p.add_argument("--max-size", type=int, default=8, help="max atoms (small clusters are more coherent)")
    args = p.parse_args()
    device = args.device

    sae = sps.load_sae(args.ckpt_path, device)
    alive_idx1, cluster_dict, img_cluster_acts, n_img = sps.discover_clusters(
        sae, args.data_path, args.layer_name, device, cache_path=args.cluster_cache)
    paths, classes = discover_images(args.image_dir)
    print(f"images on disk: {len(paths)} | images in h5: {n_img}")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", dtype=torch.bfloat16, device_map=device).eval()
    proc = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", min_pixels=256 * 256, max_pixels=256 * 256)
    hook_module = qwen.get_submodule(args.layer_name)

    def caption(image, clamps):
        hook = sps.ClampHook(sae.sae1, clamps).attach(hook_module) if clamps else None
        try:
            return sps.gen_caption(qwen, proc, image, device)
        finally:
            if hook:
                hook.remove()

    # small, coherent clusters (huge clusters are mixed "garbage collectors");
    # among eligible sizes, take the most strongly-activating ones.
    eligible = [l2 for l2 in cluster_dict
                if args.min_size <= len(cluster_dict[l2]) <= args.max_size]
    eligible.sort(key=lambda l2: float(img_cluster_acts[l2].max()), reverse=True)
    print(f"clusters total={len(cluster_dict)}, "
          f"in size [{args.min_size},{args.max_size}]={len(eligible)}")
    for l2 in eligible[:args.n_clusters]:
        abs_A = alive_idx1[cluster_dict[l2]].cpu().tolist()
        acts = img_cluster_acts[l2]
        top = np.argsort(acts)[::-1]
        concept = [classes[i] for i in top[:5] if i < len(classes)]
        sigma = sps.cluster_sigma(sae, args.data_path, args.layer_name, device, abs_A, top[:10].tolist())
        print(f"\n{'='*70}\nL2 cluster #{l2} | {len(abs_A)} atoms | sigma_A={sigma:.3f}")
        print(f"  concept (top classes): {concept}")

        print("  -- SUPPRESS (concept present -> clamp to -%.0f sigma) --" % args.alpha)
        for i in top[:args.n_images]:
            if i >= len(paths):
                continue
            try:
                img = Image.open(paths[i]).convert("RGB")
            except Exception:
                continue
            base = caption(img, [])
            supp = caption(img, [(a, -args.alpha * sigma) for a in abs_A])
            print(f"   [{classes[i]}] base: {base}")
            print(f"   [{classes[i]}] supp: {supp}")

        print("  -- INSERT (concept absent -> clamp to +%.0f sigma) --" % args.alpha)
        for i in top[::-1][:args.n_images]:
            if i >= len(paths):
                continue
            try:
                img = Image.open(paths[i]).convert("RGB")
            except Exception:
                continue
            base = caption(img, [])
            ins = caption(img, [(a, args.alpha * sigma) for a in abs_A])
            print(f"   [{classes[i]}] base: {base}")
            print(f"   [{classes[i]}] +ins: {ins}")


if __name__ == "__main__":
    main()
