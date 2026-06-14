#!/usr/bin/env python3
"""
CSAE concept steering — qualitative demo (no LLM judge).

Pick a Level-2 unit with --cluster and steer it: SUPPRESS (clamp its Level-1
atoms to -alpha*sigma_A) on images where the concept is present, and INSERT
(+alpha*sigma_A) on images where it is absent. Prints baseline vs steered
captions so the effect can be read directly. Run without --cluster to list the
available unit #s.

Requires a labelled image folder (class subdirs) aligned with the activation
HDF5 (same sorted order as data_gen/extract_activations.py).

Example:
    python steering/run_demo.py --ckpt-path ./runs/imagenet/.../ae.pt \
        --data-path ./data/imagenet_val_acts.h5 \
        --image-dir /path/to/imagenet/val --device cuda:0 --cluster 7452
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
    p.add_argument("--cluster", type=int, default=None,
                   help="Level-2 unit # to steer (omit to list available units)")
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--alpha", type=float, default=3.0)
    args = p.parse_args()
    device = args.device

    sae = sps.load_sae(args.ckpt_path, device)
    alive_idx1, cluster_dict, img_cluster_acts, n_img = sps.discover_clusters(
        sae, args.data_path, args.layer_name, device, cache_path=args.cluster_cache)
    paths, classes = discover_images(args.image_dir)

    # Print every alive Level-2 unit (id, #atoms, top concept) so the reader can
    # choose one to steer. Pick any --cluster from this list.
    if args.cluster not in cluster_dict:
        if args.cluster is not None:
            print(f"unit {args.cluster} not found. Alive Level-2 units:")
        else:
            print(f"Alive Level-2 units ({len(cluster_dict)}) — pick one with --cluster:")
        for l2 in sorted(cluster_dict, key=lambda c: float(img_cluster_acts[c].max()), reverse=True):
            top = np.argsort(img_cluster_acts[l2])[::-1]
            concept = [classes[i] for i in top[:3] if i < len(classes)]
            print(f"  --cluster {l2:<6d} ({len(cluster_dict[l2])} atoms)  {concept}")
        return

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

    l2 = args.cluster
    abs_A = alive_idx1[cluster_dict[l2]].cpu().tolist()
    top = np.argsort(img_cluster_acts[l2])[::-1]
    concept = [classes[i] for i in top[:5] if i < len(classes)]
    sigma = sps.cluster_sigma(sae, args.data_path, args.layer_name, device, abs_A, top[:10].tolist())
    print(f"\nL2 unit #{l2} | {len(abs_A)} atoms | sigma_A={sigma:.3f} | concept: {concept}")

    print(f"\n-- SUPPRESS (-{args.alpha:.0f} sigma) on images where the concept is present --")
    for i in top[:args.n_images]:
        if i >= len(paths):
            continue
        img = Image.open(paths[i]).convert("RGB")
        print(f"  [{classes[i]}] base: {caption(img, [])}")
        print(f"  [{classes[i]}] supp: {caption(img, [(a, -args.alpha * sigma) for a in abs_A])}")

    print(f"\n-- INSERT (+{args.alpha:.0f} sigma) on images where the concept is absent --")
    for i in top[::-1][:args.n_images]:
        if i >= len(paths):
            continue
        img = Image.open(paths[i]).convert("RGB")
        print(f"  [{classes[i]}] base: {caption(img, [])}")
        print(f"  [{classes[i]}] +ins: {caption(img, [(a, args.alpha * sigma) for a in abs_A])}")


if __name__ == "__main__":
    main()
