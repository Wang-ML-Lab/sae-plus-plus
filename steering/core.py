#!/usr/bin/env python3
"""
Core CSAE steering primitives (Pach et al. 2025 protocol): cluster discovery,
per-cluster scale (sigma_A), and the ClampHook intervention.
"""
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from csae.model import TwoLevelBatchTopKSAE


def load_sae(ckpt_path, device):
    import json
    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob["ae"] if "ae" in blob else blob
    H1, D = state["sae1.encoder.weight"].shape
    H2 = state["sae2.encoder.weight"].shape[0]
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    k1, k2 = 16, 1
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            c = json.load(f)["trainer"]
        k1, k2 = c.get("k1_target", 16), c.get("k2_target", 1)
    sae = TwoLevelBatchTopKSAE(activation_dim=D, dict_size1=H1, dict_size2=H2, k1=k1, k2=k2)
    sae.load_state_dict(state, strict=False)
    return sae.to(device).eval()


def find_h5_dataset(f, layer_name):
    if "X" in f and isinstance(f["X"], h5py.Group) and layer_name in f["X"]:
        return f["X"][layer_name]
    if "X" in f and isinstance(f["X"], h5py.Dataset):
        return f["X"]
    raise KeyError(f"Cannot find {layer_name}")


@torch.no_grad()
def discover_clusters(sae, data_path, layer_name, device, topk_l1=20, act_thr=1e-3,
                      cache_path=None):
    """Group alive Level-1 atoms by their Level-2 parent (argmax of sae2 on the
    decoder atoms). Returns (alive_idx1, cluster_dict, img_cluster_acts, N_img)
    where cluster_dict maps an L2 unit -> list of positions into alive_idx1."""
    if cache_path and os.path.exists(cache_path):
        print(f"[discover] loading cache {cache_path}")
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        return (c["alive_idx1"].to(device), c["cluster_dict"],
                c["img_cluster_acts"], c["N_img"])

    sae1, sae2 = sae.sae1, sae.sae2
    H1 = sae1.encoder.weight.shape[0]

    with h5py.File(data_path, "r") as f:
        ds = find_h5_dataset(f, layer_name)
        offsets = f["token_offsets"][:] if "token_offsets" in f else None
        N_tok = ds.shape[0]
        N_img = len(offsets) - 1 if offsets is not None else N_tok
        freq1 = torch.zeros(H1, device=device)
        for start in range(0, N_tok, 4096):
            x = torch.tensor(ds[start:start + 4096], device=device, dtype=torch.float32)
            post = F.relu(sae1.encoder(x - sae1.b_dec))
            vals, idxs = post.topk(topk_l1, dim=-1)
            mask = vals > act_thr
            for b in range(x.size(0)):
                active = idxs[b][mask[b]]
                if active.numel() > 0:
                    freq1.index_add_(0, active, torch.ones_like(active, dtype=torch.float32))

    alive_idx1 = (freq1 > 0).nonzero(as_tuple=False).flatten()
    atoms = sae1.decoder.weight[:, alive_idx1].T
    post2 = F.relu(sae2.encoder(atoms - sae2.b_dec))
    alive_mask2 = post2.max(dim=0).values > act_thr
    alive_idx2 = alive_mask2.nonzero(as_tuple=False).flatten()
    parent_map = post2[:, alive_idx2].argmax(dim=-1)

    cluster_dict = {}
    for l1_pos in range(len(alive_idx1)):
        cluster_dict.setdefault(parent_map[l1_pos].item(), []).append(l1_pos)
    cluster_dict = {k: v for k, v in cluster_dict.items() if len(v) >= 2}
    print(f"[discover] alive L1={len(alive_idx1)} L2={len(alive_idx2)} clusters(>=2)={len(cluster_dict)}")

    img_cluster_acts = {l2: np.zeros(N_img, dtype=np.float32) for l2 in cluster_dict}
    with h5py.File(data_path, "r") as f:
        ds = find_h5_dataset(f, layer_name)
        offsets_arr = f["token_offsets"][:] if "token_offsets" in f else None
        for img_i in tqdm(range(N_img), desc="[discover] profiling", leave=False):
            if offsets_arr is not None:
                s, e = int(offsets_arr[img_i]), int(offsets_arr[img_i + 1])
                x = torch.tensor(ds[s:e], device=device, dtype=torch.float32)
            else:
                x = torch.tensor(ds[img_i:img_i + 1], device=device, dtype=torch.float32)
            post = F.relu(sae1.encoder(x - sae1.b_dec))
            f1 = post[:, alive_idx1].mean(dim=0)
            for l2, positions in cluster_dict.items():
                img_cluster_acts[l2][img_i] = f1[positions].sum().item()

    if cache_path:
        torch.save({"alive_idx1": alive_idx1.cpu(), "cluster_dict": cluster_dict,
                    "img_cluster_acts": img_cluster_acts, "N_img": N_img}, cache_path)
        print(f"[discover] cached -> {cache_path}")
    return alive_idx1, cluster_dict, img_cluster_acts, N_img


@torch.no_grad()
def cluster_sigma(sae, data_path, layer_name, device, l1_abs_indices, top_img_idxs):
    """Per-token activation scale of the cluster's atoms on its top images.
    Clamping overrides per-token codes, so sigma is measured per-token (not
    pooled), as the mean of activations that survive the learned threshold."""
    sae1 = sae.sae1
    thr = max(float(sae1.threshold.item()), 1.0)
    with h5py.File(data_path, "r") as f:
        ds = find_h5_dataset(f, layer_name)
        offsets = f["token_offsets"][:] if "token_offsets" in f else None
        vals = []
        for img_i in top_img_idxs:
            if offsets is not None:
                s, e = int(offsets[img_i]), int(offsets[img_i + 1])
                x = torch.tensor(ds[s:e], device=device, dtype=torch.float32)
            else:
                x = torch.tensor(ds[img_i:img_i + 1], device=device, dtype=torch.float32)
            post = F.relu(sae1.encoder(x - sae1.b_dec))[:, l1_abs_indices]
            pos = post[post > thr * 0.5]
            if pos.numel() > 0:
                vals.append(pos.mean().item())
    return float(np.mean(vals)) if vals else thr


class ClampHook:
    """Set the post-ReLU code of the target atoms to fixed values at every token
    position, then decode through the SAE's sparse path. clamps = [(atom_idx, value), ...]."""

    def __init__(self, sae1, clamps):
        self.sae1 = sae1
        self.target = torch.tensor([c[0] for c in clamps], dtype=torch.long)
        self.values = torch.tensor([c[1] for c in clamps], dtype=torch.float32)
        self.handle = None

    def _hook(self, module, inputs, outputs):
        was_tuple = isinstance(outputs, tuple)
        x = outputs[0] if was_tuple else outputs
        shape, dtype, dev = x.shape, x.dtype, x.device
        x_flat = x.float().reshape(-1, shape[-1])
        post = F.relu(self.sae1.encoder(x_flat - self.sae1.b_dec))
        post[:, self.target.to(dev)] = self.values.to(dev)
        x_mod = (self.sae1.decoder(post) + self.sae1.b_dec).reshape(shape).to(dtype)
        return (x_mod,) + outputs[1:] if was_tuple else x_mod

    def attach(self, module):
        self.handle = module.register_forward_hook(self._hook)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def gen_caption(model, proc, image, device, max_new=30):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe what is in this image in one short sentence."},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new = out[:, inputs["input_ids"].shape[-1]:]
    return proc.batch_decode(new, skip_special_tokens=True)[0].strip()
