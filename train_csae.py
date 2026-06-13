#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a CSAE (cascaded two-level SAE) on streamed MLLM activations.

Activations are read from an HDF5 file produced by `data_gen/extract_activations.py`
(group "X", dataset keyed by `--layer_name`; 2D [tokens, dim] or 3D
[images, seq, dim]). Training is driven by `dictionary_learning.training.trainSAE`,
which handles activation normalization, checkpointing (`ae.pt` + `config.json`),
and optional Weights & Biases logging. The trained checkpoint is consumed by
`eval_hms.py`.

Example (Qwen3-VL-4B x ImageNet, paper default d=20000, k1=20, k2=10):

    python train_csae.py \
        --save_dir ./runs/qwen_imagenet \
        --model_name Qwen --dataset_name ImageNet \
        --data_path ./data/imagenet_qwen_block23.h5 \
        --layer_name model.visual.blocks.23 \
        --device cuda:0 --dict_size 20000 --k1 20 --k2 10 \
        --num_tokens 500000000
"""

import os
# torch <2.9 reads PYTORCH_CUDA_ALLOC_CONF; >=2.9 reads PYTORCH_ALLOC_CONF.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import time
from typing import Optional

import numpy as np
import h5py
import torch as t
import torch.multiprocessing as mp

from csae.model import TwoLevelBatchTopKTrainer

# Note: the generic training harness (`trainSAE`) is imported lazily inside
# main() so that the loader / config helpers here can be reused without
# requiring dictionary_learning to be installed (depend-don't-vendor).


def _h5_attr_json_load(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        x = x.decode("utf-8")
    return json.loads(x)


# ------------------------------ HDF5 streaming loader ------------------------------
class H5StreamingActivationLoader:
    """Infinite iterator yielding shuffled [batch_size, dim] activation batches.

    Supports 2D datasets ([tokens, dim]) and 3D datasets ([images, seq, dim]),
    with optional image-token slicing via the HDF5 attr `image_token_slice`.
    """

    def __init__(
        self,
        h5_path: str,
        layer_name: str,
        batch_size: int,
        device: str,
        dtype: t.dtype = t.float32,
        seed: int = 0,
        images_per_read: Optional[int] = None,
        block_rows: Optional[int] = None,
        h5_rdcc_nbytes: int = 512 * 1024 * 1024,
        h5_rdcc_nslots: int = 1_000_003,
        pin_memory: bool = True,
        token_filter: str = "image_only",
    ):
        self.h5_path = h5_path
        self.layer_name = layer_name
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype = dtype
        self.seed = int(seed)
        self.images_per_read = images_per_read
        self.block_rows = block_rows
        self.h5_rdcc_nbytes = int(h5_rdcc_nbytes)
        self.h5_rdcc_nslots = int(h5_rdcc_nslots)
        self.pin_memory = bool(pin_memory)
        self.token_filter = token_filter

        self._rng = np.random.default_rng(self.seed)
        self._h5 = None
        self._dset = None
        self._cache_flat = None
        self._cache_pos = 0

        self.ndim = None
        self.in_dim = None
        self.n_rows = None
        self.n_images = None
        self.seq_len = None
        self._token_slice = None

    def _ensure_open(self):
        if self._h5 is not None:
            return
        self._h5 = h5py.File(
            self.h5_path, "r",
            rdcc_nbytes=self.h5_rdcc_nbytes, rdcc_nslots=self.h5_rdcc_nslots,
        )
        if "X" not in self._h5:
            raise KeyError("HDF5 file missing group 'X'.")
        xgrp = self._h5["X"]
        if self.layer_name not in xgrp:
            raise KeyError(f"Layer '{self.layer_name}' not found under 'X'.")

        self._dset = xgrp[self.layer_name]
        self.ndim = int(self._dset.ndim)
        self.in_dim = int(self._dset.shape[-1])

        if self.token_filter == "image_only" and "image_token_slice" in self._h5.attrs:
            try:
                s = _h5_attr_json_load(self._h5.attrs["image_token_slice"])
                self._token_slice = (int(s["start"]), int(s["end"]))
            except Exception:
                self._token_slice = None
        else:
            self._token_slice = None

        if self.ndim == 2:
            self.n_rows = int(self._dset.shape[0])
            if self.block_rows is None:
                self.block_rows = max(256_000, 64 * self.batch_size)
        elif self.ndim == 3:
            self.n_images = int(self._dset.shape[0])
            self.seq_len = int(self._dset.shape[1])
            if self.images_per_read is None:
                eff_seq = self.seq_len
                if self._token_slice is not None:
                    eff_seq = max(1, self._token_slice[1] - self._token_slice[0])
                self.images_per_read = max(64, math.ceil((8 * self.batch_size) / max(1, eff_seq)))
        else:
            raise ValueError(f"Unsupported ndim={self.ndim}")

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None
                self._dset = None
        self._cache_flat = None
        self._cache_pos = 0

    def __iter__(self):
        return self

    def _refill_cache_ndim2(self):
        n = self.n_rows
        br = int(self.block_rows)
        if br >= n:
            block = self._dset[:, :].astype(np.float32, copy=False)
        else:
            start = int(self._rng.integers(0, n - br))
            block = self._dset[start:start + br, :].astype(np.float32, copy=False)
        self._rng.shuffle(block, axis=0)
        self._cache_flat = block
        self._cache_pos = 0

    def _refill_cache_ndim3(self):
        nimg = self.n_images
        bimg = int(self.images_per_read)
        if bimg >= nimg:
            block = self._dset[:, :, :].astype(np.float32, copy=False)
        else:
            start = int(self._rng.integers(0, nimg - bimg))
            block = self._dset[start:start + bimg, :, :].astype(np.float32, copy=False)

        if self._token_slice is not None:
            s0, s1 = self._token_slice
            s0 = max(0, min(s0, block.shape[1]))
            s1 = max(0, min(s1, block.shape[1]))
            block = block[:, s0:s1, :]

        flat = block.reshape(-1, self.in_dim)
        self._rng.shuffle(flat, axis=0)
        self._cache_flat = flat
        self._cache_pos = 0

    def __next__(self):
        self._ensure_open()
        if self._cache_flat is None or (self._cache_pos + self.batch_size > self._cache_flat.shape[0]):
            if self.ndim == 2:
                self._refill_cache_ndim2()
            else:
                self._refill_cache_ndim3()
        batch_np = self._cache_flat[self._cache_pos: self._cache_pos + self.batch_size]
        self._cache_pos += self.batch_size
        batch = t.from_numpy(batch_np)
        if self.pin_memory and batch.device.type == "cpu":
            batch = batch.pin_memory()
        return batch.to(device=self.device, dtype=self.dtype, non_blocking=True)


# ------------------------------ CLI ------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Train a cascaded SAE (CSAE).")
    # I/O
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--layer_name", type=str, required=True)
    p.add_argument("--model_name", type=str, required=True, help="Backbone tag for bookkeeping, e.g. Qwen.")
    p.add_argument("--dataset_name", type=str, default="dataset", help="Dataset tag for bookkeeping.")
    p.add_argument("--device", type=str, default="cuda:0")
    # CSAE architecture / sparsity
    p.add_argument("--dict_size", type=int, default=20000, help="Level-1 dictionary size (d).")
    p.add_argument("--dict2_rule", type=str, default="half", choices=["half", "same"],
                   help="Level-2 size = d (same) or max(256, d//2) (half).")
    p.add_argument("--k1", type=int, default=20, help="Level-1 BatchTopK target L0.")
    p.add_argument("--k2", type=int, default=None, help="Level-2 target L0 (default: k2_rule).")
    p.add_argument("--k2_rule", type=str, default="half", choices=["half", "same"],
                   help="Default k2 = k1 (same) or max(1, k1//2) (half) when --k2 unset.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    # Schedule (formerly demo_config constants)
    p.add_argument("--num_tokens", type=int, default=500_000_000)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--decay_start_fraction", type=float, default=0.8)
    p.add_argument("--k_anneal_end_fraction", type=float, default=0.1)
    p.add_argument("--sae2_start_step", type=int, default=10000,
                   help="Step at which the Level-2 SAE starts training (lets Level-1 stabilize first).")
    # Data loader
    p.add_argument("--sae_batch_size", type=int, default=4096)
    p.add_argument("--images_per_read", type=int, default=None)
    p.add_argument("--block_rows", type=int, default=None)
    p.add_argument("--h5_cache_mb", type=int, default=512)
    p.add_argument("--token_filter", type=str, default="image_only", choices=["all", "image_only"])
    # Logging / misc
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="csae")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="Build everything but skip training.")
    return p.parse_args()


def build_trainer_config(activation_dim, steps, args, submodule_name):
    dict_size1 = int(args.dict_size)
    dict_size2 = dict_size1 if args.dict2_rule == "same" else max(256, dict_size1 // 2)
    k1 = int(args.k1)
    k2 = int(args.k2) if args.k2 is not None else (k1 if args.k2_rule == "same" else max(1, k1 // 2))
    k_anneal = int(steps * args.k_anneal_end_fraction)
    decay_start = int(steps * args.decay_start_fraction)
    return {
        "trainer": TwoLevelBatchTopKTrainer,
        "wandb_name": args.wandb_name or f"csae_{submodule_name}",
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": int(args.warmup_steps),
        "decay_start": decay_start,
        "device": args.device,
        "layer": args.layer_name,
        "lm_name": args.model_name,
        "submodule_name": submodule_name,
        "seed": int(args.seed),
        "dict_size1": dict_size1,
        "dict_size2": dict_size2,
        "k1": k1,
        "k2": k2,
        "lr1": float(args.lr),
        "lr2": float(args.lr),
        "threshold_beta": 0.999,
        "threshold_start_step1": 1000,
        "threshold_start_step2": 1000,
        "sae2_start_step": int(args.sae2_start_step),
        "k1_anneal_steps": k_anneal,
        "k2_anneal_steps": k_anneal,
    }


def main():
    args = get_args()
    mp.set_start_method("spawn", force=True)
    start_time = time.time()

    with h5py.File(args.data_path, "r") as f:
        if "X" not in f:
            raise KeyError("HDF5 file missing group 'X'.")
        dset = f["X"][args.layer_name]
        activation_dim = int(dset.shape[-1])
        print(f"HDF5 dataset shape: {tuple(dset.shape)}  activation_dim={activation_dim}")

    steps = int(args.num_tokens / args.sae_batch_size)
    print(f"Total training steps: {steps}")

    data_iter = H5StreamingActivationLoader(
        h5_path=args.data_path,
        layer_name=args.layer_name,
        batch_size=int(args.sae_batch_size),
        device=args.device,
        dtype=t.float32,
        seed=int(args.seed),
        images_per_read=args.images_per_read,
        block_rows=args.block_rows,
        h5_rdcc_nbytes=int(args.h5_cache_mb) * 1024 * 1024,
        h5_rdcc_nslots=1_000_003,
        pin_memory=True,
        token_filter=args.token_filter,
    )

    submodule_name = f"{args.dataset_name}_{args.model_name}_layer_{args.layer_name.replace('.', '_')}"
    save_dir_layer = os.path.join(args.save_dir, submodule_name)
    trainer_configs = [build_trainer_config(activation_dim, steps, args, submodule_name)]

    if not args.dry_run:
        # depend-don't-vendor: generic training harness from dictionary_learning.
        from dictionary_learning.training import trainSAE
        # bf16 autocast on CUDA (as in the paper runs); float32 on CPU, where
        # bf16 autocast mixes dtypes in nn.Linear and errors.
        autocast_dtype = t.bfloat16 if str(args.device).startswith("cuda") else t.float32
        trainSAE(
            data=data_iter,
            trainer_configs=trainer_configs,
            use_wandb=args.use_wandb,
            steps=steps,
            save_steps=None,
            save_dir=save_dir_layer,
            log_steps=100,
            wandb_project=args.wandb_project,
            normalize_activations=True,
            verbose=True,
            autocast_dtype=autocast_dtype,
            backup_steps=1000,
        )

    data_iter.close()
    print(f"\n--- Total training time: {time.time() - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
