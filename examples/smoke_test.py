#!/usr/bin/env python3
"""
Self-contained smoke test for CSAE.

Generates a tiny synthetic HDF5 activation file (same layout as
`data_gen/extract_activations.py`: group "X" -> [tokens, dim] dataset), then
exercises the real training path — `H5StreamingActivationLoader` +
`TwoLevelBatchTopKTrainer.update` — for a handful of steps. Checks that the
reconstruction loss decreases and that the enforced BatchTopK sparsity holds.

Does NOT require `dictionary_learning` (it bypasses the trainSAE harness and
drives the trainer's own update loop). Run from the repo root:

    python examples/smoke_test.py
"""
import os
import sys
import tempfile

import h5py
import numpy as np
import torch as t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_csae import H5StreamingActivationLoader
from csae.model import TwoLevelBatchTopKTrainer

DIM, DICT1, DICT2, K1, K2 = 32, 128, 64, 8, 4
N_TOKENS, BATCH, STEPS = 20_000, 256, 60
DEVICE = "cpu"  # keep off shared GPUs; the test is tiny


def make_synthetic_h5(path, layer="model.visual.blocks.23"):
    """Low-rank + noise activations so reconstruction is learnable."""
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((DIM, 6)).astype(np.float32)
    codes = rng.standard_normal((N_TOKENS, 6)).astype(np.float32)
    x = codes @ basis.T + 0.1 * rng.standard_normal((N_TOKENS, DIM)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_group("X").create_dataset(layer, data=x)
    return layer


def main():
    tmp = tempfile.mkdtemp(prefix="csae_smoke_")
    h5_path = os.path.join(tmp, "synthetic.h5")
    layer = make_synthetic_h5(h5_path)
    print(f"[1/3] wrote synthetic activations -> {h5_path}  shape=({N_TOKENS},{DIM})")

    loader = H5StreamingActivationLoader(
        h5_path=h5_path, layer_name=layer, batch_size=BATCH,
        device=DEVICE, seed=0, token_filter="all", pin_memory=False,
    )
    trainer = TwoLevelBatchTopKTrainer(
        steps=STEPS, activation_dim=DIM, dict_size1=DICT1, dict_size2=DICT2,
        k1=K1, k2=K2, layer=0, lm_name="smoke", warmup_steps=5, device=DEVICE,
    )
    print(f"[2/3] loader + CSAE trainer built (d={DICT1}/{DICT2}, k1={K1}, k2={K2})")

    losses = []
    for step in range(1, STEPS + 1):
        x = next(loader)
        losses.append(trainer.update(step, x))

    # Verify enforced sparsity on a fresh batch (TwoLevel: SAE1 on x, SAE2 on atoms)
    ae = trainer.ae
    x = next(loader).to(DEVICE)
    f1 = ae.sae1.encode(x, use_threshold=False)
    atoms = ae.sae1.decoder.weight.T  # Level-1 decoder atoms = SAE2 input
    f2 = ae.sae2.encode(atoms, use_threshold=False)
    k1_eff = (f1 > 0).float().sum(dim=-1).mean().item()
    k2_eff = (f2 > 0).float().sum(dim=-1).mean().item()
    first, last = np.mean(losses[:5]), np.mean(losses[-5:])
    loader.close()

    print(f"[3/3] loss {first:.3f} -> {last:.3f}   "
          f"(per-row L0: f1~{k1_eff:.1f}<= {K1}? , f2~{k2_eff:.1f}<= {K2}?)")

    ok = (last < first) and (k1_eff <= K1 + 1e-6) and (k2_eff <= K2 + 1e-6)
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
