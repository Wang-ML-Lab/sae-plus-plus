"""
CSAE: Cascaded Sparse Autoencoder (canonical two-level model).

This is the model used for the paper's main results and for the steering /
HMS experiments. It is fully self-contained (the small decoder-normalization /
LR-schedule helpers are inlined), so it has no dependency on
`dictionary_learning`'s internal module layout.

Structure:
  - `BatchTopKSAE`           : a single BatchTopK SAE (encoder/decoder + bias).
  - `TwoLevelBatchTopKSAE`   : Level-1 SAE on activations `x`, and Level-2 SAE
                               on the Level-1 decoder atoms (columns of W1).
  - `TwoLevelBatchTopKTrainer`: joint end-to-end trainer.

The high-level (Level-2) SAE operates on `sae1`'s decoder weight columns
(dictionary atoms), i.e. "concepts of concepts" — see the paper, Sec. 3.2.
"""

import math
from typing import Optional

import torch as t
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Helpers (inlined from dictionary_learning.trainers.trainer for self-containment)
# --------------------------------------------------------------------------------------

class SAETrainer:
    """Generic base class for SAE training algorithms (duck-typed for trainSAE)."""

    def __init__(self, seed=None):
        self.seed = seed
        self.logging_parameters = []

    def update(self, step, activations):
        pass

    def get_logging_parameters(self):
        stats = {}
        for param in self.logging_parameters:
            if hasattr(self, param):
                stats[param] = getattr(self, param)
            else:
                print(f"Warning: {param} not found in {self}")
        return stats

    @property
    def config(self):
        return {"wandb_name": "trainer"}


@t.no_grad()
def set_decoder_norm_to_unit_norm(W_dec_DF: t.Tensor, activation_dim: int, d_sae: int) -> t.Tensor:
    """Rescale decoder columns to unit norm. Dims passed in to catch the
    transposed-nn.Linear footgun."""
    D, Fdim = W_dec_DF.shape
    assert D == activation_dim
    assert Fdim == d_sae
    eps = t.finfo(W_dec_DF.dtype).eps
    norm = t.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


@t.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    W_dec_DF: t.Tensor, W_dec_DF_grad: t.Tensor, activation_dim: int, d_sae: int
) -> t.Tensor:
    """Project out the gradient component parallel to each (unit-norm) decoder column."""
    D, Fdim = W_dec_DF.shape
    assert D == activation_dim
    assert Fdim == d_sae
    normed = W_dec_DF / (t.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)
    parallel = (W_dec_DF_grad * normed).sum(dim=0, keepdim=True)
    W_dec_DF_grad -= parallel * normed
    return W_dec_DF_grad


def get_lr_schedule(total_steps: int, warmup_steps: int, decay_start: Optional[int] = None):
    """Linear warmup, constant, then optional linear decay to 0."""
    if decay_start is not None:
        assert 0 <= decay_start < total_steps, "decay_start must be >= 0 and < steps."
        assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
    assert 0 <= warmup_steps < total_steps, "warmup_steps must be >= 0 and < steps."

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if decay_start is not None and step >= decay_start:
            return (total_steps - step) / (total_steps - decay_start)
        return 1.0

    return lr_schedule


# --------------------------------------------------------------------------------------
# Single BatchTopK SAE
# --------------------------------------------------------------------------------------

class BatchTopKSAE(nn.Module):
    """A single BatchTopK SAE: ReLU encoder + unit-norm decoder, batch-top-k sparsity."""

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = int(activation_dim)
        self.dict_size = int(dict_size)

        self.register_buffer("k", t.tensor(k, dtype=t.float32))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(self.dict_size, self.activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight.data, self.activation_dim, self.dict_size
        )
        self.encoder = nn.Linear(self.activation_dim, self.dict_size, bias=True)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(t.zeros(self.activation_dim))

    def encode(self, x: t.Tensor, return_active: bool = False, use_threshold: bool = True):
        post_relu = F.relu(self.encoder(x - self.b_dec))
        if use_threshold:
            f = post_relu * (post_relu > self.threshold)
        else:
            flat = post_relu.flatten()
            k_total = int(self.k.item() * x.size(0))
            k_total = max(1, min(k_total, flat.numel()))
            topk = flat.topk(k_total, sorted=False)
            f = t.zeros_like(flat).scatter_(-1, topk.indices, topk.values).view_as(post_relu)
        if return_active:
            active_F = (f.sum(0) > 0)
            return f, active_F, post_relu
        return f

    def decode(self, f: t.Tensor) -> t.Tensor:
        return self.decoder(f) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        f = self.encode(x, return_active=False, use_threshold=False)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        return x_hat

    def scale_biases(self, scale: float):
        scale = float(scale)
        self.encoder.bias.data.mul_(scale)
        self.b_dec.data.mul_(scale)
        if self.threshold.item() >= 0:
            self.threshold.mul_(scale)

    @classmethod
    def from_pretrained(cls, path, k=None, device=None, **kwargs) -> "BatchTopKSAE":
        state_dict = t.load(path, map_location="cpu")
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = int(state_dict["k"].item())
        ae = cls(int(activation_dim), int(dict_size), int(k))
        ae.load_state_dict(state_dict, strict=True)
        if device is not None:
            ae.to(device)
        return ae


# --------------------------------------------------------------------------------------
# Two-level cascaded SAE (CSAE)
# --------------------------------------------------------------------------------------

class TwoLevelBatchTopKSAE(nn.Module):
    """
    SAE1 learns on activations x. SAE2 learns on SAE1 decoder atoms (columns of
    W1, i.e. rows of W1^T). `live_mask*` are current-forward-pass masks stored as
    buffers so they travel inside the checkpoint.
    """

    def __init__(self, activation_dim: int, dict_size1: int, dict_size2: int, k1: int, k2: int):
        super().__init__()
        self.activation_dim = int(activation_dim)
        self.dict_size1 = int(dict_size1)
        self.dict_size2 = int(dict_size2)

        self.sae1 = BatchTopKSAE(self.activation_dim, self.dict_size1, int(k1))
        self.sae2 = BatchTopKSAE(self.activation_dim, self.dict_size2, int(k2))

        self.register_buffer("live_mask1", t.zeros(self.dict_size1, dtype=t.bool))
        self.register_buffer("live_mask2", t.zeros(self.dict_size2, dtype=t.bool))
        self.register_buffer("live_n1", t.tensor(0, dtype=t.int))
        self.register_buffer("live_n2", t.tensor(0, dtype=t.int))

    def scale_biases(self, scale: float):
        self.sae1.scale_biases(scale)
        self.sae2.scale_biases(scale)

    @t.no_grad()
    def set_live1(self, active_F: t.Tensor):
        self.live_mask1.copy_(active_F)
        self.live_n1.fill_(int(active_F.sum().item()))

    @t.no_grad()
    def set_live2(self, active_F: t.Tensor):
        self.live_mask2.copy_(active_F)
        self.live_n2.fill_(int(active_F.sum().item()))

    @t.no_grad()
    def clear_live2(self):
        self.live_mask2.zero_()
        self.live_n2.zero_()

    @property
    def live_idx1(self) -> t.Tensor:
        return self.live_mask1.nonzero(as_tuple=False).flatten()

    @property
    def live_idx2(self) -> t.Tensor:
        return self.live_mask2.nonzero(as_tuple=False).flatten()


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------

class TwoLevelBatchTopKTrainer(SAETrainer):
    """Joint end-to-end trainer for the two-level CSAE."""

    def __init__(
        self,
        steps: int,
        activation_dim: int,
        dict_size1: int,
        dict_size2: int,
        k1: int,
        k2: int,
        layer,
        lm_name: str,
        submodule_name: Optional[str] = None,
        wandb_name: str = "TwoLevelBatchTopK",
        lr1: Optional[float] = None,
        lr2: Optional[float] = None,
        auxk_alpha1: float = 1 / 32,
        auxk_alpha2: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,
        threshold_beta: float = 0.999,
        threshold_start_step1: int = 1000,
        threshold_start_step2: int = 1000,
        sae2_start_step: int = 0,
        sae2_update_every: int = 1,
        sae2_use_alive_only: bool = True,
        sae2_min_alive: int = 1,
        sae2_loss_alpha: float = 1.0,
        k1_anneal_steps: Optional[int] = None,
        k2_anneal_steps: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        freeze_l1_ckpt: Optional[str] = None,
    ):
        super().__init__(seed)
        assert lm_name is not None and layer is not None

        self.steps = int(steps)
        self.activation_dim = int(activation_dim)
        self.dict_size1 = int(dict_size1)
        self.dict_size2 = int(dict_size2)
        self.k1_target = int(k1)
        self.k2_target = int(k2)

        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name

        self.auxk_alpha1 = float(auxk_alpha1)
        self.auxk_alpha2 = float(auxk_alpha2)
        self.sae2_loss_alpha = float(sae2_loss_alpha)

        self.warmup_steps = int(warmup_steps)
        self.decay_start = int(decay_start) if decay_start is not None else None
        self.threshold_beta = float(threshold_beta)
        self.threshold_start_step1 = int(threshold_start_step1)
        self.threshold_start_step2 = int(threshold_start_step2)

        self.sae2_start_step = int(sae2_start_step)
        self.sae2_update_every = int(max(1, sae2_update_every))
        self.sae2_use_alive_only = bool(sae2_use_alive_only)
        self.sae2_min_alive = int(max(1, sae2_min_alive))

        self.k1_anneal_steps = int(k1_anneal_steps) if k1_anneal_steps is not None else None
        self.k2_anneal_steps = int(k2_anneal_steps) if k2_anneal_steps is not None else None

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.device = device if device is not None else ("cuda" if t.cuda.is_available() else "cpu")

        self.ae = TwoLevelBatchTopKSAE(
            activation_dim=self.activation_dim,
            dict_size1=self.dict_size1,
            dict_size2=self.dict_size2,
            k1=self.k1_target,
            k2=self.k2_target,
        ).to(self.device)

        # Optionally load a pretrained, frozen Level-1 SAE
        self.freeze_l1 = freeze_l1_ckpt is not None
        if freeze_l1_ckpt is not None:
            state = t.load(freeze_l1_ckpt, map_location="cpu")
            l1_state = {k.replace("sae1.", ""): v for k, v in state.items() if k.startswith("sae1.")}
            self.ae.sae1.load_state_dict(l1_state, strict=False)
            for p in self.ae.sae1.parameters():
                p.requires_grad_(False)
            print(f"[Freeze L1] Loaded from {freeze_l1_ckpt}, L1 frozen")

        if lr1 is None:
            lr1 = 2e-4 / math.sqrt(self.dict_size1 / (2 ** 14))
        if lr2 is None:
            lr2 = 2e-4 / math.sqrt(self.dict_size2 / (2 ** 14))
        self.lr1 = float(lr1)
        self.lr2 = float(lr2)

        self.logging_parameters = ["k1", "k2", "alive1", "alive2", "alive_for_sae2", "do_sae2"]
        self.k1 = int(self.ae.sae1.k.item())
        self.k2 = int(self.ae.sae2.k.item())
        self.alive1 = 0
        self.alive2 = 0
        self.alive_for_sae2 = 0
        self.do_sae2 = 0

        l1_lr = 0.0 if self.freeze_l1 else self.lr1
        self.optimizer = t.optim.Adam(
            [
                {"params": self.ae.sae1.parameters(), "lr": l1_lr},
                {"params": self.ae.sae2.parameters(), "lr": self.lr2},
            ],
            betas=(0.9, 0.999),
        )
        lr_fn = get_lr_schedule(self.steps, self.warmup_steps, decay_start=self.decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    @staticmethod
    def _update_threshold(sae: BatchTopKSAE, f: t.Tensor, beta: float):
        device_type = "cuda" if f.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = f[f > 0]
            min_activation = 0.0 if active.numel() == 0 else active.min().detach().to(dtype=t.float32)
            if sae.threshold.item() < 0:
                sae.threshold = min_activation
            else:
                sae.threshold = beta * sae.threshold + (1 - beta) * min_activation

    def _anneal_k(self, sae: BatchTopKSAE, step: int, k_target: int, k_anneal_steps: Optional[int]):
        if k_anneal_steps is None or k_anneal_steps <= 0:
            return
        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        sae.k.fill_(float(self.activation_dim * (1 - ratio) + k_target * ratio))

    def _sae2_training_matrix_no_sg(self) -> t.Tensor:
        """Level-2 input = Level-1 decoder atoms (W1 columns), restricted to the
        current-step live atoms (dynamic masking of dead latents)."""
        W1 = self.ae.sae1.decoder.weight.T
        if not self.sae2_use_alive_only:
            self.alive_for_sae2 = int(W1.shape[0])
            return W1
        live = self.ae.live_mask1
        n_live = int(live.sum().item())
        if n_live >= self.sae2_min_alive:
            self.alive_for_sae2 = n_live
            return W1[live]
        self.alive_for_sae2 = int(W1.shape[0])
        return W1

    def loss(self, x: t.Tensor, step: Optional[int] = None, logging: bool = False):
        if step is None:
            step = 0
        x = x.to(self.device)

        # ---- SAE1 (on activations) ----
        f1, active1_F, _ = self.ae.sae1.encode(x, return_active=True, use_threshold=False)
        self.ae.set_live1(active1_F)
        self.alive1 = int(self.ae.live_n1.item())
        if step > self.threshold_start_step1:
            self._update_threshold(self.ae.sae1, f1, self.threshold_beta)
        x_hat1 = self.ae.sae1.decode(f1)
        loss1 = (x - x_hat1).pow(2).sum(dim=-1).mean()

        # ---- SAE2 (on Level-1 decoder atoms, scheduled) ----
        self.do_sae2 = int(
            step >= self.sae2_start_step
            and ((step - self.sae2_start_step) % self.sae2_update_every == 0)
        )
        l2_2 = t.tensor(0.0, device=self.device, dtype=x.dtype)
        loss2 = t.tensor(0.0, device=self.device, dtype=x.dtype)
        if self.do_sae2:
            M = self._sae2_training_matrix_no_sg()
            f2, active2_F, _ = self.ae.sae2.encode(M, return_active=True, use_threshold=False)
            self.ae.set_live2(active2_F)
            self.alive2 = int(self.ae.live_n2.item())
            if step > self.threshold_start_step2:
                self._update_threshold(self.ae.sae2, f2, self.threshold_beta)
            M_hat = self.ae.sae2.decode(f2)
            l2_2 = (M - M_hat).pow(2).sum(dim=-1).mean()
            loss2 = l2_2
        else:
            self.ae.clear_live2()
            self.alive2 = 0
            self.alive_for_sae2 = 0

        total = loss1 + self.sae2_loss_alpha * loss2
        self.k1 = int(self.ae.sae1.k.item())
        self.k2 = int(self.ae.sae2.k.item())

        if not logging:
            return total
        return (
            x, x_hat1, f1,
            {
                "loss_total": float(total.item()),
                "loss_1": float(loss1.item()),
                "loss_2": float(loss2.item()),
                "l2_2": float(l2_2.item()),
                "k1": int(self.k1), "k2": int(self.k2),
                "alive1": int(self.alive1), "alive2": int(self.alive2),
                "alive_for_sae2": int(self.alive_for_sae2), "do_sae2": int(self.do_sae2),
                "lr_group0": float(self.optimizer.param_groups[0]["lr"]),
                "lr_group1": float(self.optimizer.param_groups[1]["lr"]),
            },
        )

    def update(self, step: int, x: t.Tensor):
        x = x.to(self.device)

        # Geometric-median initialization of the decoder biases
        if step == 1:
            with t.no_grad():
                self.ae.sae1.b_dec.data = self.geometric_median(x)
        if step == self.sae2_start_step:
            with t.no_grad():
                W1_data = self.ae.sae1.decoder.weight.T.detach()
                self.ae.sae2.b_dec.data = self.geometric_median(W1_data)
                print(f"[Init] SAE2 bias = geometric median of W1 at step {step}")

        self.optimizer.zero_grad(set_to_none=True)
        total = self.loss(x, step=step, logging=False)
        total.backward()

        if self.ae.sae1.decoder.weight.grad is not None:
            self.ae.sae1.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.sae1.decoder.weight, self.ae.sae1.decoder.weight.grad,
                self.ae.sae1.activation_dim, self.ae.sae1.dict_size,
            )
        if self.ae.sae2.decoder.weight.grad is not None:
            self.ae.sae2.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.sae2.decoder.weight, self.ae.sae2.decoder.weight.grad,
                self.ae.sae2.activation_dim, self.ae.sae2.dict_size,
            )

        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self._anneal_k(self.ae.sae1, step, self.k1_target, self.k1_anneal_steps)
        self._anneal_k(self.ae.sae2, step, self.k2_target, self.k2_anneal_steps)

        self.ae.sae1.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.sae1.decoder.weight.data, self.ae.sae1.activation_dim, self.ae.sae1.dict_size
        )
        self.ae.sae2.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.sae2.decoder.weight.data, self.ae.sae2.activation_dim, self.ae.sae2.dict_size
        )
        return float(total.item())

    @property
    def config(self):
        return {
            "trainer_class": "TwoLevelBatchTopKTrainer",
            "wandb_name": self.wandb_name,
            "lm_name": self.lm_name,
            "layer": self.layer,
            "submodule_name": self.submodule_name,
            "device": self.device,
            "steps": self.steps,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "activation_dim": self.activation_dim,
            "dict_size1": self.dict_size1,
            "dict_size2": self.dict_size2,
            "k1_target": self.k1_target,
            "k2_target": self.k2_target,
            "lr1": self.lr1,
            "lr2": self.lr2,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step1": self.threshold_start_step1,
            "threshold_start_step2": self.threshold_start_step2,
            "sae2_start_step": self.sae2_start_step,
            "sae2_update_every": self.sae2_update_every,
            "sae2_use_alive_only": self.sae2_use_alive_only,
            "sae2_min_alive": self.sae2_min_alive,
            "k1_anneal_steps": self.k1_anneal_steps,
            "k2_anneal_steps": self.k2_anneal_steps,
            "sae2_loss_alpha": self.sae2_loss_alpha,
            "seed": self.seed,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break
        return guess
