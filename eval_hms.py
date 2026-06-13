"""
SAE++ evaluation using the EXACT notebook protocol.
Directly adapted from user's notebook cells.
"""
import os, sys, json, h5py, torch, argparse
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from csae.model import TwoLevelBatchTopKSAE

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--ckpt-path", required=True)
parser.add_argument("--data-path", required=True)
parser.add_argument("--embedding-path", required=True)
parser.add_argument("--layer-name", default="model.visual.blocks.23")
parser.add_argument("--method", default="SAE++")
args = parser.parse_args()

DEVICE = args.device
CKPT_PATH = args.ckpt_path
DATA_PATH = args.data_path
LAYER_NAME = args.layer_name

DATA_SCALE = 1.0
ACT_THR_L1 = 1e-3
ACT_THR_L2 = 1e-3
TOPK_L1_FOR_ALIVE = 12
TOPK_L2_FOR_ALIVE = 1
BATCH_IMAGES_FOR_ALIVE = 128
CHUNK_TOKENS_FOR_ALIVE = 8192
SCAN_BATCH_IMAGES = 16
TOP_K_IMAGES = 10

# ============================================================
# TopKTracker (exact copy from notebook)
# ============================================================
class TopKTracker:
    def __init__(self, top_k, n_units, device):
        self.top_k = int(top_k)
        self.n_units = int(n_units)
        self.device = device
        self.scores = torch.full((n_units, self.top_k), -float("inf"), device=device)
        self.img_ids = torch.full((n_units, self.top_k), -1, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, batch_scores, batch_start_img):
        B, U = batch_scores.shape
        assert U == self.n_units
        img_ids = torch.arange(batch_start_img, batch_start_img + B, device=self.device, dtype=torch.long)
        cand_scores = batch_scores.transpose(0, 1).contiguous()
        cand_imgids = img_ids.view(1, -1).expand(U, -1).contiguous()
        all_scores = torch.cat([self.scores, cand_scores], dim=1)
        all_imgids = torch.cat([self.img_ids, cand_imgids], dim=1)
        topv, topi = torch.topk(all_scores, k=self.top_k, dim=1, largest=True, sorted=True)
        self.scores = topv
        self.img_ids = torch.gather(all_imgids, 1, topi)

# ============================================================
# Load model (exact copy from notebook)
# ============================================================
def load_trained_twolevel_batchtopk(ckpt_path, device="cpu"):
    run_dir = os.path.dirname(ckpt_path)
    trainer_cfg = {}
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            trainer_cfg = json.load(f).get("trainer", {}) or {}
    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob["ae"] if isinstance(blob, dict) and "ae" in blob else blob
    meta = {k: v for k, v in blob.items() if k != "ae"} if isinstance(blob, dict) and "ae" in blob else {}
    w1 = state["sae1.encoder.weight"]
    w2 = state["sae2.encoder.weight"]
    D, H1, H2 = int(w1.shape[1]), int(w1.shape[0]), int(w2.shape[0])
    k1 = int(state["sae1.k"].item()) if "sae1.k" in state else int(trainer_cfg.get("k1", trainer_cfg.get("k1_target")))
    k2 = int(state["sae2.k"].item()) if "sae2.k" in state else int(trainer_cfg.get("k2", trainer_cfg.get("k2_target")))
    model = TwoLevelBatchTopKSAE(activation_dim=D, dict_size1=H1, dict_size2=H2, k1=k1, k2=k2)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), trainer_cfg, meta, (k1, k2)

# ============================================================
# Alive inference (exact copy from notebook)
# ============================================================
@torch.no_grad()
def _post_relu(sae, x):
    return F.relu(sae.encoder(x - sae.b_dec))

@torch.no_grad()
def infer_alive_indices_from_dataset(data_path, layer_name, model, device,
    data_scale=1.0, batch_images=256, chunk_tokens=4096,
    act_thr1=1e-3, act_thr2=1e-3, topk1=20, topk2=20):
    model = model.to(device).eval()
    sae1, sae2 = model.sae1, model.sae2
    H1, H2 = int(sae1.dict_size), int(sae2.dict_size)
    topk1, topk2 = min(int(topk1), H1), min(int(topk2), H2)
    freq1 = torch.zeros((H1,), dtype=torch.long, device="cpu")
    freq2 = torch.zeros((H2,), dtype=torch.long, device="cpu")

    with h5py.File(data_path, "r") as f:
        ds = f["X"][layer_name]
        if "token_offsets" in f:
            offsets = f["token_offsets"][:]
            N_images = len(offsets) - 1
            mode = "flat"
        elif ds.ndim == 3:
            N_images = int(ds.shape[0])
            mode = "3d"
        else:
            raise ValueError(f"Unsupported data shape: {tuple(ds.shape)}")

        for img_start in tqdm(range(0, N_images, batch_images), desc="infer-alive"):
            img_end = min(img_start + batch_images, N_images)
            if mode == "flat":
                x_flat = torch.from_numpy(ds[int(offsets[img_start]):int(offsets[img_end])]).to(device=device, dtype=torch.float32)
            else:
                x_flat = torch.from_numpy(ds[img_start:img_end]).to(device=device, dtype=torch.float32).flatten(0, 1)
            if data_scale != 1.0:
                x_flat = x_flat / data_scale
            for x_chunk in x_flat.split(chunk_tokens, dim=0):
                post1 = _post_relu(sae1, x_chunk)
                v1, i1 = torch.topk(post1, k=topk1, dim=1, largest=True, sorted=False)
                m1 = v1 > act_thr1
                if m1.any():
                    freq1 += torch.bincount(i1[m1].detach().cpu(), minlength=H1)

    alive_idx1 = (freq1 > 0).nonzero(as_tuple=False).flatten().to(device=device, dtype=torch.long)

    # SAE2 alive: from L1 decoder atoms (NOT data)
    atoms = sae1.decoder.weight[:, alive_idx1].T.contiguous()
    post2 = _post_relu(sae2, atoms)
    v2, i2 = torch.topk(post2, k=topk2, dim=1, largest=True, sorted=False)
    m2 = v2 > act_thr2
    if m2.any():
        freq2[i2[m2].unique().cpu()] = 1

    alive_idx2 = (freq2 > 0).nonzero(as_tuple=False).flatten().to(device=device, dtype=torch.long)
    print(f"[Infer alive] L1 alive: {alive_idx1.numel()} / {H1}")
    print(f"[Infer alive] L2 alive: {alive_idx2.numel()} / {H2}")
    return alive_idx1, alive_idx2

# ============================================================
# L1→L2 mapping (exact copy from notebook)
# ============================================================
@torch.no_grad()
def compute_L1_to_L2_map_post2(model, alive_idx1, alive_idx2, device):
    model = model.to(device).eval()
    alive_idx1 = alive_idx1.to(device=device, dtype=torch.long)
    alive_idx2 = alive_idx2.to(device=device, dtype=torch.long)
    W1 = model.sae1.decoder.weight
    atoms = W1[:, alive_idx1].T.contiguous()
    z2_in = atoms - model.sae2.b_dec.view(1, -1)
    post2 = F.relu(model.sae2.encoder(z2_in))
    return post2[:, alive_idx2].contiguous()

# ============================================================
# Scan + collect (exact copy from notebook)
# ============================================================
@torch.no_grad()
def scan_dataset_and_track_and_collect(data_path, layer_name, model,
    alive_idx1, alive_idx2, tracker1, tracker2, mapping_matrix,
    batch_images=256, device="cuda", data_scale=1.0, collect_dtype=torch.float16):
    model = model.to(device).eval()
    alive_idx1 = alive_idx1.to(device=device, dtype=torch.long)
    alive_idx2 = alive_idx2.to(device=device, dtype=torch.long)
    mapping_matrix = mapping_matrix.to(device=device)

    with h5py.File(data_path, "r") as f:
        ds = f["X"][layer_name]
        if "token_offsets" in f:
            offsets = f["token_offsets"][:]
            N_images = len(offsets) - 1
            mode = "flat"
        elif ds.ndim == 3:
            N_images = int(ds.shape[0])
            mode = "3d"
        else:
            raise ValueError(f"Unsupported data shape: {tuple(ds.shape)}")

        L1, L2 = int(alive_idx1.numel()), int(alive_idx2.numel())
        activations_sae1 = torch.empty((N_images, L1), dtype=collect_dtype, device="cpu")

        for img_start in tqdm(range(0, N_images, batch_images), desc="scan+collect"):
            img_end = min(img_start + batch_images, N_images)
            cur_bs = img_end - img_start

            if mode == "flat":
                t0, t1 = int(offsets[img_start]), int(offsets[img_end])
                x_flat = torch.from_numpy(ds[t0:t1]).to(device=device, dtype=torch.float32)
                if data_scale != 1.0: x_flat = x_flat / data_scale
                local_offsets = offsets[img_start:img_end + 1] - t0
            else:
                x_3d = torch.from_numpy(ds[img_start:img_end]).to(device=device, dtype=torch.float32)
                if data_scale != 1.0: x_3d = x_3d / data_scale
                T = int(x_3d.shape[1])
                x_flat = x_3d.flatten(0, 1)

            # sae1.encode() with batch-top-k - chunk to avoid OOM on large d
            chunk_size = 4096  # tokens per chunk
            f1_alive_chunks = []
            f2_alive_chunks = []
            for ci in range(0, x_flat.shape[0], chunk_size):
                ce = min(ci + chunk_size, x_flat.shape[0])
                f1_full_chunk = model.sae1.encode(x_flat[ci:ce], return_active=False, use_threshold=False)
                f1_alive_chunk = f1_full_chunk[:, alive_idx1]
                map_dev = mapping_matrix.to(dtype=f1_alive_chunk.dtype)
                f2_alive_chunk = f1_alive_chunk @ map_dev
                f1_alive_chunks.append(f1_alive_chunk)
                f2_alive_chunks.append(f2_alive_chunk)
                del f1_full_chunk
            f1_alive_tok = torch.cat(f1_alive_chunks, dim=0)
            f2_alive_tok = torch.cat(f2_alive_chunks, dim=0)
            del f1_alive_chunks, f2_alive_chunks

            # Pool tokens -> images
            if mode == "flat":
                f1_batch, f2_batch = [], []
                for i in range(cur_bs):
                    s, e = int(local_offsets[i]), int(local_offsets[i + 1])
                    if e > s:
                        f1_batch.append(f1_alive_tok[s:e].mean(dim=0))
                        f2_batch.append(f2_alive_tok[s:e].mean(dim=0))
                    else:
                        f1_batch.append(torch.zeros((L1,), device=device))
                        f2_batch.append(torch.zeros((L2,), device=device))
                f1_batch = torch.stack(f1_batch)
                f2_batch = torch.stack(f2_batch)
            else:
                f1_batch = f1_alive_tok.view(cur_bs, T, L1).mean(dim=1)
                f2_batch = f2_alive_tok.view(cur_bs, T, L2).mean(dim=1)

            tracker1.update(f1_batch, img_start)
            tracker2.update(f2_batch, img_start)
            activations_sae1[img_start:img_end].copy_(f1_batch.detach().to("cpu", dtype=collect_dtype))

    return activations_sae1

# ============================================================
# HMS Score (exact copy from notebook)
# ============================================================
def calculate_hms_score(tracker, mapping_matrix, alive_idx1, alive_idx2, embeddings, activations):
    l1_strength, l1_parents_idx = mapping_matrix.max(dim=1)
    l1_strength_np = l1_strength.cpu().numpy()
    l1_parents_idx_np = l1_parents_idx.cpu().numpy()
    real_l1_ids = alive_idx1.cpu().numpy()
    real_l2_ids = alive_idx2.cpu().numpy()

    l2_clusters = {}
    for l1_idx, l2_parent in enumerate(l1_parents_idx_np):
        if l1_strength_np[l1_idx] > 1e-3:
            if l2_parent not in l2_clusters: l2_clusters[l2_parent] = []
            l2_clusters[l2_parent].append(l1_idx)

    print("Generating L1 Semantic Prototypes...")
    num_l1 = len(real_l1_ids)
    dino_dim = embeddings.shape[1]
    l1_prototypes = torch.zeros((num_l1, dino_dim), device=embeddings.device)

    for i in range(num_l1):
        acts = activations[:, i]
        a_min, a_max = acts.min(), acts.max()
        range_val = a_max - a_min
        weights = (acts - a_min) / range_val if range_val > 1e-8 else torch.ones_like(acts)
        weights = weights.unsqueeze(1)
        proto = (weights * embeddings).sum(dim=0) / (weights.sum() + 1e-8)
        l1_prototypes[i] = proto

    l1_prototypes = F.normalize(l1_prototypes, p=2, dim=1)

    hms_results = {}
    print("Computing HMS Scores...")
    for l2_idx, children in l2_clusters.items():
        if len(children) < 2: continue
        protos = l1_prototypes[children]
        sim_matrix = torch.mm(protos, protos.t())
        m = len(children)
        indices = torch.triu_indices(m, m, offset=1)
        hms_score = sim_matrix[indices[0], indices[1]].mean().item()
        hms_results[int(real_l2_ids[l2_idx])] = {
            "hms_score": hms_score,
            "cluster_size": m,
            "child_ids": [int(real_l1_ids[c]) for c in children]
        }
    return hms_results

# ============================================================
# Main (exact notebook flow)
# ============================================================
print(f"Loading model from {CKPT_PATH}...")
model, trainer_cfg, meta, (k1, k2) = load_trained_twolevel_batchtopk(CKPT_PATH, device=DEVICE)
print(f"Loaded TwoLevelBatchTopKSAE: k1={k1}, k2={k2}")

print(f"\nLoading embeddings from {args.embedding_path}...")
with h5py.File(args.embedding_path, 'r') as f:
    if 'X' in f:
        x = f['X']
        if isinstance(x, h5py.Group):
            key = list(x.keys())[0]
            embeddings_all = torch.from_numpy(x[key][:]).float()
        else:
            embeddings_all = torch.from_numpy(x[:]).float()
    else:
        key = list(f.keys())[0]
        embeddings_all = torch.from_numpy(f[key][:]).float()
print(f"  Embeddings: {embeddings_all.shape}")

print("\nStep A) Infer alive indices from dataset...")
alive_idx1, alive_idx2 = infer_alive_indices_from_dataset(
    data_path=DATA_PATH, layer_name=LAYER_NAME, model=model, device=DEVICE,
    data_scale=DATA_SCALE, batch_images=BATCH_IMAGES_FOR_ALIVE,
    chunk_tokens=CHUNK_TOKENS_FOR_ALIVE,
    act_thr1=ACT_THR_L1, act_thr2=ACT_THR_L2,
    topk1=TOPK_L1_FOR_ALIVE, topk2=TOPK_L2_FOR_ALIVE)

print("\nStep B) Build L1->L2 map (post2 on SAE1 atoms)...")
L1_to_L2_Map = compute_L1_to_L2_map_post2(model, alive_idx1, alive_idx2, device=DEVICE)
print(f"Map shape: {tuple(L1_to_L2_Map.shape)}")

print("\nStep C) Init trackers...")
tracker1 = TopKTracker(TOP_K_IMAGES, alive_idx1.numel(), device=DEVICE)
tracker2 = TopKTracker(TOP_K_IMAGES, alive_idx2.numel(), device=DEVICE)

print("\nStep D) Scan + track...")
activations_sae1 = scan_dataset_and_track_and_collect(
    data_path=DATA_PATH, layer_name=LAYER_NAME, model=model,
    alive_idx1=alive_idx1, alive_idx2=alive_idx2,
    tracker1=tracker1, tracker2=tracker2, mapping_matrix=L1_to_L2_Map,
    batch_images=SCAN_BATCH_IMAGES, device=DEVICE, data_scale=DATA_SCALE)

print("\nStep E) Compute HMS...")
device_hms = DEVICE
HMS = calculate_hms_score(tracker1, L1_to_L2_Map, alive_idx1, alive_idx2,
                          embeddings_all.to(device_hms), activations_sae1.to(device_hms))

hms_scores = [data['hms_score'] for data in HMS.values()]
if hms_scores:
    mean_hms = np.mean(hms_scores)
    median_hms = np.median(hms_scores)
    min_hms = np.min(hms_scores)
    max_hms = np.max(hms_scores)
else:
    mean_hms = median_hms = min_hms = max_hms = 0.0

n_clusters = len(hms_scores)
print(f"\n{'='*60}")
print(f"HMS STATISTICS ({args.method})")
print(f"-"*60)
print(f"Alive L1:   {alive_idx1.numel()}")
print(f"Alive L2:   {alive_idx2.numel()}")
print(f"L2 Clusters (>=2 children): {n_clusters}")
print(f"HMS_min:    {min_hms:.4f}")
print(f"HMS_med:    {median_hms:.4f}")
print(f"HMS_max:    {max_hms:.4f}")
print(f"HMS_mean:   {mean_hms:.4f}")
print(f"{'='*60}")

for l2_id, data in sorted(HMS.items()):
    print(f"  Parent {l2_id:5d}: HMS={data['hms_score']:.4f}, Size={data['cluster_size']}, Children={data['child_ids']}")
