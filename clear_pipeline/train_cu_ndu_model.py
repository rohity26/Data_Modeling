"""train_cu_ndu_model.py — Retrain CalibratedTopoAR with N_DU added as an 8th CU
feature, on the updated (150,437-row / ~50h) train_normal_tcp.csv.

WHY N_DU AS A FEATURE
----------------------
Today, N_DU (how many DUs a CU serves) is only used OUTSIDE the model, as a
fixed divisor applied to CU net_tx/net_rx before the model ever sees the data
(see slice_features() below: cu[:,5] /= N_DU). The model itself never knows
how many DUs a given CU serves, so it has to trust that a straight linear
per-DU division is the right way to make CU traffic comparable across
topologies with 1, 2, 3, or 4 DUs. Feeding N_DU in explicitly lets the shared
model learn that scaling relationship itself instead of us assuming it's
linear -- this was the senior's suggestion, and it requires retraining because
it changes cu_dim from 7 to 8 (the existing checkpoints cannot be reused).

TRAINING ONLY
-------------
This script does not evaluate -- it only builds the 4 leave-one-out
checkpoints. Use run_experiment_cpu_mem_70pct.py (pointed at CU_NDU_TCP_STRESS
and CKPT_DIR below) to evaluate them afterward.

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python3 -u train_cu_ndu_model.py 2>&1 | tee train_cu_ndu_model.log
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR
from dataset import TopologySequenceDataset, MultiTopologyBatchSampler, collate_windows

# =============================================================================
# USER INPUTS
# =============================================================================

ALL_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]

BASE_DIR = Path("CU_NDU_TCP_STRESS")     # train.npz per topology, built from the updated 150,437-row CSV
CKPT_DIR = Path("model_TCP_NDU_backup")  # NEW folder -- does not touch model_TCP_70pct_backup
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_VARIANT = "random"                  # keep the same prefix convention as existing checkpoints
CKPT_PREFIX = f"{DATA_VARIANT}_"

# Same feature slices as the current evaluation pipeline (run_experiment_cpu_mem_70pct.py).
CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6,
                 7, 8, 9, 10, 11, 12, 13, 14,
                 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]

CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]

PREPROCESS_VERSION = "v0"
CU_ZV_IDX = []
DU_ZV_IDX = []

EMBED_DIM  = 32
WINDOW_LEN = 64
BATCH_SIZE = 256
EPOCHS     = 150
PATIENCE   = 5
LR         = 5e-4
VAL_FRAC   = 0.1
CAL_FRAC   = 0.2     # kept for parity with the eval pipeline's fit/cal split; unused for training loss
SEED       = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_DU_OF = {"cu0_du0du1": 2, "cu1_du2": 1, "cu2_du3du4du5": 3, "cu3_du6du7du8du9": 4}

# =============================================================================
# HELPERS
# =============================================================================

def load_npz(topo: str) -> dict:
    p = BASE_DIR / topo / "train.npz"
    assert p.exists(), f"File not found: {p}"
    return dict(np.load(p))


def impute_cpu_glitch(arr: np.ndarray, irate_idx: list, eps: float = 1e-6) -> np.ndarray:
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
        else:
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t - 1, :, irate_idx], arr[t, :, irate_idx])
    return arr


def slice_features(z: dict, topo: str):
    """Same as run_experiment_cpu_mem_70pct.py's slice_features(), plus N_DU
    appended as an 8th CU feature (constant per topology, broadcast over T)."""
    cu = z["cu"].astype(np.float32)
    du = z["du"].astype(np.float32)
    N_DU = du.shape[1]

    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]

    cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
    du = impute_cpu_glitch(du, DU_IRATE_IDX)

    _tx = cu[:, 3:4]
    _rx = cu[:, 4:5]
    net_diff = _tx - _rx
    net_ratio = _tx / (_rx + 1e-6)

    # NEW: N_DU as an 8th CU feature -- constant per topology, one column, same
    # value for every row. RobustScaler (v0) will fit its own scale for this
    # column just like any other; the model is what actually learns to use it.
    n_du_col = np.full((cu.shape[0], 1), float(N_DU_OF[topo]), dtype=np.float32)

    cu = np.concatenate([cu, net_diff, net_ratio, n_du_col], axis=1)   # (T, 8)

    _du_tx = du[:, :, 4:5]
    _du_rx = du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)   # (T, N, 30) -- unchanged

    block_id = np.zeros(len(cu), dtype=np.int64)
    return cu, du, block_id


def phase_preprocess(train_zs, train_topos):
    raw_streams = []
    for z, topo in zip(train_zs, train_topos):
        cu, du, bid = slice_features(z, topo)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})

    bundle = fit_bundle(raw_streams, CU_ZV_IDX, DU_ZV_IDX, version=PREPROCESS_VERSION)

    streams = []
    for i, raw in enumerate(raw_streams):
        cu_s, du_s, _, kept_bid = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        print(f"  topo[{i}] {train_topos[i]:18s}  cu_s {cu_s.shape}  du_s {du_s.shape}  "
              f"(mu={cu_s.mean():+.3f}, sigma={cu_s.std():.3f})")
        streams.append((cu_s, du_s, kept_bid))
    return bundle, streams


def phase_train(fit_streams, train_topos, model_ckpt):
    cu_dim = fit_streams[0][0].shape[1]
    du_dim = fit_streams[0][1].shape[2]
    print(f"  cu_dim={cu_dim} (expect 8, was 7 before adding N_DU)  du_dim={du_dim}")

    train_subsets, val_subsets = [], []
    train_lens, val_lens = [], []
    rng = np.random.RandomState(SEED)
    for i, (cu_s, du_s, bid) in enumerate(fit_streams):
        ds = TopologySequenceDataset(cu_s, du_s, bid, window_len=WINDOW_LEN, stride=1)
        n = len(ds)
        perm = rng.permutation(n)
        n_val = max(1, int(round(VAL_FRAC * n)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_subsets.append(torch.utils.data.Subset(ds, train_idx))
        val_subsets.append(torch.utils.data.Subset(ds, val_idx))
        train_lens.append(len(train_idx))
        val_lens.append(len(val_idx))
        print(f"  topo[{i}] {train_topos[i]:18s} N_DU={du_s.shape[1]}  "
              f"windows={n}  (train={len(train_idx)}, val={len(val_idx)})")

    train_concat = torch.utils.data.ConcatDataset(train_subsets)
    val_concat = torch.utils.data.ConcatDataset(val_subsets)

    train_loader = DataLoader(
        train_concat,
        batch_sampler=MultiTopologyBatchSampler(train_lens, BATCH_SIZE, shuffle=True, seed=SEED),
        collate_fn=collate_windows,
    )
    val_loader = DataLoader(
        val_concat,
        batch_sampler=MultiTopologyBatchSampler(val_lens, BATCH_SIZE, shuffle=False, seed=SEED),
        collate_fn=collate_windows,
    )

    torch.manual_seed(SEED)
    model = CalibratedTopoAR(cu_dim=cu_dim, du_dim=du_dim, embed_dim=EMBED_DIM).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            cu_b = batch["cu"].to(DEVICE)
            du_b = batch["du"].to(DEVICE)
            cu_hat, du_hat = model(cu_b, du_b)
            loss = (
                ((cu_hat[:, :-1] - cu_b[:, 1:]) ** 2).mean() +
                ((du_hat[:, :-1] - du_b[:, 1:]) ** 2).mean()
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cu_b = batch["cu"].to(DEVICE)
                du_b = batch["du"].to(DEVICE)
                cu_hat, du_hat = model(cu_b, du_b)
                val_loss += (
                    ((cu_hat[:, :-1] - cu_b[:, 1:]) ** 2).mean() +
                    ((du_hat[:, :-1] - du_b[:, 1:]) ** 2).mean()
                ).item()
        val_loss /= max(len(val_loader), 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  tr={tr_loss/len(train_loader):.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  (best_val={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    n_train_rows = sum(len(f[0]) for f in fit_streams)
    torch.save({"state_dict": best_state,
                "cu_dim": cu_dim, "du_dim": du_dim, "embed_dim": EMBED_DIM,
                "cal_frac": CAL_FRAC, "n_train_rows": n_train_rows,
                "topos": list(train_topos),
                "preprocess": PREPROCESS_VERSION,
                "has_n_du_feature": True}, model_ckpt)
    print(f"  Model saved -> {model_ckpt}")
    return model


def run_one_fold(test_topo: str):
    train_topos = [t for t in ALL_TOPOS if t != test_topo]
    model_ckpt = CKPT_DIR / f"{CKPT_PREFIX}model_ckpt_test_{test_topo}.pt"

    print(f"\n{'='*70}")
    print(f"  LOO fold: held out = {test_topo}")
    print(f"  Train on: {train_topos}")
    print(f"  Device  : {DEVICE}")
    print(f"{'='*70}")

    print(f"\n[1] Loading {len(train_topos)} train topologies, fitting {PREPROCESS_VERSION} scaler ...")
    train_zs = [load_npz(t) for t in train_topos]
    bundle, train_streams = phase_preprocess(train_zs, train_topos)

    print(f"\n[2] Per-topology fit/cal split (cal_frac={CAL_FRAC}, cal unused here -- training only) ...")
    fit_streams = []
    for i, (cu_s, du_s, kept_bid) in enumerate(train_streams):
        n_total = len(cu_s)
        n_cal = int(round(CAL_FRAC * n_total))
        n_fit = n_total - n_cal
        fit_streams.append((cu_s[:n_fit], du_s[:n_fit], kept_bid[:n_fit]))
        print(f"  topo[{i}] {train_topos[i]:18s} total={n_total}  fit={n_fit}  cal(unused)={n_cal}")

    if model_ckpt.exists():
        print(f"\n[3] Checkpoint already exists at {model_ckpt} -- skipping (delete it to retrain).")
        return

    print(f"\n[3] Training CalibratedTopoAR ...")
    phase_train(fit_streams, train_topos, model_ckpt)


def main():
    for test_topo in ALL_TOPOS:
        run_one_fold(test_topo)
    print(f"\n\nAll 4 LOO folds done. Checkpoints in {CKPT_DIR}/")


if __name__ == "__main__":
    main()
