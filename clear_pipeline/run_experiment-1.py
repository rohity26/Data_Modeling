"""run_experiment.py — Cross-topology CPU stress detection (Step 3, v0 preprocessing).

WHAT IT DOES
------------
  Train on one topology's normal data (train.npz from TRAIN_TOPO).
  Test  on a DIFFERENT topology's mixed data (test.npz from TEST_TOPO).
  Report F1 / precision / recall for CU and each DU.

KEY DESIGN DECISIONS
--------------------
  Preprocessing v0: raw → RobustScaler, NO block_diff.
    Why no block_diff: stress is constant (sustained 80% CPU). After diff, both
    normal-delta≈0 and stress-delta≈0 look identical — the level shift disappears.
    RobustScaler alone preserves the level shift as a large positive scaled value.

  Traffic-invariant features only (DU_FEAT_SLICE = slice(0, 2) = cpu + mem_pct):
    net_tx and dl_brate scale ~2× when traffic doubles — unreliable across topologies.
    cpu and mem_pct are fractions of hardware capacity, comparable across machines.

  Prometheus irate glitch imputation (applied BEFORE RobustScaler):
    irate(container_cpu_user_seconds_total[5m]) returns exactly 0.0 at 5-minute
    clock boundaries when the sliding window crosses a counter reset. These appear
    every 300 seconds in both train and test. The raw 0.0 is physically impossible
    (a process cannot use zero CPU for an entire 1-second scrape interval) and is a
    pure measurement artifact. After RobustScaler, 0.0 maps to (0-median)/IQR ≈ -27
    for DU and ≈ -5 for CU — the model fails to predict these, spiking lift scores
    to ~286 (DU) and ~304 (CU). This inflates the p99.9 threshold to values set
    entirely by artifacts rather than real normal data. Fix: forward-fill (replace
    raw 0.0 with the previous timestep's value) before scaling. The imputation is
    applied identically to train and test so the scaler never sees the outlier.

  Train-side held-out calibration (NO test label leakage):
    The last CAL_FRAC of the TRAIN stream is held out before training. The model
    is fit on the remaining train-train portion only. After training, that held-out
    cal stream (which the model has NEVER seen) is used to compute feat_norm and
    the lift-score threshold. This matches deployment: at inference time we have
    no labels on the live stream, so we cannot use test-normal rows for the
    threshold. Cross-topology baseline shift is now a real failure mode for the
    model, not something the threshold is allowed to hide.

FLOW
----
  [1] Fit preprocessing on TRAIN topology normal data (with CPU glitch imputation).
  [2] Split train stream into (train_fit, cal) by row index — last CAL_FRAC is cal.
  [3] Train CalibratedTopoAR on train_fit only.
  [4] Run inference on the held-out cal stream → calibrate feat_norm + threshold.
  [5] Apply the SAME scalers to TEST topology (no refit; imputation also applied).
  [6] Run sequential LSTM inference on the full TEST stream.
  [7] Score all TEST rows, compare vs ground-truth labels, print metrics.

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python run_experiment.py
"""

import sys
import os
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import fit_bundle, transform_stream, causal_rolling_normalize
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated, dual_threshold_from_val
from dataset import TopologySequenceDataset, MultiTopologyBatchSampler, collate_windows
from scoring import lift_score

# =============================================================================
# USER INPUTS — change these to try different cross-topology pairs
# =============================================================================

ALL_TOPOS     = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]   # all available topologies
TEST_TOPO     = "cu1_du2"                    # held-out topology; change to try a different split
TRAIN_TOPOS   = [t for t in ALL_TOPOS if t != TEST_TOPO]      # auto-derived: all except TEST_TOPO
RUN_ALL_LOO   = True   # True → run all 3 leave-one-out splits sequentially and print a summary table

BASE_DIR      = Path("CU_NET_random_STRESS")
STRESS_TYPE   = 3         # 1=CPU | 2=MEM | 3=NET  — must match the test dataset
STRESS_NAMES  = {1: "CPU", 2: "MEM", 3: "NET"}

# Feature slices — all KPIs minus permanently-zero features.
CU_FEAT_SLICE = [0, 1, 2, 5, 6]           # cpu, mem_pct, mem_bytes, net_tx, net_rx  (5 features)
DU_FEAT_SLICE = list(range(25))

# Irate/counter indices in the POST-SLICE arrays
CU_IRATE_IDX = [0, 3, 4]    # cpu, net_tx, net_rx
DU_IRATE_IDX = [0, 3, 4, 5] # cpu, fs_writes, net_tx, net_rx

# Preprocessing: RobustScaler v0
PREPROCESS_VERSION = "v0"
CU_ZV_IDX = []
DU_ZV_IDX = []

# Model hyperparameters
EMBED_DIM     = 32
WINDOW_LEN    = 64
BATCH_SIZE    = 256
EPOCHS        = 150
PATIENCE      = 5
LR            = 5e-4
VAL_FRAC      = 0.1             # window-level val for early stopping inside training
CAL_FRAC      = 0.2             # tail fraction of train stream held out for threshold calibration
SEED          = 42

COLD_START_K  = WINDOW_LEN

# Separate thresholds per entity type
# Set CU threshold to 99.5 to optimize Precision/Recall after masking net_ratio
CU_THRESHOLD_PCT = 99.5         # percentile for CU threshold
DU_THRESHOLD_PCT = 99.9         # percentile for DU threshold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BIDIR = "bidir" in BASE_DIR.name   # derived from BASE_DIR name; True when directory contains "bidir"

DATA_VARIANT = "_".join(BASE_DIR.name.split("_")[2:-1])
CKPT_PREFIX  = "bidr_" if BIDIR else (f"{DATA_VARIANT}_" if DATA_VARIANT else "")

IMPUTE = True
SAVE_ERRORS = True

# =============================================================================
# HELPERS
# =============================================================================

def load_npz(topo: str, split: str) -> dict:
    p = BASE_DIR / f"{topo}_stress{STRESS_TYPE}" / f"{split}.npz"
    assert p.exists(), f"File not found: {p}"
    return dict(np.load(p))


def impute_cpu_glitch(arr: np.ndarray, irate_idx: list, eps: float = 1e-6) -> np.ndarray:
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:  # CU: (T, dim)
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
        else:              # DU: (T, N, dim)
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t - 1, :, irate_idx], arr[t, :, irate_idx])
    return arr

def slice_features(z: dict):
    """Apply CU/DU feature slices and return (cu, du, block_id)."""
    cu       = z["cu"].astype(np.float32)
    du       = z["du"].astype(np.float32)
    
    N_DU = du.shape[1]

    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]
    if IMPUTE:
        cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
        du = impute_cpu_glitch(du, DU_IRATE_IDX)

    _tx = cu[:, 3:4]
    _rx = cu[:, 4:5]
    cu = np.concatenate([cu,
                         _tx - _rx,                 # net_diff  → position 5
                         _tx / (_rx + 1e-6)],       # net_ratio → position 6
                        axis=1)

    _du_tx = du[:, :, 4:5]
    _du_rx = du[:, :, 5:6]
    du = np.concatenate([du,
                         _du_tx - _du_rx,             # net_diff  → last-1 position
                         _du_tx / (_du_rx + 1e-6)],   # net_ratio → last position
                        axis=2)

    block_id = z["block_id"].astype(np.int64)
    return cu, du, block_id

# =============================================================================
# PHASE 1: PREPROCESSING
# =============================================================================

def phase_preprocess(train_zs, train_topos):
    raw_streams = []
    for z in train_zs:
        cu, du, bid = slice_features(z)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})

    bundle = fit_bundle(raw_streams, CU_ZV_IDX, DU_ZV_IDX, version=PREPROCESS_VERSION)

    streams = []
    for i, raw in enumerate(raw_streams):
        cu_s, du_s, _, kept_bid = transform_stream(
            bundle, raw["cu"], raw["du"], raw["block_id"]
        )
        print(f"  topo[{i}] {train_topos[i]:18s}  cu_s {cu_s.shape}  du_s {du_s.shape}  "
              f"(μ={cu_s.mean():+.3f}, σ={cu_s.std():.3f} after {PREPROCESS_VERSION})")
        streams.append((cu_s, du_s, kept_bid))
    return bundle, streams

# =============================================================================
# PHASE 2: TRAIN MODEL
# =============================================================================

def phase_train(fit_streams, train_topos, model_ckpt) -> CalibratedTopoAR:
    cu_dim = fit_streams[0][0].shape[1]
    du_dim = fit_streams[0][1].shape[2]

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
    val_concat   = torch.utils.data.ConcatDataset(val_subsets)

    train_loader = DataLoader(
        train_concat,
        batch_sampler=MultiTopologyBatchSampler(train_lens, BATCH_SIZE, shuffle=True,  seed=SEED),
        collate_fn=collate_windows,
        num_workers=12,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_concat,
        batch_sampler=MultiTopologyBatchSampler(val_lens,   BATCH_SIZE, shuffle=False, seed=SEED),
        collate_fn=collate_windows,
        num_workers=12,
        pin_memory=True
    )

    torch.manual_seed(SEED)
    model = CalibratedTopoAR(cu_dim=cu_dim, du_dim=du_dim, embed_dim=EMBED_DIM).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
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
            if batch_idx % 500 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)} completed... loss: {loss.item():.4f}")

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

        if epoch % 1 == 0:
            print(f"  epoch {epoch:3d}  tr={tr_loss/len(train_loader):.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
                "preprocess": PREPROCESS_VERSION}, model_ckpt)
    print(f"  Model saved → {model_ckpt}")
    return model

# =============================================================================
# PHASE 3: SEQUENTIAL INFERENCE
# =============================================================================

def phase_infer(model: CalibratedTopoAR, cu_s: np.ndarray, du_s: np.ndarray):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)
    du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)

    cu_sqerr = (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()
    du_sqerr = (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()
    return cu_sqerr, du_sqerr

# =============================================================================
# PHASE 4: CALIBRATE THRESHOLDS (Mask channel 6 for CU)
# =============================================================================

def phase_calibrate(cu_sqerr, du_sqerr):
    cu_sqerr_n = cu_sqerr[COLD_START_K:]
    du_sqerr_n = du_sqerr[COLD_START_K:]

    n_normal = len(cu_sqerr_n)
    print(f"  Normal-calibration rows (held-out train tail): {n_normal}")
    if n_normal < 100:
        raise RuntimeError(f"Too few cal rows for threshold calibration: {n_normal}")

    # Exclude channel 6 (net_ratio) for CU to prevent explosive scale from distorting scores
    cu_sqerr_n_masked = cu_sqerr_n[:, :6]

    cu_feat_norm  = feat_norm_calibrated(cu_sqerr_n_masked)
    du_sqerr_flat = du_sqerr_n.reshape(-1, du_sqerr_n.shape[-1])
    du_feat_norm  = feat_norm_calibrated(du_sqerr_flat)

    cu_norm_scores = lift_score(cu_sqerr_n_masked, cu_feat_norm)
    du_norm_scores = lift_score(du_sqerr_flat, du_feat_norm)

    cu_thr = float(np.percentile(cu_norm_scores, CU_THRESHOLD_PCT))
    du_thr = float(np.percentile(du_norm_scores, DU_THRESHOLD_PCT))
    print(f"  CU threshold (p{CU_THRESHOLD_PCT:.1f}): {cu_thr:.4f}")
    print(f"  DU threshold (p{DU_THRESHOLD_PCT:.1f}): {du_thr:.4f}")
    return cu_feat_norm, du_feat_norm, cu_thr, du_thr

# =============================================================================
# PHASE 5: EVALUATE (Mask channel 6 for CU)
# =============================================================================

def phase_evaluate(cu_sqerr, du_sqerr, cu_feat_norm, du_feat_norm,
                   cu_thr, du_thr, cu_stress, du_stress):
    N = du_sqerr.shape[1]

    start = COLD_START_K
    cu_sqerr_ev = cu_sqerr[start:]
    du_sqerr_ev = du_sqerr[start:]

    cu_lbl = (cu_stress[start + 1:] == STRESS_TYPE).astype(int)
    du_lbl = (du_stress[start + 1:] == STRESS_TYPE)

    assert len(cu_sqerr_ev) == len(cu_lbl), \
        f"Length mismatch: sqerr {len(cu_sqerr_ev)} vs label {len(cu_lbl)}"

    # Mask channel 6 for CU evaluation
    cu_scores = lift_score(cu_sqerr_ev[:, :6], cu_feat_norm)
    cu_pred   = (cu_scores > cu_thr).astype(int)

    du_scores = np.stack(
        [lift_score(du_sqerr_ev[:, i, :], du_feat_norm) for i in range(N)],
        axis=1,
    )
    du_pred = (du_scores > du_thr).astype(int)

    all_metrics = {}

    def metrics(name, pred, lbl):
        tp = int(((pred == 1) & (lbl == 1)).sum())
        fp = int(((pred == 1) & (lbl == 0)).sum())
        fn = int(((pred == 0) & (lbl == 1)).sum())
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        total_anom = int((lbl == 1).sum())
        print(f"  {name:<12s}  anom={total_anom:>6d}  TP={tp:>6d}  FP={fp:>6d}  "
              f"FN={fn:>6d}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        all_metrics[name] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1, "anom": total_anom}

    print(f"\n  {'Entity':<12s}  {'anom':>6s}  {'TP':>6s}  {'FP':>6s}  "
          f"{'FN':>6s}  {'P':>5s}  {'R':>5s}  {'F1':>5s}")
    print(f"  {'-'*72}")
    metrics("CU", cu_pred, cu_lbl)
    for i in range(N):
        metrics(f"DU_{i}", du_pred[:, i], du_lbl[:, i].astype(int))

    any_pred  = (cu_pred == 1) | du_pred.any(axis=1)
    any_lbl   = (cu_lbl  == 1) | du_lbl.any(axis=1)
    print(f"  {'-'*72}")
    metrics("ANY", any_pred.astype(int), any_lbl.astype(int))

    return cu_scores, du_scores, cu_pred, du_pred, all_metrics

# =============================================================================
# ROOT CAUSE KPI
# =============================================================================

_CU_FEAT_NAMES = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_diff"]
_DU_FEAT_NAMES = (["cpu", "mem_pct", "mem_bytes", "fs_reads", "fs_writes", "net_tx", "net_rx"]
                  + [f"pci_{i}" for i in range(18)]
                  + ["net_diff", "net_ratio"])

_RC_FEAT_GROUPS = {
    1: {"CU": {0},              "DU": {0}},
    2: {"CU": {2},              "DU": {2}},
    3: {"CU": {3, 4, 5},        "DU": {5, 6, 25, 26}}, # Masked out channel 6 from CU RCA
}


def phase_root_cause(cu_sqerr_ev, du_sqerr_ev, cu_fn, du_fn, cu_pred, du_pred, cu_lbl, du_lbl):
    N = du_sqerr_ev.shape[1]
    rc_groups = _RC_FEAT_GROUPS.get(STRESS_TYPE)
    if rc_groups is None:
        print(f"\n[8a] Root cause KPI: no mapping defined for STRESS_TYPE={STRESS_TYPE}")
        return

    rca_metrics = {}

    def _rc_acc(sqerr_ev, feat_norm, pred, lbl, valid_feats, feat_names, ent_name):
        tp_mask = pred.astype(bool) & lbl.astype(bool)
        n_tp = int(tp_mask.sum())
        if n_tp == 0:
            print(f"  {ent_name:<12s}  TPs=     0  RC_correct=       0/0   (  N/A)  dominant_pred_RC=—")
            rca_metrics[ent_name] = {"rc_correct": 0, "rc_total": 0, "rc_pct": float("nan"), "dominant_rc": "—"}
            return
        norm_err  = sqerr_ev[tp_mask] / feat_norm
        rc_idx    = np.argmax(norm_err, axis=1)
        rc_correct = np.isin(rc_idx, list(valid_feats))
        n_correct  = int(rc_correct.sum())
        pct        = 100.0 * n_correct / n_tp
        counts     = np.bincount(rc_idx, minlength=len(feat_names))
        top_feat   = feat_names[int(np.argmax(counts))]
        print(f"  {ent_name:<12s}  TPs={n_tp:>6d}  RC_correct={n_correct:>6d}/{n_tp:<6d}"
              f"  ({pct:5.1f}%)  dominant_pred_RC={top_feat}")
        rca_metrics[ent_name] = {"rc_correct": n_correct, "rc_total": n_tp, "rc_pct": pct, "dominant_rc": top_feat}

    expected_cu = "|".join(_CU_FEAT_NAMES[f] for f in sorted(rc_groups["CU"]))
    print(f"\n[8a] Root cause KPI accuracy (stress={STRESS_NAMES[STRESS_TYPE]}, "
          f"expected_CU={expected_cu}) ...")
    print(f"  {'Entity':<12s}  {'TPs':>6s}  {'RC_correct':>20s}  {'dominant_pred_RC':>18s}")
    print(f"  {'-'*64}")
    _rc_acc(cu_sqerr_ev[:, :6], cu_fn, cu_pred, cu_lbl, rc_groups["CU"], _CU_FEAT_NAMES, "CU")
    for i in range(N):
        _rc_acc(du_sqerr_ev[:, i, :], du_fn, du_pred[:, i], du_lbl[:, i],
                rc_groups["DU"], _DU_FEAT_NAMES, f"DU_{i}")
    return rca_metrics

# =============================================================================
# PLOTTING
# =============================================================================

def _shade(ax, t_array, mask, color, alpha, label=None):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    in_block = False
    block_start = None
    first = True
    for i in range(len(mask)):
        if mask[i] and not in_block:
            block_start = t_array[i]
            in_block = True
        elif not mask[i] and in_block:
            ax.axvspan(block_start, t_array[i], color=color, alpha=alpha,
                       label=(label if first else None))
            in_block = False
            first = False
    if in_block:
        ax.axvspan(block_start, t_array[-1] + 1, color=color, alpha=alpha,
                   label=(label if first else None))


def phase_plot(cu_s_te, du_s_te, cu_stress, du_stress,
               cu_scores, du_scores, cu_pred, du_pred, cu_thr, du_thr,
               train_topos, test_topo):
    T      = len(cu_s_te)
    t_full = np.arange(T)
    score_t = np.arange(COLD_START_K + 1, T)

    n_du = du_s_te.shape[1]
    entities = [("CU",   cu_s_te,           cu_stress,
                          cu_scores,          cu_pred,          cu_thr)]
    for i in range(n_du):
        entities.append((f"DU_{i}", du_s_te[:, i, :], du_stress[:, i],
                         du_scores[:, i],    du_pred[:, i],    du_thr))

    fig, axes = plt.subplots(len(entities), 2, figsize=(18, 4 * len(entities)),
                             sharex=False)
    fig.suptitle(
        f"Cross-topology {STRESS_NAMES[STRESS_TYPE]} stress detection\n"
        f"Train: {'+'.join(train_topos)}  →  Test: {test_topo}  "
        f"(v0 preprocessing, CU={len(np.arange(7)[CU_FEAT_SLICE])}feat DU={len(np.arange(37)[DU_FEAT_SLICE])}feat)",
        fontsize=12, y=1.01,
    )

    for row, (name, feat, stress_lbl, scores, pred, thr) in enumerate(entities):
        ax_f  = axes[row, 0]
        ax_sc = axes[row, 1]

        _cu_labels = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_diff", "net_ratio"]
        _du_labels = (["cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx"]
                      + [f"pci_{i}" for i in range(22)]
                      + ["net_diff", "net_ratio"])
        feat_labels = _cu_labels if name == "CU" else _du_labels
        _colors_base = ["steelblue", "darkorange", "green", "purple", "brown", "crimson", "teal"]
        feat_colors = (_colors_base + [f"C{i}" for i in range(feat.shape[1] - len(_colors_base))])
        for fi in range(feat.shape[1]):
            ax_f.plot(t_full, feat[:, fi], color=feat_colors[fi], lw=0.7,
                      label=feat_labels[fi])
        _shade(ax_f, t_full,  stress_lbl == STRESS_TYPE, "red",    0.20, "GT anomaly")
        _shade(ax_f, score_t, pred.astype(bool),        "yellow", 0.40, "Detected")

        lo = np.percentile(feat[COLD_START_K:], 1)
        hi = np.percentile(feat[COLD_START_K:], 99)
        pad = max((hi - lo) * 0.3, 0.5)
        ax_f.set_ylim(lo - pad, hi + pad)
        ax_f.set_ylabel(name, fontsize=10)
        ax_f.set_title(f"{name} — scaled features", fontsize=10)
        ax_f.legend(loc="upper right", fontsize=7, framealpha=0.7)

        ax_sc.plot(score_t, scores, color="navy", lw=0.7, label="lift score")
        ax_sc.axhline(thr, color="red", ls="--", lw=1.2, label=f"threshold={thr:.4f}")
        _shade(ax_sc, score_t, stress_lbl[COLD_START_K + 1:] == STRESS_TYPE, "red", 0.20, "GT anomaly")
        ax_sc.set_yscale("log")
        ax_sc.set_ylim(bottom=max(scores[scores > 0].min() * 0.5, 1e-4))
        ax_sc.set_ylabel("lift score (log)", fontsize=9)
        ax_sc.set_title(f"{name} — anomaly score", fontsize=10)
        ax_sc.legend(loc="upper right", fontsize=7, framealpha=0.7)

    for col in range(2):
        axes[-1, col].set_xlabel("Timestep", fontsize=9)

    plt.tight_layout()
    out = Path(f"cross_anomaly_plot_{CKPT_PREFIX}{test_topo}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out.resolve()}")
    plt.close()

# =============================================================================
# MAIN EXECUTOR
# =============================================================================

def run_one(train_topos, test_topo):
    model_ckpt = Path(f"{CKPT_PREFIX}model_ckpt_test_{test_topo}.pt")
    cu_dim_info = len(np.arange(7)[CU_FEAT_SLICE])
    du_dim_info = len(np.arange(37)[DU_FEAT_SLICE])

    print(f"\n{'='*70}")
    print(f"  Cross-topology {STRESS_NAMES[STRESS_TYPE]} stress detection (multi-train + RobustScaler {PREPROCESS_VERSION})")
    print(f"  Train : {train_topos}  (normal only, pooled)")
    print(f"  Test  : {test_topo}    (unseen topology)")
    print(f"  Preprocessing : RobustScaler {PREPROCESS_VERSION}")
    print(f"  CU features   : {cu_dim_info}   DU features: {du_dim_info}")
    print(f"  Device        : {DEVICE}")
    print(f"{'='*70}")

    print(f"\n[1] Loading {len(train_topos)} train topologies, fitting {PREPROCESS_VERSION} scaler ...")
    train_zs = [load_npz(t, "train") for t in train_topos]
    bundle, train_streams = phase_preprocess(train_zs, train_topos)

    print(f"\n[2] Per-topology fit/cal split (cal_frac={CAL_FRAC}) ...")
    fit_streams, cal_streams = [], []
    for i, (cu_s, du_s, kept_bid) in enumerate(train_streams):
        n_total = len(cu_s)
        n_cal   = int(round(CAL_FRAC * n_total))
        n_fit   = n_total - n_cal
        fit_streams.append((cu_s[:n_fit], du_s[:n_fit], kept_bid[:n_fit]))
        cal_streams.append((cu_s[n_fit:], du_s[n_fit:]))
        print(f"  topo[{i}] {train_topos[i]:18s} total={n_total}  fit={n_fit}  cal={n_cal}")
    n_fit_total = sum(len(f[0]) for f in fit_streams)

    cu_dim = fit_streams[0][0].shape[1]
    du_dim = fit_streams[0][1].shape[2]
    if model_ckpt.exists():
        print(f"\n[3] Loading model from checkpoint ({model_ckpt}) ...")
        ckpt = torch.load(model_ckpt, map_location='cpu')
        mismatches = []
        if ckpt.get("topos") != list(train_topos):
            mismatches.append(f"topos: ckpt={ckpt.get('topos')} vs current={list(train_topos)}")
        if ckpt.get("n_train_rows") != n_fit_total:
            mismatches.append(f"n_train_rows: ckpt={ckpt.get('n_train_rows')} vs current={n_fit_total}")
        if ckpt.get("cu_dim") != cu_dim:
            mismatches.append(f"cu_dim: ckpt={ckpt.get('cu_dim')} vs current={cu_dim}")
        if ckpt.get("du_dim") != du_dim:
            mismatches.append(f"du_dim: ckpt={ckpt.get('du_dim')} vs current={du_dim}")
        if ckpt.get("preprocess") != PREPROCESS_VERSION:
            mismatches.append(f"preprocess: ckpt={ckpt.get('preprocess')} vs current={PREPROCESS_VERSION}")
        if mismatches:
            raise SystemExit(
                f"\n  Refusing to load incompatible checkpoint {model_ckpt}:\n    "
                + "\n    ".join(mismatches)
                + f"\n  Delete {model_ckpt} and rerun to retrain from scratch."
            )
        model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"],
                                 embed_dim=ckpt["embed_dim"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"\n[3] Training CalibratedTopoAR on {len(train_topos)} topologies ...")
        model = phase_train(fit_streams, train_topos, model_ckpt)

    print(f"\n[4] Inference on held-out CAL streams ...")
    cu_sqerr_pool, du_sqerr_pool = [], []
    for i, (cu_s_cal, du_s_cal) in enumerate(cal_streams):
        cu_sq, du_sq = phase_infer(model, cu_s_cal, du_s_cal)
        cu_sqerr_pool.append(cu_sq[COLD_START_K:])
        du_sqerr_pool.append(du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1]))
        print(f"  topo[{i}] {train_topos[i]:18s} cu_sqerr {cu_sq.shape}  du_sqerr {du_sq.shape}")
    cu_sqerr_n   = np.concatenate(cu_sqerr_pool, axis=0)
    du_sqerr_flt = np.concatenate(du_sqerr_pool, axis=0)

    print(f"\n[5] Calibrating thresholds on POOLED cal stream (channel 6 masked for CU) ...")
    cu_fn = feat_norm_calibrated(cu_sqerr_n[:, :6])
    du_fn = feat_norm_calibrated(du_sqerr_flt)
    cu_norm_scores = lift_score(cu_sqerr_n[:, :6], cu_fn)
    du_norm_scores = lift_score(du_sqerr_flt, du_fn)
    cu_thr = float(np.percentile(cu_norm_scores, CU_THRESHOLD_PCT))
    du_thr = float(np.percentile(du_norm_scores, DU_THRESHOLD_PCT))
    print(f"  CU threshold (p{CU_THRESHOLD_PCT:.1f}): {cu_thr:.4f}")
    print(f"  DU threshold (p{DU_THRESHOLD_PCT:.1f}): {du_thr:.4f}")

    print(f"\n[6] Transforming TEST topology ({test_topo}) ...")
    test_z  = load_npz(test_topo, "test")
    cu_te, du_te, bid_te = slice_features(test_z)
    cu_s_te, du_s_te, kept_mask, _ = transform_stream(bundle, cu_te, du_te, bid_te)

    cu_stress = test_z["cu_stress"][kept_mask].astype(np.int32)
    du_stress = test_z["du_stress"][kept_mask].astype(np.int32)
    n_du_te   = du_s_te.shape[1]

    print("\n[7] Running inference on test stream ...")
    cu_sqerr, du_sqerr = phase_infer(model, cu_s_te, du_s_te)

    if SAVE_ERRORS:
        feat_tag = f"f{cu_dim}"
        _bparts = BASE_DIR.name.split("_")
        _stress_label = "_".join(_bparts[:2] + ([DATA_VARIANT] if DATA_VARIANT else []))
        save_dir = Path(__file__).parent / "exp_runs" / f"{_stress_label}_my_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        err_path = save_dir / f"recon_errors_{test_topo}_{feat_tag}.npz"
        np.savez(err_path,
                 cu_sqerr=cu_sqerr, du_sqerr=du_sqerr,
                 cu_stress=cu_stress, du_stress=du_stress,
                 cu_feat_norm=cu_fn, du_feat_norm=du_fn,
                 cu_thr=np.array([cu_thr]),
                 du_thr=np.array([du_thr]))

    print("\n[8] Evaluation results ...")
    cu_scores, du_scores, cu_pred, du_pred, eval_metrics = phase_evaluate(
        cu_sqerr, du_sqerr, cu_fn, du_fn, cu_thr, du_thr, cu_stress, du_stress
    )

    _start = COLD_START_K
    cu_lbl_ev = (cu_stress[_start + 1:] == STRESS_TYPE)
    du_lbl_ev = (du_stress[_start + 1:] == STRESS_TYPE)
    rca_metrics = phase_root_cause(
        cu_sqerr[_start:], du_sqerr[_start:],
        cu_fn, du_fn,
        cu_pred, du_pred,
        cu_lbl_ev, du_lbl_ev,
    )
    for ent, rca in (rca_metrics or {}).items():
        if ent in eval_metrics:
            eval_metrics[ent].update(rca)

    cu_scores_cal = lift_score(cu_sqerr_n[:, :6], cu_fn)
    du_scores_cal = lift_score(du_sqerr_flt, du_fn)
    pcts = [50, 90, 99, 99.9]
    print("\n[8b] Score distribution: CAL vs TEST")
    print(f"  {'':6s}  " + "  ".join(f"p{p:4.1f}" for p in pcts))
    cu_te_scores = lift_score(cu_sqerr[:, :6], cu_fn)
    print(f"  {'CU cal':6s}  " + "  ".join(f"{np.percentile(cu_scores_cal, p):6.3f}" for p in pcts))
    print(f"  {'CU te ':6s}  " + "  ".join(f"{np.percentile(cu_te_scores, p):6.3f}" for p in pcts))
    du_sqerr_flat_te = du_sqerr.reshape(-1, du_sqerr.shape[-1])
    du_te_scores = lift_score(du_sqerr_flat_te, du_fn)
    print(f"  {'DU cal':6s}  " + "  ".join(f"{np.percentile(du_scores_cal, p):6.3f}" for p in pcts))
    print(f"  {'DU te ':6s}  " + "  ".join(f"{np.percentile(du_te_scores, p):6.3f}" for p in pcts))

    print("\n[9] Generating plot ...")
    phase_plot(cu_s_te, du_s_te, cu_stress, du_stress,
               cu_scores, du_scores, cu_pred, du_pred, cu_thr, du_thr,
               train_topos, test_topo)

    print("\nDone.")
    return eval_metrics


def main():
    import torch
    torch.set_num_threads(48)
    torch.set_num_interop_threads(4)
    
    if RUN_ALL_LOO:
        all_results = []
        for test_t in ALL_TOPOS:
            train_ts = [t for t in ALL_TOPOS if t != test_t]
            result = run_one(train_ts, test_t)
            all_results.append({"test_topo": test_t, "metrics": result})

        entity_keys: list = []
        for r in all_results:
            for k in r["metrics"]:
                if k not in entity_keys:
                    entity_keys.append(k)

        print(f"\n\n{'='*70}")
        print(f"  LEAVE-ONE-OUT SUMMARY  ({len(ALL_TOPOS)} configurations)")
        print(f"{'='*70}")
        header = f"  {'Test topology':<22s}" + "".join(f"  {e+' F1':>10s}" for e in entity_keys)
        print(header)
        print(f"  {'-'*68}")
        for r in all_results:
            row = f"  {r['test_topo']:<22s}"
            for e in entity_keys:
                if e in r["metrics"]:
                    row += f"  {r['metrics'][e]['f1']:>10.3f}"
                else:
                    row += f"  {'N/A':>10s}"
            print(row)
        print(f"  {'-'*68}")

        csv_path = Path(f"loo_results_{DATA_VARIANT}.csv" if DATA_VARIANT else "loo_results.csv")

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["stress_type", "test_topo", "entity",
                                 "anom", "tp", "fp", "fn", "precision", "recall", "f1",
                                 "rc_correct", "rc_total", "rc_pct", "dominant_rc"])
            for r in all_results:
                for entity, m in r["metrics"].items():
                    writer.writerow([
                        STRESS_NAMES[STRESS_TYPE], r["test_topo"], entity,
                        m["anom"], m["tp"], m["fp"], m["fn"],
                        f"{m['p']:.4f}", f"{m['r']:.4f}", f"{m['f1']:.4f}",
                        m.get("rc_correct", ""), m.get("rc_total", ""),
                        f"{m['rc_pct']:.1f}" if isinstance(m.get("rc_pct"), float) and not (m["rc_pct"] != m["rc_pct"]) else "",
                        m.get("dominant_rc", ""),
                    ])
        print(f"\n  Results appended → {csv_path.resolve()}")
    else:
        run_one(TRAIN_TOPOS, TEST_TOPO)


if __name__ == "__main__":
    main()
