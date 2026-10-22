"""Validate the Mahalanobis joint-anomaly score (mahalanobis_score.py) as a
drop-in replacement for the project's existing max-pool lift_score, on the
same 12 cells used throughout this investigation: all 4 new-registry
own-held-out checkpoints x CPU/MEM/NET.

Same model (no retraining), same features, same N_DU correction, same
RobustScaler. Only the CU-level scoring statistic changes: D^2 (Mahalanobis,
joint over all 7 CU channels' SIGNED residuals) instead of
max_channel(sqerr/feat_norm). The NDT threshold-selection procedure itself
(EWMA alpha=0.05, per-training-topology z-sweep + average) is kept IDENTICAL
to the baseline and just applied to the new D^2 stream, so this isolates
the effect of the scoring statistic alone.

Read-only; does not touch any existing file, checkpoint, or training process.
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import pandas as pd
import torch

import build_dataset as bd_orig
import build_dataset_1_4_7_10 as bd_new
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
import ndt_threshold as ndt
from mahalanobis_score import fit_mahalanobis, mahalanobis_distance

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
TRAIN_BASE_DIR = f"{CLEAR}/CU_1_4_7_10_STRESS"
CKPT_DIR = f"{CLEAR}/model_1_4_7_10_normcorrected"

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]
ALPHA = 0.05  # kept identical to baseline -- isolating the scoring-statistic change only

ALL_NEW_TOPOS = ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]
with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    BETA1_PER_SPLIT = json.load(f)
OWN_TARGET_CFG = {
    "cu1_du2": {"bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    "cu3_du6du7du8du9": {"bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    "cu0_7du": {"bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu7du_11gb_all"},
    "cu0_10du": {"bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_11gb_all"},
}


def impute(arr, idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            g = arr[t, idx] < eps; arr[t, idx] = np.where(g, arr[t - 1, idx], arr[t, idx])
        else:
            g = arr[t, :, idx] < eps; arr[t, :, idx] = np.where(g, arr[t - 1, :, idx], arr[t, :, idx])
    return arr


def make_slice_features(beta1):
    def slice_features(cu_raw, du_raw):
        cu = cu_raw.astype(np.float32).copy(); du = du_raw.astype(np.float32).copy()
        N_DU = du.shape[1]
        cu[:, 5] = cu[:, 5] / N_DU; cu[:, 6] = cu[:, 6] / N_DU
        cu = cu[:, CU_FEAT_SLICE]; du = du[:, :, DU_FEAT_SLICE]
        cu = impute(cu, CU_IRATE_IDX); du = impute(du, DU_IRATE_IDX)
        cu[:, 0] -= beta1["cpu"] * (N_DU - 1); cu[:, 1] -= beta1["mem_pct"] * (N_DU - 1); cu[:, 2] -= beta1["mem_bytes"] * (N_DU - 1)
        tx, rx = cu[:, 3:4], cu[:, 4:5]
        cu = np.concatenate([cu, tx - rx, tx / (rx + 1e-6)], axis=1)
        dtx, drx = du[:, :, 4:5], du[:, :, 5:6]
        du = np.concatenate([du, dtx - drx, dtx / (drx + 1e-6)], axis=2)
        return cu, du
    return slice_features


def load_npz(slice_features, topo):
    z = dict(np.load(f"{TRAIN_BASE_DIR}/{topo}_stress3/train.npz"))
    cu, du = slice_features(z["cu"], z["du"])
    return cu, du, z["block_id"].astype(np.int64)


def phase_infer_signed(model, cu_s, du_s):
    """Same forward pass as the project's phase_infer, but returns the
    SIGNED residual (pred - actual), not squared -- Mahalanobis needs
    direction, lift_score only needed magnitude."""
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE); du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    return (cu_hat[0, :-1] - cu_t[0, 1:]).cpu().numpy(), (du_hat[0, :-1] - du_t[0, 1:]).cpu().numpy()


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


print(f"{'ckpt':20s} {'stress':5s}  {'F1(lift_score, baseline)':>25s}  {'F1(Mahalanobis)':>15s}")

for held_out in ALL_NEW_TOPOS:
    slice_features = make_slice_features(BETA1_PER_SPLIT[held_out])
    train_topos = [t for t in ALL_NEW_TOPOS if t != held_out]
    ckpt = torch.load(f"{CKPT_DIR}/normfix_model_ckpt_test_{held_out}.pt", map_location="cpu")
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(slice_features, t))) for t in train_topos]
    bundle = fit_bundle(raw_streams, [], [], version="v0")
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    # --- baseline (squared error) + new (signed residual) calibration pools ---
    pooled_cu_sq, pooled_cu_signed, per_topo_sq, per_topo_signed = [], [], {}, {}
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
        cu_signed, _ = phase_infer_signed(model, cu_s[n_fit:], du_s[n_fit:])
        cu_signed = cu_signed[COLD_START_K:]
        cu_sq = cu_signed ** 2
        pooled_cu_sq.append(cu_sq); pooled_cu_signed.append(cu_signed)
        per_topo_sq[t] = cu_sq; per_topo_signed[t] = cu_signed

    cu_sqerr_pool = np.concatenate(pooled_cu_sq, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    cu_scores_pool_baseline = (cu_sqerr_pool / cu_fn_pool).max(axis=-1)

    cu_signed_pool = np.concatenate(pooled_cu_signed, axis=0)
    maha_fit = fit_mahalanobis(cu_signed_pool)
    cu_scores_pool_maha = mahalanobis_distance(cu_signed_pool, maha_fit)

    def eps_from_pool(pool_scores, per_topo_lens):
        i = 0; per_topo_eps = {}
        for t, n in per_topo_lens.items():
            per_topo_eps[t] = ndt.select_threshold(ndt.ewma_smooth(pool_scores[i:i + n], alpha=ALPHA))["eps"]
            i += n
        return float(np.mean(list(per_topo_eps.values())))

    lens = {t: len(per_topo_sq[t]) for t in train_topos}
    eps_baseline = eps_from_pool(cu_scores_pool_baseline, lens)
    eps_maha = eps_from_pool(cu_scores_pool_maha, lens)

    own_cfg = OWN_TARGET_CFG[held_out]
    bdmod, csv_dir = own_cfg["bd"], own_cfg["csv_dir"]
    topo_cfg = bdmod.TOPOLOGY_REGISTRY[held_out]
    cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]

    for stress in STRESS_ORDER:
        path = f"{csv_dir}/test_{cu_id}_{stress}_rep1.csv"
        df = pd.read_csv(path); cols = df.columns.tolist()
        cu_cols = bdmod.get_cu_columns(cu_id, cols)
        cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
        du_raw_list = [df[bdmod.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                       for d, p in zip(du_ids, pci_ids)]
        du_raw = np.stack(du_raw_list, axis=1)
        lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
        cu_feat, du_feat = slice_features(cu_raw, du_raw)
        block_id = np.zeros(len(cu_feat), dtype=np.int64)
        cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
        cu_signed, _ = phase_infer_signed(model, cu_s, du_s)
        start = COLD_START_K
        lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress]).astype(int)

        cu_sqerr = cu_signed[start:] ** 2
        scores_baseline = (cu_sqerr / cu_fn_pool).max(axis=-1)
        m_baseline = metrics((ndt.ewma_smooth(scores_baseline, alpha=ALPHA) > eps_baseline).astype(int), lbl_ev)

        scores_maha = mahalanobis_distance(cu_signed[start:], maha_fit)
        m_maha = metrics((ndt.ewma_smooth(scores_maha, alpha=ALPHA) > eps_maha).astype(int), lbl_ev)

        print(f"{held_out:20s} {stress:5s}  {m_baseline['f1']:25.3f}  {m_maha['f1']:15.3f}   "
              f"(baseline: tp={m_baseline['tp']} fp={m_baseline['fp']} fn={m_baseline['fn']} | "
              f"maha: tp={m_maha['tp']} fp={m_maha['fp']} fn={m_maha['fn']})")

print(f"\nmaha shrinkage used per checkpoint recalculated each loop (Ledoit-Wolf, data-driven)")
print("DONE")
