"""Test a SCOPED alpha change for the NDT threshold: only cu1_du2-held-out's
own eps computation gets a different alpha; the other 3 checkpoints keep the
default alpha=0.05 for THEIR OWN eps computation. Each checkpoint's eps
depends only on its own 3 training topologies -- cu1_du2-held-out's eps
never factors into any other checkpoint's eps at all, so this is expected to
carry zero collateral-damage risk, unlike the earlier GLOBAL alpha-for-
everyone test which broke 3 of 4 checkpoints (because cu1_du2 sits inside
THEIR training mix, so a global alpha change inflated ITS contribution to
THEIR eps too).

Sweeps a range of alphas for cu1_du2-held-out specifically, including small
perturbations around the 0.05 default (0.04, 0.06) as well as the earlier
0.3 probe -- per user request, to see if a modest nudge (not a drastic jump)
is enough, and to understand the sensitivity more granularly.

Full re-inference (not reusing cached scores) for an independent check.
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

ALPHA_SWEEP = [0.04, 0.05, 0.06, 0.1, 0.2, 0.3]
DEFAULT_ALPHA = 0.05

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


def phase_infer(model, cu_s, du_s):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE); du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy(), (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


def run_checkpoint(held_out):
    slice_features = make_slice_features(BETA1_PER_SPLIT[held_out])
    train_topos = [t for t in ALL_NEW_TOPOS if t != held_out]
    ckpt = torch.load(f"{CKPT_DIR}/normfix_model_ckpt_test_{held_out}.pt", map_location="cpu")
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(slice_features, t))) for t in train_topos]
    bundle = fit_bundle(raw_streams, [], [], version="v0")
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled_cu, per_topo_sq = [], {}
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
        cu_sq, _ = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        cu_sq = cu_sq[COLD_START_K:]
        pooled_cu.append(cu_sq); per_topo_sq[t] = cu_sq
    cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    scores_pool = (cu_sqerr_pool / cu_fn_pool).max(axis=-1)

    own_cfg = OWN_TARGET_CFG[held_out]
    bdmod, csv_dir = own_cfg["bd"], own_cfg["csv_dir"]
    topo_cfg = bdmod.TOPOLOGY_REGISTRY[held_out]
    cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]

    raw_scores = {}
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
        cu_sqerr, _ = phase_infer(model, cu_s, du_s)
        start = COLD_START_K
        lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress]).astype(int)
        scores = (cu_sqerr[start:] / cu_fn_pool).max(axis=-1)
        raw_scores[stress] = (scores, lbl_ev)

    return train_topos, per_topo_sq, scores_pool, raw_scores


def eps_at_alpha(train_topos, per_topo_sq, scores_pool, alpha):
    i = 0; per_topo_eps = {}
    for t in train_topos:
        n = len(per_topo_sq[t])
        per_topo_eps[t] = ndt.select_threshold(ndt.ewma_smooth(scores_pool[i:i + n], alpha=alpha))["eps"]
        i += n
    return float(np.mean(list(per_topo_eps.values())))


print("=== cu1_du2-held-out: alpha sweep (scoped -- only THIS checkpoint's eps changes) ===")
train_topos, per_topo_sq, scores_pool, raw_scores = run_checkpoint("cu1_du2")
print(f"{'alpha':>6s}  {'CPU F1':>8s}  {'MEM F1':>8s}  {'NET F1':>8s}   NET(tp/fp/fn)  eps")
for alpha in ALPHA_SWEEP:
    eps = eps_at_alpha(train_topos, per_topo_sq, scores_pool, alpha)
    row = {}
    for stress in STRESS_ORDER:
        scores, lbl = raw_scores[stress]
        smoothed = ndt.ewma_smooth(scores, alpha=alpha)
        row[stress] = metrics((smoothed > eps).astype(int), lbl)
    print(f"{alpha:6.2f}  {row['CPU']['f1']:8.3f}  {row['MEM']['f1']:8.3f}  {row['NET']['f1']:8.3f}   "
          f"tp={row['NET']['tp']} fp={row['NET']['fp']} fn={row['NET']['fn']}  eps={eps:.2f}")

print("\n=== other 3 checkpoints: confirming ZERO impact (alpha stays default=0.05, never touched) ===")
for held_out in ["cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]:
    train_topos2, per_topo_sq2, scores_pool2, raw_scores2 = run_checkpoint(held_out)
    eps = eps_at_alpha(train_topos2, per_topo_sq2, scores_pool2, DEFAULT_ALPHA)
    for stress in STRESS_ORDER:
        scores, lbl = raw_scores2[stress]
        smoothed = ndt.ewma_smooth(scores, alpha=DEFAULT_ALPHA)
        m = metrics((smoothed > eps).astype(int), lbl)
        print(f"{held_out:20s} {stress:5s}  F1={m['f1']:.3f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})  eps={eps:.2f}")

print("\nDONE")
