"""Threshold-method-only fix attempt: sweep the EWMA smoothing alpha used by
NDT (ndt_threshold.py, currently fixed at 0.05) for the cu1_du2-held-out
new-registry checkpoint, recalibrating eps per alpha the normal way (NDT
select_threshold on each training topology's own calibration stream,
smoothed with that alpha, then averaged), and re-score CPU/MEM/NET on its
own held-out cu1_du2. No feature changes, no retraining -- same trained
model, same 7-channel lift_score, only the smoothing/threshold-selection
stage changes.
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import pandas as pd
import torch

import build_dataset as bd_orig
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
TRAIN_BASE_DIR = f"{CLEAR}/CU_1_4_7_10_STRESS"
CKPT_PATH = f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_cu1_du2.pt"
TRAIN_TOPOS = ["cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]

with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    beta1 = json.load(f)["cu1_du2"]


def impute(arr, idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            g = arr[t, idx] < eps; arr[t, idx] = np.where(g, arr[t - 1, idx], arr[t, idx])
        else:
            g = arr[t, :, idx] < eps; arr[t, :, idx] = np.where(g, arr[t - 1, :, idx], arr[t, :, idx])
    return arr


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


def load_npz(topo):
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


ckpt = torch.load(CKPT_PATH, map_location="cpu")
raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(t))) for t in TRAIN_TOPOS]
bundle = fit_bundle(raw_streams, [], [], version="v0")
model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])

pooled_cu, per_topo_sq = [], {}
for t, raw in zip(TRAIN_TOPOS, raw_streams):
    cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
    n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
    cu_sq, _ = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
    cu_sq = cu_sq[COLD_START_K:]
    pooled_cu.append(cu_sq); per_topo_sq[t] = cu_sq
cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
scores_pool = (cu_sqerr_pool / cu_fn_pool).max(axis=-1)

# score cu1_du2 own held-out, all 3 stress types (raw, unsmoothed score + label)
topo_cfg = bd_orig.TOPOLOGY_REGISTRY["cu1_du2"]
cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
eval_data = {}
for stress in STRESS_ORDER:
    path = f"{DATA_DIR}/test_csv_dl_udp_2iface/test_{cu_id}_{stress}_rep1.csv"
    df = pd.read_csv(path); cols = df.columns.tolist()
    cu_cols = bd_orig.get_cu_columns(cu_id, cols)
    cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
    du_raw_list = [df[bd_orig.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
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
    eval_data[stress] = (scores, lbl_ev)

ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
print(f"{'alpha':>6s}  {'CPU F1':>8s}  {'MEM F1':>8s}  {'NET F1':>8s}   NET(tp/fp/fn)")
for alpha in ALPHAS:
    i = 0
    per_topo_eps = {}
    for t in TRAIN_TOPOS:
        n = len(per_topo_sq[t])
        pick = ndt.select_threshold(ndt.ewma_smooth(scores_pool[i:i + n], alpha=alpha))
        per_topo_eps[t] = pick["eps"]
        i += n
    eps = float(np.mean(list(per_topo_eps.values())))

    results = {}
    for stress in STRESS_ORDER:
        scores, lbl = eval_data[stress]
        smoothed = ndt.ewma_smooth(scores, alpha=alpha)
        m = metrics((smoothed > eps).astype(int), lbl)
        results[stress] = m
    print(f"{alpha:6.2f}  {results['CPU']['f1']:8.3f}  {results['MEM']['f1']:8.3f}  {results['NET']['f1']:8.3f}   "
          f"tp={results['NET']['tp']} fp={results['NET']['fp']} fn={results['NET']['fn']}  (eps={eps:.2f})")
