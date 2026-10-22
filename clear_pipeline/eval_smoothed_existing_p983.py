"""Isolates the EWMA-smoothing piece of NDT on top of the plain pooled-
percentile 'existing' method, at p98.3 (the best percentile found by the
sweep). One variable changed vs. what's already in the artifact: the raw
CU score is EWMA-smoothed (alpha=0.05, same as ndt_threshold.ewma_smooth)
on BOTH the calibration side (before taking the p98.3 percentile) and the
evaluation side (before comparing to that frozen threshold) -- same
calibration/eval split and same threshold-selection convention (single
pooled percentile, not per-topology averaging) as the rest of this artifact.

CU-level, own-registry targets only (the three cells this was built to
explain), but computed for all 8 checkpoints' own target for a full picture.
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
from ndt_threshold import ewma_smooth

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
PCTL = 98.3

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]

with open(f"{CLEAR}/ndu_beta1_per_split.json") as f:
    BETA1_ORIG = json.load(f)
with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    BETA1_NEW = json.load(f)

ORIG_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
NEW_TOPOS = ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]

NEWREG_OWN_CFG = {
    "cu1_du2": {"bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    "cu3_du6du7du8du9": {"bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    "cu0_7du": {"bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu7du_11gb_all"},
    "cu0_10du": {"bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_11gb_all"},
}
NEWREG_KEY_MAP = {"cu1_du2": "cu1_du2__newreg", "cu3_du6du7du8du9": "cu3_du6du7du8du9__newreg",
                  "cu0_7du": "cu0_7du", "cu0_10du": "cu0_10du"}

CKPTS = []
for held_out in ORIG_TOPOS:
    CKPTS.append({
        "key": held_out, "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
        "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS",
        "train_topos": [t for t in ORIG_TOPOS if t != held_out],
        "beta1": BETA1_ORIG[held_out], "own": held_out,
        "bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface",
    })
for held_out in NEW_TOPOS:
    own_cfg = NEWREG_OWN_CFG[held_out]
    CKPTS.append({
        "key": NEWREG_KEY_MAP[held_out], "ckpt": f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
        "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS",
        "train_topos": [t for t in NEW_TOPOS if t != held_out],
        "beta1": BETA1_NEW[held_out], "own": held_out,
        "bd": own_cfg["bd"], "csv_dir": own_cfg["csv_dir"],
    })


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


def load_npz(slice_features, base_dir, topo):
    z = dict(np.load(f"{base_dir}/{topo}_stress3/train.npz"))
    cu, du = slice_features(z["cu"], z["du"])
    return cu, du, z["block_id"].astype(np.int64)


def phase_infer(model, cu_s, du_s):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE); du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


all_results = {}
for ck in CKPTS:
    print(f"\n=== {ck['key']} (own={ck['own']}) ===")
    slice_features = make_slice_features(ck["beta1"])
    ckpt = torch.load(ck["ckpt"], map_location="cpu")
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(slice_features, ck["base_dir"], t))) for t in ck["train_topos"]]
    bundle = fit_bundle(raw_streams, [], [], version="v0")
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled_cu = []
    for t, raw in zip(ck["train_topos"], raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
        cu_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        pooled_cu.append(cu_sq[COLD_START_K:])
    cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    cu_scores_pool_raw = (cu_sqerr_pool / cu_fn_pool).max(axis=-1)
    cu_scores_pool_smooth = ewma_smooth(cu_scores_pool_raw)

    thr_raw = float(np.percentile(cu_scores_pool_raw, PCTL))
    thr_smooth = float(np.percentile(cu_scores_pool_smooth, PCTL))
    print(f"  thr_raw={thr_raw:.3f}  thr_smooth={thr_smooth:.3f}")

    topo_cfg = ck["bd"].TOPOLOGY_REGISTRY[ck["own"]]
    cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
    stress_results = {}
    for stress in STRESS_ORDER:
        path = f"{ck['csv_dir']}/test_{cu_id}_{stress}_rep1.csv"
        df = pd.read_csv(path); cols = df.columns.tolist()
        cu_cols = ck["bd"].get_cu_columns(cu_id, cols)
        cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
        du_raw_list = [df[ck["bd"].get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                       for d, p in zip(du_ids, pci_ids)]
        du_raw = np.stack(du_raw_list, axis=1)
        lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
        cu_feat, du_feat = slice_features(cu_raw, du_raw)
        block_id = np.zeros(len(cu_feat), dtype=np.int64)
        cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
        cu_sqerr = phase_infer(model, cu_s, du_s)
        start = COLD_START_K
        lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress]).astype(int)
        scores_raw = (cu_sqerr[start:] / cu_fn_pool).max(axis=-1)
        scores_smooth = ewma_smooth(scores_raw)

        m_raw = metrics((scores_raw > thr_raw).astype(int), lbl_ev)
        m_smooth = metrics((scores_smooth > thr_smooth).astype(int), lbl_ev)
        stress_results[stress] = {"raw": m_raw, "smoothed": m_smooth}
        print(f"    {stress}: raw f1={m_raw['f1']:.3f} (fp={m_raw['fp']},fn={m_raw['fn']})   "
              f"smoothed f1={m_smooth['f1']:.3f} (fp={m_smooth['fp']},fn={m_smooth['fn']})")

        if stress == "NET":
            N_DS = 90
            idx = np.linspace(0, len(scores_smooth) - 1, N_DS).astype(int)
            stress_results[stress]["plot"] = {
                "scores_smooth": scores_smooth[idx].tolist(),
                "lbl": lbl_ev[idx].tolist(),
                "thr_smooth": thr_smooth,
            }

    all_results[ck["key"]] = {"own": ck["own"], "thr_raw": thr_raw, "thr_smooth": thr_smooth, "stress": stress_results}

with open(f"{CLEAR}/eval_smoothed_existing_p983_results.json", "w") as f:
    json.dump(all_results, f, indent=1)
print("\nsaved -> eval_smoothed_existing_p983_results.json")
print("DONE")
