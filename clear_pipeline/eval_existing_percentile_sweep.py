"""CU-level only, existing (pooled percentile) method, N_DU-corrected checkpoints.
Sweep the CU threshold percentile from 98.0 to 99.0 (step 0.1) to find the value
that gives the best overall CU F1 across all scopes (own-registry, small->big,
big->small), all 8 checkpoints, all targets, all 3 stress types.

Reuses the exact same model-loading / slice_features / calibration-pooling /
inference pipeline as eval_ndu_extrap_normcorrected.py -- just sweeps the
percentile cheaply after scores are computed once per target/stress, instead
of only evaluating 99.9 and 98.5.

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

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
PCTL_SWEEP = [round(p, 2) for p in np.arange(98.0, 99.901, 0.1)]

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

CKPTS = []
for held_out in ORIG_TOPOS:
    CKPTS.append({
        "key": held_out, "registry": "orig",
        "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
        "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS",
        "train_topos": [t for t in ORIG_TOPOS if t != held_out],
        "beta1": BETA1_ORIG[held_out],
        "targets": [
            {"name": held_out, "bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
            {"name": "cu0_7du", "bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu7du_11gb_all"},
            {"name": "cu0_10du", "bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_11gb_all"},
        ],
    })

NEWREG_OWN_CFG = {
    "cu1_du2": {"bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    "cu3_du6du7du8du9": {"bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    "cu0_7du": {"bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu7du_11gb_all"},
    "cu0_10du": {"bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_11gb_all"},
}
NEWREG_KEY_MAP = {"cu1_du2": "cu1_du2__newreg", "cu3_du6du7du8du9": "cu3_du6du7du8du9__newreg",
                  "cu0_7du": "cu0_7du", "cu0_10du": "cu0_10du"}
for held_out in NEW_TOPOS:
    own_cfg = NEWREG_OWN_CFG[held_out]
    CKPTS.append({
        "key": NEWREG_KEY_MAP[held_out], "registry": "new",
        "ckpt": f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
        "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS",
        "train_topos": [t for t in NEW_TOPOS if t != held_out],
        "beta1": BETA1_NEW[held_out],
        "targets": [
            {"name": held_out, "bd": own_cfg["bd"], "csv_dir": own_cfg["csv_dir"]},
            {"name": "cu0_du0du1", "bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
            {"name": "cu2_du3du4du5", "bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
        ],
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
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy(), (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


all_raw = {}   # key -> {"cu_scores_pool": [...], "targets": {t_name: {stress: {"scores":[...], "lbl":[...]}}}}

for ck in CKPTS:
    print(f"\n{'='*90}\nCheckpoint: {ck['key']}\n{'='*90}")
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
        cu_sq, _ = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        pooled_cu.append(cu_sq[COLD_START_K:])

    cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    cu_scores_pool = (cu_sqerr_pool / cu_fn_pool).max(axis=-1)

    ck_targets = {}
    for target in ck["targets"]:
        t_name, bdmod, csv_dir = target["name"], target["bd"], target["csv_dir"]
        topo_cfg = bdmod.TOPOLOGY_REGISTRY[t_name]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        t_stress = {}
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
            t_stress[stress] = {"scores": scores, "lbl": lbl_ev}
        ck_targets[t_name] = t_stress
    all_raw[ck["key"]] = {"cu_scores_pool": cu_scores_pool, "targets": ck_targets}

# ---- sweep percentiles ----
sweep_summary = {}
per_cell_by_pctl = {}
for pctl in PCTL_SWEEP:
    f1s = []
    cell_f1 = {}
    for key, rec in all_raw.items():
        thr = float(np.percentile(rec["cu_scores_pool"], pctl))
        for t_name, t_stress in rec["targets"].items():
            for stress, sl in t_stress.items():
                m = metrics((sl["scores"] > thr).astype(int), sl["lbl"])
                f1s.append(m["f1"])
                cell_f1[f"{key}|{t_name}|{stress}"] = m["f1"]
    sweep_summary[pctl] = {"mean_f1": float(np.mean(f1s)), "min_f1": float(np.min(f1s)), "median_f1": float(np.median(f1s))}
    per_cell_by_pctl[pctl] = cell_f1
    print(f"p{pctl}: mean_f1={sweep_summary[pctl]['mean_f1']:.4f}  median={sweep_summary[pctl]['median_f1']:.4f}  min={sweep_summary[pctl]['min_f1']:.4f}")

best_pctl = max(sweep_summary, key=lambda p: sweep_summary[p]["mean_f1"])
print(f"\nBEST by mean F1: p{best_pctl} -> mean={sweep_summary[best_pctl]['mean_f1']:.4f}")

# ---- build full per-cell/per-stress result at the best percentile, same shape as before ----
best_results = {}
for key, rec in all_raw.items():
    thr = float(np.percentile(rec["cu_scores_pool"], best_pctl))
    best_results[key] = {"threshold": thr, "targets": {}}
    for t_name, t_stress in rec["targets"].items():
        best_results[key]["targets"][t_name] = {}
        for stress, sl in t_stress.items():
            m = metrics((sl["scores"] > thr).astype(int), sl["lbl"])
            best_results[key]["targets"][t_name][stress] = m

out = {
    "pctl_sweep": PCTL_SWEEP,
    "sweep_summary": sweep_summary,
    "best_pctl": best_pctl,
    "best_results": best_results,
}
out_path = f"{CLEAR}/eval_existing_percentile_sweep_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print(f"\nsaved -> {out_path}")
print("DONE")
