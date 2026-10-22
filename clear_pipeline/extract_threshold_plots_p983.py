"""Extract real NET score streams (+ p98.3 existing threshold) for the three
own-registry cells sitting in the 0.7-0.8 F1 band on the 'Best Percentile'
heatmap (p98.3): cu1_du2 (orig registry), cu2_du3du4du5 (orig registry),
cu1_du2 (new registry). Downsamples + log-scales into SVG polyline points,
same style as the earlier rootcause-plot.html threshold illustrations.
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

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
CU_FEAT_SLICE = [0, 1, 2, 5, 6]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}

with open(f"{CLEAR}/ndu_beta1_per_split.json") as f:
    BETA1_ORIG = json.load(f)
with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    BETA1_NEW = json.load(f)

ORIG_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
NEW_TOPOS = ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]

CASES = [
    {"label": "cu1_du2 (original registry)", "key": "cu1_du2", "registry": "orig",
     "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu1_du2.pt",
     "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS",
     "train_topos": [t for t in ORIG_TOPOS if t != "cu1_du2"],
     "beta1": BETA1_ORIG["cu1_du2"], "own": "cu1_du2", "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface",
     "thr": 25.149},
    {"label": "cu2_du3du4du5 (original registry)", "key": "cu2_du3du4du5", "registry": "orig",
     "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu2_du3du4du5.pt",
     "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS",
     "train_topos": [t for t in ORIG_TOPOS if t != "cu2_du3du4du5"],
     "beta1": BETA1_ORIG["cu2_du3du4du5"], "own": "cu2_du3du4du5", "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface",
     "thr": 32.938},
    {"label": "cu1_du2 (new registry, trained on cu3_du6du7du8du9+cu0_7du+cu0_10du)", "key": "cu1_du2__newreg", "registry": "new",
     "ckpt": f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_cu1_du2.pt",
     "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS",
     "train_topos": [t for t in NEW_TOPOS if t != "cu1_du2"],
     "beta1": BETA1_NEW["cu1_du2"], "own": "cu1_du2", "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface",
     "thr": 24.616},
]


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


results = []
for c in CASES:
    print(f"\n=== {c['label']} ===")
    slice_features = make_slice_features(c["beta1"])
    ckpt = torch.load(c["ckpt"], map_location="cpu")
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(slice_features, c["base_dir"], t))) for t in c["train_topos"]]
    bundle = fit_bundle(raw_streams, [], [], version="v0")
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled_cu = []
    for t, raw in zip(c["train_topos"], raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
        cu_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        pooled_cu.append(cu_sq[COLD_START_K:])
    cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)

    topo_cfg = bd_orig.TOPOLOGY_REGISTRY[c["own"]]
    cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
    path = f"{c['csv_dir']}/test_{cu_id}_NET_rep1.csv"
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
    cu_sqerr = phase_infer(model, cu_s, du_s)
    start = COLD_START_K
    lbl_ev = (lbl[start + 1:] == STYPE_NUM["NET"]).astype(int)
    scores = (cu_sqerr[start:] / cu_fn_pool).max(axis=-1)
    thr = c["thr"]
    pred = (scores > thr).astype(int)
    tp = int(((pred == 1) & (lbl_ev == 1)).sum()); fp = int(((pred == 1) & (lbl_ev == 0)).sum()); fn = int(((pred == 0) & (lbl_ev == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); f1 = 2 * p * r / (p + r + 1e-9)
    print(f"  thr={thr}  tp={tp} fp={fp} fn={fn} f1={f1:.3f}  n={len(scores)}")

    # downsample to ~90 points for a compact SVG polyline
    N_DS = 90
    idx = np.linspace(0, len(scores) - 1, N_DS).astype(int)
    ds_scores = scores[idx]
    ds_lbl = lbl_ev[idx]
    results.append({
        "label": c["label"], "key": c["key"], "thr": thr, "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
        "scores": ds_scores.tolist(), "lbl": ds_lbl.tolist(),
    })

with open(f"{CLEAR}/threshold_plots_p983_results.json", "w") as f:
    json.dump(results, f, indent=1)
print("\nsaved threshold_plots_p983_results.json")
