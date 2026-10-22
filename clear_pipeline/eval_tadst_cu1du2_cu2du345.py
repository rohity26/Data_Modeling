"""Test the full TADST method (EWMA-smoothed GPD tail-fit + online rate-
matching threshold update, tadst_threshold.py) on the two own-registry cells
stuck at 0.76-0.80 F1 at the best plain-percentile cutoff: cu1_du2 and
cu2_du3du4du5 NET. Same N_DU-corrected checkpoints/calibration pooling as
every other eval in this project.

Sweeps alpha (TADST's "expected anomaly proportion") over two regimes: the
textbook-small values EVT/POT methods are designed for, and values matched to
this data's ACTUAL contamination ratio (~36-37% positives in these short test
streams -- these are NOT rare tail events, which is itself informative about
whether POT-family methods are even the right tool here).
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
from tadst_threshold import fit_tadst_initial, run_tadst_online

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
ALPHA_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.36]
BETA = 0.1
INIT_PCTL = 98.0

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

ORIG_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]

CASES = [
    {"key": "cu1_du2", "own": "cu1_du2",
     "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu1_du2.pt",
     "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "train_topos": [t for t in ORIG_TOPOS if t != "cu1_du2"],
     "beta1": BETA1_ORIG["cu1_du2"], "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    {"key": "cu2_du3du4du5", "own": "cu2_du3du4du5",
     "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu2_du3du4du5.pt",
     "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "train_topos": [t for t in ORIG_TOPOS if t != "cu2_du3du4du5"],
     "beta1": BETA1_ORIG["cu2_du3du4du5"], "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
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


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


all_results = {}
for c in CASES:
    print(f"\n{'='*90}\n{c['key']}\n{'='*90}")
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
    cu_scores_pool = (cu_sqerr_pool / cu_fn_pool).max(axis=-1)

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
    actual_contam = float(lbl_ev.mean())
    print(f"  actual NET contamination ratio in this test stream: {actual_contam:.3f}")

    c_results = {"actual_contam": actual_contam, "alpha": {}}
    for alpha in ALPHA_SWEEP:
        init = fit_tadst_initial(cu_scores_pool, alpha, INIT_PCTL, BETA)
        pred, tau_trace = run_tadst_online(scores, init["sigma"], init["gamma"], init["z_q"], alpha, BETA)
        m = metrics(pred, lbl_ev)
        c_results["alpha"][str(alpha)] = {
            "tau_0": init["z_q"], "tau_final": float(tau_trace[-1]), "tau_min": float(tau_trace.min()),
            "tau_max": float(tau_trace.max()), **m,
        }
        print(f"    alpha={alpha}: tau_0={init['z_q']:.2f} tau_final={tau_trace[-1]:.2f} "
              f"(min={tau_trace.min():.2f} max={tau_trace.max():.2f})  "
              f"tp={m['tp']} fp={m['fp']} fn={m['fn']} f1={m['f1']:.3f}")

    all_results[c["key"]] = c_results

with open(f"{CLEAR}/eval_tadst_cu1du2_cu2du345_results.json", "w") as f:
    json.dump(all_results, f, indent=1)
print("\nsaved -> eval_tadst_cu1du2_cu2du345_results.json")
print("DONE")
