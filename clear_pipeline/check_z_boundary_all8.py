"""Calibration-only check: for each of the 8 checkpoints, does the z-sweep's
own chosen z (for each training topology) ever land AT the edge of the
search range we give it? If a chosen z pins at the boundary, that's a sign
(visible from calibration data alone, no test labels needed) that the range
was too narrow and should be widened further. If every chosen z sits
comfortably inside the range, that's calibration-only evidence the range was
already wide enough.

Tests z ranges up to 20 (what we actually use) AND up to 50 (a generous
upper bound) to see whether 20 is already wide enough or whether some
topology secretly wants to go higher.
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import torch

import build_dataset as bd_orig
import build_dataset_1_4_7_10 as bd_new
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
BASE_ALPHA = 0.05
Z_UP_TO_20 = np.arange(2.5, 20.01, 0.5)
Z_UP_TO_50 = np.arange(2.5, 50.01, 0.5)

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(f"{CLEAR}/ndu_beta1_per_split.json") as f:
    BETA1_ORIG = json.load(f)
with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    BETA1_NEW = json.load(f)

ORIG_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
NEW_TOPOS = ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]

CKPTS = []
for held_out in ORIG_TOPOS:
    CKPTS.append({"key": held_out, "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
                  "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "train_topos": [t for t in ORIG_TOPOS if t != held_out],
                  "beta1": BETA1_ORIG[held_out]})
NEWREG_KEY_MAP = {"cu1_du2": "cu1_du2__newreg", "cu3_du6du7du8du9": "cu3_du6du7du8du9__newreg",
                  "cu0_7du": "cu0_7du", "cu0_10du": "cu0_10du"}
for held_out in NEW_TOPOS:
    CKPTS.append({"key": NEWREG_KEY_MAP[held_out], "ckpt": f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
                  "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS", "train_topos": [t for t in NEW_TOPOS if t != held_out],
                  "beta1": BETA1_NEW[held_out]})


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


print(f"{'checkpoint':28} {'topology':22} {'z@range<=20':12} {'z@range<=50':12} {'pinned at 20?':14}")
for ck in CKPTS:
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
    cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)

    i = 0
    for t, cu_sq in zip(ck["train_topos"], pooled_cu):
        n = len(cu_sq)
        smoothed = ndt.ewma_smooth(cu_scores_pool[i:i + n], alpha=BASE_ALPHA)
        pick20 = ndt.select_threshold(smoothed, z_values=Z_UP_TO_20)
        pick50 = ndt.select_threshold(smoothed, z_values=Z_UP_TO_50)
        pinned = "YES <-- widen!" if pick20["z"] >= 20.0 else "no"
        print(f"{ck['key']:28} {t:22} {pick20['z']:<12} {pick50['z']:<12} {pinned}")
        i += n
