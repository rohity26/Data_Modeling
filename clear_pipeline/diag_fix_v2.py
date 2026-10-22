"""Diagnostic v2: apply BOTH fixes together, across all 8 existing checkpoints,
no retraining.

Fix 1 -- mem_pct glitch imputation: mem_pct occasionally reads an impossible
0.000 (measurement dropout), same pattern already fixed for cpu/net_tx/net_rx
but never extended to mem_pct. Add it to the glitch-imputation index list.

Fix 2 -- net_ratio scaler-IQR floor: net_ratio's natural spread is tiny, so
RobustScaler's IQR for this channel is near-zero, amplifying trivial noise
into huge errors. Floor the scaler's scale_ for this channel (last index,
both CU and DU) at a sane minimum instead of letting it approach zero.
Channel stays in the model (cu_dim/du_dim unchanged) -- safe for frozen
checkpoints, no retraining.
"""
import sys
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import torch

from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX_FIXED = [0, 1, 3, 4]   # cpu, mem_pct (NEW), net_tx, net_rx
DU_IRATE_IDX_FIXED = [0, 1, 3, 4, 5]  # cpu, mem_pct (NEW), fs_writes, net_tx, net_rx
PREPROCESS_VERSION = "v0"
CAL_FRAC = 0.2
WINDOW_LEN = 64
COLD_START_K = WINDOW_LEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NET_RATIO_SCALE_FLOOR = 0.01

EXPERIMENTS = [
    {
        "name": "original", "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS",
        "ckpt_dir": f"{CLEAR}/model_DLUDP_NET", "ckpt_prefix": "dlv1_",
        "all_topos": ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"],
    },
    {
        "name": "new", "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS",
        "ckpt_dir": f"{CLEAR}/model_1_4_7_10", "ckpt_prefix": "v2_",
        "all_topos": ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"],
    },
]


def impute_glitch(arr, irate_idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
        else:
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t - 1, :, irate_idx], arr[t, :, irate_idx])
    return arr


def slice_features(cu_raw, du_raw, fixed):
    cu = cu_raw.astype(np.float32).copy()
    du = du_raw.astype(np.float32).copy()
    N_DU = du.shape[1]
    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]
    cu_idx = CU_IRATE_IDX_FIXED if fixed else [0, 3, 4]
    du_idx = DU_IRATE_IDX_FIXED if fixed else [0, 3, 4, 5]
    cu = impute_glitch(cu, cu_idx)
    du = impute_glitch(du, du_idx)
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)
    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)
    return cu, du


def load_train_npz(base_dir, topo, fixed):
    p = f"{base_dir}/{topo}_stress3/train.npz"
    z = dict(np.load(p))
    cu, du = slice_features(z["cu"], z["du"], fixed)
    bid = z["block_id"].astype(np.int64)
    return cu, du, bid


def phase_infer(model, cu_s, du_s):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)
    du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    cu_sqerr = (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()
    du_sqerr = (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()
    return cu_sqerr, du_sqerr


def get_pooled_scores(base_dir, ckpt_path, train_topos, fixed):
    raw_streams = []
    for t in train_topos:
        cu, du, bid = load_train_npz(base_dir, t, fixed)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)

    if fixed:
        bundle.cu_scaler.scale_[-1] = max(bundle.cu_scaler.scale_[-1], NET_RATIO_SCALE_FLOOR)
        bundle.du_scaler.scale_[-1] = max(bundle.du_scaler.scale_[-1], NET_RATIO_SCALE_FLOOR)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled = []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s)
        n_cal = int(round(CAL_FRAC * n_total))
        n_fit = n_total - n_cal
        cu_sq, _ = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        pooled.append(cu_sq[COLD_START_K:])
    cu_sqerr = np.concatenate(pooled, axis=0)
    cu_fn = feat_norm_calibrated(cu_sqerr)
    scores = lift_score(cu_sqerr, cu_fn)
    return scores


PCTS = [50, 90, 99, 99.5, 99.9, 99.99, 100]
print(f"{'checkpoint':<24s}{'variant':>10s}" + "".join(f"{'p'+str(p):>10s}" for p in PCTS) + f"{'n_out200':>10s}")

for exp_cfg in EXPERIMENTS:
    for test_topo in exp_cfg["all_topos"]:
        ckpt_path = f"{exp_cfg['ckpt_dir']}/{exp_cfg['ckpt_prefix']}model_ckpt_test_{test_topo}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        train_topos = ckpt["topos"]

        for variant, fixed in [("original", False), ("both-fixes", True)]:
            scores = get_pooled_scores(exp_cfg["base_dir"], ckpt_path, train_topos, fixed)
            n_out = int((scores > 200).sum())
            row = f"{exp_cfg['name']+':'+test_topo:<24s}{variant:>10s}"
            for p in PCTS:
                row += f"{np.percentile(scores, p):>10.2f}"
            row += f"{n_out:>10d}"
            print(row)
