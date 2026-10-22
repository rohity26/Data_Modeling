"""Test a third candidate for calibration-only alpha selection: split each
training topology's calibration stream into two halves (fit / held-out).
Since calibration data is, by construction, 100% normal (no injected stress),
ANY flag raised on the held-out half is unambiguously a false positive --
no labels needed at all. Pick alpha to minimize that held-out FPR.

Checks whether this is a non-degenerate (interior-optimum) criterion, or
whether -- like the direct score-maximization attempt -- it just walks
monotonically to one edge of the search grid (which would mean it is ALSO
not usable for honest automatic alpha selection).
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import torch

from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
TRAIN_BASE_DIR = f"{CLEAR}/CU_1_4_7_10_STRESS"

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    BETA1_NEW = json.load(f)

HELD_OUT = "cu1_du2"
TRAIN_TOPOS = ["cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]
CKPT = f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_{HELD_OUT}.pt"

Z_VALUES = np.arange(2.5, 20.01, 0.5)
ALPHA_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


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


slice_features = make_slice_features(BETA1_NEW[HELD_OUT])
ckpt = torch.load(CKPT, map_location="cpu")
raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(slice_features, TRAIN_BASE_DIR, t))) for t in TRAIN_TOPOS]
bundle = fit_bundle(raw_streams, [], [], version="v0")
model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])

# Build a LONGER calibration stream per topology (whole normal train.npz, not
# just the last 20%) so there's enough to split into fit/held-out halves
# without starving either side -- still 100% normal data, never test CSVs.
per_topo_full_sqerr = {}
for t, raw in zip(TRAIN_TOPOS, raw_streams):
    cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
    cu_sq = phase_infer(model, cu_s, du_s)
    per_topo_full_sqerr[t] = cu_sq[COLD_START_K:]

# feat_norm fit on the pooled last-20% calibration (same as everywhere else in
# this project), applied to the longer stream too
pooled_last20 = []
for t, raw in zip(TRAIN_TOPOS, raw_streams):
    cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
    n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
    cu_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
    pooled_last20.append(cu_sq[COLD_START_K:])
cu_fn_pool = feat_norm_calibrated(np.concatenate(pooled_last20, axis=0))

per_topo_scores_full = {t: lift_score(sq, cu_fn_pool) for t, sq in per_topo_full_sqerr.items()}

print("Calibration stream lengths (full, per topology):")
for t, s in per_topo_scores_full.items():
    print(f"  {t}: {len(s)} points")

print("\n" + "=" * 90)
print("Cross-calibration-fold false-positive-rate vs alpha (NO test labels used)")
print("=" * 90)
results = {}
for alpha in ALPHA_GRID:
    fold_fprs = []
    for t, scores in per_topo_scores_full.items():
        n = len(scores)
        half = n // 2
        fit_part, held_part = scores[:half], scores[half:]
        smoothed_fit = ndt.ewma_smooth(fit_part, alpha=alpha)
        pick = ndt.select_threshold(smoothed_fit, z_values=Z_VALUES)
        eps = pick["eps"]
        smoothed_held = ndt.ewma_smooth(held_part, alpha=alpha)
        fpr = float((smoothed_held > eps).mean())
        fold_fprs.append(fpr)
    mean_fpr = float(np.mean(fold_fprs))
    results[alpha] = {"mean_fpr": mean_fpr, "per_topo_fpr": dict(zip(per_topo_scores_full.keys(), fold_fprs))}
    print(f"  alpha={alpha}: mean held-out-calibration FPR={mean_fpr:.5f}  per_topo={dict(zip(per_topo_scores_full.keys(), [round(f,5) for f in fold_fprs]))}")

with open(f"{CLEAR}/eval_calibfold_alpha_cu1du2_results.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=1)
print("\nsaved -> eval_calibfold_alpha_cu1du2_results.json")
print("DONE")
