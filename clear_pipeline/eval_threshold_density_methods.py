"""Density-based alternatives to the raw empirical per-topology percentile,
feeding the SAME N_DU-extrapolation line fit. Two methods:
  - KDE-smoothed percentile: fit a kernel density estimate to each training
    topology's own calibration scores, read the threshold off the smooth CDF
    instead of counting raw order statistics.
  - GPD tail fit (extreme value theory): fit a Generalized Pareto distribution
    to the exceedances above the 90th percentile, solve for the p99.0 quantile
    from the fitted tail instead of the empirical one.

Reuses the test score arrays already saved by eval_threshold_plot_evidence.py
(threshold_plot_evidence_full.json) -- only the calibration side is recomputed
(cheap: no re-run of model inference on test data).
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import torch
from scipy.stats import gaussian_kde, genpareto

import build_dataset as bd_orig
import build_dataset_1_4_7_10 as bd_new
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
OUT_DIR = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad"

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
PREPROCESS_VERSION = "v0"
CAL_FRAC = 0.2
WINDOW_LEN = 64
COLD_START_K = WINDOW_LEN
CU_PCT = 99.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = [
    {"name": "original", "bd": bd_orig, "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS",
     "ckpt_dir": f"{CLEAR}/model_DLUDP_NET", "ckpt_prefix": "dlv1_",
     "all_topos": ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"],
     "n_du_of": {"cu0_du0du1": 2, "cu1_du2": 1, "cu2_du3du4du5": 3, "cu3_du6du7du8du9": 4}},
    {"name": "new", "bd": bd_new, "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS",
     "ckpt_dir": f"{CLEAR}/model_1_4_7_10", "ckpt_prefix": "v2_",
     "all_topos": ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"],
     "n_du_of": {"cu1_du2": 1, "cu3_du6du7du8du9": 4, "cu0_7du": 7, "cu0_10du": 10}},
]


def impute_cpu_glitch(arr, irate_idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
        else:
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t - 1, :, irate_idx], arr[t, :, irate_idx])
    return arr


def slice_features(cu_raw, du_raw):
    cu = cu_raw.astype(np.float32).copy()
    du = du_raw.astype(np.float32).copy()
    N_DU = du.shape[1]
    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]
    cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
    du = impute_cpu_glitch(du, DU_IRATE_IDX)
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)
    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)
    return cu, du


def load_train_npz(base_dir, topo):
    p = f"{base_dir}/{topo}_stress3/train.npz"
    z = dict(np.load(p))
    cu, du = slice_features(z["cu"], z["du"])
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


def kde_quantile(scores, q):
    scores = np.asarray(scores)
    kde = gaussian_kde(scores)
    lo, hi = scores.min() - 3 * scores.std(), scores.max() + 3 * scores.std()
    grid = np.linspace(lo, hi, 2000)
    pdf = kde(grid)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    idx = np.searchsorted(cdf, q)
    idx = min(idx, len(grid) - 1)
    return float(grid[idx])


def gpd_quantile(scores, q, u_pct=90.0):
    scores = np.asarray(scores)
    u = np.percentile(scores, u_pct)
    exceed = scores[scores > u] - u
    p_u = len(exceed) / len(scores)
    if len(exceed) < 10 or q <= 1 - p_u:
        # not enough tail data, or target quantile isn't actually beyond u: fall back to empirical
        return float(np.percentile(scores, q * 100))
    shape, loc, scale = genpareto.fit(exceed, floc=0)
    cdf_target = 1 - (1 - q) / p_u
    y = genpareto.ppf(cdf_target, shape, loc=0, scale=scale)
    return float(u + y)


def metrics_from_scores(scores, lbl, thr):
    pred = (np.array(scores) > thr).astype(int)
    lbl = np.array(lbl)
    tp = int(((pred == 1) & (lbl == 1)).sum())
    fp = int(((pred == 1) & (lbl == 0)).sum())
    fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


with open(f"{OUT_DIR}/threshold_plot_evidence_full.json") as f:
    saved = json.load(f)

results = {}
for exp_cfg in EXPERIMENTS:
    bd = exp_cfg["bd"]; base_dir = exp_cfg["base_dir"]; ckpt_dir = exp_cfg["ckpt_dir"]
    ckpt_prefix = exp_cfg["ckpt_prefix"]; n_du_of = exp_cfg["n_du_of"]

    for test_topo in exp_cfg["all_topos"]:
        key = f"{exp_cfg['name']}|{test_topo}"
        ckpt_path = f"{ckpt_dir}/{ckpt_prefix}model_ckpt_test_{test_topo}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        train_topos = ckpt["topos"]
        print(f"\n### {key} trained on {train_topos}")

        raw_streams = []
        for t in train_topos:
            cu, du, bid = load_train_npz(base_dir, t)
            raw_streams.append({"cu": cu, "du": du, "block_id": bid})
        bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
        model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])

        pts_raw, pts_kde, pts_gpd = {}, {}, {}
        for t, raw in zip(train_topos, raw_streams):
            cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
            n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
            cu_s_cal, du_s_cal = cu_s[n_fit:], du_s[n_fit:]
            cu_sq, _ = phase_infer(model, cu_s_cal, du_s_cal)
            cu_sq = cu_sq[COLD_START_K:]
            cu_fn_t = feat_norm_calibrated(cu_sq)
            cu_scores_t = lift_score(cu_sq, cu_fn_t)

            pts_raw[t] = float(np.percentile(cu_scores_t, CU_PCT))
            pts_kde[t] = kde_quantile(cu_scores_t, CU_PCT / 100.0)
            pts_gpd[t] = gpd_quantile(cu_scores_t, CU_PCT / 100.0)
            print(f"  {t:20s} N_DU={n_du_of[t]:>2d}  raw={pts_raw[t]:.2f}  kde={pts_kde[t]:.2f}  gpd={pts_gpd[t]:.2f}")

        def fit_line_and_extrap(pts):
            xs = np.array([1.0 / n_du_of[t] for t in train_topos])
            ys = np.array([pts[t] for t in train_topos])
            A = np.vstack([np.ones_like(xs), xs]).T
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            a, b = coef
            return float(a + b * (1.0 / n_du_of[test_topo]))

        extrap_raw = fit_line_and_extrap(pts_raw)
        extrap_kde = fit_line_and_extrap(pts_kde)
        extrap_gpd = fit_line_and_extrap(pts_gpd)
        print(f"  extrapolated: raw={extrap_raw:.2f}  kde={extrap_kde:.2f}  gpd={extrap_gpd:.2f}")

        saved_rows = saved[key]["rows"]
        row_out = {}
        for stress, rec in saved_rows.items():
            scores, lbl = rec["score"], rec["lbl"]
            m_raw = metrics_from_scores(scores, lbl, extrap_raw)
            m_kde = metrics_from_scores(scores, lbl, extrap_kde)
            m_gpd = metrics_from_scores(scores, lbl, extrap_gpd)
            row_out[stress] = {
                "raw": {"thr": extrap_raw, "f1": m_raw["f1"], "tp": m_raw["tp"], "fp": m_raw["fp"], "fn": m_raw["fn"]},
                "kde": {"thr": extrap_kde, "f1": m_kde["f1"], "tp": m_kde["tp"], "fp": m_kde["fp"], "fn": m_kde["fn"]},
                "gpd": {"thr": extrap_gpd, "f1": m_gpd["f1"], "tp": m_gpd["tp"], "fp": m_gpd["fp"], "fn": m_gpd["fn"]},
            }
            print(f"  {stress}: raw F1={m_raw['f1']:.3f}  kde F1={m_kde['f1']:.3f}  gpd F1={m_gpd['f1']:.3f}")
        results[key] = row_out

with open(f"{OUT_DIR}/data_density_methods.js", "w") as f:
    f.write("window.DATA_DENSITY_METHODS = " + json.dumps(results) + ";\n")
with open(f"{OUT_DIR}/density_methods_results.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved -> {OUT_DIR}/data_density_methods.js")
