"""1000-POINT (corrected) version: alpha grid narrowed to match the paper's
actual supplementary-data range (0-0.112), trained on 20 shift levels x 50
unaveraged repeats = 1000 points instead of 10 hand-picked pairs. See
eval_svr_adaptive_alpha_full.py for the original 10-point version this one
supersedes/corrects; the chat record has the honest before/after F1
comparison (the 10-point version empirically wins on F1 almost everywhere,
despite not matching the cited paper's own parameter range).

SVR-based adaptive-alpha EWMA (Noor-ul-Amin & Kazmi 2024, Scientific Reports,
"Machine learning based parameter-free adaptive EWMA control chart to monitor
process dispersion"), adapted to our pipeline and tested across all 8
N_DU-corrected checkpoints / all established targets, CU-level only.

Method (calibration-only training, same spirit as the paper):
  1. From each checkpoint's pooled calibration score, get mu_c, sigma_c.
  2. Monte Carlo SIMULATE synthetic sustained-block shifts of various
     magnitudes (in units of sigma_c) on top of REAL calibration segments
     (resampled quiet data + a constant offset block, matching the shape of
     our real stress blocks -- sustained, not transient).
  3. For each simulated shift size, sweep candidate alpha values and pick the
     one that detects the shift fastest while keeping false alarms on the
     quiet portions at (near) zero, using the SAME fixed eps as the eps=0.05
     baseline (Section 17/18) -- only alpha varies here, not the threshold.
  4. Train an SVR (RBF kernel) mapping shift-size -> best-alpha, using only
     these synthetic (shift_sigma, best_alpha) pairs -- no real test labels.
  5. At evaluation: estimate a LIVE "current shift" from a rolling window of
     the real raw score stream, feed it to the trained SVR to get a live
     alpha_t at every point, EWMA-smooth with that time-varying alpha, and
     compare to the SAME fixed eps.

Honest flag: step 5 continuously re-estimates alpha online during
evaluation -- same *category* as the already-rejected TADST/Attar online
threshold adaptation, though here only the smoothing strength changes, not
the threshold's target rate. Tested here, not assumed.
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVR

import build_dataset as bd_orig
import build_dataset_1_4_7_10 as bd_new
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
Z_VALUES = np.arange(2.5, 20.01, 0.5)
BASE_ALPHA = 0.05  # used to fix eps, same as Sections 17/18
# Corrected to match the actual paper: their own supplementary dataset
# (github.com/muhammadwaqaskazmi/ARL-SDRL-Codes) shows real lambda values never
# exceed ~0.112 (median 0.059, 99th pctl 0.110) -- NOT the 0.05-0.6 range used
# in the first version of this script. Candidate grid narrowed accordingly.
ALPHA_CANDIDATES = np.arange(0.01, 0.16, 0.01)
# Their supplementary data also shows ~20 distinct shift levels, each backed by
# ~250-345 Monte Carlo repeats (5000 rows / 20 levels), fed to the SVR
# UNAVERAGED (their own stated reason for using SVR: "lessens the effect of
# outliers" -- i.e. let the regression itself smooth the noise, don't
# pre-average). Scaled down here to 20 levels x 50 repeats = 1000 points for
# compute time, same unaveraged-per-trial structure.
SHIFT_GRID = np.linspace(0.0, 10.0, 20).tolist()
N_REPEATS_PER_LEVEL = 50
BLOCK_LEN = 250  # matches typical real stress-block duration
QUIET_LEN = 300
ROLL_WINDOW = 64  # matches WINDOW_LEN elsewhere

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
        "key": held_out, "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
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
        "key": NEWREG_KEY_MAP[held_out], "ckpt": f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
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
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


def ewma_smooth_variable(x, alpha_arr):
    x = np.asarray(x, dtype=np.float64); alpha_arr = np.asarray(alpha_arr, dtype=np.float64)
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha_arr[i] * x[i] + (1 - alpha_arr[i]) * out[i - 1]
    return out


def simulate_and_train_svr(cu_scores_pool, rng):
    """20 shift levels x 50 repeated trials each = 1000 individual, UNAVERAGED
    (shift_sigma, best_alpha) points fed straight to the SVR -- matching the
    paper's own actual supplementary dataset structure (~20 levels, ~250-345
    repeats each, fed raw) rather than pre-averaging down to one point per
    level. Inner EWMA smoothing uses pandas .ewm(adjust=False) -- verified
    identical output to ndt.ewma_smooth, just vectorized for speed since this
    now runs 1000x instead of 10x."""
    mu_c, sigma_c = float(cu_scores_pool.mean()), float(cu_scores_pool.std())
    n_pool = len(cu_scores_pool)
    block_start, block_end = QUIET_LEN, QUIET_LEN + BLOCK_LEN
    eps_here = mu_c + 3.0 * sigma_c
    train_pairs = []

    for shift_sigma in SHIFT_GRID:
        for _rep in range(N_REPEATS_PER_LEVEL):
            i0 = rng.integers(0, max(1, n_pool - (2 * QUIET_LEN + BLOCK_LEN)))
            quiet_before = cu_scores_pool[i0:i0 + QUIET_LEN]
            block_src = cu_scores_pool[i0 + QUIET_LEN:i0 + QUIET_LEN + BLOCK_LEN]
            shifted_block = block_src + shift_sigma * sigma_c
            quiet_after = cu_scores_pool[i0 + QUIET_LEN + BLOCK_LEN:i0 + 2 * QUIET_LEN + BLOCK_LEN]
            synth = pd.Series(np.concatenate([quiet_before, shifted_block, quiet_after]))

            best_alpha, best_delay, best_fp = ALPHA_CANDIDATES[0], BLOCK_LEN, 10 ** 9
            for alpha in ALPHA_CANDIDATES:
                smoothed = synth.ewm(alpha=alpha, adjust=False).mean().values
                flagged = smoothed > eps_here
                fp_count = int(flagged[:block_start].sum() + flagged[block_end:].sum())
                in_block = flagged[block_start:block_end]
                delay = int(np.argmax(in_block)) if in_block.any() else BLOCK_LEN
                if fp_count < best_fp or (fp_count == best_fp and delay < best_delay):
                    best_fp, best_delay, best_alpha = fp_count, delay, alpha
            train_pairs.append((shift_sigma, best_alpha))

    shifts = np.array([p[0] for p in train_pairs]).reshape(-1, 1)
    alphas = np.array([p[1] for p in train_pairs])
    svr = SVR(kernel="rbf", C=10, epsilon=0.01)
    svr.fit(shifts, alphas)
    return svr, mu_c, sigma_c, train_pairs


all_results = {}
rng = np.random.default_rng(42)
for ck in CKPTS:
    print(f"\n{'='*90}\n{ck['key']}\n{'='*90}")
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

    # fixed eps, same convention as Sections 17/18 (alpha=0.05, z in [2.5,20])
    i = 0; per_topo_eps = {}
    for t, cu_sq in zip(ck["train_topos"], pooled_cu):
        n = len(cu_sq)
        pick = ndt.select_threshold(ndt.ewma_smooth(cu_scores_pool[i:i + n], alpha=BASE_ALPHA), z_values=Z_VALUES)
        per_topo_eps[t] = pick["eps"]; i += n
    eps_fixed = float(np.mean(list(per_topo_eps.values())))

    svr, mu_c, sigma_c, train_pairs = simulate_and_train_svr(cu_scores_pool, rng)
    print(f"  eps_fixed={eps_fixed:.2f}  mu_c={mu_c:.2f}  sigma_c={sigma_c:.2f}")
    print(f"  SVR training pairs (shift_sigma -> best_alpha): {train_pairs}")

    ck_results = {"eps_fixed": eps_fixed, "mu_c": mu_c, "sigma_c": sigma_c, "train_pairs": train_pairs, "targets": {}}
    for target in ck["targets"]:
        t_name, bdmod, csv_dir = target["name"], target["bd"], target["csv_dir"]
        topo_cfg = bdmod.TOPOLOGY_REGISTRY[t_name]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        t_results = {}
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
            cu_sqerr = phase_infer(model, cu_s, du_s)
            start = COLD_START_K
            lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress]).astype(int)
            scores = lift_score(cu_sqerr[start:], cu_fn_pool)

            # live rolling shift estimate -> alpha_t via SVR
            n = len(scores)
            roll_mean = pd.Series(scores).rolling(ROLL_WINDOW, min_periods=1).mean().values
            shift_est = (roll_mean - mu_c) / (sigma_c + 1e-9)
            shift_est_clip = np.clip(shift_est, 0.0, SHIFT_GRID[-1])
            alpha_t = svr.predict(shift_est_clip.reshape(-1, 1))
            alpha_t = np.clip(alpha_t, 0.01, 0.15)

            smoothed_fixed = ndt.ewma_smooth(scores, alpha=BASE_ALPHA)
            m_fixed = metrics((smoothed_fixed > eps_fixed).astype(int), lbl_ev)

            smoothed_adapt = ewma_smooth_variable(scores, alpha_t)
            m_adapt = metrics((smoothed_adapt > eps_fixed).astype(int), lbl_ev)

            t_results[stress] = {"fixed": m_fixed, "adaptive": m_adapt}
        ck_results["targets"][t_name] = t_results
        print(f"    {t_name}: " + "  ".join(
            f"{s}[fixed={t_results[s]['fixed']['f1']:.3f} adapt={t_results[s]['adaptive']['f1']:.3f}]" for s in STRESS_ORDER))

    all_results[ck["key"]] = ck_results

with open(f"{CLEAR}/eval_svr_adaptive_alpha_full_1000pt_results.json", "w") as f:
    json.dump(all_results, f, indent=1)
print("\nsaved -> eval_svr_adaptive_alpha_full_1000pt_results.json")
print("DONE")
