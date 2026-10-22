"""Same SVR-adaptive-alpha method as eval_svr_adaptive_alpha_full.py
(Noor-ul-Amin & Kazmi 2024), now applied at DU level instead of CU level.

Key difference from the CU pipeline, confirmed from every established script
in this project: DU features (cpu, mem_pct, mem_bytes) are NEVER N_DU-corrected
-- that correction only ever applies to the CU's own features (since only the
CU's resource usage scales with how many DUs are attached to it; a DU's own
resource usage does not depend on N_DU). slice_features below mirrors this
exactly -- it touches du only for glitch-imputation and the tx-rx/tx/rx
derived ratios, never for N_DU subtraction.

Also mirrors an existing asymmetry already established in this project's own
DU-evaluation scripts: feat_norm for the z-sweep/eps-fit step is recomputed
PER TRAINING TOPOLOGY (du_fn_t), while feat_norm used to score real test
targets is the POOLED one (du_fn_pool) -- not something invented here, just
carried over unchanged from eval_newreg_4ckpt_on_unseen.py / eval_all4_loo_on_unseen.py.
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
BASE_ALPHA = 0.05
ALPHA_CANDIDATES = np.arange(0.05, 0.65, 0.05)
SHIFT_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
BLOCK_LEN = 250
QUIET_LEN = 300
ROLL_WINDOW = 64

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
        # N_DU correction applies ONLY to CU features -- never to DU features
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


def ewma_smooth_variable(x, alpha_arr):
    x = np.asarray(x, dtype=np.float64); alpha_arr = np.asarray(alpha_arr, dtype=np.float64)
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha_arr[i] * x[i] + (1 - alpha_arr[i]) * out[i - 1]
    return out


def simulate_and_train_svr(scores_pool, rng):
    mu_c, sigma_c = float(scores_pool.mean()), float(scores_pool.std())
    n_pool = len(scores_pool)
    train_pairs = []
    for shift_sigma in SHIFT_GRID:
        i0 = rng.integers(0, max(1, n_pool - (2 * QUIET_LEN + BLOCK_LEN)))
        quiet_before = scores_pool[i0:i0 + QUIET_LEN]
        block_src = scores_pool[i0 + QUIET_LEN:i0 + QUIET_LEN + BLOCK_LEN]
        shifted_block = block_src + shift_sigma * sigma_c
        quiet_after = scores_pool[i0 + QUIET_LEN + BLOCK_LEN:i0 + 2 * QUIET_LEN + BLOCK_LEN]
        synth = np.concatenate([quiet_before, shifted_block, quiet_after])
        block_start, block_end = QUIET_LEN, QUIET_LEN + BLOCK_LEN

        best_alpha, best_delay, best_fp = ALPHA_CANDIDATES[0], BLOCK_LEN, 10 ** 9
        for alpha in ALPHA_CANDIDATES:
            smoothed = ndt.ewma_smooth(synth, alpha=alpha)
            eps_here = mu_c + 3.0 * sigma_c
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

    pooled_du = []
    for t, raw in zip(ck["train_topos"], raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
        _, du_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        du_sq = du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1])
        pooled_du.append(du_sq)
    du_sqerr_pool = np.concatenate(pooled_du, axis=0)
    du_fn_pool = feat_norm_calibrated(du_sqerr_pool)
    du_scores_pool = lift_score(du_sqerr_pool, du_fn_pool)

    # fixed eps: per-topology feat_norm for z-sweep (matches established DU pattern), alpha=0.05, z in [2.5,20]
    per_topo_eps = {}
    for t, du_sq in zip(ck["train_topos"], pooled_du):
        du_fn_t = feat_norm_calibrated(du_sq)
        du_scores_t = lift_score(du_sq, du_fn_t)
        pick = ndt.select_threshold(ndt.ewma_smooth(du_scores_t, alpha=BASE_ALPHA), z_values=Z_VALUES)
        per_topo_eps[t] = pick["eps"]
    eps_fixed = float(np.mean(list(per_topo_eps.values())))

    svr, mu_c, sigma_c, train_pairs = simulate_and_train_svr(du_scores_pool, rng)
    print(f"  eps_fixed={eps_fixed:.2f}  mu_c={mu_c:.2f}  sigma_c={sigma_c:.2f}")
    print(f"  SVR training pairs (shift_sigma -> best_alpha): {train_pairs}")

    ck_results = {"eps_fixed": eps_fixed, "mu_c": mu_c, "sigma_c": sigma_c, "targets": {}}
    for target in ck["targets"]:
        t_name, bdmod, csv_dir = target["name"], target["bd"], target["csv_dir"]
        topo_cfg = bdmod.TOPOLOGY_REGISTRY[t_name]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        t_results = {}
        for stress in STRESS_ORDER:
            per_du_fixed_f1, per_du_adapt_f1 = [], []
            per_du_fixed, per_du_adapt = {}, {}
            for du_id in du_ids:
                path = f"{csv_dir}/test_{du_id}_{stress}_rep1.csv"
                df = pd.read_csv(path); cols = df.columns.tolist()
                cu_cols = bdmod.get_cu_columns(cu_id, cols)
                cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
                du_raw_list = [df[bdmod.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                               for d, p in zip(du_ids, pci_ids)]
                du_raw = np.stack(du_raw_list, axis=1)
                my_idx = du_ids.index(du_id)
                lbl = df[f"{du_id}_stressType"].fillna(0).astype(int).values
                cu_feat, du_feat = slice_features(cu_raw, du_raw)
                block_id = np.zeros(len(cu_feat), dtype=np.int64)
                cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
                _, du_sqerr = phase_infer(model, cu_s, du_s)
                start = COLD_START_K
                lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress]).astype(int)
                scores = lift_score(du_sqerr[start:, my_idx, :], du_fn_pool)

                n = len(scores)
                roll_mean = pd.Series(scores).rolling(ROLL_WINDOW, min_periods=1).mean().values
                shift_est = (roll_mean - mu_c) / (sigma_c + 1e-9)
                shift_est_clip = np.clip(shift_est, 0.0, SHIFT_GRID[-1])
                alpha_t = svr.predict(shift_est_clip.reshape(-1, 1))
                alpha_t = np.clip(alpha_t, 0.02, 0.6)

                smoothed_fixed = ndt.ewma_smooth(scores, alpha=BASE_ALPHA)
                m_fixed = metrics((smoothed_fixed > eps_fixed).astype(int), lbl_ev)
                smoothed_adapt = ewma_smooth_variable(scores, alpha_t)
                m_adapt = metrics((smoothed_adapt > eps_fixed).astype(int), lbl_ev)

                per_du_fixed[du_id] = m_fixed; per_du_adapt[du_id] = m_adapt
                per_du_fixed_f1.append(m_fixed["f1"]); per_du_adapt_f1.append(m_adapt["f1"])

            t_results[stress] = {
                "mean_fixed_f1": float(np.mean(per_du_fixed_f1)), "mean_adapt_f1": float(np.mean(per_du_adapt_f1)),
                "per_du_fixed": {k: v["f1"] for k, v in per_du_fixed.items()},
                "per_du_adapt": {k: v["f1"] for k, v in per_du_adapt.items()},
            }
        ck_results["targets"][t_name] = t_results
        print(f"    {t_name}: " + "  ".join(
            f"{s}[fixed={t_results[s]['mean_fixed_f1']:.3f} adapt={t_results[s]['mean_adapt_f1']:.3f}]" for s in STRESS_ORDER))

    all_results[ck["key"]] = ck_results

with open(f"{CLEAR}/eval_svr_adaptive_alpha_du_full_results.json", "w") as f:
    json.dump(all_results, f, indent=1)
print("\nsaved -> eval_svr_adaptive_alpha_du_full_results.json")
print("DONE")
