"""Threshold-vs-N_DU extrapolation experiment, using ALL 8 EXISTING LOO
checkpoints (4 from the original 1/2/3/4-DU registry + 4 from the new
1/4/7/10-DU registry) -- no retraining.

For each checkpoint (trained on 3 topologies, held out the 4th):
  1. Compute the CU calibration threshold SEPARATELY per training topology
     (not pooled) at p99.0 -- gives 3 points: (N_DU, thr).
  2. Fit a straight line: thr = a + b * (1/N_DU), least squares through those
     3 points. (Functional form fixed in advance: net_tx/net_rx are divided
     by N_DU in slice_features(), so 1/N_DU is the natural covariate.)
  3. Evaluate the line at the REAL held-out topology's N_DU -> predicted
     threshold. This uses ONLY the 3 training topologies' own calibration
     data + the held-out topology's N_DU (a config fact, not its data).
  4. Score the real held-out topology's own test files (CPU/MEM/NET, CU,
     own-file-only) with: (a) current pooled-p99.0 baseline, (b) the new
     extrapolated threshold. Compare P/R/F1.

Does not modify build_dataset.py / run_experiment_original.py /
build_dataset_1_4_7_10.py / run_experiment_1_4_7_10.py -- reads their
TOPOLOGY_REGISTRY / get_cu_columns / get_du_columns only.
"""
import sys
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import pandas as pd
import torch

import build_dataset as bd_orig
import build_dataset_1_4_7_10 as bd_new
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
PREPROCESS_VERSION = "v0"
CAL_FRAC = 0.2
WINDOW_LEN = 64
COLD_START_K = WINDOW_LEN
CU_THRESHOLD_PCT = 99.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}

# ---- the two registries, each with its own checkpoints / topology metadata ----
EXPERIMENTS = [
    {
        "name": "original (1/2/3/4-DU)",
        "bd": bd_orig,
        "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS",
        "ckpt_dir": f"{CLEAR}/model_DLUDP_NET",
        "ckpt_prefix": "dlv1_",
        "all_topos": ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"],
        "n_du_of": {"cu0_du0du1": 2, "cu1_du2": 1, "cu2_du3du4du5": 3, "cu3_du6du7du8du9": 4},
        "test_source": {t: f"{DATA_DIR}/test_csv_dl_udp_2iface" for t in
                         ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]},
    },
    {
        "name": "new (1/4/7/10-DU)",
        "bd": bd_new,
        "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS",
        "ckpt_dir": f"{CLEAR}/model_1_4_7_10",
        "ckpt_prefix": "v2_",
        "all_topos": ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"],
        "n_du_of": {"cu1_du2": 1, "cu3_du6du7du8du9": 4, "cu0_7du": 7, "cu0_10du": 10},
        "test_source": {
            "cu1_du2": f"{DATA_DIR}/test_csv_dl_udp_2iface",
            "cu3_du6du7du8du9": f"{DATA_DIR}/test_csv_dl_udp_2iface",
            "cu0_7du": f"{DATA_DIR}/test_csv_1cu7du",
            "cu0_10du": f"{DATA_DIR}/test_csv_1cu10du_124",
        },
    },
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


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum())
    fp = int(((pred == 1) & (lbl == 0)).sum())
    fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    anom = int((lbl == 1).sum())
    return {"anom": anom, "tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}


summary_rows = []

print(f"{'='*100}")
print("THRESHOLD-VS-N_DU EXTRAPOLATION EXPERIMENT -- all 8 existing checkpoints, no retraining")
print(f"{'='*100}")

for exp_cfg in EXPERIMENTS:
    bd = exp_cfg["bd"]
    base_dir = exp_cfg["base_dir"]
    ckpt_dir = exp_cfg["ckpt_dir"]
    ckpt_prefix = exp_cfg["ckpt_prefix"]
    n_du_of = exp_cfg["n_du_of"]
    test_source = exp_cfg["test_source"]

    print(f"\n{'#'*100}\nREGISTRY: {exp_cfg['name']}\n{'#'*100}")

    for test_topo in exp_cfg["all_topos"]:
        ckpt_path = f"{ckpt_dir}/{ckpt_prefix}model_ckpt_test_{test_topo}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        train_topos = ckpt["topos"]
        print(f"\n### held-out={test_topo} (N_DU={n_du_of[test_topo]})  trained on {train_topos}")

        raw_streams = []
        for t in train_topos:
            cu, du, bid = load_train_npz(base_dir, t)
            raw_streams.append({"cu": cu, "du": du, "block_id": bid})
        bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)

        model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])

        per_topo_thr = {}
        pooled_cu_sqerr = []
        for t, raw in zip(train_topos, raw_streams):
            cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
            n_total = len(cu_s)
            n_cal = int(round(CAL_FRAC * n_total))
            n_fit = n_total - n_cal
            cu_s_cal, du_s_cal = cu_s[n_fit:], du_s[n_fit:]
            cu_sq, _ = phase_infer(model, cu_s_cal, du_s_cal)
            cu_sq = cu_sq[COLD_START_K:]
            pooled_cu_sqerr.append(cu_sq)

            cu_fn_t = feat_norm_calibrated(cu_sq)
            cu_scores_t = lift_score(cu_sq, cu_fn_t)
            thr_t = float(np.percentile(cu_scores_t, CU_THRESHOLD_PCT))
            per_topo_thr[t] = thr_t
            print(f"  per-topology thr: {t:20s} N_DU={n_du_of[t]:>2d}  thr(p{CU_THRESHOLD_PCT})={thr_t:.4f}")

        xs = np.array([1.0 / n_du_of[t] for t in train_topos])
        ys = np.array([per_topo_thr[t] for t in train_topos])
        A = np.vstack([np.ones_like(xs), xs]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        a, b = coef
        print(f"  fitted line: thr = {a:.4f} + {b:.4f} * (1/N_DU)")

        target_n_du = n_du_of[test_topo]
        extrap_thr = a + b * (1.0 / target_n_du)
        print(f"  extrapolated threshold at held-out N_DU={target_n_du}: {extrap_thr:.4f}")

        cu_sqerr_pool = np.concatenate(pooled_cu_sqerr, axis=0)
        cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
        cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)
        baseline_thr = float(np.percentile(cu_scores_pool, CU_THRESHOLD_PCT))
        print(f"  baseline pooled threshold (current method): {baseline_thr:.4f}")

        topo_cfg = bd.TOPOLOGY_REGISTRY[test_topo]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = test_source[test_topo]

        for stress_name in ["CPU", "MEM", "NET"]:
            path = f"{csv_dir}/test_{cu_id}_{stress_name}_rep1.csv"
            df = pd.read_csv(path)
            cols = df.columns.tolist()
            cu_cols = bd.get_cu_columns(cu_id, cols)
            cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
            du_raw_list = [df[bd.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                           for d, p in zip(du_ids, pci_ids)]
            du_raw = np.stack(du_raw_list, axis=1)
            lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
            cu_feat, du_feat = slice_features(cu_raw, du_raw)
            block_id = np.zeros(len(cu_feat), dtype=np.int64)
            cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
            cu_sqerr, _ = phase_infer(model, cu_s, du_s)
            start = COLD_START_K
            stype_num = STYPE_NUM[stress_name]
            lbl_ev = (lbl[start + 1:] == stype_num).astype(int)
            scores = lift_score(cu_sqerr[start:], cu_fn_pool)

            pred_base = (scores > baseline_thr).astype(int)
            pred_extrap = (scores > extrap_thr).astype(int)
            m_base = metrics(pred_base, lbl_ev)
            m_extrap = metrics(pred_extrap, lbl_ev)
            print(f"  {stress_name}: baseline(thr={baseline_thr:.2f}) P={m_base['p']:.3f} R={m_base['r']:.3f} F1={m_base['f1']:.3f}"
                  f"   |   extrap(thr={extrap_thr:.2f}) P={m_extrap['p']:.3f} R={m_extrap['r']:.3f} F1={m_extrap['f1']:.3f}")
            summary_rows.append({
                "registry": exp_cfg["name"], "held_out": test_topo, "n_du": target_n_du, "stress": stress_name,
                "baseline_thr": baseline_thr, "extrap_thr": extrap_thr,
                "base_f1": m_base["f1"], "extrap_f1": m_extrap["f1"],
                "base_r": m_base["r"], "extrap_r": m_extrap["r"],
                "base_p": m_base["p"], "extrap_p": m_extrap["p"],
                "score": [round(float(x), 4) for x in scores],
                "lbl": [int(x) for x in lbl_ev],
                "m_base": m_base, "m_extrap": m_extrap,
            })

print(f"\n\n{'='*100}\nSUMMARY: baseline (pooled p99.0) vs. extrapolated (per-topology-fit) threshold\n{'='*100}")
print(f"{'registry':<22s}{'held_out':<20s}{'N_DU':>5s}{'stress':>7s}{'base_F1':>9s}{'extrap_F1':>11s}{'delta':>8s}")
for r in summary_rows:
    delta = r["extrap_f1"] - r["base_f1"]
    print(f"{r['registry']:<22s}{r['held_out']:<20s}{r['n_du']:>5d}{r['stress']:>7s}{r['base_f1']:>9.3f}{r['extrap_f1']:>11.3f}{delta:>+8.3f}")

print(f"\nDone. {len(summary_rows)} rows total.")

import json
OUT_DIR = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad"
DATA = {}
for r in summary_rows:
    key = f"{r['registry']}|{r['held_out']}"
    DATA.setdefault(key, {})
    DATA[key][r["stress"]] = {
        "held_out": r["held_out"], "n_du": r["n_du"], "stress": r["stress"],
        "baseline_thr": r["baseline_thr"], "extrap_thr": r["extrap_thr"],
        "base": r["m_base"], "extrap": r["m_extrap"],
        "score": r["score"], "lbl": r["lbl"],
    }
with open(f"{OUT_DIR}/data_loo_extrap.js", "w") as f:
    f.write("window.DATA_LOO_EXTRAP = " + json.dumps(DATA) + ";\n")
print(f"saved -> {OUT_DIR}/data_loo_extrap.js")
