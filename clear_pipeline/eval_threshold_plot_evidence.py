"""Generate full evidence (score/lbl arrays + both thresholds) for the
threshold-extrapolation report, across all 8 existing checkpoints and all 3
stress types. For NET stress, uses the 2.5%-loss recollection
(CU_25_loss_net_stress_test/) wherever it applies (cu1_du2, cu3_du6du7du8du9,
cu0_7du, cu0_10du); falls back to the original (~4.4-4.8%-loss) NET file for
the 2 original-registry-only topologies (cu0_du0du1, cu2_du3du4du5) which have
no 2.5%-loss equivalent collected.

No retraining -- reuses existing checkpoints. Does not modify build_dataset.py /
run_experiment_original.py / build_dataset_1_4_7_10.py / run_experiment_1_4_7_10.py.
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
from scoring import lift_score

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
LOSS25_DIR = f"{DATA_DIR}/CU_25_loss_net_stress_test"
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
CU_THRESHOLD_PCT = 99.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}

EXPERIMENTS = [
    {
        "name": "original", "bd": bd_orig,
        "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "ckpt_dir": f"{CLEAR}/model_DLUDP_NET",
        "ckpt_prefix": "dlv1_",
        "all_topos": ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"],
        "n_du_of": {"cu0_du0du1": 2, "cu1_du2": 1, "cu2_du3du4du5": 3, "cu3_du6du7du8du9": 4},
        "test_source": {t: f"{DATA_DIR}/test_csv_dl_udp_2iface" for t in
                         ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]},
    },
    {
        "name": "new", "bd": bd_new,
        "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS", "ckpt_dir": f"{CLEAR}/model_1_4_7_10",
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

# NET-stress 2.5%-loss override: held-out topology -> 2.5%-loss NET file
NET_25LOSS_FILE = {
    "cu1_du2": f"{LOSS25_DIR}/test_srscu1_NET_rep1.csv",
    "cu3_du6du7du8du9": f"{LOSS25_DIR}/test_srscu4_NET_rep1.csv",
    "cu0_7du": f"{LOSS25_DIR}/test_srscu7_NET_rep1.csv",
    "cu0_10du": f"{LOSS25_DIR}/test_srscu10_NET_rep1.csv",
}


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


all_data = {}

for exp_cfg in EXPERIMENTS:
    bd = exp_cfg["bd"]
    base_dir = exp_cfg["base_dir"]
    ckpt_dir = exp_cfg["ckpt_dir"]
    ckpt_prefix = exp_cfg["ckpt_prefix"]
    n_du_of = exp_cfg["n_du_of"]
    test_source = exp_cfg["test_source"]

    for test_topo in exp_cfg["all_topos"]:
        key = f"{exp_cfg['name']}|{test_topo}"
        ckpt_path = f"{ckpt_dir}/{ckpt_prefix}model_ckpt_test_{test_topo}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        train_topos = ckpt["topos"]
        print(f"\n### {key} (N_DU={n_du_of[test_topo]}) trained on {train_topos}")

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

        xs = np.array([1.0 / n_du_of[t] for t in train_topos])
        ys = np.array([per_topo_thr[t] for t in train_topos])
        A = np.vstack([np.ones_like(xs), xs]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        a, b = coef
        target_n_du = n_du_of[test_topo]
        extrap_thr = float(a + b * (1.0 / target_n_du))

        cu_sqerr_pool = np.concatenate(pooled_cu_sqerr, axis=0)
        cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
        cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)
        baseline_thr = float(np.percentile(cu_scores_pool, CU_THRESHOLD_PCT))
        print(f"  baseline={baseline_thr:.4f}  extrap={extrap_thr:.4f}")

        topo_cfg = bd.TOPOLOGY_REGISTRY[test_topo]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]

        rows = {}
        for stress_name in ["CPU", "MEM", "NET"]:
            if stress_name == "NET" and test_topo in NET_25LOSS_FILE:
                path = NET_25LOSS_FILE[test_topo]
                net_variant = "2.5pct_loss"
            else:
                csv_dir = test_source[test_topo]
                path = f"{csv_dir}/test_{cu_id}_{stress_name}_rep1.csv"
                net_variant = "original_loss" if stress_name == "NET" else None

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
            print(f"  {stress_name} ({net_variant or 'n/a'}): base F1={m_base['f1']:.3f}  extrap F1={m_extrap['f1']:.3f}")

            rows[stress_name] = {
                "score": [round(float(x), 4) for x in scores],
                "lbl": [int(x) for x in lbl_ev],
                "baseline_thr": baseline_thr, "extrap_thr": extrap_thr,
                "base": {"tp": m_base["tp"], "fp": m_base["fp"], "fn": m_base["fn"], "anom": m_base["anom"], "f1": m_base["f1"]},
                "extrap": {"tp": m_extrap["tp"], "fp": m_extrap["fp"], "fn": m_extrap["fn"], "anom": m_extrap["anom"], "f1": m_extrap["f1"]},
                "net_variant": net_variant, "source_file": path.split("/")[-1],
                "entity_id": cu_id, "topo": test_topo, "n_du": target_n_du,
            }

        all_data[key] = {"n_du": target_n_du, "registry": exp_cfg["name"], "rows": rows}

with open(f"{OUT_DIR}/threshold_plot_evidence_full.json", "w") as f:
    json.dump(all_data, f)
print(f"\nsaved -> {OUT_DIR}/threshold_plot_evidence_full.json")

# per-checkpoint JS files for the artifact
for key, v in all_data.items():
    safe = key.replace("|", "_")
    with open(f"{OUT_DIR}/data_thrplot_{safe}.js", "w") as f:
        f.write(f"window.DATA_THRPLOT_{safe} = " + json.dumps(v) + ";\n")
print("saved per-checkpoint JS files")
