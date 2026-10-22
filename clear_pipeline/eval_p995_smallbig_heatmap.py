"""Small->Big cross-registry evaluation at p99.5 (existing pooled vs new
N_DU-extrapolated), all 4 original-registry checkpoints scored fresh on
cu0_7du and cu0_10du. Same pattern as eval_p985_full_crossreg.py's small->big
section, just at a different percentile, heatmap data only (no full score
arrays needed this time)."""
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
CU_PCT = 99.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}

N_DU_ORIG = {"cu0_du0du1": 2, "cu1_du2": 1, "cu2_du3du4du5": 3, "cu3_du6du7du8du9": 4}
N_DU_NEW = {"cu1_du2": 1, "cu3_du6du7du8du9": 4, "cu0_7du": 7, "cu0_10du": 10}

ORIG_CFG = {"bd": bd_orig, "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "ckpt_dir": f"{CLEAR}/model_DLUDP_NET",
            "ckpt_prefix": "dlv1_", "all_topos": list(N_DU_ORIG), "n_du_of": N_DU_ORIG}
NEW_TEST_SOURCE = {"cu0_7du": f"{DATA_DIR}/test_csv_1cu7du", "cu0_10du": f"{DATA_DIR}/test_csv_1cu10du_124"}
BIG_TARGETS = ["cu0_7du", "cu0_10du"]


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
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


def score_topology(bundle, model, cu_fn_pool, bd, csv_dir, cu_id, du_ids, pci_ids, stress_name):
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
    return scores, lbl_ev


def build_checkpoint(cfg, test_topo):
    ckpt = torch.load(f"{cfg['ckpt_dir']}/{cfg['ckpt_prefix']}model_ckpt_test_{test_topo}.pt", map_location="cpu")
    train_topos = ckpt["topos"]
    raw_streams = []
    for t in train_topos:
        cu, du, bid = load_train_npz(cfg["base_dir"], t)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    per_topo_thr = {}
    pooled_cu_sqerr = []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        cu_s_cal, du_s_cal = cu_s[n_fit:], du_s[n_fit:]
        cu_sq, _ = phase_infer(model, cu_s_cal, du_s_cal)
        cu_sq = cu_sq[COLD_START_K:]
        pooled_cu_sqerr.append(cu_sq)
        cu_fn_t = feat_norm_calibrated(cu_sq)
        cu_scores_t = lift_score(cu_sq, cu_fn_t)
        per_topo_thr[t] = float(np.percentile(cu_scores_t, CU_PCT))

    cu_sqerr_pool = np.concatenate(pooled_cu_sqerr, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)
    baseline_thr = float(np.percentile(cu_scores_pool, CU_PCT))

    return {"bundle": bundle, "model": model, "cu_fn_pool": cu_fn_pool,
            "train_topos": train_topos, "per_topo_thr": per_topo_thr, "baseline_thr": baseline_thr}


def extrapolate_thr(ckpt_info, n_du_of, target_n_du):
    train_topos = ckpt_info["train_topos"]
    xs = np.array([1.0 / n_du_of[t] for t in train_topos])
    ys = np.array([ckpt_info["per_topo_thr"][t] for t in train_topos])
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a, b = coef
    return float(a + b * (1.0 / target_n_du))


results = {}
for test_topo in ORIG_CFG["all_topos"]:
    key = test_topo
    ck = build_checkpoint(ORIG_CFG, test_topo)
    print(f"\n### smallbig|{key} trained_topos={ck['train_topos']}")
    rows_by_target = {}
    for target in BIG_TARGETS:
        target_n_du = N_DU_NEW[target]
        extrap_thr = extrapolate_thr(ck, ORIG_CFG["n_du_of"], target_n_du)
        topo_cfg = bd_new.TOPOLOGY_REGISTRY[target]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = NEW_TEST_SOURCE[target]
        rows = {}
        for stress in ["CPU", "MEM", "NET"]:
            scores, lbl = score_topology(ck["bundle"], ck["model"], ck["cu_fn_pool"], bd_new, csv_dir, cu_id, du_ids, pci_ids, stress)
            m_base = metrics((scores > ck["baseline_thr"]).astype(int), lbl)
            m_new = metrics((scores > extrap_thr).astype(int), lbl)
            rows[stress] = {"baseline": {"thr": ck["baseline_thr"], **m_base}, "new": {"thr": extrap_thr, **m_new}}
            print(f"  -> {target} {stress}: baseline F1={m_base['f1']:.3f}  new F1={m_new['f1']:.3f}")
        rows_by_target[target] = {"n_du": target_n_du, "rows": rows}
    results[key] = {"source_n_du": ORIG_CFG["n_du_of"][test_topo], "targets": rows_by_target}

with open(f"{OUT_DIR}/data_p995_smallbig_heatmap.js", "w") as f:
    f.write("window.DATA_P995_SMALLBIG = " + json.dumps(results) + ";\n")
print(f"\nsaved -> {OUT_DIR}/data_p995_smallbig_heatmap.js")
print("DONE")
