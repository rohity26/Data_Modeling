"""Remove anomalous/glitch rows from calibration data (per-channel robust
outlier rule, MAD-based, no labels involved), then use a single constant
pooled percentile (p99.9) on the cleaned calibration distribution.

Glitch rule (pre-registered, decided before looking at any result below):
for each of the 7 CU channels (cpu, mem_pct, mem_bytes, net_tx, net_rx,
net_diff, net_ratio), compute median + MAD of that channel's squared
prediction error over the calibration split; flag a calibration row as a
glitch if ANY channel's error exceeds median + K*MAD with K=10 (a very
conservative "clearly broken, not just a tail sample" cutoff). Drop flagged
rows before computing the pooled feat-norm and the p99.9 threshold.

Tests the cleaned threshold against: own-registry LOO (8), small->big (8),
big->small (8) -- same checkpoints/targets as every other experiment in this
set, no retraining.
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
CU_PCT = 99.9
GLITCH_K = 10.0  # pre-registered before any result was seen
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}

N_DU_ORIG = {"cu0_du0du1": 2, "cu1_du2": 1, "cu2_du3du4du5": 3, "cu3_du6du7du8du9": 4}
N_DU_NEW = {"cu1_du2": 1, "cu3_du6du7du8du9": 4, "cu0_7du": 7, "cu0_10du": 10}
ORIG_CFG = {"bd": bd_orig, "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "ckpt_dir": f"{CLEAR}/model_DLUDP_NET",
            "ckpt_prefix": "dlv1_", "all_topos": list(N_DU_ORIG), "n_du_of": N_DU_ORIG,
            "test_source": {t: f"{DATA_DIR}/test_csv_dl_udp_2iface" for t in N_DU_ORIG}}
NEW_CFG = {"bd": bd_new, "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS", "ckpt_dir": f"{CLEAR}/model_1_4_7_10",
           "ckpt_prefix": "v2_", "all_topos": list(N_DU_NEW), "n_du_of": N_DU_NEW,
           "test_source": {"cu1_du2": f"{DATA_DIR}/test_csv_dl_udp_2iface",
                            "cu3_du6du7du8du9": f"{DATA_DIR}/test_csv_dl_udp_2iface",
                            "cu0_7du": f"{DATA_DIR}/test_csv_1cu7du",
                            "cu0_10du": f"{DATA_DIR}/test_csv_1cu10du_124"}}
BIG_TARGETS = ["cu0_7du", "cu0_10du"]
SMALL_TARGETS = ["cu0_du0du1", "cu2_du3du4du5"]


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
    cu = cu_raw.astype(np.float32).copy(); du = du_raw.astype(np.float32).copy()
    N_DU = du.shape[1]
    cu[:, 5] = cu[:, 5] / N_DU; cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]; du = du[:, :, DU_FEAT_SLICE]
    cu = impute_cpu_glitch(cu, CU_IRATE_IDX); du = impute_cpu_glitch(du, DU_IRATE_IDX)
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)
    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)
    return cu, du


def load_train_npz(base_dir, topo):
    p = f"{base_dir}/{topo}_stress3/train.npz"; z = dict(np.load(p))
    cu, du = slice_features(z["cu"], z["du"])
    return cu, du, z["block_id"].astype(np.int64)


def phase_infer(model, cu_s, du_s):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE); du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy(), (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()


def remove_glitch_rows(cu_sqerr, k=GLITCH_K):
    med = np.median(cu_sqerr, axis=0)
    mad = np.median(np.abs(cu_sqerr - med), axis=0) + 1e-9
    bad = (np.abs(cu_sqerr - med) > k * mad).any(axis=1)
    return cu_sqerr[~bad], int(bad.sum()), int(len(bad))


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


def score_topology(bundle, model, cu_fn_pool, bd, csv_dir, cu_id, du_ids, pci_ids, stress_name):
    path = f"{csv_dir}/test_{cu_id}_{stress_name}_rep1.csv"
    df = pd.read_csv(path); cols = df.columns.tolist()
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
    return metrics((scores > thr_global).astype(int) if False else scores, lbl_ev)  # placeholder, real call below


def build_checkpoint_cleaned(cfg, test_topo):
    ckpt = torch.load(f"{cfg['ckpt_dir']}/{cfg['ckpt_prefix']}model_ckpt_test_{test_topo}.pt", map_location="cpu")
    train_topos = ckpt["topos"]
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(cfg["base_dir"], t))) for t in train_topos]
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled_clean, pooled_raw = [], []
    glitch_counts = []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        cu_sq, _ = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        cu_sq = cu_sq[COLD_START_K:]
        cleaned, n_bad, n_tot = remove_glitch_rows(cu_sq)
        glitch_counts.append((t, n_bad, n_tot))
        pooled_clean.append(cleaned)
        pooled_raw.append(cu_sq)

    cu_sqerr_clean = np.concatenate(pooled_clean, axis=0)
    cu_sqerr_raw = np.concatenate(pooled_raw, axis=0)
    cu_fn_clean = feat_norm_calibrated(cu_sqerr_clean)
    cu_fn_raw = feat_norm_calibrated(cu_sqerr_raw)
    thr_clean = float(np.percentile(lift_score(cu_sqerr_clean, cu_fn_clean), CU_PCT))
    thr_raw = float(np.percentile(lift_score(cu_sqerr_raw, cu_fn_raw), CU_PCT))

    return {"bundle": bundle, "model": model, "cu_fn_clean": cu_fn_clean, "cu_fn_raw": cu_fn_raw,
            "thr_clean": thr_clean, "thr_raw": thr_raw, "train_topos": train_topos, "glitch_counts": glitch_counts}


def score_and_metrics(ck, bd, csv_dir, cu_id, du_ids, pci_ids, stress_name):
    path = f"{csv_dir}/test_{cu_id}_{stress_name}_rep1.csv"
    df = pd.read_csv(path); cols = df.columns.tolist()
    cu_cols = bd.get_cu_columns(cu_id, cols)
    cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
    du_raw_list = [df[bd.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                   for d, p in zip(du_ids, pci_ids)]
    du_raw = np.stack(du_raw_list, axis=1)
    lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
    cu_feat, du_feat = slice_features(cu_raw, du_raw)
    block_id = np.zeros(len(cu_feat), dtype=np.int64)
    cu_s, du_s, _, _ = transform_stream(ck["bundle"], cu_feat, du_feat, block_id)
    cu_sqerr, _ = phase_infer(ck["model"], cu_s, du_s)
    start = COLD_START_K
    stype_num = STYPE_NUM[stress_name]
    lbl_ev = (lbl[start + 1:] == stype_num).astype(int)

    scores_clean = lift_score(cu_sqerr[start:], ck["cu_fn_clean"])
    scores_raw = lift_score(cu_sqerr[start:], ck["cu_fn_raw"])
    m_clean = metrics((scores_clean > ck["thr_clean"]).astype(int), lbl_ev)
    m_raw = metrics((scores_raw > ck["thr_raw"]).astype(int), lbl_ev)
    return m_raw, m_clean


results = {"own": {}, "small_to_big": {}, "big_to_small": {}}

print(f"{'='*100}\nOWN-REGISTRY LOO at p{CU_PCT}, raw vs glitch-cleaned calibration\n{'='*100}")
for cfg, regname in [(ORIG_CFG, "original"), (NEW_CFG, "new")]:
    for test_topo in cfg["all_topos"]:
        key = f"{regname}|{test_topo}"
        ck = build_checkpoint_cleaned(cfg, test_topo)
        print(f"\n### {key}  thr_raw={ck['thr_raw']:.2f}  thr_clean={ck['thr_clean']:.2f}")
        for t, n_bad, n_tot in ck["glitch_counts"]:
            print(f"    glitch rows in {t}: {n_bad}/{n_tot} ({100*n_bad/n_tot:.2f}%)")
        topo_cfg = cfg["bd"].TOPOLOGY_REGISTRY[test_topo]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = cfg["test_source"][test_topo]
        rows = {}
        for stress in ["CPU", "MEM", "NET"]:
            m_raw, m_clean = score_and_metrics(ck, cfg["bd"], csv_dir, cu_id, du_ids, pci_ids, stress)
            rows[stress] = {"raw": m_raw, "clean": m_clean}
            print(f"    {stress}: raw F1={m_raw['f1']:.3f}  clean F1={m_clean['f1']:.3f}")
        results["own"][key] = rows

print(f"\n{'='*100}\nSMALL->BIG at p{CU_PCT}\n{'='*100}")
for test_topo in ORIG_CFG["all_topos"]:
    ck = build_checkpoint_cleaned(ORIG_CFG, test_topo)
    print(f"\n### smallbig|{test_topo}  thr_raw={ck['thr_raw']:.2f}  thr_clean={ck['thr_clean']:.2f}")
    for target in BIG_TARGETS:
        topo_cfg = bd_new.TOPOLOGY_REGISTRY[target]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = NEW_CFG["test_source"][target]
        rows = {}
        for stress in ["CPU", "MEM", "NET"]:
            m_raw, m_clean = score_and_metrics(ck, bd_new, csv_dir, cu_id, du_ids, pci_ids, stress)
            rows[stress] = {"raw": m_raw, "clean": m_clean}
            print(f"    -> {target} {stress}: raw F1={m_raw['f1']:.3f}  clean F1={m_clean['f1']:.3f}")
        results["small_to_big"][f"{test_topo}->{target}"] = rows

print(f"\n{'='*100}\nBIG->SMALL at p{CU_PCT}\n{'='*100}")
for test_topo in NEW_CFG["all_topos"]:
    ck = build_checkpoint_cleaned(NEW_CFG, test_topo)
    print(f"\n### bigsmall|{test_topo}  thr_raw={ck['thr_raw']:.2f}  thr_clean={ck['thr_clean']:.2f}")
    for target in SMALL_TARGETS:
        topo_cfg = bd_orig.TOPOLOGY_REGISTRY[target]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = ORIG_CFG["test_source"][target]
        rows = {}
        for stress in ["CPU", "MEM", "NET"]:
            m_raw, m_clean = score_and_metrics(ck, bd_orig, csv_dir, cu_id, du_ids, pci_ids, stress)
            rows[stress] = {"raw": m_raw, "clean": m_clean}
            print(f"    -> {target} {stress}: raw F1={m_raw['f1']:.3f}  clean F1={m_clean['f1']:.3f}")
        results["big_to_small"][f"{test_topo}->{target}"] = rows

with open(f"{OUT_DIR}/p999_cleaned_results.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved -> {OUT_DIR}/p999_cleaned_results.json")
print("DONE")
