"""Evaluate the 4 ORIGINAL (small-registry) checkpoints on the newly
recollected cu0_10du test data (data_input/test_csv_1cu10du_11gb_all, fixed
~11Gb traffic), both CU-level and DU-level (per-DU rows), existing
(pooled-percentile) vs NDT -- reusing already-computed thresholds/eps (no
recalibration), only fresh model inference on the new CSV folder.

No existing files modified.
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
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
OUT_DIR = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad"
LOG_FILE = f"{CLEAR}/eval_results_log.jsonl"
NEW_TRAFFIC_DIR = f"{DATA_DIR}/test_csv_1cu10du_11gb_all"

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
PREPROCESS_VERSION = "v0"
CAL_FRAC = 0.2
WINDOW_LEN = 64
COLD_START_K = WINDOW_LEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]

ORIG_CFG = {"base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "ckpt_dir": f"{CLEAR}/model_DLUDP_NET", "ckpt_prefix": "dlv1_"}
TEST_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
TARGET = "cu0_10du"


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
    cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)
    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)
    return cu, du


def load_train_npz(base_dir, topo):
    z = dict(np.load(f"{base_dir}/{topo}_stress3/train.npz"))
    cu, du = slice_features(z["cu"], z["du"])
    return cu, du, z["block_id"].astype(np.int64)


def phase_infer(model, cu_s, du_s):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE); du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy(), (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()


def build_checkpoint(test_topo):
    ckpt = torch.load(f"{ORIG_CFG['ckpt_dir']}/{ORIG_CFG['ckpt_prefix']}model_ckpt_test_{test_topo}.pt", map_location="cpu")
    train_topos = ckpt["topos"]
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(ORIG_CFG["base_dir"], t))) for t in train_topos]
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled_cu, pooled_du = [], []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        cu_sq, du_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        cu_sq = cu_sq[COLD_START_K:]
        du_sq = du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1])
        pooled_cu.append(cu_sq); pooled_du.append(du_sq)
    cu_fn_pool = feat_norm_calibrated(np.concatenate(pooled_cu, axis=0))
    du_fn_pool = feat_norm_calibrated(np.concatenate(pooled_du, axis=0))
    return {"bundle": bundle, "model": model, "cu_fn_pool": cu_fn_pool, "du_fn_pool": du_fn_pool, "train_topos": train_topos}


def score_cu(ck, csv_dir, cu_id, du_ids, pci_ids, stress_name):
    path = f"{csv_dir}/test_{cu_id}_{stress_name}_rep1.csv"
    df = pd.read_csv(path); cols = df.columns.tolist()
    cu_cols = bd_new.get_cu_columns(cu_id, cols)
    cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
    du_raw_list = [df[bd_new.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                   for d, p in zip(du_ids, pci_ids)]
    du_raw = np.stack(du_raw_list, axis=1)
    lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
    cu_feat, du_feat = slice_features(cu_raw, du_raw)
    block_id = np.zeros(len(cu_feat), dtype=np.int64)
    cu_s, du_s, _, _ = transform_stream(ck["bundle"], cu_feat, du_feat, block_id)
    cu_sqerr, _ = phase_infer(ck["model"], cu_s, du_s)
    start = COLD_START_K
    lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress_name]).astype(int)
    scores = lift_score(cu_sqerr[start:], ck["cu_fn_pool"])
    return np.array(scores), lbl_ev


def score_all_dus(ck, csv_dir, cu_id, du_ids, pci_ids, stress_name):
    per_du = {}
    for du_id in du_ids:
        path = f"{csv_dir}/test_{du_id}_{stress_name}_rep1.csv"
        df = pd.read_csv(path); cols = df.columns.tolist()
        cu_cols = bd_new.get_cu_columns(cu_id, cols)
        cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
        du_raw_list = [df[bd_new.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                       for d, p in zip(du_ids, pci_ids)]
        du_raw = np.stack(du_raw_list, axis=1)
        my_idx = du_ids.index(du_id)
        lbl = df[f"{du_id}_stressType"].fillna(0).astype(int).values
        cu_feat, du_feat = slice_features(cu_raw, du_raw)
        block_id = np.zeros(len(cu_feat), dtype=np.int64)
        cu_s, du_s, _, _ = transform_stream(ck["bundle"], cu_feat, du_feat, block_id)
        _, du_sqerr = phase_infer(ck["model"], cu_s, du_s)
        start = COLD_START_K
        lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress_name]).astype(int)
        scores = lift_score(du_sqerr[start:, my_idx, :], ck["du_fn_pool"])
        per_du[du_id] = (np.array(scores), lbl_ev)
    return per_du


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


with open(f"{OUT_DIR}/ndt_crossreg_results.json") as f:
    cross = json.load(f)
with open(f"{OUT_DIR}/ndt_du_results.json") as f:
    dures = json.load(f)

results = {"cu": {}, "du": {}}
topo_cfg = bd_new.TOPOLOGY_REGISTRY[TARGET]
cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]

print(f"{'='*100}\nCU evaluation on NEW cu0_10du test data (fixed ~11Gb) -- small-registry checkpoints\n{'='*100}")
for ck_name in TEST_TOPOS:
    ck = build_checkpoint(ck_name)
    cu_eps = cross["small_to_big"][ck_name]["eps"]
    cu_existing_thr = cross["small_to_big"][ck_name]["targets"][TARGET]["rows"]["CPU"]["existing"]["thr"]
    old_existing_f1 = {s: cross["small_to_big"][ck_name]["targets"][TARGET]["rows"][s]["existing"]["f1"] for s in STRESS_ORDER}
    old_ndt_f1 = {s: cross["small_to_big"][ck_name]["targets"][TARGET]["rows"][s]["ndt_pruned"]["f1"] for s in STRESS_ORDER}

    print(f"\n### {ck_name}  cu_existing_thr={cu_existing_thr:.2f}  cu_eps={cu_eps:.2f}")
    results["cu"][ck_name] = {}
    for stress in STRESS_ORDER:
        scores, lbl = score_cu(ck, NEW_TRAFFIC_DIR, cu_id, du_ids, pci_ids, stress)
        m_existing = metrics((scores > cu_existing_thr).astype(int), lbl)
        smoothed = ndt.ewma_smooth(scores)
        m_ndt = metrics((smoothed > cu_eps).astype(int), lbl)
        print(f"  {stress}: OLD(124) existing={old_existing_f1[stress]:.3f} ndt={old_ndt_f1[stress]:.3f}  |  "
              f"NEW(11gb) existing F1={m_existing['f1']:.3f}  ndt F1={m_ndt['f1']:.3f}")
        results["cu"][ck_name][stress] = {
            "old_existing_f1": old_existing_f1[stress], "old_ndt_f1": old_ndt_f1[stress],
            "new_existing": m_existing, "new_ndt": m_ndt}

print(f"\n{'='*100}\nDU evaluation on NEW cu0_10du test data (fixed ~11Gb) -- small-registry checkpoints\n{'='*100}")
for ck_name in TEST_TOPOS:
    ck = build_checkpoint(ck_name)
    du_key = f"{ck_name}->{TARGET}"
    print(f"\n### {ck_name}")
    results["du"][ck_name] = {}
    for stress in STRESS_ORDER:
        per_du = score_all_dus(ck, NEW_TRAFFIC_DIR, cu_id, du_ids, pci_ids, stress)
        results["du"][ck_name][stress] = {}
        for du_id, (scores, lbl) in per_du.items():
            old_rec = dures["small_to_big"][du_key][stress][du_id]
            du_existing_thr = old_rec["existing"]["thr"]
            du_eps = old_rec["ndt"]["thr"]
            m_existing = metrics((scores > du_existing_thr).astype(int), lbl)
            smoothed = ndt.ewma_smooth(scores)
            m_ndt = metrics((smoothed > du_eps).astype(int), lbl)
            print(f"  {stress} {du_id}: OLD(124) existing={old_rec['existing']['f1']:.3f} ndt={old_rec['ndt']['f1']:.3f}  |  "
                  f"NEW(11gb) existing F1={m_existing['f1']:.3f}  ndt F1={m_ndt['f1']:.3f}")
            results["du"][ck_name][stress][du_id] = {
                "old_existing_f1": old_rec["existing"]["f1"], "old_ndt_f1": old_rec["ndt"]["f1"],
                "new_existing": m_existing, "new_ndt": m_ndt}

with open(f"{OUT_DIR}/cu0_10du_11gb_results.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved -> {OUT_DIR}/cu0_10du_11gb_results.json")

log_entry = {
    "ts": "2026-10-10T00:00:00",
    "script": "eval_cu0_10du_11gb.py",
    "desc": "Re-evaluated the 4 original small-registry checkpoints on a freshly recollected cu0_10du test "
            "set (data_input/test_csv_1cu10du_11gb_all, fixed ~11Gb traffic), CU-level and DU-level "
            "(per-DU), existing vs NDT, compared against the old test_csv_1cu10du_124 (fixed 12.4Gb) "
            "results. Reused already-computed thresholds/eps, no recalibration.",
    "results": results,
}
with open(LOG_FILE, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
print(f"logged -> {LOG_FILE}")
print("\nDONE")
