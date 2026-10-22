"""Evaluate the NEW N_DU-corrected checkpoint
(model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu0_du0du1.pt,
trained on cu1_du2+cu2_du3du4du5+cu3_du6du7du8du9 with the cpu/mem_pct/
mem_bytes N_DU correction applied in slice_features) on ALL THREE unseen
topologies: cu0_du0du1 (N_DU=2, own-registry held-out), cu0_7du (N_DU=7,
small->big), cu0_10du (N_DU=10, small->big, fixed ~11Gb recollection).

CU-level AND DU-level (per-DU), using the SAME official convention already
validated elsewhere in this project:
  - existing = pooled-percentile threshold fit on this checkpoint's own
    pooled calibration stream (CU: p99.9, DU: p99.5)
  - NDT = per-training-topology EWMA-smoothed calibration stream -> NDT
    (Hundman et al. 2018) threshold selection, eps averaged across the 3
    training topologies (CU and DU each have their own eps)
Both thresholds/eps are computed FRESH for this checkpoint (score
distributions differ from the uncorrected model because the cpu/mem_pct/
mem_bytes features themselves differ) -- no reuse of other checkpoints'
thresholds.

Baseline (uncorrected dlv1_model_ckpt_test_cu0_du0du1.pt) CU F1s are pulled
from already-logged results for side-by-side context:
  - cu0_du0du1: ndt_own_registry_results.json['original|cu0_du0du1']
  - cu0_7du:    ndt_crossreg_results.json['small_to_big']['cu0_du0du1']['targets']['cu0_7du']
  - cu0_10du:   cu0_10du_11gb_results.json['cu']['cu0_du0du1']  (fixed ~11Gb, apples-to-apples)

No existing files modified; no retraining.
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
OUT_JSON = f"{CLEAR}/eval_corrected_on_unseen_results.json"

CKPT_PATH = f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu0_du0du1.pt"
TRAIN_BASE_DIR = f"{CLEAR}/CU_DL_UDP_STRESS"
TRAIN_TOPOS = ["cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
PREPROCESS_VERSION = "v0"
CAL_FRAC = 0.2
WINDOW_LEN = 64
COLD_START_K = WINDOW_LEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]

CPU_NDU_BETA1 = 0.004793
MEM_PCT_NDU_BETA1 = 0.0277313
MEM_BYTES_NDU_BETA1 = 1.89963e7

TARGETS = [
    {"name": "cu0_du0du1", "bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    {"name": "cu0_7du", "bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu7du"},
    {"name": "cu0_10du", "bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_11gb_all"},
]

# Baseline (uncorrected) CU F1s pulled from already-logged results, for context only.
with open(f"{OUT_DIR}/ndt_own_registry_results.json") as f:
    _own = json.load(f)
with open(f"{OUT_DIR}/ndt_crossreg_results.json") as f:
    _cross = json.load(f)
with open(f"{OUT_DIR}/cu0_10du_11gb_results.json") as f:
    _11gb = json.load(f)

BASELINE_CU_F1 = {
    "cu0_du0du1": {s: _own["original|cu0_du0du1"]["rows"][s]["existing"]["f1"] for s in STRESS_ORDER},
    "cu0_7du": {s: _cross["small_to_big"]["cu0_du0du1"]["targets"]["cu0_7du"]["rows"][s]["existing"]["f1"] for s in STRESS_ORDER},
    "cu0_10du": {s: _11gb["cu"]["cu0_du0du1"][s]["old_existing_f1"] for s in STRESS_ORDER},
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
    """Exact mirror of run_experiment_cpu_corrected.py's slice_features."""
    cu = cu_raw.astype(np.float32).copy(); du = du_raw.astype(np.float32).copy()
    N_DU = du.shape[1]
    cu[:, 5] = cu[:, 5] / N_DU; cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]; du = du[:, :, DU_FEAT_SLICE]
    cu = impute_cpu_glitch(cu, CU_IRATE_IDX); du = impute_cpu_glitch(du, DU_IRATE_IDX)

    cu[:, 0] = cu[:, 0] - CPU_NDU_BETA1 * (N_DU - 1)
    cu[:, 1] = cu[:, 1] - MEM_PCT_NDU_BETA1 * (N_DU - 1)
    cu[:, 2] = cu[:, 2] - MEM_BYTES_NDU_BETA1 * (N_DU - 1)

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


print(f"{'='*100}\nBuilding N_DU-corrected checkpoint: {CKPT_PATH}\n{'='*100}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(TRAIN_BASE_DIR, t))) for t in TRAIN_TOPOS]
bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
print(f"  loaded ckpt (cu_dim={ckpt['cu_dim']}, du_dim={ckpt['du_dim']}, embed_dim={ckpt['embed_dim']})")

pooled_cu, pooled_du = [], []
for t, raw in zip(TRAIN_TOPOS, raw_streams):
    cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
    n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
    cu_sq, du_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
    cu_sq = cu_sq[COLD_START_K:]
    du_sq = du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1])
    pooled_cu.append(cu_sq); pooled_du.append(du_sq)
    print(f"  {t}: cal rows (cu)={len(cu_sq)}  (du pooled)={len(du_sq)}")

# ---- CU threshold/eps (official per-topo-averaged NDT convention) ----
cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)
cu_existing_thr = float(np.percentile(cu_scores_pool, 99.9))
i = 0
cu_per_topo_eps = {}
for t, cu_sq in zip(TRAIN_TOPOS, pooled_cu):
    n = len(cu_sq)
    pick = ndt.select_threshold(ndt.ewma_smooth(cu_scores_pool[i:i + n]))
    cu_per_topo_eps[t] = pick["eps"]
    i += n
cu_eps = float(np.mean(list(cu_per_topo_eps.values())))

# ---- DU threshold/eps (official per-topo-averaged NDT convention) ----
du_sqerr_pool = np.concatenate(pooled_du, axis=0)
du_fn_pool = feat_norm_calibrated(du_sqerr_pool)
du_scores_pool = lift_score(du_sqerr_pool, du_fn_pool)
du_existing_thr = float(np.percentile(du_scores_pool, 99.5))
du_per_topo_eps = {}
for t, du_sq in zip(TRAIN_TOPOS, pooled_du):
    du_fn_t = feat_norm_calibrated(du_sq)
    du_scores_t = lift_score(du_sq, du_fn_t)
    pick = ndt.select_threshold(ndt.ewma_smooth(du_scores_t))
    du_per_topo_eps[t] = pick["eps"]
du_eps = float(np.mean(list(du_per_topo_eps.values())))

print(f"\n  CU existing thr (p99.9) = {cu_existing_thr:.4f}   CU NDT eps (mean of 3) = {cu_eps:.4f}")
print(f"  DU existing thr (p99.5) = {du_existing_thr:.4f}   DU NDT eps (mean of 3) = {du_eps:.4f}")


def score_cu(csv_dir, cu_id, du_ids, pci_ids, stress_name):
    path = f"{csv_dir}/test_{cu_id}_{stress_name}_rep1.csv"
    df = pd.read_csv(path); cols = df.columns.tolist()
    bdmod = bd_cur
    cu_cols = bdmod.get_cu_columns(cu_id, cols)
    cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
    du_raw_list = [df[bdmod.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                   for d, p in zip(du_ids, pci_ids)]
    du_raw = np.stack(du_raw_list, axis=1)
    lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
    cu_feat, du_feat = slice_features(cu_raw, du_raw)
    block_id = np.zeros(len(cu_feat), dtype=np.int64)
    cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
    cu_sqerr, _ = phase_infer(model, cu_s, du_s)
    start = COLD_START_K
    lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress_name]).astype(int)
    scores = lift_score(cu_sqerr[start:], cu_fn_pool)
    return np.array(scores), lbl_ev


def score_all_dus(csv_dir, cu_id, du_ids, pci_ids, stress_name):
    bdmod = bd_cur
    per_du = {}
    for du_id in du_ids:
        path = f"{csv_dir}/test_{du_id}_{stress_name}_rep1.csv"
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
        lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress_name]).astype(int)
        scores = lift_score(du_sqerr[start:, my_idx, :], du_fn_pool)
        per_du[du_id] = (np.array(scores), lbl_ev)
    return per_du


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


results = {"cu": {}, "du": {}, "thresholds": {
    "cu_existing_thr": cu_existing_thr, "cu_eps": cu_eps,
    "du_existing_thr": du_existing_thr, "du_eps": du_eps}}

for target in TARGETS:
    name, bd_cur, csv_dir = target["name"], target["bd"], target["csv_dir"]
    topo_cfg = bd_cur.TOPOLOGY_REGISTRY[name]
    cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]

    print(f"\n{'='*100}\nCU evaluation on {name}  (N_DU={len(du_ids)}, csv_dir={csv_dir})\n{'='*100}")
    results["cu"][name] = {}
    for stress in STRESS_ORDER:
        scores, lbl = score_cu(csv_dir, cu_id, du_ids, pci_ids, stress)
        m_existing = metrics((scores > cu_existing_thr).astype(int), lbl)
        smoothed = ndt.ewma_smooth(scores)
        m_ndt = metrics((smoothed > cu_eps).astype(int), lbl)
        base = BASELINE_CU_F1[name][stress]
        print(f"  {stress}: baseline(uncorrected,existing) F1={base:.3f}  |  "
              f"corrected existing F1={m_existing['f1']:.3f} (tp={m_existing['tp']} fp={m_existing['fp']} fn={m_existing['fn']})  "
              f"corrected ndt F1={m_ndt['f1']:.3f} (tp={m_ndt['tp']} fp={m_ndt['fp']} fn={m_ndt['fn']})")
        results["cu"][name][stress] = {"baseline_existing_f1": base, "corrected_existing": m_existing, "corrected_ndt": m_ndt}

    print(f"\n{'='*100}\nDU evaluation on {name} (per-DU)\n{'='*100}")
    results["du"][name] = {}
    for stress in STRESS_ORDER:
        per_du = score_all_dus(csv_dir, cu_id, du_ids, pci_ids, stress)
        results["du"][name][stress] = {}
        f1_existing_list, f1_ndt_list = [], []
        for du_id, (scores, lbl) in per_du.items():
            m_existing = metrics((scores > du_existing_thr).astype(int), lbl)
            smoothed = ndt.ewma_smooth(scores)
            m_ndt = metrics((smoothed > du_eps).astype(int), lbl)
            results["du"][name][stress][du_id] = {"existing": m_existing, "ndt": m_ndt}
            f1_existing_list.append(m_existing["f1"]); f1_ndt_list.append(m_ndt["f1"])
        print(f"  {stress}: avg-over-{len(per_du)}-DUs  existing F1={np.mean(f1_existing_list):.3f}  ndt F1={np.mean(f1_ndt_list):.3f}")
        for du_id in per_du:
            r = results["du"][name][stress][du_id]
            print(f"      {du_id}: existing F1={r['existing']['f1']:.3f}  ndt F1={r['ndt']['f1']:.3f}")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved -> {OUT_JSON}")

log_entry = {
    "ts": "2026-10-10T00:00:00",
    "script": "eval_corrected_on_unseen.py",
    "desc": "Evaluated the NEW N_DU-corrected checkpoint (normfix_model_ckpt_test_cu0_du0du1.pt, trained "
            "on cu1_du2+cu2_du3du4du5+cu3_du6du7du8du9 with cpu/mem_pct/mem_bytes N_DU correction applied) "
            "on all 3 unseen topologies (cu0_du0du1, cu0_7du, cu0_10du[fixed 11gb]), CU-level and DU-level "
            "(per-DU), fresh own existing(p99.9/p99.5)/NDT(per-topo-averaged) thresholds. CU baseline F1s "
            "(uncorrected dlv1 checkpoint, existing method) included for context.",
    "results": results,
}
with open(LOG_FILE, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
print(f"logged -> {LOG_FILE}")
print("\nDONE")
