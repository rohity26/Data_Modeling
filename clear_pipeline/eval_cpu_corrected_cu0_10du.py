"""Evaluate the NEW N_DU-corrected checkpoint
(model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu0_du0du1.pt,
trained on cu1_du2+cu2_du3du4du5+cu3_du6du7du8du9 with the cpu/mem_pct/
mem_bytes N_DU correction applied in slice_features) on the cu0_10du target
(data_input/test_csv_1cu10du_11gb_all, fixed ~11Gb traffic), CU-level only.

Computes its OWN existing (pooled p99.9) threshold and its OWN NDT eps fresh
(no reuse of the uncorrected-model's precomputed thresholds -- the score
distributions differ because the features differ). Compares directly against
the uncorrected dlv1_model_ckpt_test_cu0_du0du1.pt baseline's already-logged
F1 on the same cu0_10du 11gb test set (eval_results_log.jsonl entry for
eval_cu0_10du_11gb.py: CPU=0.543, MEM=0.541, NET=0.538).

No existing files modified.
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import pandas as pd
import torch

import build_dataset_1_4_7_10 as bd_new
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
NEW_TRAFFIC_DIR = f"{DATA_DIR}/test_csv_1cu10du_11gb_all"
LOG_FILE = f"{CLEAR}/eval_results_log.jsonl"
OUT_JSON = f"{CLEAR}/eval_cpu_corrected_cu0_10du_results.json"

CKPT_PATH = f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu0_du0du1.pt"
TRAIN_BASE_DIR = f"{CLEAR}/CU_DL_UDP_STRESS"
TRAIN_TOPOS = ["cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]

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
TARGET = "cu0_10du"

# Same N_DU-correction constants as run_experiment_cpu_corrected.py, fit ONLY
# on the 3 small training topologies.
CPU_NDU_BETA1 = 0.004793
MEM_PCT_NDU_BETA1 = 0.0277313
MEM_BYTES_NDU_BETA1 = 1.89963e7

# Baseline (uncorrected) F1s on this exact target, from eval_cu0_10du_11gb.py's
# logged run (checkpoint dlv1_model_ckpt_test_cu0_du0du1.pt).
BASELINE_F1 = {"CPU": 0.543093269969097, "MEM": 0.5414364636923414, "NET": 0.5378704716143825}


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
    """Exact mirror of run_experiment_cpu_corrected.py's slice_features:
    slice -> impute -> apply N_DU correction (cpu/mem_pct/mem_bytes) -> net_diff/net_ratio."""
    cu = cu_raw.astype(np.float32).copy(); du = du_raw.astype(np.float32).copy()
    N_DU = du.shape[1]
    cu[:, 5] = cu[:, 5] / N_DU; cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]; du = du[:, :, DU_FEAT_SLICE]
    cu = impute_cpu_glitch(cu, CU_IRATE_IDX)

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

pooled_cu = []
for t, raw in zip(TRAIN_TOPOS, raw_streams):
    cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
    n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
    cu_sq, _ = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
    cu_sq = cu_sq[COLD_START_K:]
    pooled_cu.append(cu_sq)
    print(f"  {t}: cal rows={len(cu_sq)}")
pooled_cu = np.concatenate(pooled_cu, axis=0)
cu_fn_pool = feat_norm_calibrated(pooled_cu)
cu_cal_scores = lift_score(pooled_cu, cu_fn_pool)

cu_existing_thr = float(np.percentile(cu_cal_scores, 99.9))
smoothed_cal = ndt.ewma_smooth(cu_cal_scores)
ndt_sel = ndt.select_threshold(smoothed_cal)
cu_eps = ndt_sel["eps"]
print(f"\n  pooled cal rows={len(cu_cal_scores)}")
print(f"  existing threshold (p99.9) = {cu_existing_thr:.4f}")
print(f"  NDT eps (z={ndt_sel['z']:.1f})     = {cu_eps:.4f}")


def score_cu(csv_dir, cu_id, du_ids, pci_ids, stress_name):
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
    cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
    cu_sqerr, _ = phase_infer(model, cu_s, du_s)
    start = COLD_START_K
    lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress_name]).astype(int)
    scores = lift_score(cu_sqerr[start:], cu_fn_pool)
    return np.array(scores), lbl_ev


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


print(f"\n{'='*100}\nCU evaluation on cu0_10du (fixed ~11Gb) -- N_DU-corrected checkpoint\n{'='*100}")
topo_cfg = bd_new.TOPOLOGY_REGISTRY[TARGET]
cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]

results = {}
for stress in STRESS_ORDER:
    scores, lbl = score_cu(NEW_TRAFFIC_DIR, cu_id, du_ids, pci_ids, stress)
    m_existing = metrics((scores > cu_existing_thr).astype(int), lbl)
    smoothed = ndt.ewma_smooth(scores)
    m_ndt = metrics((smoothed > cu_eps).astype(int), lbl)
    base = BASELINE_F1[stress]
    print(f"  {stress}: baseline(uncorrected) F1={base:.3f}  |  "
          f"corrected existing F1={m_existing['f1']:.3f} (tp={m_existing['tp']} fp={m_existing['fp']} fn={m_existing['fn']})  "
          f"corrected ndt F1={m_ndt['f1']:.3f} (tp={m_ndt['tp']} fp={m_ndt['fp']} fn={m_ndt['fn']})")
    results[stress] = {"baseline_f1": base, "corrected_existing": m_existing, "corrected_ndt": m_ndt}

with open(OUT_JSON, "w") as f:
    json.dump({"existing_thr": cu_existing_thr, "ndt_eps": cu_eps, "results": results}, f, indent=1)
print(f"\nsaved -> {OUT_JSON}")

log_entry = {
    "ts": "2026-10-10T00:00:00",
    "script": "eval_cpu_corrected_cu0_10du.py",
    "desc": "Evaluated the NEW N_DU-corrected checkpoint (normfix_model_ckpt_test_cu0_du0du1.pt, "
            "trained on cu1_du2+cu2_du3du4du5+cu3_du6du7du8du9 with cpu/mem_pct/mem_bytes N_DU "
            "correction applied) on cu0_10du (fixed ~11Gb), CU-level, fresh own existing(p99.9)/NDT "
            "thresholds. Compared against uncorrected baseline F1 from eval_cu0_10du_11gb.py log entry.",
    "existing_thr": cu_existing_thr, "ndt_eps": cu_eps,
    "results": results,
}
with open(LOG_FILE, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
print(f"logged -> {LOG_FILE}")
print("\nDONE")
