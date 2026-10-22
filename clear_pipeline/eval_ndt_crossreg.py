"""NDT (Hundman et al. 2018) cross-registry evaluation, both directions:
  Small->Big: 4 original-registry checkpoints scored on cu0_7du / cu0_10du
              (reuses saved full score arrays, data_p985_smallbig_fullscores.js
              -- no new model inference needed).
  Big->Small: 4 new-registry checkpoints scored on cu0_du0du1 / cu2_du3du4du5
              (fresh model inference -- no saved full score arrays exist for
              this direction yet).

NDT thresholds (per training-topology eps, averaged) are REUSED from
ndt_own_registry_results.json -- already computed per checkpoint from
calibration-only data in eval_ndt_own_registry.py, not recomputed here.
Existing-method comparison numbers are REUSED from p985_partb.json.
Standalone script -- does not modify or import any existing project file.
"""
import sys, re, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import pandas as pd
import torch

import build_dataset as bd_orig
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR
from scoring import lift_score
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
OUT_DIR = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad"
LOG_FILE = f"{CLEAR}/eval_results_log.jsonl"

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
PREPROCESS_VERSION = "v0"
COLD_START_K = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRESS_ORDER = ["CPU", "MEM", "NET"]
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}

NEW_CFG = {"base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS", "ckpt_dir": f"{CLEAR}/model_1_4_7_10", "ckpt_prefix": "v2_"}
BIGSMALL_TARGETS = ["cu0_du0du1", "cu2_du3du4du5"]  # scored by NEW checkpoints
BIGSMALL_CKPTS = ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]


def impute_cpu_glitch(arr, irate_idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        glitch = arr[t, irate_idx] < eps
        arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
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
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)
    du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()


def score_topology(bundle, model, cu_fn_pool, csv_dir, cu_id, du_ids, pci_ids, stress_name):
    path = f"{csv_dir}/test_{cu_id}_{stress_name}_rep1.csv"
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    cu_cols = bd_orig.get_cu_columns(cu_id, cols)
    cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
    du_raw_list = [df[bd_orig.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                   for d, p in zip(du_ids, pci_ids)]
    du_raw = np.stack(du_raw_list, axis=1)
    lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
    cu_feat, du_feat = slice_features(cu_raw, du_raw)
    block_id = np.zeros(len(cu_feat), dtype=np.int64)
    cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
    cu_sqerr = phase_infer(model, cu_s, du_s)
    start = COLD_START_K
    lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress_name]).astype(int)
    scores = lift_score(cu_sqerr[start:], cu_fn_pool)
    return [round(float(x), 4) for x in scores], [int(x) for x in lbl_ev]


def eval_ndt(scores, lbl, eps):
    smoothed = ndt.ewma_smooth(scores)
    pred_simple = (smoothed > eps).astype(int)
    pred_pruned = ndt.prune(smoothed, eps).astype(int)
    return ndt.metrics(pred_simple, lbl), ndt.metrics(pred_pruned, lbl)


with open(f"{OUT_DIR}/ndt_own_registry_results.json") as f:
    ndt_own = json.load(f)
with open(f"{OUT_DIR}/p985_partb.json") as f:
    partb = json.load(f)

results = {"small_to_big": {}, "big_to_small": {}}

# ============================================================
# SMALL -> BIG: reuse saved full score arrays, reuse saved NDT eps
# ============================================================
print(f"{'='*95}\nSMALL -> BIG  (4 original-registry checkpoints, scored on cu0_7du / cu0_10du)\n{'='*95}")
with open(f"{OUT_DIR}/data_p985_smallbig_fullscores.js") as f:
    txt = f.read()
txt = re.sub(r"^window\.DATA_P985_SMALLBIG_FULL\s*=\s*", "", txt.strip())
txt = re.sub(r";\s*$", "", txt)
SB_FULL = json.loads(txt)

for ckpt in ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]:
    eps = ndt_own[f"original|{ckpt}"]["ndt_eps_mean"]
    print(f"\n### {ckpt}  (eps={eps:.4f})")
    results["small_to_big"][ckpt] = {"eps": eps, "targets": {}}
    for target, rec in SB_FULL[ckpt]["targets"].items():
        results["small_to_big"][ckpt]["targets"][target] = {"n_du": rec["n_du"], "rows": {}}
        for stress in STRESS_ORDER:
            row = rec["rows"][stress]
            scores, lbl = np.array(row["score"]), np.array(row["lbl"])
            m_simple, m_pruned = eval_ndt(scores, lbl, eps)
            m_existing = partb[f"smallbig|{ckpt}"]["targets"][target]["rows"][stress]["baseline"]
            print(f"  -> {target:10s} {stress}: existing f1={m_existing['f1']:.4f}  "
                  f"| NDT f1={m_simple['f1']:.4f}  | NDT+pruned f1={m_pruned['f1']:.4f}")
            results["small_to_big"][ckpt]["targets"][target]["rows"][stress] = {
                "existing": m_existing, "ndt_simple": m_simple, "ndt_pruned": m_pruned}

# ============================================================
# BIG -> SMALL: fresh model inference, reuse saved NDT eps
# ============================================================
print(f"\n{'='*95}\nBIG -> SMALL  (4 new-registry checkpoints, scored on cu0_du0du1 / cu2_du3du4du5)\n{'='*95}")
for ckpt in BIGSMALL_CKPTS:
    eps = ndt_own[f"new|{ckpt}"]["ndt_eps_mean"]
    ckpt_data = torch.load(f"{NEW_CFG['ckpt_dir']}/{NEW_CFG['ckpt_prefix']}model_ckpt_test_{ckpt}.pt", map_location="cpu")
    train_topos = ckpt_data["topos"]
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(NEW_CFG["base_dir"], t))) for t in train_topos]
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
    model = CalibratedTopoAR(cu_dim=ckpt_data["cu_dim"], du_dim=ckpt_data["du_dim"], embed_dim=ckpt_data["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt_data["state_dict"])

    # need cu_fn_pool (same normalizer used when this checkpoint's existing
    # thresholds/scores were computed) -- recompute from its own pooled calibration
    from model_calibrated import feat_norm_calibrated
    CAL_FRAC = 0.2
    pooled = []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        cu_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])[COLD_START_K:]
        pooled.append(cu_sq)
    cu_fn_pool = feat_norm_calibrated(np.concatenate(pooled, axis=0))

    print(f"\n### {ckpt}  (eps={eps:.4f}, trained on {train_topos})")
    results["big_to_small"][ckpt] = {"eps": eps, "targets": {}}
    for target in BIGSMALL_TARGETS:
        topo_cfg = bd_orig.TOPOLOGY_REGISTRY[target]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        results["big_to_small"][ckpt]["targets"][target] = {"rows": {}}
        for stress in STRESS_ORDER:
            scores, lbl = score_topology(bundle, model, cu_fn_pool, f"{DATA_DIR}/test_csv_dl_udp_2iface",
                                          cu_id, du_ids, pci_ids, stress)
            scores, lbl = np.array(scores), np.array(lbl)
            m_simple, m_pruned = eval_ndt(scores, lbl, eps)
            m_existing = partb[f"bigsmall|{ckpt}"]["targets"][target]["rows"][stress]["baseline"]
            print(f"  -> {target:15s} {stress}: existing f1={m_existing['f1']:.4f}  "
                  f"| NDT f1={m_simple['f1']:.4f}  | NDT+pruned f1={m_pruned['f1']:.4f}")
            results["big_to_small"][ckpt]["targets"][target]["rows"][stress] = {
                "existing": m_existing, "ndt_simple": m_simple, "ndt_pruned": m_pruned}

with open(f"{OUT_DIR}/ndt_crossreg_results.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved -> {OUT_DIR}/ndt_crossreg_results.json")

log_entry = {
    "ts": "2026-10-09T01:00:00",
    "script": "eval_ndt_crossreg.py",
    "desc": "NDT (Hundman et al. 2018, per-topology eps averaged + EWMA-smoothed, reused from "
            "ndt_own_registry_results.json) vs existing pooled-percentile baseline, cross-registry both "
            "directions: small->big (4 original checkpoints -> cu0_7du/cu0_10du, reused saved full score "
            "arrays) and big->small (4 new checkpoints -> cu0_du0du1/cu2_du3du4du5, fresh model inference). "
            "CPU/MEM/NET.",
    "results": results,
}
with open(LOG_FILE, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
print(f"logged -> {LOG_FILE}")
print("\nDONE")
