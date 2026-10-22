"""Evaluate BOTH unseen stress topologies the user has fixed-traffic (~11Gb)
test data for -- cu0_7du (data_input/test_csv_1cu7du_11gb_all) and cu0_10du
(data_input/test_csv_1cu10du_11gb_all) -- against the TWO N_DU-corrected
checkpoints that have finished training so far:
  - normfix_model_ckpt_test_cu0_du0du1.pt       (trained on cu1_du2+cu2_du3du4du5+cu3_du6du7du8du9)
  - normfix_model_ckpt_test_cu3_du6du7du8du9.pt (trained on cu0_du0du1+cu1_du2+cu2_du3du4du5)

Both cu0_7du (N_DU=7) and cu0_10du (N_DU=10) are unseen to BOTH checkpoints
(small->big cross-registry). Each checkpoint uses its OWN beta1 N_DU-correction
constants (fit only on its own training topologies, never touching either
target) and its OWN freshly-computed existing(p99.9 CU / p99.5 DU) and NDT
(per-training-topology-averaged) thresholds -- no threshold or beta1 reuse
across checkpoints. Bundle/model/thresholds are built ONCE per checkpoint,
then reused for both targets.

This script does NOT touch the still-running cu1_du2 / cu2_du3du4du5 training
processes or any of their files -- read-only evaluation of the 2 already-
saved checkpoints only.
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
TRAIN_BASE_DIR = f"{CLEAR}/CU_DL_UDP_STRESS"
LOG_FILE = f"{CLEAR}/eval_results_log.jsonl"
OUT_JSON = f"{CLEAR}/eval_cu7du_cu10du_11gb_two_ckpts_results.json"

TARGETS = [
    {"name": "cu0_7du", "csv_dir": f"{DATA_DIR}/test_csv_1cu7du_11gb_all"},
    {"name": "cu0_10du", "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_11gb_all"},
]

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

CHECKPOINTS = [
    {
        "name": "cu0_du0du1",
        "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu0_du0du1.pt",
        "train_topos": ["cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"],
        "beta1": {"cpu": 0.004793, "mem_pct": 0.0277313, "mem_bytes": 1.89963e7},
    },
    {
        "name": "cu3_du6du7du8du9",
        "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_cu3_du6du7du8du9.pt",
        "train_topos": ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"],
        "beta1": {"cpu": 0.00410854, "mem_pct": 0.0280129, "mem_bytes": 1.89904e7},
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


def make_slice_features(beta1):
    def slice_features(cu_raw, du_raw):
        cu = cu_raw.astype(np.float32).copy(); du = du_raw.astype(np.float32).copy()
        N_DU = du.shape[1]
        cu[:, 5] = cu[:, 5] / N_DU; cu[:, 6] = cu[:, 6] / N_DU
        cu = cu[:, CU_FEAT_SLICE]; du = du[:, :, DU_FEAT_SLICE]
        cu = impute_cpu_glitch(cu, CU_IRATE_IDX); du = impute_cpu_glitch(du, DU_IRATE_IDX)

        cu[:, 0] = cu[:, 0] - beta1["cpu"] * (N_DU - 1)
        cu[:, 1] = cu[:, 1] - beta1["mem_pct"] * (N_DU - 1)
        cu[:, 2] = cu[:, 2] - beta1["mem_bytes"] * (N_DU - 1)

        _tx, _rx = cu[:, 3:4], cu[:, 4:5]
        cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)
        _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
        du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)
        return cu, du
    return slice_features


def load_train_npz(slice_features, base_dir, topo):
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


for t in TARGETS:
    cfg = bd_new.TOPOLOGY_REGISTRY[t["name"]]
    print(f"{t['name']} registry: cu_id={cfg['cu_id']}  du_ids={cfg['du_ids']}  N_DU={len(cfg['du_ids'])}")

all_results = {}

for ck in CHECKPOINTS:
    ck_name = ck["name"]
    print(f"\n{'='*100}\nCheckpoint: {ck_name}  (train={ck['train_topos']})\n{'='*100}")
    slice_features = make_slice_features(ck["beta1"])

    ckpt = torch.load(ck["ckpt"], map_location="cpu")
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(slice_features, TRAIN_BASE_DIR, t)))
                   for t in ck["train_topos"]]
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  loaded ckpt (cu_dim={ckpt['cu_dim']}, du_dim={ckpt['du_dim']}, embed_dim={ckpt['embed_dim']})")

    pooled_cu, pooled_du = [], []
    for t, raw in zip(ck["train_topos"], raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        cu_sq, du_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        cu_sq = cu_sq[COLD_START_K:]
        du_sq = du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1])
        pooled_cu.append(cu_sq); pooled_du.append(du_sq)
        print(f"  {t}: cal rows (cu)={len(cu_sq)}  (du pooled)={len(du_sq)}")

    # ---- CU threshold/eps ----
    cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)
    cu_existing_thr = float(np.percentile(cu_scores_pool, 99.9))
    i = 0
    cu_per_topo_eps = {}
    for t, cu_sq in zip(ck["train_topos"], pooled_cu):
        n = len(cu_sq)
        pick = ndt.select_threshold(ndt.ewma_smooth(cu_scores_pool[i:i + n]))
        cu_per_topo_eps[t] = pick["eps"]
        i += n
    cu_eps = float(np.mean(list(cu_per_topo_eps.values())))

    # ---- DU threshold/eps ----
    du_sqerr_pool = np.concatenate(pooled_du, axis=0)
    du_fn_pool = feat_norm_calibrated(du_sqerr_pool)
    du_scores_pool = lift_score(du_sqerr_pool, du_fn_pool)
    du_existing_thr = float(np.percentile(du_scores_pool, 99.5))
    du_per_topo_eps = {}
    for t, du_sq in zip(ck["train_topos"], pooled_du):
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

    def score_all_dus(csv_dir, cu_id, du_ids, pci_ids, stress_name):
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
            cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
            _, du_sqerr = phase_infer(model, cu_s, du_s)
            start = COLD_START_K
            lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress_name]).astype(int)
            scores = lift_score(du_sqerr[start:, my_idx, :], du_fn_pool)
            per_du[du_id] = (np.array(scores), lbl_ev)
        return per_du

    ck_results = {"thresholds": {
        "cu_existing_thr": cu_existing_thr, "cu_eps": cu_eps,
        "du_existing_thr": du_existing_thr, "du_eps": du_eps}}

    for target in TARGETS:
        t_name, csv_dir = target["name"], target["csv_dir"]
        topo_cfg = bd_new.TOPOLOGY_REGISTRY[t_name]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        t_results = {"cu": {}, "du": {}}

        print(f"\n{'='*100}\nCU evaluation on {t_name} (11gb, N_DU={len(du_ids)})  [ckpt={ck_name}]\n{'='*100}")
        for stress in STRESS_ORDER:
            scores, lbl = score_cu(csv_dir, cu_id, du_ids, pci_ids, stress)
            m_existing = metrics((scores > cu_existing_thr).astype(int), lbl)
            smoothed = ndt.ewma_smooth(scores)
            m_ndt = metrics((smoothed > cu_eps).astype(int), lbl)
            print(f"  {stress}: existing F1={m_existing['f1']:.3f} (tp={m_existing['tp']} fp={m_existing['fp']} fn={m_existing['fn']})  "
                  f"ndt F1={m_ndt['f1']:.3f} (tp={m_ndt['tp']} fp={m_ndt['fp']} fn={m_ndt['fn']})")
            t_results["cu"][stress] = {"existing": m_existing, "ndt": m_ndt}

        print(f"\n{'='*100}\nDU evaluation on {t_name} (11gb, per-DU)  [ckpt={ck_name}]\n{'='*100}")
        for stress in STRESS_ORDER:
            per_du = score_all_dus(csv_dir, cu_id, du_ids, pci_ids, stress)
            t_results["du"][stress] = {}
            f1_existing_list, f1_ndt_list = [], []
            for du_id, (scores, lbl) in per_du.items():
                m_existing = metrics((scores > du_existing_thr).astype(int), lbl)
                smoothed = ndt.ewma_smooth(scores)
                m_ndt = metrics((smoothed > du_eps).astype(int), lbl)
                t_results["du"][stress][du_id] = {"existing": m_existing, "ndt": m_ndt}
                f1_existing_list.append(m_existing["f1"]); f1_ndt_list.append(m_ndt["f1"])
            print(f"  {stress}: avg-over-{len(per_du)}-DUs  existing F1={np.mean(f1_existing_list):.3f}  ndt F1={np.mean(f1_ndt_list):.3f}")
            for du_id in per_du:
                r = t_results["du"][stress][du_id]
                print(f"      {du_id}: existing F1={r['existing']['f1']:.3f}  ndt F1={r['ndt']['f1']:.3f}")

        ck_results[t_name] = t_results

    all_results[ck_name] = ck_results

with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=1)
print(f"\nsaved -> {OUT_JSON}")

log_entry = {
    "ts": "2026-10-10T23:30:00",
    "script": "eval_cu7du_11gb_two_ckpts.py",
    "desc": "Evaluated cu0_7du (data_input/test_csv_1cu7du_11gb_all, fixed ~11Gb, user-collected) AND "
            "cu0_10du (data_input/test_csv_1cu10du_11gb_all) against the 2 N_DU-corrected checkpoints finished "
            "so far (normfix_model_ckpt_test_cu0_du0du1.pt and normfix_model_ckpt_test_cu3_du6du7du8du9.pt), "
            "each using its own beta1 correction constants and freshly-computed existing(p99.9 CU/p99.5 DU)/NDT "
            "(per-training-topology-averaged) thresholds. CU and DU (per-DU) level, all stress types. Read-only; "
            "did not touch the still-running cu1_du2/cu2_du3du4du5 training processes.",
    "results": all_results,
}
with open(LOG_FILE, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
print(f"logged -> {LOG_FILE}")
print("\nDONE")
