"""DU-level NDT (Hundman et al. 2018) evaluation -- same 3-part scope as
eval_du_perdu_full.py (own-registry LOO, small->big, big->small), but
replacing the pooled-percentile threshold (existing, p99.5) with NDT: per
training-topology DU calibration stream (pooled across that topology's own
DUs), EWMA-smoothed, NDT-selected eps, then averaged across the checkpoint's
training topologies -- same recipe already validated for CU in
eval_ndt_own_registry.py / eval_ndt_crossreg.py.

Per-DU rows (not pooled across DUs), own-file-only. No retraining; no
existing files modified.
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


def build_checkpoint(cfg, test_topo):
    ckpt = torch.load(f"{cfg['ckpt_dir']}/{cfg['ckpt_prefix']}model_ckpt_test_{test_topo}.pt", map_location="cpu")
    train_topos = ckpt["topos"]
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(cfg["base_dir"], t))) for t in train_topos]
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    per_topo_eps = {}
    pooled_du_sqerr = []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        _, du_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        du_sq = du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1])  # pool all DUs of this topology together
        pooled_du_sqerr.append(du_sq)
        du_fn_t = feat_norm_calibrated(du_sq)
        du_scores_t = lift_score(du_sq, du_fn_t)
        pick = ndt.select_threshold(ndt.ewma_smooth(du_scores_t))
        per_topo_eps[t] = pick

    du_sqerr_pool = np.concatenate(pooled_du_sqerr, axis=0)
    du_fn_pool = feat_norm_calibrated(du_sqerr_pool)
    du_scores_pool = lift_score(du_sqerr_pool, du_fn_pool)
    baseline_thr = float(np.percentile(du_scores_pool, 99.5))  # existing convention, unchanged
    ndt_eps_mean = float(np.mean([p["eps"] for p in per_topo_eps.values()]))

    return {"bundle": bundle, "model": model, "du_fn_pool": du_fn_pool,
            "train_topos": train_topos, "per_topo_eps": per_topo_eps,
            "baseline_thr": baseline_thr, "ndt_eps_mean": ndt_eps_mean}


def metrics(tp, fp, fn):
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


def score_all_dus_real(ck, bd, csv_dir, cu_id, du_ids, pci_ids, stress_name, thr, smooth=False, pruned=False):
    """Own-file-only score every DU. Returns dict du_id -> (tp, fp, fn).
    smooth=True applies the same EWMA smoothing NDT's eps was selected on
    (use for thr=ndt_eps_mean); smooth=False thresholds the raw score
    (use for thr=baseline_thr, which was picked on the raw/unsmoothed stream)."""
    per_du = {}
    for du_id, pci in zip(du_ids, pci_ids):
        path = f"{csv_dir}/test_{du_id}_{stress_name}_rep1.csv"
        df = pd.read_csv(path); cols = df.columns.tolist()
        cu_cols = bd.get_cu_columns(cu_id, cols)
        cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
        du_raw_list = [df[bd.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                       for d, p in zip(du_ids, pci_ids)]
        du_raw = np.stack(du_raw_list, axis=1)
        my_idx = du_ids.index(du_id)
        lbl = df[f"{du_id}_stressType"].fillna(0).astype(int).values
        cu_feat, du_feat = slice_features(cu_raw, du_raw)
        block_id = np.zeros(len(cu_feat), dtype=np.int64)
        cu_s, du_s, _, _ = transform_stream(ck["bundle"], cu_feat, du_feat, block_id)
        _, du_sqerr = phase_infer(ck["model"], cu_s, du_s)
        start = COLD_START_K
        stype_num = STYPE_NUM[stress_name]
        lbl_ev = (lbl[start + 1:] == stype_num).astype(int)
        scores = lift_score(du_sqerr[start:, my_idx, :], ck["du_fn_pool"])
        scored = ndt.ewma_smooth(scores) if smooth else scores
        if pruned:
            pred = ndt.prune(scored, thr).astype(int)
        else:
            pred = (scored > thr).astype(int)
        tp = int(((pred == 1) & (lbl_ev == 1)).sum())
        fp = int(((pred == 1) & (lbl_ev == 0)).sum())
        fn = int(((pred == 0) & (lbl_ev == 1)).sum())
        per_du[du_id] = (tp, fp, fn)
    return per_du


results = {"own": {}, "small_to_big": {}, "big_to_small": {}}

print(f"{'='*100}\nOWN-REGISTRY DU LOO -- existing (pooled p99.5) vs NDT\n{'='*100}")
for cfg, regname in [(ORIG_CFG, "original"), (NEW_CFG, "new")]:
    for test_topo in cfg["all_topos"]:
        key = f"{regname}|{test_topo}"
        ck = build_checkpoint(cfg, test_topo)
        print(f"\n### {key}  baseline={ck['baseline_thr']:.2f}  ndt_eps_mean={ck['ndt_eps_mean']:.2f}")
        topo_cfg = cfg["bd"].TOPOLOGY_REGISTRY[test_topo]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = cfg["test_source"][test_topo]
        rows = {}
        for stress in STRESS_ORDER:
            per_b = score_all_dus_real(ck, cfg["bd"], csv_dir, cu_id, du_ids, pci_ids, stress, ck["baseline_thr"])
            per_ndt = score_all_dus_real(ck, cfg["bd"], csv_dir, cu_id, du_ids, pci_ids, stress, ck["ndt_eps_mean"], smooth=True)
            rows[stress] = {}
            for du_id in du_ids:
                tp_b, fp_b, fn_b = per_b[du_id]
                tp_n, fp_n, fn_n = per_ndt[du_id]
                m_b, m_n = metrics(tp_b, fp_b, fn_b), metrics(tp_n, fp_n, fn_n)
                rows[stress][du_id] = {"existing": {"thr": ck["baseline_thr"], **m_b},
                                        "ndt": {"thr": ck["ndt_eps_mean"], **m_n}}
                print(f"    {stress} {du_id}: existing F1={m_b['f1']:.3f}  NDT F1={m_n['f1']:.3f}")
        results["own"][key] = rows

print(f"\n{'='*100}\nSMALL->BIG DU -- existing vs NDT\n{'='*100}")
for test_topo in ORIG_CFG["all_topos"]:
    ck = build_checkpoint(ORIG_CFG, test_topo)
    print(f"\n### smallbig|{test_topo}  baseline={ck['baseline_thr']:.2f}  ndt_eps_mean={ck['ndt_eps_mean']:.2f}")
    for target in BIG_TARGETS:
        topo_cfg = bd_new.TOPOLOGY_REGISTRY[target]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = NEW_CFG["test_source"][target]
        rows = {}
        for stress in STRESS_ORDER:
            per_b = score_all_dus_real(ck, bd_new, csv_dir, cu_id, du_ids, pci_ids, stress, ck["baseline_thr"])
            per_ndt = score_all_dus_real(ck, bd_new, csv_dir, cu_id, du_ids, pci_ids, stress, ck["ndt_eps_mean"], smooth=True)
            rows[stress] = {}
            for du_id in du_ids:
                tp_b, fp_b, fn_b = per_b[du_id]
                tp_n, fp_n, fn_n = per_ndt[du_id]
                m_b, m_n = metrics(tp_b, fp_b, fn_b), metrics(tp_n, fp_n, fn_n)
                rows[stress][du_id] = {"existing": {"thr": ck["baseline_thr"], **m_b},
                                        "ndt": {"thr": ck["ndt_eps_mean"], **m_n}}
                print(f"    -> {target} {stress} {du_id}: existing F1={m_b['f1']:.3f}  NDT F1={m_n['f1']:.3f}")
        results["small_to_big"][f"{test_topo}->{target}"] = rows

print(f"\n{'='*100}\nBIG->SMALL DU -- existing vs NDT\n{'='*100}")
for test_topo in NEW_CFG["all_topos"]:
    ck = build_checkpoint(NEW_CFG, test_topo)
    print(f"\n### bigsmall|{test_topo}  baseline={ck['baseline_thr']:.2f}  ndt_eps_mean={ck['ndt_eps_mean']:.2f}")
    for target in SMALL_TARGETS:
        topo_cfg = bd_orig.TOPOLOGY_REGISTRY[target]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = ORIG_CFG["test_source"][target]
        rows = {}
        for stress in STRESS_ORDER:
            per_b = score_all_dus_real(ck, bd_orig, csv_dir, cu_id, du_ids, pci_ids, stress, ck["baseline_thr"])
            per_ndt = score_all_dus_real(ck, bd_orig, csv_dir, cu_id, du_ids, pci_ids, stress, ck["ndt_eps_mean"], smooth=True)
            rows[stress] = {}
            for du_id in du_ids:
                tp_b, fp_b, fn_b = per_b[du_id]
                tp_n, fp_n, fn_n = per_ndt[du_id]
                m_b, m_n = metrics(tp_b, fp_b, fn_b), metrics(tp_n, fp_n, fn_n)
                rows[stress][du_id] = {"existing": {"thr": ck["baseline_thr"], **m_b},
                                        "ndt": {"thr": ck["ndt_eps_mean"], **m_n}}
                print(f"    -> {target} {stress} {du_id}: existing F1={m_b['f1']:.3f}  NDT F1={m_n['f1']:.3f}")
        results["big_to_small"][f"{test_topo}->{target}"] = rows

with open(f"{OUT_DIR}/data_ndt_du.js", "w") as f:
    f.write("window.DATA_NDT_DU = " + json.dumps(results) + ";\n")
with open(f"{OUT_DIR}/ndt_du_results.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved -> {OUT_DIR}/data_ndt_du.js")

log_entry = {
    "ts": "2026-10-09T02:00:00",
    "script": "eval_ndt_du.py",
    "desc": "DU-level NDT (Hundman et al. 2018) vs existing pooled-percentile (p99.5) baseline, "
            "3-part scope (own-registry LOO, small->big, big->small), per-DU rows, own-file-only. "
            "NDT: per training-topology pooled-DU calibration stream, EWMA-smoothed, NDT-selected eps, "
            "averaged across the checkpoint's training topologies -- same recipe as CU NDT.",
    "results": results,
}
with open(LOG_FILE, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
print(f"logged -> {LOG_FILE}")
print("\nDONE")
