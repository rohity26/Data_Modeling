"""Re-score DUs for (A) Own-Registry new-registry side (cu1_du2,
cu3_du6du7du8du9, cu0_7du, cu0_10du -- own held-out topology) and (B)
Small->Big (4 original checkpoints -> cu0_7du/cu0_10du), this time saving
FULL per-DU score/lbl arrays (not just scalar tp/fp/fn) for threshold-plot
panels. Reuses NDT eps already computed and saved in
ndt_own_registry_results.json / ndt_crossreg_results.json -- no recalibration,
just fresh model inference to capture the raw arrays.
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]

N_DU_NEW = {"cu1_du2": 1, "cu3_du6du7du8du9": 4, "cu0_7du": 7, "cu0_10du": 10}
NEW_CFG = {"bd": bd_new, "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS", "ckpt_dir": f"{CLEAR}/model_1_4_7_10",
           "ckpt_prefix": "v2_", "all_topos": list(N_DU_NEW), "n_du_of": N_DU_NEW,
           "test_source": {"cu1_du2": f"{DATA_DIR}/test_csv_dl_udp_2iface",
                            "cu3_du6du7du8du9": f"{DATA_DIR}/test_csv_dl_udp_2iface",
                            "cu0_7du": f"{DATA_DIR}/test_csv_1cu7du",
                            "cu0_10du": f"{DATA_DIR}/test_csv_1cu10du_124"}}
ORIG_CFG = {"bd": bd_orig, "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "ckpt_dir": f"{CLEAR}/model_DLUDP_NET",
            "ckpt_prefix": "dlv1_", "all_topos": ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]}
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


def build_checkpoint_du_fn(cfg, test_topo):
    ckpt = torch.load(f"{cfg['ckpt_dir']}/{cfg['ckpt_prefix']}model_ckpt_test_{test_topo}.pt", map_location="cpu")
    train_topos = ckpt["topos"]
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(cfg["base_dir"], t))) for t in train_topos]
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled_du_sqerr = []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        _, du_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        du_sq = du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1])
        pooled_du_sqerr.append(du_sq)
    du_fn_pool = feat_norm_calibrated(np.concatenate(pooled_du_sqerr, axis=0))
    return {"bundle": bundle, "model": model, "du_fn_pool": du_fn_pool, "train_topos": train_topos}


def score_all_dus_full(ck, bd, csv_dir, cu_id, du_ids, pci_ids, stress_name):
    """Own-file-only score every DU. Returns dict du_id -> (score_list, lbl_list)."""
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
        per_du[du_id] = ([round(float(x), 4) for x in scores], [int(x) for x in lbl_ev])
    return per_du


with open(f"{OUT_DIR}/ndt_own_registry_results.json") as f:
    ndt_own = json.load(f)
with open(f"{OUT_DIR}/ndt_crossreg_results.json") as f:
    ndt_cross = json.load(f)

results = {"own_new": {}, "small_to_big": {}}

print(f"{'='*95}\nDU full-score arrays: Own-Registry (new-registry side)\n{'='*95}")
for test_topo in NEW_CFG["all_topos"]:
    key = f"new|{test_topo}"
    eps = ndt_own[key]["ndt_eps_mean"]
    thr_existing = ndt_own[key]["rows"]["CPU"]["existing"]["thr"]
    ck = build_checkpoint_du_fn(NEW_CFG, test_topo)
    topo_cfg = bd_new.TOPOLOGY_REGISTRY[test_topo]
    cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
    csv_dir = NEW_CFG["test_source"][test_topo]
    print(f"\n### {key}  eps={eps:.2f}  thr_existing={thr_existing:.2f}")
    rows = {}
    for stress in STRESS_ORDER:
        per_du = score_all_dus_full(ck, bd_new, csv_dir, cu_id, du_ids, pci_ids, stress)
        rows[stress] = {du_id: {"score": s, "lbl": l} for du_id, (s, l) in per_du.items()}
        print(f"  {stress}: {list(per_du.keys())}")
    results["own_new"][test_topo] = {"eps": eps, "thr_existing": thr_existing, "n_du": N_DU_NEW[test_topo], "rows": rows}

print(f"\n{'='*95}\nDU full-score arrays: Small->Big\n{'='*95}")
for test_topo in ORIG_CFG["all_topos"]:
    eps = ndt_cross["small_to_big"][test_topo]["eps"]
    thr_existing = ndt_cross["small_to_big"][test_topo]["targets"]["cu0_7du"]["rows"]["CPU"]["existing"]["thr"]
    ck = build_checkpoint_du_fn(ORIG_CFG, test_topo)
    print(f"\n### smallbig|{test_topo}  eps={eps:.2f}  thr_existing={thr_existing:.2f}")
    results["small_to_big"][test_topo] = {"eps": eps, "thr_existing": thr_existing, "targets": {}}
    for target in BIG_TARGETS:
        topo_cfg = bd_new.TOPOLOGY_REGISTRY[target]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        csv_dir = NEW_CFG["test_source"][target]
        rows = {}
        for stress in STRESS_ORDER:
            per_du = score_all_dus_full(ck, bd_new, csv_dir, cu_id, du_ids, pci_ids, stress)
            rows[stress] = {du_id: {"score": s, "lbl": l} for du_id, (s, l) in per_du.items()}
            print(f"  -> {target} {stress}: {list(per_du.keys())}")
        results["small_to_big"][test_topo]["targets"][target] = {"n_du": N_DU_NEW[target], "rows": rows}

with open(f"{OUT_DIR}/data_ndt_du_fullscores.js", "w") as f:
    f.write("window.DATA_NDT_DU_FULL = " + json.dumps(results) + ";\n")
print(f"\nsaved -> {OUT_DIR}/data_ndt_du_fullscores.js")
print("DONE")
