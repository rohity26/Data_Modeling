"""Score real held-out topologies at p99.9, comparing OLD (contaminated
features) vs NEW (both fixes: mem_pct glitch imputation + net_ratio scaler-
IQR floor) -- no retraining, same frozen checkpoints.

Covers both directions already on the page:
  - small->big: 4 original checkpoints -> cu0_7du / cu0_10du
  - big->small: {4,7,10}-DU checkpoint -> cu1_du2 / cu0_du0du1 / cu2_du3du4du5
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

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX_OLD = [0, 3, 4]
DU_IRATE_IDX_OLD = [0, 3, 4, 5]
CU_IRATE_IDX_NEW = [0, 1, 3, 4]
DU_IRATE_IDX_NEW = [0, 1, 3, 4, 5]
PREPROCESS_VERSION = "v0"
CAL_FRAC = 0.2
WINDOW_LEN = 64
COLD_START_K = WINDOW_LEN
CU_THRESHOLD_PCT = 99.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
NET_RATIO_SCALE_FLOOR = 0.01

TEST_SOURCE_NEW = {
    "cu1_du2": f"{DATA_DIR}/test_csv_dl_udp_2iface",
    "cu3_du6du7du8du9": f"{DATA_DIR}/test_csv_dl_udp_2iface",
    "cu0_7du": f"{DATA_DIR}/test_csv_1cu7du",
    "cu0_10du": f"{DATA_DIR}/test_csv_1cu10du_124",
}
TEST_SOURCE_ORIG = {
    "cu1_du2": f"{DATA_DIR}/test_csv_dl_udp_2iface",
    "cu0_du0du1": f"{DATA_DIR}/test_csv_dl_udp_2iface",
    "cu2_du3du4du5": f"{DATA_DIR}/test_csv_dl_udp_2iface",
    "cu3_du6du7du8du9": f"{DATA_DIR}/test_csv_dl_udp_2iface",
}


def impute_glitch(arr, idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            g = arr[t, idx] < eps
            arr[t, idx] = np.where(g, arr[t - 1, idx], arr[t, idx])
        else:
            g = arr[t, :, idx] < eps
            arr[t, :, idx] = np.where(g, arr[t - 1, :, idx], arr[t, :, idx])
    return arr


def slice_features(cu_raw, du_raw, fixed):
    cu = cu_raw.astype(np.float32).copy()
    du = du_raw.astype(np.float32).copy()
    N_DU = du.shape[1]
    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]
    cu_idx = CU_IRATE_IDX_NEW if fixed else CU_IRATE_IDX_OLD
    du_idx = DU_IRATE_IDX_NEW if fixed else DU_IRATE_IDX_OLD
    cu = impute_glitch(cu, cu_idx)
    du = impute_glitch(du, du_idx)
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)
    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)
    return cu, du


def load_train_npz(base_dir, topo, fixed):
    z = dict(np.load(f"{base_dir}/{topo}_stress3/train.npz"))
    cu, du = slice_features(z["cu"], z["du"], fixed)
    return cu, du, z["block_id"].astype(np.int64)


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


def run_checkpoint(base_dir, ckpt_path, bd_mod, test_source, targets):
    """targets: list of (target_topo,) to score, using the checkpoint's own
    held-out-pool. Returns dict: target_topo -> stress -> {old:{...}, new:{...}}"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_topos = ckpt["topos"]
    out = {}

    for fixed, key in [(False, "old"), (True, "new")]:
        raw_streams = []
        for t in train_topos:
            cu, du, bid = load_train_npz(base_dir, t, fixed)
            raw_streams.append({"cu": cu, "du": du, "block_id": bid})
        bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
        if fixed:
            bundle.cu_scaler.scale_[-1] = max(bundle.cu_scaler.scale_[-1], NET_RATIO_SCALE_FLOOR)
            bundle.du_scaler.scale_[-1] = max(bundle.du_scaler.scale_[-1], NET_RATIO_SCALE_FLOOR)

        model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])

        pooled = []
        for t, raw in zip(train_topos, raw_streams):
            cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
            n_total = len(cu_s)
            n_cal = int(round(CAL_FRAC * n_total))
            n_fit = n_total - n_cal
            cu_sq, _ = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
            pooled.append(cu_sq[COLD_START_K:])
        cu_sqerr_pool = np.concatenate(pooled, axis=0)
        cu_fn = feat_norm_calibrated(cu_sqerr_pool)
        cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn)
        thr = float(np.percentile(cu_scores_pool, CU_THRESHOLD_PCT))

        for target_topo in targets:
            topo_cfg = bd_mod.TOPOLOGY_REGISTRY[target_topo]
            cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
            csv_dir = test_source[target_topo]
            out.setdefault(target_topo, {})

            for stress_name in ["CPU", "MEM", "NET"]:
                path = f"{csv_dir}/test_{cu_id}_{stress_name}_rep1.csv"
                df = pd.read_csv(path)
                cols = df.columns.tolist()
                cu_cols = bd_mod.get_cu_columns(cu_id, cols)
                cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
                du_raw_list = [df[bd_mod.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                               for d, p in zip(du_ids, pci_ids)]
                du_raw = np.stack(du_raw_list, axis=1)
                lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
                cu_feat, du_feat = slice_features(cu_raw, du_raw, fixed)
                block_id = np.zeros(len(cu_feat), dtype=np.int64)
                cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
                cu_sqerr, _ = phase_infer(model, cu_s, du_s)
                start = COLD_START_K
                stype_num = STYPE_NUM[stress_name]
                lbl_ev = (lbl[start + 1:] == stype_num).astype(int)
                scores = lift_score(cu_sqerr[start:], cu_fn)
                pred = (scores > thr).astype(int)
                m = metrics(pred, lbl_ev)
                out[target_topo].setdefault(stress_name, {})
                out[target_topo][stress_name][key] = {
                    "thr": thr, "score": [round(float(x), 4) for x in scores],
                    "lbl": [int(x) for x in lbl_ev], **m,
                }
                print(f"  [{key}] {target_topo} {stress_name}: thr={thr:.2f} P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f}")
    return out


ALL_RESULTS = {}

print("=" * 100)
print("SMALL -> BIG (4 original checkpoints -> cu0_7du / cu0_10du)")
print("=" * 100)
for test_topo in ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]:
    ckpt_path = f"{CLEAR}/model_DLUDP_NET/dlv1_model_ckpt_test_{test_topo}.pt"
    print(f"\n### checkpoint={test_topo}")
    res = run_checkpoint(f"{CLEAR}/CU_DL_UDP_STRESS", ckpt_path, bd_new, TEST_SOURCE_NEW, ["cu0_7du", "cu0_10du"])
    ALL_RESULTS[f"small2big|{test_topo}"] = res

print("\n" + "=" * 100)
print("BIG -> SMALL ({4,7,10}-DU checkpoint -> cu1_du2 / cu0_du0du1 / cu2_du3du4du5)")
print("=" * 100)
ckpt_path = f"{CLEAR}/model_1_4_7_10/v2_model_ckpt_test_cu1_du2.pt"
res = run_checkpoint(f"{CLEAR}/CU_1_4_7_10_STRESS", ckpt_path, bd_orig, TEST_SOURCE_ORIG,
                      ["cu1_du2", "cu0_du0du1", "cu2_du3du4du5"])
ALL_RESULTS["big2small|cu4710"] = res

OUT_DIR = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad"
with open(f"{OUT_DIR}/data_bothfixes_p999.js", "w") as f:
    f.write("window.DATA_BOTHFIXES_P999 = " + json.dumps(ALL_RESULTS) + ";\n")
print(f"\nsaved -> {OUT_DIR}/data_bothfixes_p999.js")
