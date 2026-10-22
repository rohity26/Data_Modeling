"""Full NDT evaluation, CU-level, across all 8 N_DU-corrected checkpoints and
all their established targets -- same scope/format as every other section in
the artifact. Identical to the project's standard NDT method (per-topology
z-sweep + pruning-free CU thresholding + eps averaging across training
topologies), EXCEPT the z-sweep range is widened from the paper's own
[2.5,10] to [2.5,20] (verified to be the plateau point -- going further to 30
or 50 changes nothing, confirmed in eval_ndt_zrange_expand_cu1du2.py). alpha
stays at 0.05 everywhere, uniformly, no scoped exception.
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
Z_VALUES = np.arange(2.5, 20.01, 0.5)

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]

with open(f"{CLEAR}/ndu_beta1_per_split.json") as f:
    BETA1_ORIG = json.load(f)
with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    BETA1_NEW = json.load(f)

ORIG_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
NEW_TOPOS = ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]

CKPTS = []
for held_out in ORIG_TOPOS:
    CKPTS.append({
        "key": held_out, "ckpt": f"{CLEAR}/model_DLUDP_NET_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
        "base_dir": f"{CLEAR}/CU_DL_UDP_STRESS",
        "train_topos": [t for t in ORIG_TOPOS if t != held_out],
        "beta1": BETA1_ORIG[held_out],
        "targets": [
            {"name": held_out, "bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
            {"name": "cu0_7du", "bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu7du_11gb_all"},
            {"name": "cu0_10du", "bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_11gb_all"},
        ],
    })

NEWREG_OWN_CFG = {
    "cu1_du2": {"bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    "cu3_du6du7du8du9": {"bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    "cu0_7du": {"bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu7du_11gb_all"},
    "cu0_10du": {"bd": bd_new, "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_11gb_all"},
}
NEWREG_KEY_MAP = {"cu1_du2": "cu1_du2__newreg", "cu3_du6du7du8du9": "cu3_du6du7du8du9__newreg",
                  "cu0_7du": "cu0_7du", "cu0_10du": "cu0_10du"}
for held_out in NEW_TOPOS:
    own_cfg = NEWREG_OWN_CFG[held_out]
    CKPTS.append({
        "key": NEWREG_KEY_MAP[held_out], "ckpt": f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_{held_out}.pt",
        "base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS",
        "train_topos": [t for t in NEW_TOPOS if t != held_out],
        "beta1": BETA1_NEW[held_out],
        "targets": [
            {"name": held_out, "bd": own_cfg["bd"], "csv_dir": own_cfg["csv_dir"]},
            {"name": "cu0_du0du1", "bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
            {"name": "cu2_du3du4du5", "bd": bd_orig, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
        ],
    })


def impute(arr, idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            g = arr[t, idx] < eps; arr[t, idx] = np.where(g, arr[t - 1, idx], arr[t, idx])
        else:
            g = arr[t, :, idx] < eps; arr[t, :, idx] = np.where(g, arr[t - 1, :, idx], arr[t, :, idx])
    return arr


def make_slice_features(beta1):
    def slice_features(cu_raw, du_raw):
        cu = cu_raw.astype(np.float32).copy(); du = du_raw.astype(np.float32).copy()
        N_DU = du.shape[1]
        cu[:, 5] = cu[:, 5] / N_DU; cu[:, 6] = cu[:, 6] / N_DU
        cu = cu[:, CU_FEAT_SLICE]; du = du[:, :, DU_FEAT_SLICE]
        cu = impute(cu, CU_IRATE_IDX); du = impute(du, DU_IRATE_IDX)
        cu[:, 0] -= beta1["cpu"] * (N_DU - 1); cu[:, 1] -= beta1["mem_pct"] * (N_DU - 1); cu[:, 2] -= beta1["mem_bytes"] * (N_DU - 1)
        tx, rx = cu[:, 3:4], cu[:, 4:5]
        cu = np.concatenate([cu, tx - rx, tx / (rx + 1e-6)], axis=1)
        dtx, drx = du[:, :, 4:5], du[:, :, 5:6]
        du = np.concatenate([du, dtx - drx, dtx / (drx + 1e-6)], axis=2)
        return cu, du
    return slice_features


def load_npz(slice_features, base_dir, topo):
    z = dict(np.load(f"{base_dir}/{topo}_stress3/train.npz"))
    cu, du = slice_features(z["cu"], z["du"])
    return cu, du, z["block_id"].astype(np.int64)


def phase_infer(model, cu_s, du_s):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE); du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


all_results = {}
for ck in CKPTS:
    print(f"\n{'='*90}\n{ck['key']}\n{'='*90}")
    slice_features = make_slice_features(ck["beta1"])
    ckpt = torch.load(ck["ckpt"], map_location="cpu")
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(slice_features, ck["base_dir"], t))) for t in ck["train_topos"]]
    bundle = fit_bundle(raw_streams, [], [], version="v0")
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled_cu = []
    for t, raw in zip(ck["train_topos"], raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
        cu_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
        pooled_cu.append(cu_sq[COLD_START_K:])
    cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)

    i = 0
    per_topo_eps = {}
    for t, cu_sq in zip(ck["train_topos"], pooled_cu):
        n = len(cu_sq)
        pick = ndt.select_threshold(ndt.ewma_smooth(cu_scores_pool[i:i + n]), z_values=Z_VALUES)
        per_topo_eps[t] = pick["eps"]
        i += n
    eps = float(np.mean(list(per_topo_eps.values())))
    print(f"  per_topo_eps={per_topo_eps}  final eps={eps:.3f}")

    ck_results = {"eps": eps, "per_topo_eps": per_topo_eps, "targets": {}}
    for target in ck["targets"]:
        t_name, bdmod, csv_dir = target["name"], target["bd"], target["csv_dir"]
        topo_cfg = bdmod.TOPOLOGY_REGISTRY[t_name]
        cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
        t_results = {}
        for stress in STRESS_ORDER:
            path = f"{csv_dir}/test_{cu_id}_{stress}_rep1.csv"
            df = pd.read_csv(path); cols = df.columns.tolist()
            cu_cols = bdmod.get_cu_columns(cu_id, cols)
            cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
            du_raw_list = [df[bdmod.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                           for d, p in zip(du_ids, pci_ids)]
            du_raw = np.stack(du_raw_list, axis=1)
            lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
            cu_feat, du_feat = slice_features(cu_raw, du_raw)
            block_id = np.zeros(len(cu_feat), dtype=np.int64)
            cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
            cu_sqerr = phase_infer(model, cu_s, du_s)
            start = COLD_START_K
            lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress]).astype(int)
            scores = lift_score(cu_sqerr[start:], cu_fn_pool)
            smoothed = ndt.ewma_smooth(scores)
            m = metrics((smoothed > eps).astype(int), lbl_ev)
            t_results[stress] = m
        ck_results["targets"][t_name] = t_results
        print(f"    {t_name}: " + "  ".join(f"{s}={t_results[s]['f1']:.3f}" for s in STRESS_ORDER))

    all_results[ck["key"]] = ck_results

with open(f"{CLEAR}/eval_ndt_zrange20_full_results.json", "w") as f:
    json.dump(all_results, f, indent=1)
print("\nsaved -> eval_ndt_zrange20_full_results.json")
print("DONE")
