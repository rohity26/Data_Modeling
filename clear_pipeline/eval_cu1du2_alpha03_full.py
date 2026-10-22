"""Full CU + DU re-evaluation of the cu1_du2-held-out new-registry checkpoint
at alpha=0.3 (scoped fix, validated safe/beneficial), reporting the SAME
targets shown in the LOO Cross-Evaluation artifact for this checkpoint:
  - CU: own held-out (cu1_du2), + Big->Small cross-targets (cu0_du0du1,
    cu2_du3du4du5)
  - DU: per-DU rows for all 3 of the above targets

Computes BOTH alpha=0.05 (matching what's currently published) and
alpha=0.3 (candidate fix) side by side, for CU and DU, all 3 stress types,
so the full before/after picture can be shown as chat-only heatmaps. Does
NOT publish or touch the artifact -- read-only diagnostic script.
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import pandas as pd
import torch

import build_dataset as bd_orig
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
TRAIN_BASE_DIR = f"{CLEAR}/CU_1_4_7_10_STRESS"
CKPT_PATH = f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_cu1_du2.pt"
TRAIN_TOPOS = ["cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]
TARGETS = ["cu1_du2", "cu0_du0du1", "cu2_du3du4du5"]
ALPHAS = [0.05, 0.3]

with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    beta1 = json.load(f)["cu1_du2"]


def impute(arr, idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            g = arr[t, idx] < eps; arr[t, idx] = np.where(g, arr[t - 1, idx], arr[t, idx])
        else:
            g = arr[t, :, idx] < eps; arr[t, :, idx] = np.where(g, arr[t - 1, :, idx], arr[t, :, idx])
    return arr


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


def load_npz(topo):
    z = dict(np.load(f"{TRAIN_BASE_DIR}/{topo}_stress3/train.npz"))
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


ckpt = torch.load(CKPT_PATH, map_location="cpu")
raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(t))) for t in TRAIN_TOPOS]
bundle = fit_bundle(raw_streams, [], [], version="v0")
model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])

pooled_cu, pooled_du, per_topo_cu, per_topo_du = [], [], {}, {}
for t, raw in zip(TRAIN_TOPOS, raw_streams):
    cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
    n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
    cu_sq, du_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
    cu_sq = cu_sq[COLD_START_K:]
    du_sq = du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1])
    pooled_cu.append(cu_sq); pooled_du.append(du_sq)
    per_topo_cu[t] = cu_sq; per_topo_du[t] = du_sq

cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
cu_scores_pool = (cu_sqerr_pool / cu_fn_pool).max(axis=-1)

du_sqerr_pool = np.concatenate(pooled_du, axis=0)
du_fn_pool = feat_norm_calibrated(du_sqerr_pool)
du_scores_pool = (du_sqerr_pool / du_fn_pool).max(axis=-1)


def eps_from_pool(pool_scores, per_topo_lens, alpha):
    i = 0; per_topo_eps = {}
    for t, n in per_topo_lens.items():
        per_topo_eps[t] = ndt.select_threshold(ndt.ewma_smooth(pool_scores[i:i + n], alpha=alpha))["eps"]
        i += n
    return float(np.mean(list(per_topo_eps.values())))


cu_lens = {t: len(per_topo_cu[t]) for t in TRAIN_TOPOS}
du_lens = {t: len(per_topo_du[t]) for t in TRAIN_TOPOS}
cu_eps = {a: eps_from_pool(cu_scores_pool, cu_lens, a) for a in ALPHAS}
du_eps = {a: eps_from_pool(du_scores_pool, du_lens, a) for a in ALPHAS}
print("CU eps:", cu_eps, "  DU eps:", du_eps)

results = {"cu": {}, "du": {}}
for target in TARGETS:
    topo_cfg = bd_orig.TOPOLOGY_REGISTRY[target]
    cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
    results["cu"][target] = {}
    results["du"][target] = {}
    for stress in STRESS_ORDER:
        path = f"{DATA_DIR}/test_csv_dl_udp_2iface/test_{cu_id}_{stress}_rep1.csv"
        df = pd.read_csv(path); cols = df.columns.tolist()
        cu_cols = bd_orig.get_cu_columns(cu_id, cols)
        cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
        du_raw_list = [df[bd_orig.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                       for d, p in zip(du_ids, pci_ids)]
        du_raw = np.stack(du_raw_list, axis=1)
        lbl = df[f"{cu_id}_stressType"].fillna(0).astype(int).values
        cu_feat, du_feat = slice_features(cu_raw, du_raw)
        block_id = np.zeros(len(cu_feat), dtype=np.int64)
        cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
        cu_sqerr, du_sqerr = phase_infer(model, cu_s, du_s)
        start = COLD_START_K
        lbl_ev_cu = (lbl[start + 1:] == STYPE_NUM[stress]).astype(int)
        cu_scores = (cu_sqerr[start:] / cu_fn_pool).max(axis=-1)

        cu_row = {}
        for alpha in ALPHAS:
            sm = ndt.ewma_smooth(cu_scores, alpha=alpha)
            cu_row[alpha] = metrics((sm > cu_eps[alpha]).astype(int), lbl_ev_cu)
        results["cu"][target][stress] = cu_row

        du_row = {}
        for du_idx, du_id in enumerate(du_ids):
            # Each DU is scored from its OWN dedicated test file (same
            # pattern as eval_newreg_4ckpt_on_unseen.py's score_all_dus) --
            # row counts differ slightly file-to-file, so cu_sqerr/du_sqerr
            # from the main cu_id file cannot be reused across DUs.
            du_path = f"{DATA_DIR}/test_csv_dl_udp_2iface/test_{du_id}_{stress}_rep1.csv"
            df_d = pd.read_csv(du_path); cols_d = df_d.columns.tolist()
            cu_cols_d = bd_orig.get_cu_columns(cu_id, cols_d)
            cu_raw_d = df_d[cu_cols_d].fillna(0.0).astype(np.float32).values
            du_raw_list_d = [df_d[bd_orig.get_du_columns(d, p, cols_d)].fillna(0.0).astype(np.float32).values
                              for d, p in zip(du_ids, pci_ids)]
            du_raw_d = np.stack(du_raw_list_d, axis=1)
            lbl_du = df_d[f"{du_id}_stressType"].fillna(0).astype(int).values
            cu_feat_d, du_feat_d = slice_features(cu_raw_d, du_raw_d)
            block_id_d = np.zeros(len(cu_feat_d), dtype=np.int64)
            cu_s_d, du_s_d, _, _ = transform_stream(bundle, cu_feat_d, du_feat_d, block_id_d)
            _, du_sqerr_d = phase_infer(model, cu_s_d, du_s_d)
            lbl_ev_du = (lbl_du[start + 1:] == STYPE_NUM[stress]).astype(int)
            du_scores = (du_sqerr_d[start:, du_idx, :] / du_fn_pool).max(axis=-1)
            per_alpha = {}
            for alpha in ALPHAS:
                sm = ndt.ewma_smooth(du_scores, alpha=alpha)
                per_alpha[alpha] = metrics((sm > du_eps[alpha]).astype(int), lbl_ev_du)
            du_row[du_id] = per_alpha
        results["du"][target][stress] = du_row

out_path = f"{CLEAR}/eval_cu1du2_alpha03_full_results.json"
with open(out_path, "w") as f:
    json.dump({
        "cu_eps": {str(k): v for k, v in cu_eps.items()},
        "du_eps": {str(k): v for k, v in du_eps.items()},
        "results": results,
    }, f, indent=1)
print("saved ->", out_path)

print("\n=== CU ===")
for target in TARGETS:
    for stress in STRESS_ORDER:
        r = results["cu"][target][stress]
        print(f"{target:16s} {stress:5s}  a.05 F1={r[0.05]['f1']:.3f}  a.3 F1={r[0.3]['f1']:.3f}")

print("\n=== DU ===")
for target in TARGETS:
    for stress in STRESS_ORDER:
        for du_id, r in results["du"][target][stress].items():
            print(f"{target:16s} {stress:5s} {du_id:8s}  a.05 F1={r[0.05]['f1']:.3f}  a.3 F1={r[0.3]['f1']:.3f}")

print("DONE")
