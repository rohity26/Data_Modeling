"""Extract the full score/label time series for the one poor cell flagged by the
user: new-registry checkpoint held-out=cu1_du2 (trained on cu3_du6du7du8du9,
cu0_7du, cu0_10du), scored on its own held-out topology cu1_du2, CU-level NET
stress (NDT F1=0.607, tp=243/fp=314/fn=0 -- a precision problem from training
only on N_DU=4,7,10 and testing on N_DU=1's much noisier per-DU traffic).

Reuses the exact same scoring pipeline as eval_newreg_4ckpt_on_unseen.py (same
beta1, same bundle fit on the 3 training topologies, same cu_fn_pool/eps), but
additionally saves the raw per-timestep score + label arrays (not just
aggregate metrics) so a client-side threshold-plot panel can be drawn.

Read-only; does not touch any training process or file.
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
from scoring import lift_score
import ndt_threshold as ndt

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
TRAIN_BASE_DIR = f"{CLEAR}/CU_1_4_7_10_STRESS"
CKPT_PATH = f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_cu1_du2.pt"
TRAIN_TOPOS = ["cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]
OUT_JS = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad/loo4_artifact/data_cu1du2_newreg_net_plot.js"

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

with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    beta1 = json.load(f)["cu1_du2"]


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

    cu[:, 0] = cu[:, 0] - beta1["cpu"] * (N_DU - 1)
    cu[:, 1] = cu[:, 1] - beta1["mem_pct"] * (N_DU - 1)
    cu[:, 2] = cu[:, 2] - beta1["mem_bytes"] * (N_DU - 1)

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


ckpt = torch.load(CKPT_PATH, map_location="cpu")
raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(TRAIN_BASE_DIR, t))) for t in TRAIN_TOPOS]
bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
print(f"loaded ckpt cu1_du2-held-out (cu_dim={ckpt['cu_dim']}, du_dim={ckpt['du_dim']}, embed_dim={ckpt['embed_dim']})")

pooled_cu = []
cu_per_topo_sq = {}
for t, raw in zip(TRAIN_TOPOS, raw_streams):
    cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
    n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
    cu_sq, du_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
    cu_sq = cu_sq[COLD_START_K:]
    pooled_cu.append(cu_sq)
    cu_per_topo_sq[t] = cu_sq
    print(f"  {t}: cal rows (cu)={len(cu_sq)}")

cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)
i = 0
cu_per_topo_eps = {}
for t in TRAIN_TOPOS:
    n = len(cu_per_topo_sq[t])
    pick = ndt.select_threshold(ndt.ewma_smooth(cu_scores_pool[i:i + n]))
    cu_per_topo_eps[t] = pick["eps"]
    i += n
cu_eps = float(np.mean(list(cu_per_topo_eps.values())))
print(f"CU NDT eps (mean of 3 training topos) = {cu_eps:.4f}")
print(f"  per-topo eps: {cu_per_topo_eps}")

# ---- score the own held-out target (cu1_du2) on NET ----
topo_cfg = bd_orig.TOPOLOGY_REGISTRY["cu1_du2"]
cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]

path = f"{DATA_DIR}/test_csv_dl_udp_2iface/test_{cu_id}_NET_rep1.csv"
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
cu_sqerr, _ = phase_infer(model, cu_s, du_s)
start = COLD_START_K
lbl_ev = (lbl[start + 1:] == STYPE_NUM["NET"]).astype(int)
scores = lift_score(cu_sqerr[start:], cu_fn_pool)
scores = np.array(scores)

smoothed = ndt.ewma_smooth(scores)
tp = int(((smoothed > cu_eps) & (lbl_ev == 1)).sum())
fp = int(((smoothed > cu_eps) & (lbl_ev == 0)).sum())
fn = int(((smoothed <= cu_eps) & (lbl_ev == 1)).sum())
p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
f1 = 2 * p * r / (p + r + 1e-9)
print(f"NET on cu1_du2: tp={tp} fp={fp} fn={fn} f1={f1:.4f}  (expect f1~0.607)")
print(f"score stream length = {len(scores)}  eps={cu_eps:.4f}")

payload = {
    "score": [float(x) for x in scores],
    "lbl": [int(x) for x in lbl_ev],
    "eps": cu_eps,
    "tp": tp, "fp": fp, "fn": fn, "f1": f1,
}
with open(OUT_JS, "w") as f:
    f.write("window.DATA_CU1DU2_NEWREG_NET_PLOT = ")
    json.dump(payload, f)
    f.write(";\n")
print(f"saved -> {OUT_JS}")
