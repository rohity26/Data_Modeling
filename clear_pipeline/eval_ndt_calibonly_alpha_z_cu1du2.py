"""Select BOTH alpha and z for cu1_du2 (new-registry) using ONLY calibration
data -- no test labels anywhere in the selection process. This directly
answers: can alpha/z be chosen in a way that's honest (not test-set-tuned)?

Method: extend NDT's own unsupervised score function (Hundman et al. Eq. in
Section 3.2 -- Delta-mu/mu + Delta-sigma/sigma, penalized by flagged count and
run count, computed ONLY on the calibration error stream) to also search over
alpha, not just z. For each candidate alpha:
  1. Smooth each of the 3 training topologies' own calibration score stream
     at that alpha.
  2. Run the existing z-sweep (wide range, [2.5,50]) on each -- this already
     returns the NDT "score" (the SAME calibration-only quality criterion)
     for whichever z it picked for that topology.
  3. Average that calibration-only score across the 3 training topologies.
Pick the alpha with the best average calibration-only score. Then -- and ONLY
then -- evaluate the resulting (alpha, eps) on the test targets, purely to
report/verify, never to choose among candidates.
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

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
WINDOW_LEN = 64; COLD_START_K = WINDOW_LEN; CAL_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}
STRESS_ORDER = ["CPU", "MEM", "NET"]

with open(f"{CLEAR}/ndu_beta1_per_split_new_registry.json") as f:
    BETA1_NEW = json.load(f)

HELD_OUT = "cu1_du2"
TRAIN_TOPOS = ["cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]
CKPT = f"{CLEAR}/model_1_4_7_10_normcorrected/normfix_model_ckpt_test_{HELD_OUT}.pt"
TARGETS = [
    {"name": "cu1_du2", "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    {"name": "cu2_du3du4du5", "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    {"name": "cu0_du0du1", "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
]

Z_VALUES_WIDE = np.arange(2.5, 50.01, 0.5)
ALPHA_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


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


slice_features = make_slice_features(BETA1_NEW[HELD_OUT])
ckpt = torch.load(CKPT, map_location="cpu")
raw_streams = [dict(zip(["cu", "du", "block_id"], load_npz(slice_features, TRAIN_BASE_DIR, t))) for t in TRAIN_TOPOS]
bundle = fit_bundle(raw_streams, [], [], version="v0")
model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])

pooled_cu = []
for t, raw in zip(TRAIN_TOPOS, raw_streams):
    cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
    n_total = len(cu_s); n_fit = n_total - int(round(CAL_FRAC * n_total))
    cu_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])
    pooled_cu.append(cu_sq[COLD_START_K:])
cu_sqerr_pool = np.concatenate(pooled_cu, axis=0)
cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)

print("=" * 90)
print("STEP 1: pick alpha using ONLY calibration data (no test labels touched)")
print("=" * 90)
calib_only_results = {}
for alpha in ALPHA_GRID:
    i = 0
    per_topo_score, per_topo_eps, per_topo_z = {}, {}, {}
    for t, cu_sq in zip(TRAIN_TOPOS, pooled_cu):
        n = len(cu_sq)
        pick = ndt.select_threshold(ndt.ewma_smooth(cu_scores_pool[i:i + n], alpha=alpha), z_values=Z_VALUES_WIDE)
        per_topo_score[t] = pick["score"]; per_topo_eps[t] = pick["eps"]; per_topo_z[t] = pick["z"]
        i += n
    mean_calib_score = float(np.mean(list(per_topo_score.values())))
    eps = float(np.mean(list(per_topo_eps.values())))
    calib_only_results[alpha] = {"mean_calib_score": mean_calib_score, "eps": eps,
                                  "per_topo_score": per_topo_score, "per_topo_z": per_topo_z, "per_topo_eps": per_topo_eps}
    print(f"  alpha={alpha}: mean calib-only score={mean_calib_score:.5f}  "
          f"per_topo_z={per_topo_z}  eps={eps:.2f}")

best_alpha = max(calib_only_results, key=lambda a: calib_only_results[a]["mean_calib_score"])
best_eps = calib_only_results[best_alpha]["eps"]
print(f"\n>>> Calibration-only choice: alpha={best_alpha}  eps={best_eps:.3f} "
      f"(mean calib score={calib_only_results[best_alpha]['mean_calib_score']:.5f})")
print("    (this choice never looked at test labels or test F1)")

print("\n" + "=" * 90)
print("STEP 2: evaluate the calibration-chosen (alpha, eps) on test data -- for")
print("verification/reporting ONLY, not used to select among candidates")
print("=" * 90)

target_scores = {}
for target in TARGETS:
    t_name, csv_dir = target["name"], target["csv_dir"]
    topo_cfg = bd_orig.TOPOLOGY_REGISTRY[t_name]
    cu_id, du_ids, pci_ids = topo_cfg["cu_id"], topo_cfg["du_ids"], topo_cfg["pci_ids"]
    target_scores[t_name] = {}
    for stress in STRESS_ORDER:
        path = f"{csv_dir}/test_{cu_id}_{stress}_rep1.csv"
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
        cu_sqerr = phase_infer(model, cu_s, du_s)
        start = COLD_START_K
        lbl_ev = (lbl[start + 1:] == STYPE_NUM[stress]).astype(int)
        scores = lift_score(cu_sqerr[start:], cu_fn_pool)
        target_scores[t_name][stress] = (scores, lbl_ev)

final_check = {}
for t_name in target_scores:
    t_res = {}
    for stress in STRESS_ORDER:
        scores, lbl_ev = target_scores[t_name][stress]
        smoothed = ndt.ewma_smooth(scores, alpha=best_alpha)
        m = metrics((smoothed > best_eps).astype(int), lbl_ev)
        t_res[stress] = m
    final_check[t_name] = t_res
    print(f"  {t_name}: " + "  ".join(f"{s}={t_res[s]['f1']:.3f}" for s in STRESS_ORDER))

out = {
    "calib_only_sweep": {str(a): {"mean_calib_score": v["mean_calib_score"], "eps": v["eps"]}
                          for a, v in calib_only_results.items()},
    "chosen_alpha": best_alpha, "chosen_eps": best_eps,
    "test_verification": final_check,
}
with open(f"{CLEAR}/eval_ndt_calibonly_alpha_z_cu1du2_results.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nsaved -> eval_ndt_calibonly_alpha_z_cu1du2_results.json")
print("DONE")
