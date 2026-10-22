"""Own-Registry LOO evaluation of NDT (Nonparametric Dynamic Thresholding,
Hundman et al. 2018) vs our existing pooled-percentile baseline, for all 8
checkpoints (4 original + 4 new registry), CPU/MEM/NET.

NDT threshold is computed ONLY from each checkpoint's pooled calibration
score stream (own training-topology data, no test/labels touched) -- same
calibration data our existing baseline_thr/extrap_thr already use. Test-side
scoring reuses the already-saved score/lbl arrays from
threshold_plot_evidence_full.json (no new model inference needed for test
scoring, same convention as eval_p985_full_crossreg.py PART A).

Standalone script -- does not modify or import any existing project file.
"""
import sys, json
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
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
PREPROCESS_VERSION = "v0"
CAL_FRAC = 0.2
WINDOW_LEN = 64
COLD_START_K = WINDOW_LEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRESS_ORDER = ["CPU", "MEM", "NET"]

N_DU_ORIG = {"cu0_du0du1": 2, "cu1_du2": 1, "cu2_du3du4du5": 3, "cu3_du6du7du8du9": 4}
N_DU_NEW = {"cu1_du2": 1, "cu3_du6du7du8du9": 4, "cu0_7du": 7, "cu0_10du": 10}
ORIG_CFG = {"base_dir": f"{CLEAR}/CU_DL_UDP_STRESS", "ckpt_dir": f"{CLEAR}/model_DLUDP_NET",
            "ckpt_prefix": "dlv1_", "all_topos": list(N_DU_ORIG)}
NEW_CFG = {"base_dir": f"{CLEAR}/CU_1_4_7_10_STRESS", "ckpt_dir": f"{CLEAR}/model_1_4_7_10",
           "ckpt_prefix": "v2_", "all_topos": list(N_DU_NEW)}


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


def build_checkpoint_with_pooled_cal(cfg, test_topo):
    ckpt = torch.load(f"{cfg['ckpt_dir']}/{cfg['ckpt_prefix']}model_ckpt_test_{test_topo}.pt", map_location="cpu")
    train_topos = ckpt["topos"]
    raw_streams = [dict(zip(["cu", "du", "block_id"], load_train_npz(cfg["base_dir"], t))) for t in train_topos]
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    pooled = []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        cu_sq = phase_infer(model, cu_s[n_fit:], du_s[n_fit:])[COLD_START_K:]
        pooled.append(cu_sq)

    cu_sqerr_pool = np.concatenate(pooled, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    # Same normalizer (cu_fn_pool) used for the saved test-score arrays, so NDT
    # candidates stay on the identical scale -- but split back per-topology so
    # mu/sigma are computed on ONE coherent stream at a time (not a blend of
    # several different topologies' distributions, which inflates sigma).
    cal_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)
    per_topo_scores = {}
    i = 0
    for t, cu_sq in zip(train_topos, pooled):
        n = len(cu_sq)
        per_topo_scores[t] = cal_scores_pool[i:i + n]
        i += n
    return {"train_topos": train_topos, "cal_scores_pool": cal_scores_pool, "per_topo_scores": per_topo_scores}


print(f"{'='*95}\nNDT (Nonparametric Dynamic Thresholding) vs existing pooled-percentile baseline\nOwn-Registry LOO, all 8 checkpoints, CPU/MEM/NET\n{'='*95}")

with open(f"{OUT_DIR}/threshold_plot_evidence_full.json") as f:
    saved = json.load(f)

results = {}
log_summary = {}
for cfg, regname in [(ORIG_CFG, "original"), (NEW_CFG, "new")]:
    for test_topo in cfg["all_topos"]:
        key = f"{regname}|{test_topo}"
        ck = build_checkpoint_with_pooled_cal(cfg, test_topo)

        # NDT computed separately per training topology (one coherent stream
        # at a time), then averaged into the single threshold applied at test
        # time -- avoids inflating sigma by blending different topologies'
        # score distributions into one array before picking z. Each stream is
        # EWMA-smoothed first (Hundman Sec 3.1 operates on smoothed errors,
        # not the raw per-timestep stream).
        per_topo_picks = {t: ndt.select_threshold(ndt.ewma_smooth(s)) for t, s in ck["per_topo_scores"].items()}
        eps_per_topo = {t: p["eps"] for t, p in per_topo_picks.items()}
        eps = float(np.mean(list(eps_per_topo.values())))

        print(f"\n### {key}  (trained on {ck['train_topos']})")
        for t, p in per_topo_picks.items():
            print(f"  NDT[{t}]: z={p['z']:.1f}  eps={p['eps']:.4f}  score={p['score']:.4f}  "
                  f"n_flagged={p['n_flagged']}  n_runs={p['n_runs']}")
        print(f"  -> mean eps across training topologies = {eps:.4f}")

        row_out = {"train_topos": ck["train_topos"], "ndt_per_topo": per_topo_picks,
                   "ndt_eps_mean": eps, "rows": {}}
        for stress in STRESS_ORDER:
            rec = saved[key]["rows"][stress]
            scores_raw, lbl = np.array(rec["score"]), np.array(rec["lbl"])
            scores = ndt.ewma_smooth(scores_raw)
            pred_simple = (scores > eps).astype(int)
            pred_pruned = ndt.prune(scores, eps).astype(int)
            m_simple = ndt.metrics(pred_simple, lbl)
            m_pruned = ndt.metrics(pred_pruned, lbl)
            m_existing = rec["base"]
            print(f"  {stress}: existing(pooled %ile) f1={m_existing['f1']:.4f} (thr={rec['baseline_thr']:.2f})  "
                  f"| NDT f1={m_simple['f1']:.4f}  | NDT+pruned f1={m_pruned['f1']:.4f}")
            row_out["rows"][stress] = {
                "existing": {"thr": rec["baseline_thr"], **m_existing},
                "ndt_simple": {"thr": eps, **m_simple},
                "ndt_pruned": {"thr": eps, **m_pruned},
            }
        results[key] = row_out

with open(f"{OUT_DIR}/ndt_own_registry_results.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved -> {OUT_DIR}/ndt_own_registry_results.json")

# log entry for future reference
log_entry = {
    "ts": "2026-10-09T00:00:00",
    "script": "eval_ndt_own_registry.py",
    "desc": "NDT v3 (Hundman et al. 2018) vs existing pooled-percentile baseline, Own-Registry LOO, "
            "all 8 checkpoints, CPU/MEM/NET. v1->v2: NDT z/eps computed separately per training "
            "topology (one coherent calibration stream at a time, same cu_fn_pool normalization as "
            "existing/test scores), then averaged into one applied threshold -- v1 pooled all training "
            "topologies into one array before computing mu/sigma, which inflated sigma and pushed z to "
            "the sweep's upper bound (10.0) every time, collapsing NET F1 (v2 still mostly hit the z=10 "
            "boundary though, NET still collapsed). v2->v3: added EWMA smoothing (alpha=0.05) of both "
            "calibration and test score streams before NDT selection/thresholding, per Hundman Sec 3.1 "
            "(method assumes smoothed errors, not raw per-timestep scores). Test scoring reuses saved "
            "arrays from threshold_plot_evidence_full.json.",
    "results": {k: {stress: v["rows"][stress] for stress in STRESS_ORDER} for k, v in results.items()},
}
with open(LOG_FILE, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
print(f"logged -> {LOG_FILE}")
print("\nDONE")
