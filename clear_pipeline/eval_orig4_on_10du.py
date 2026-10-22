"""Test each of the 4 ORIGINAL (1/2/3/4-DU) LOO checkpoints against the
completely external cu0_10du (10-DU) test set -- a topology that was never
in any of these checkpoints' training rotation at all (it's from the NEW
registry, max DU in the original registry was 4).

For each original checkpoint:
  - baseline: today's pooled-calibration threshold (already computed from
    its own 3 training topologies).
  - extrapolated: per-topology threshold (same 3 training topologies, NOT
    pooled) fit as thr = a + b*(1/N_DU), evaluated at N_DU=10.
Score both against cu0_10du's real CPU/MEM/NET test files (own-file CU).
"""
import sys
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
ORIG_BASE_DIR = f"{CLEAR}/CU_DL_UDP_STRESS"
ORIG_CKPT_DIR = f"{CLEAR}/model_DLUDP_NET"
ORIG_CKPT_PREFIX = "dlv1_"
ORIG_ALL_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
ORIG_N_DU_OF = {"cu0_du0du1": 2, "cu1_du2": 1, "cu2_du3du4du5": 3, "cu3_du6du7du8du9": 4}

TARGETS = [
    {"topo": "cu0_10du", "n_du": 10, "csv_dir": f"{DATA_DIR}/test_csv_1cu10du_124",
     "net_override": f"{DATA_DIR}/CU_25_loss_net_stress_test/test_srscu10_NET_rep1.csv"},
    {"topo": "cu0_7du", "n_du": 7, "csv_dir": f"{DATA_DIR}/test_csv_1cu7du",
     "net_override": f"{DATA_DIR}/CU_25_loss_net_stress_test/test_srscu7_NET_rep1.csv"},
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
CU_THRESHOLD_PCT = 99.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}


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
    cu = cu_raw.astype(np.float32).copy()
    du = du_raw.astype(np.float32).copy()
    N_DU = du.shape[1]
    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]
    cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
    du = impute_cpu_glitch(du, DU_IRATE_IDX)
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)
    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)
    return cu, du


def load_train_npz(base_dir, topo):
    p = f"{base_dir}/{topo}_stress3/train.npz"
    z = dict(np.load(p))
    cu, du = slice_features(z["cu"], z["du"])
    bid = z["block_id"].astype(np.int64)
    return cu, du, bid


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


rows = []
print(f"{'='*100}\nTesting 4 ORIGINAL (1/2/3/4-DU) checkpoints against external targets: {[t['topo'] for t in TARGETS]}\n{'='*100}")

for test_topo in ORIG_ALL_TOPOS:
    ckpt_path = f"{ORIG_CKPT_DIR}/{ORIG_CKPT_PREFIX}model_ckpt_test_{test_topo}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_topos = ckpt["topos"]
    print(f"\n### original checkpoint 'test={test_topo}'  trained on {train_topos}  (held out {test_topo} internally, now also probing external targets)")

    raw_streams = []
    for t in train_topos:
        cu, du, bid = load_train_npz(ORIG_BASE_DIR, t)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})
    bundle = fit_bundle(raw_streams, [], [], version=PREPROCESS_VERSION)

    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    per_topo_thr = {}
    pooled_cu_sqerr = []
    for t, raw in zip(train_topos, raw_streams):
        cu_s, du_s, _, _ = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        n_total = len(cu_s)
        n_cal = int(round(CAL_FRAC * n_total))
        n_fit = n_total - n_cal
        cu_s_cal, du_s_cal = cu_s[n_fit:], du_s[n_fit:]
        cu_sq, _ = phase_infer(model, cu_s_cal, du_s_cal)
        cu_sq = cu_sq[COLD_START_K:]
        pooled_cu_sqerr.append(cu_sq)

        cu_fn_t = feat_norm_calibrated(cu_sq)
        cu_scores_t = lift_score(cu_sq, cu_fn_t)
        thr_t = float(np.percentile(cu_scores_t, CU_THRESHOLD_PCT))
        per_topo_thr[t] = thr_t
        print(f"  per-topology thr: {t:20s} N_DU={ORIG_N_DU_OF[t]:>2d}  thr(p{CU_THRESHOLD_PCT})={thr_t:.4f}")

    xs = np.array([1.0 / ORIG_N_DU_OF[t] for t in train_topos])
    ys = np.array([per_topo_thr[t] for t in train_topos])
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a, b = coef
    print(f"  fitted line: thr = {a:.4f} + {b:.4f} * (1/N_DU)")

    cu_sqerr_pool = np.concatenate(pooled_cu_sqerr, axis=0)
    cu_fn_pool = feat_norm_calibrated(cu_sqerr_pool)
    cu_scores_pool = lift_score(cu_sqerr_pool, cu_fn_pool)
    baseline_thr = float(np.percentile(cu_scores_pool, CU_THRESHOLD_PCT))
    print(f"  baseline pooled threshold (current method): {baseline_thr:.4f}")

    for target in TARGETS:
        target_topo, target_n_du, target_csv_dir = target["topo"], target["n_du"], target["csv_dir"]
        topo_cfg_target = bd_new.TOPOLOGY_REGISTRY[target_topo]
        cu_id_t, du_ids_t, pci_ids_t = topo_cfg_target["cu_id"], topo_cfg_target["du_ids"], topo_cfg_target["pci_ids"]

        extrap_thr = a + b * (1.0 / target_n_du)
        print(f"  -- target={target_topo} (N_DU={target_n_du}) -- extrapolated threshold: {extrap_thr:.4f}")

        for stress_name in ["CPU", "MEM", "NET"]:
            path = (target["net_override"] if stress_name == "NET" and "net_override" in target
                    else f"{target_csv_dir}/test_{cu_id_t}_{stress_name}_rep1.csv")
            df = pd.read_csv(path)
            cols = df.columns.tolist()
            cu_cols = bd_new.get_cu_columns(cu_id_t, cols)
            cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
            du_raw_list = [df[bd_new.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
                           for d, p in zip(du_ids_t, pci_ids_t)]
            du_raw = np.stack(du_raw_list, axis=1)
            lbl = df[f"{cu_id_t}_stressType"].fillna(0).astype(int).values
            cu_feat, du_feat = slice_features(cu_raw, du_raw)
            block_id = np.zeros(len(cu_feat), dtype=np.int64)
            cu_s, du_s, _, _ = transform_stream(bundle, cu_feat, du_feat, block_id)
            cu_sqerr, _ = phase_infer(model, cu_s, du_s)
            start = COLD_START_K
            stype_num = STYPE_NUM[stress_name]
            lbl_ev = (lbl[start + 1:] == stype_num).astype(int)
            scores = lift_score(cu_sqerr[start:], cu_fn_pool)

            pred_base = (scores > baseline_thr).astype(int)
            pred_extrap = (scores > extrap_thr).astype(int)
            m_base = metrics(pred_base, lbl_ev)
            m_extrap = metrics(pred_extrap, lbl_ev)
            print(f"  {target_topo:10s} {stress_name}: baseline(thr={baseline_thr:.2f}) P={m_base['p']:.3f} R={m_base['r']:.3f} F1={m_base['f1']:.3f}"
                  f"   |   extrap(thr={extrap_thr:.2f}) P={m_extrap['p']:.3f} R={m_extrap['r']:.3f} F1={m_extrap['f1']:.3f}")
            rows.append({
                "ckpt": test_topo, "target": target_topo, "n_du": target_n_du, "stress": stress_name,
                "baseline_thr": baseline_thr, "extrap_thr": extrap_thr,
                "m_base": m_base, "m_extrap": m_extrap,
                "score": [round(float(x), 4) for x in scores],
                "lbl": [int(x) for x in lbl_ev],
            })

print(f"\n\n{'='*100}\nEXISTING THRESHOLD (pooled, current method)\n{'='*100}")
print(f"{'checkpoint':<22s}{'target':>10s}{'N_DU':>5s}{'stress':>7s}{'thr':>9s}{'P':>7s}{'R':>7s}{'F1':>7s}")
for r in rows:
    m = r["m_base"]
    print(f"{r['ckpt']:<22s}{r['target']:>10s}{r['n_du']:>5d}{r['stress']:>7s}{r['baseline_thr']:>9.2f}{m['p']:>7.3f}{m['r']:>7.3f}{m['f1']:>7.3f}")

print(f"\n{'='*100}\nNEW THRESHOLD (N_DU-extrapolated)\n{'='*100}")
print(f"{'checkpoint':<22s}{'target':>10s}{'N_DU':>5s}{'stress':>7s}{'thr':>9s}{'P':>7s}{'R':>7s}{'F1':>7s}")
for r in rows:
    m = r["m_extrap"]
    print(f"{r['ckpt']:<22s}{r['target']:>10s}{r['n_du']:>5d}{r['stress']:>7s}{r['extrap_thr']:>9.2f}{m['p']:>7.3f}{m['r']:>7.3f}{m['f1']:>7.3f}")

print(f"\nDone. {len(rows)} rows.")

import json
OUT_DIR = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad"
DATA = {}
for r in rows:
    key = r["ckpt"]
    DATA.setdefault(key, {})
    tkey = f"{r['target']}|{r['stress']}"
    DATA[key][tkey] = {
        "ckpt": r["ckpt"], "target": r["target"], "n_du": r["n_du"], "stress": r["stress"],
        "baseline_thr": r["baseline_thr"], "extrap_thr": r["extrap_thr"],
        "base": r["m_base"], "extrap": r["m_extrap"],
        "score": r["score"], "lbl": r["lbl"],
    }
for ckpt, v in DATA.items():
    safe = ckpt.replace("-", "_")
    out_path = f"{OUT_DIR}/data_orig4_ext_{safe}.js"
    with open(out_path, "w") as f:
        f.write(f"window.DATA_ORIG4EXT_{safe} = " + json.dumps(v) + ";\n")
    print(f"saved -> {out_path}")
