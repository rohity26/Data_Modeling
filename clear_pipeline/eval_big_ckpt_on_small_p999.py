"""Test the existing checkpoint trained on {cu3_du6du7du8du9 (4DU), cu0_7du
(7DU), cu0_10du (10DU)} -- model_1_4_7_10/v2_model_ckpt_test_cu1_du2.pt --
against the two SMALL topologies it has never been scored on:
cu0_du0du1 (2DU) and cu2_du3du4du5 (3DU). (cu1_du2, 1DU, was already its
official held-out target -- that result already exists.)

Reverse-direction check: does training on a wider/higher DU range generalize
DOWN to small topologies better than the forward direction (small->big)
generalized up? Zero new training -- this checkpoint already exists.
"""
import sys
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline")
sys.path.insert(0, "/home/cs26mtech02007/Rohit_data_modeling-2/src")

import numpy as np
import pandas as pd
import torch

import build_dataset as bd_orig
from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
DATA_DIR = f"{CLEAR}/data_input"
NEW_BASE_DIR = f"{CLEAR}/CU_1_4_7_10_STRESS"
CKPT_PATH = f"{CLEAR}/model_1_4_7_10/v2_model_ckpt_test_cu1_du2.pt"

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]
PREPROCESS_VERSION = "v0"
CAL_FRAC = 0.2
WINDOW_LEN = 64
COLD_START_K = WINDOW_LEN
CU_THRESHOLD_PCT = 99.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYPE_NUM = {"CPU": 1, "MEM": 2, "NET": 3}

TRAIN_N_DU_OF = {"cu3_du6du7du8du9": 4, "cu0_7du": 7, "cu0_10du": 10}

TARGETS = [
    {"topo": "cu1_du2", "n_du": 1, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    {"topo": "cu0_du0du1", "n_du": 2, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
    {"topo": "cu2_du3du4du5", "n_du": 3, "csv_dir": f"{DATA_DIR}/test_csv_dl_udp_2iface"},
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


ckpt = torch.load(CKPT_PATH, map_location="cpu")
train_topos = ckpt["topos"]
print(f"{'='*100}\nCheckpoint: v2_model_ckpt_test_cu1_du2.pt  trained on {train_topos}\nScoring on SMALL topologies it has never seen: {[t['topo'] for t in TARGETS]}\n{'='*100}")

raw_streams = []
for t in train_topos:
    cu, du, bid = load_train_npz(NEW_BASE_DIR, t)
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
    print(f"  per-topology thr: {t:20s} N_DU={TRAIN_N_DU_OF[t]:>2d}  thr(p{CU_THRESHOLD_PCT})={thr_t:.4f}")

xs = np.array([1.0 / TRAIN_N_DU_OF[t] for t in train_topos])
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

rows = []
for target in TARGETS:
    target_topo, target_n_du, target_csv_dir = target["topo"], target["n_du"], target["csv_dir"]
    topo_cfg_target = bd_orig.TOPOLOGY_REGISTRY[target_topo]
    cu_id_t, du_ids_t, pci_ids_t = topo_cfg_target["cu_id"], topo_cfg_target["du_ids"], topo_cfg_target["pci_ids"]

    extrap_thr = a + b * (1.0 / target_n_du)
    print(f"\n  -- target={target_topo} (N_DU={target_n_du}) -- extrapolated threshold: {extrap_thr:.4f}")

    # Also report where this target's own normal-period score sits, for diagnosis
    for stress_name in ["CPU", "MEM", "NET"]:
        path = f"{target_csv_dir}/test_{cu_id_t}_{stress_name}_rep1.csv"
        df = pd.read_csv(path)
        cols = df.columns.tolist()
        cu_cols = bd_orig.get_cu_columns(cu_id_t, cols)
        cu_raw = df[cu_cols].fillna(0.0).astype(np.float32).values
        du_raw_list = [df[bd_orig.get_du_columns(d, p, cols)].fillna(0.0).astype(np.float32).values
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

        normal_scores = scores[lbl_ev == 0]
        stress_scores = scores[lbl_ev == 1]
        print(f"  {target_topo} {stress_name}: normal[median={np.median(normal_scores):.1f} p90={np.percentile(normal_scores,90):.1f}]  "
              f"stress[median={np.median(stress_scores):.1f} p10={np.percentile(stress_scores,10):.1f}]")

        pred_base = (scores > baseline_thr).astype(int)
        pred_extrap = (scores > extrap_thr).astype(int)
        m_base = metrics(pred_base, lbl_ev)
        m_extrap = metrics(pred_extrap, lbl_ev)
        print(f"    baseline(thr={baseline_thr:.2f}) P={m_base['p']:.3f} R={m_base['r']:.3f} F1={m_base['f1']:.3f}"
              f"   |   extrap(thr={extrap_thr:.2f}) P={m_extrap['p']:.3f} R={m_extrap['r']:.3f} F1={m_extrap['f1']:.3f}")
        rows.append({
            "target": target_topo, "n_du": target_n_du, "stress": stress_name,
            "baseline_thr": baseline_thr, "extrap_thr": extrap_thr,
            "m_base": m_base, "m_extrap": m_extrap,
            "score": [round(float(x), 4) for x in scores],
            "lbl": [int(x) for x in lbl_ev],
        })

print(f"\n\n{'='*100}\nSUMMARY: existing checkpoint (trained on 4/7/10-DU) scored on 2/3-DU targets\n{'='*100}")
print(f"{'target':<16s}{'N_DU':>5s}{'stress':>7s}{'P(base)':>9s}{'P(new)':>8s}{'R(base)':>9s}{'R(new)':>8s}{'F1(base)':>10s}{'F1(new)':>9s}")
for r in rows:
    mb, me = r["m_base"], r["m_extrap"]
    print(f"{r['target']:<16s}{r['n_du']:>5d}{r['stress']:>7s}{mb['p']:>9.3f}{me['p']:>8.3f}{mb['r']:>9.3f}{me['r']:>8.3f}{mb['f1']:>10.3f}{me['f1']:>9.3f}")

print(f"\nDone. {len(rows)} rows.")

import json
OUT_DIR = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad"
DATA = {}
for r in rows:
    key = r["target"]
    DATA.setdefault(key, {})
    DATA[key][r["stress"]] = {
        "target": r["target"], "n_du": r["n_du"], "stress": r["stress"],
        "baseline_thr": r["baseline_thr"], "extrap_thr": r["extrap_thr"],
        "base": r["m_base"], "extrap": r["m_extrap"],
        "score": r["score"], "lbl": r["lbl"],
    }
for target, v in DATA.items():
    safe = target.replace("-", "_")
    out_path = f"{OUT_DIR}/data_bigsmall_p999_{safe}.js"
    with open(out_path, "w") as f:
        f.write(f"window.DATA_BIGSMALL_P999_{safe} = " + json.dumps(v) + ";\n")
    print(f"saved -> {out_path}")
