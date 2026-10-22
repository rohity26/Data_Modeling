"""run_experiment_udp_native.py — Evaluate the already-trained model_UDP
checkpoints against their OWN native test data: data_input/test_csv/ (built
into CU_NET_random_STRESS/test.npz by build_dataset.py), i.e. the original
UDP-traffic collection the model was trained and validated on.

This is a straight copy of run_experiment_udp_on_tcpnet.py (same scoring
method, same checkpoints) with two changes:
  [1] BASE_DIR -> CU_NET_random_STRESS (native UDP data) instead of
      UDP_MODEL_ON_TCPNET_STRESS (TCP-NET cross-eval data).
  [2] test split file -> "test.npz" (native) instead of "test_newdata.npz".
  [3] STRESS_TYPES_TO_RUN -> [1, 2, 3] (full CPU/MEM/NET), matching the
      original run_experiment_newdata.py methodology this model was tuned
      under (DU_EXCLUDE_IDX=[12,21], p99.9/p99.9 thresholds, no CU exclusion,
      no CU/NET gating) — this is the model's native, apples-to-apples eval.

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python run_experiment_udp_native.py
"""

import sys
import os
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

import numpy as np
import torch

from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score

# =============================================================================
# USER INPUTS
# =============================================================================

ALL_TOPOS     = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
EVAL_TOPOS    = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
STRESS_TYPES_TO_RUN = [1, 2, 3]  # CPU, MEM, NET

BASE_DIR      = Path("CU_NET_random_STRESS")   # native UDP train.npz + test.npz
STRESS_NAMES  = {1: "CPU", 2: "MEM", 3: "NET"}

STRESS_TYPE   = STRESS_TYPES_TO_RUN[0]

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6,
                 7, 8, 9, 10, 11, 12, 13, 14,
                 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]

CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]

PREPROCESS_VERSION = "v0"
CU_ZV_IDX = []
DU_ZV_IDX = []

EMBED_DIM     = 32
WINDOW_LEN    = 64
CAL_FRAC      = 0.2

COLD_START_K  = WINDOW_LEN
CU_THRESHOLD_PCT = 99.9
DU_THRESHOLD_PCT = 99.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_VARIANT = "random"
CKPT_PREFIX  = f"{DATA_VARIANT}_"

IMPUTE = True
SAVE_ERRORS = True

DU_EXCLUDE_IDX = [12, 21]   # pci_6 (dl_bs), pci_15 (pusch_rsrp_db) — native exclusion set
CU_EXCLUDE_IDX = []

ENABLE_CU_NET_GATING   = False
CU_NET_GATE_FEAT_IDX   = 5
CU_ADAPTIVE_THRESHOLD_PCT = 90.0

RUN_TAG = "udp_native"

# =============================================================================
# HELPERS
# =============================================================================

def load_npz(topo: str, split: str) -> dict:
    fname = "test.npz" if split == "test" else "train.npz"
    p = BASE_DIR / f"{topo}_stress{STRESS_TYPE}" / fname
    assert p.exists(), f"File not found: {p}"
    return dict(np.load(p))


def impute_cpu_glitch(arr: np.ndarray, irate_idx: list, eps: float = 1e-6) -> np.ndarray:
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
        else:
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t - 1, :, irate_idx], arr[t, :, irate_idx])
    return arr

def slice_features(z: dict):
    cu       = z["cu"].astype(np.float32)
    du       = z["du"].astype(np.float32)
    N_DU = du.shape[1]

    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]
    if IMPUTE:
        cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
        du = impute_cpu_glitch(du, DU_IRATE_IDX)

    _tx = cu[:, 3:4]
    _rx = cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)

    _du_tx = du[:, :, 4:5]
    _du_rx = du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)

    block_id = z["block_id"].astype(np.int64)
    return cu, du, block_id

# =============================================================================
# PHASE 1: PREPROCESSING
# =============================================================================

def phase_preprocess(train_zs, train_topos):
    raw_streams = []
    for z in train_zs:
        cu, du, bid = slice_features(z)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})

    bundle = fit_bundle(raw_streams, CU_ZV_IDX, DU_ZV_IDX, version=PREPROCESS_VERSION)

    streams = []
    for i, raw in enumerate(raw_streams):
        cu_s, du_s, _, kept_bid = transform_stream(
            bundle, raw["cu"], raw["du"], raw["block_id"]
        )
        print(f"  topo[{i}] {train_topos[i]:18s}  cu_s {cu_s.shape}  du_s {du_s.shape}  "
              f"(μ={cu_s.mean():+.3f}, σ={cu_s.std():.3f} after {PREPROCESS_VERSION})")
        streams.append((cu_s, du_s, kept_bid))
    return bundle, streams

# =============================================================================
# PHASE 3: SEQUENTIAL INFERENCE
# =============================================================================

def phase_infer(model: CalibratedTopoAR, cu_s: np.ndarray, du_s: np.ndarray):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)
    du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)

    cu_sqerr = (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()
    du_sqerr = (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()
    return cu_sqerr, du_sqerr

# =============================================================================
# PHASE 5: EVALUATE
# =============================================================================

def phase_evaluate(cu_sqerr, du_sqerr, cu_feat_norm, du_feat_norm,
                   cu_thr, du_thr, cu_stress, du_stress):
    N = du_sqerr.shape[1]

    start = COLD_START_K
    cu_sqerr_ev = cu_sqerr[start:]
    du_sqerr_ev = du_sqerr[start:]

    cu_lbl = (cu_stress[start + 1:] == STRESS_TYPE).astype(int)
    du_lbl = (du_stress[start + 1:] == STRESS_TYPE)

    assert len(cu_sqerr_ev) == len(cu_lbl), \
        f"Length mismatch: sqerr {len(cu_sqerr_ev)} vs label {len(cu_lbl)}"

    du_scores = np.stack(
        [lift_score(du_sqerr_ev[:, i, :], du_feat_norm) for i in range(N)],
        axis=1,
    )
    du_pred = (du_scores > du_thr).astype(int)

    cu_lift = cu_sqerr_ev / cu_feat_norm
    cu_thr_used = cu_thr
    if ENABLE_CU_NET_GATING and STRESS_TYPE == 3:
        any_du_flagged = du_pred.any(axis=1).astype(bool)
        cu_lift = cu_lift.copy()
        cu_lift[any_du_flagged, CU_NET_GATE_FEAT_IDX] = 0.0
        cu_thr_used = float(np.percentile(cu_lift.max(axis=1), CU_ADAPTIVE_THRESHOLD_PCT))
    cu_scores = cu_lift.max(axis=1)
    cu_pred   = (cu_scores > cu_thr_used).astype(int)

    all_metrics = {}

    def metrics(name, pred, lbl):
        tp = int(((pred == 1) & (lbl == 1)).sum())
        fp = int(((pred == 1) & (lbl == 0)).sum())
        fn = int(((pred == 0) & (lbl == 1)).sum())
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        total_anom = int((lbl == 1).sum())
        print(f"  {name:<12s}  anom={total_anom:>6d}  TP={tp:>6d}  FP={fp:>6d}  "
              f"FN={fn:>6d}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        all_metrics[name] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1, "anom": total_anom}

    print(f"\n  {'Entity':<12s}  {'anom':>6s}  {'TP':>6s}  {'FP':>6s}  "
          f"{'FN':>6s}  {'P':>5s}  {'R':>5s}  {'F1':>5s}")
    print(f"  {'-'*72}")
    metrics("CU", cu_pred, cu_lbl)
    for i in range(N):
        metrics(f"DU_{i}", du_pred[:, i], du_lbl[:, i].astype(int))

    any_pred  = (cu_pred == 1) | du_pred.any(axis=1)
    any_lbl   = (cu_lbl  == 1) | du_lbl.any(axis=1)
    print(f"  {'-'*72}")
    metrics("ANY", any_pred.astype(int), any_lbl.astype(int))

    return cu_scores, du_scores, cu_pred, du_pred, all_metrics, cu_thr_used

# =============================================================================
# MAIN
# =============================================================================

def run_one(train_topos, test_topo):
    model_ckpt = Path("model_UDP") / f"{CKPT_PREFIX}model_ckpt_test_{test_topo}.pt"

    print(f"\n{'='*70}")
    print(f"  [UDP NATIVE] Cross-topology {STRESS_NAMES[STRESS_TYPE]} stress detection")
    print(f"  Train : {train_topos}  (normal only, pooled)")
    print(f"  Test  : {test_topo}    (native UDP test_csv data, unseen topology)")
    print(f"  Device        : {DEVICE}")
    print(f"{'='*70}")

    print(f"\n[1] Loading {len(train_topos)} train topologies, fitting {PREPROCESS_VERSION} scaler ...")
    train_zs = [load_npz(t, "train") for t in train_topos]
    bundle, train_streams = phase_preprocess(train_zs, train_topos)

    print(f"\n[2] Per-topology fit/cal split (cal_frac={CAL_FRAC}) ...")
    fit_streams, cal_streams = [], []
    for i, (cu_s, du_s, kept_bid) in enumerate(train_streams):
        n_total = len(cu_s)
        n_cal   = int(round(CAL_FRAC * n_total))
        n_fit   = n_total - n_cal
        fit_streams.append((cu_s[:n_fit], du_s[:n_fit], kept_bid[:n_fit]))
        cal_streams.append((cu_s[n_fit:], du_s[n_fit:]))
        print(f"  topo[{i}] {train_topos[i]:18s} total={n_total}  fit={n_fit}  cal={n_cal}")
    n_fit_total = sum(len(f[0]) for f in fit_streams)

    cu_dim = fit_streams[0][0].shape[1]
    du_dim = fit_streams[0][1].shape[2]
    if not model_ckpt.exists():
        raise SystemExit(
            f"\n  Checkpoint {model_ckpt} not found. This script only EVALUATES "
            f"existing checkpoints on the new data — it does not train."
        )
    print(f"\n[3] Loading model from checkpoint ({model_ckpt}) ...")
    ckpt = torch.load(model_ckpt, map_location='cpu')
    mismatches = []
    if ckpt.get("topos") != list(train_topos):
        mismatches.append(f"topos: ckpt={ckpt.get('topos')} vs current={list(train_topos)}")
    if ckpt.get("n_train_rows") != n_fit_total:
        mismatches.append(f"n_train_rows: ckpt={ckpt.get('n_train_rows')} vs current={n_fit_total}")
    if ckpt.get("cu_dim") != cu_dim:
        mismatches.append(f"cu_dim: ckpt={ckpt.get('cu_dim')} vs current={cu_dim}")
    if ckpt.get("du_dim") != du_dim:
        mismatches.append(f"du_dim: ckpt={ckpt.get('du_dim')} vs current={du_dim}")
    if ckpt.get("preprocess") != PREPROCESS_VERSION:
        mismatches.append(f"preprocess: ckpt={ckpt.get('preprocess')} vs current={PREPROCESS_VERSION}")
    if mismatches:
        raise SystemExit(
            f"\n  Refusing to load incompatible checkpoint {model_ckpt}:\n    "
            + "\n    ".join(mismatches)
        )
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"],
                             embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    print(f"\n[4] Inference on held-out CAL streams (per topology, pooled for calibration) ...")
    cu_sqerr_pool, du_sqerr_pool = [], []
    for i, (cu_s_cal, du_s_cal) in enumerate(cal_streams):
        cu_sq, du_sq = phase_infer(model, cu_s_cal, du_s_cal)
        cu_sqerr_pool.append(cu_sq[COLD_START_K:])
        du_sqerr_pool.append(du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1]))
        print(f"  topo[{i}] {train_topos[i]:18s} cu_sqerr {cu_sq.shape}  du_sqerr {du_sq.shape}")
    cu_sqerr_n   = np.concatenate(cu_sqerr_pool, axis=0)
    du_sqerr_flt = np.concatenate(du_sqerr_pool, axis=0)
    print(f"  Pooled cal rows: cu={cu_sqerr_n.shape[0]}  du(flat)={du_sqerr_flt.shape[0]}")

    print(f"\n[5] Calibrating thresholds on POOLED cal stream (no test labels) ...")
    cu_fn = feat_norm_calibrated(cu_sqerr_n)
    du_fn = feat_norm_calibrated(du_sqerr_flt)
    if DU_EXCLUDE_IDX:
        du_fn[DU_EXCLUDE_IDX] = np.inf
        print(f"  Excluding DU feature indices {DU_EXCLUDE_IDX} from scoring (feat_norm set to inf)")
    if CU_EXCLUDE_IDX:
        cu_fn[CU_EXCLUDE_IDX] = np.inf
        print(f"  Excluding CU feature indices {CU_EXCLUDE_IDX} from scoring (feat_norm set to inf)")
    cu_norm_scores = lift_score(cu_sqerr_n,   cu_fn)
    du_norm_scores = lift_score(du_sqerr_flt, du_fn)
    du_thr = float(np.percentile(du_norm_scores, DU_THRESHOLD_PCT))
    cu_thr = float(np.percentile(cu_norm_scores, CU_THRESHOLD_PCT))
    print(f"  CU threshold (p{CU_THRESHOLD_PCT:.1f}): {cu_thr:.4f}")
    print(f"  DU threshold (p{DU_THRESHOLD_PCT:.1f}): {du_thr:.4f}")

    print(f"\n[6] Transforming NATIVE TEST data ({test_topo}) with train-fitted scaler ...")
    test_z  = load_npz(test_topo, "test")
    cu_te, du_te, bid_te = slice_features(test_z)
    cu_s_te, du_s_te, kept_mask, _ = transform_stream(bundle, cu_te, du_te, bid_te)

    cu_stress = test_z["cu_stress"][kept_mask].astype(np.int32)
    du_stress = test_z["du_stress"][kept_mask].astype(np.int32)
    n_du_te   = du_s_te.shape[1]
    print(f"  cu_s_te {cu_s_te.shape}  du_s_te {du_s_te.shape}  "
          f"(μ={cu_s_te.mean():+.3f}, σ={cu_s_te.std():.3f})")
    print(f"  CU {STRESS_NAMES[STRESS_TYPE]}-stress rows : {(cu_stress == STRESS_TYPE).sum()}")
    for i in range(n_du_te):
        print(f"  DU_{i} {STRESS_NAMES[STRESS_TYPE]}-stress rows: {(du_stress[:, i] == STRESS_TYPE).sum()}")

    print("\n[7] Running inference on new test stream ...")
    cu_sqerr, du_sqerr = phase_infer(model, cu_s_te, du_s_te)
    print(f"  cu_sqerr {cu_sqerr.shape}  du_sqerr {du_sqerr.shape}")

    if SAVE_ERRORS:
        feat_tag = f"f{cu_dim}"
        tag_suffix = f"_{RUN_TAG}" if RUN_TAG else ""
        save_dir = Path(__file__).parent / "exp_runs" / f"UDP_{STRESS_NAMES[STRESS_TYPE]}{tag_suffix}_my_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        err_path = save_dir / f"recon_errors_{test_topo}_{feat_tag}.npz"
        np.savez(err_path,
                 cu_sqerr=cu_sqerr, du_sqerr=du_sqerr,
                 cu_stress=cu_stress, du_stress=du_stress,
                 cu_feat_norm=cu_fn, du_feat_norm=du_fn,
                 cu_thr=np.array([cu_thr]),
                 du_thr=np.array([du_thr]),
                 cu_sqerr_cal=cu_sqerr_n, du_sqerr_cal=du_sqerr_flt)
        print(f"  Errors saved → {err_path}")

    print("\n[8] Evaluation results ...")
    cu_scores, du_scores, cu_pred, du_pred, eval_metrics, cu_thr_used = phase_evaluate(
        cu_sqerr, du_sqerr, cu_fn, du_fn, cu_thr, du_thr, cu_stress, du_stress
    )

    print("\nDone.")
    return eval_metrics


def run_all_loo_for_stress(stress_type: int):
    global STRESS_TYPE
    STRESS_TYPE = stress_type

    all_results = []
    for test_t in EVAL_TOPOS:
        train_ts = [t for t in ALL_TOPOS if t != test_t]
        result = run_one(train_ts, test_t)
        all_results.append({"test_topo": test_t, "metrics": result})

    entity_keys: list = []
    for r in all_results:
        for k in r["metrics"]:
            if k not in entity_keys:
                entity_keys.append(k)

    print(f"\n\n{'='*70}")
    print(f"  [UDP NATIVE] LEAVE-ONE-OUT SUMMARY — {STRESS_NAMES[stress_type]}  ({len(EVAL_TOPOS)} configurations)")
    print(f"{'='*70}")
    header = f"  {'Test topology':<22s}" + "".join(f"  {e+' F1':>10s}" for e in entity_keys)
    print(header)
    print(f"  {'-'*68}")
    for r in all_results:
        row = f"  {r['test_topo']:<22s}"
        for e in entity_keys:
            if e in r["metrics"]:
                row += f"  {r['metrics'][e]['f1']:>10.3f}"
            else:
                row += f"  {'N/A':>10s}"
        print(row)
    print(f"  {'-'*68}")

    csv_path = Path(f"loo_results_{RUN_TAG}.csv")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["stress_type", "test_topo", "entity",
                             "anom", "tp", "fp", "fn", "precision", "recall", "f1"])
        for r in all_results:
            for entity, m in r["metrics"].items():
                writer.writerow([
                    STRESS_NAMES[stress_type], r["test_topo"], entity,
                    m["anom"], m["tp"], m["fp"], m["fn"],
                    f"{m['p']:.4f}", f"{m['r']:.4f}", f"{m['f1']:.4f}",
                ])
    print(f"\n  Results appended → {csv_path.resolve()}")
    return all_results


def main():
    for stress_type in STRESS_TYPES_TO_RUN:
        run_all_loo_for_stress(stress_type)


if __name__ == "__main__":
    main()
