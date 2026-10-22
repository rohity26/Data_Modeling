"""N_DU-correction coefficient fitter for the NEW registry (cu1_du2, cu3_du6du7du8du9,
cu0_7du, cu0_10du -- N_DU = 1, 4, 7, 10), mirroring fit_ndu_beta1.py exactly but
pointing at CU_1_4_7_10_STRESS instead of CU_DL_UDP_STRESS. Separate topology set
(different data collection), so betas must be refit independently -- the original
registry's ndu_beta1_per_split.json values do NOT apply here.

Same methodology as fit_ndu_beta1.py:
  - mem_pct, mem_bytes: raw (unimputed), simple OLS  value ~ beta0 + beta1*N_DU
  - cpu: glitch-imputed (CU_IRATE_IDX=[0,3,4] on raw array), joint OLS
         cpu ~ beta0 + beta1*N_DU + beta2*traffic (traffic = raw net_tx, column 5)

Writes a NEW file (ndu_beta1_per_split_new_registry.json) -- never touches the
original registry's ndu_beta1_per_split.json.
"""
import numpy as np
import json

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
BASE_DIR = f"{CLEAR}/CU_1_4_7_10_STRESS"

N_DU_MAP = {"cu1_du2": 1, "cu3_du6du7du8du9": 4, "cu0_7du": 7, "cu0_10du": 10}
ALL_TOPOS = list(N_DU_MAP)


def impute(arr, idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        g = arr[t, idx] < eps
        arr[t, idx] = np.where(g, arr[t - 1, idx], arr[t, idx])
    return arr


def fit_betas(train_pairs, verbose=True):
    """train_pairs: list of (topo_name, n_du). Returns dict of beta1 values."""
    mempct_all, membytes_all, ndu_plain = [], [], []
    cpu_all, tx_all, ndu_cpu = [], [], []

    for topo, n_du in train_pairs:
        z = dict(np.load(f"{BASE_DIR}/{topo}_stress3/train.npz"))
        cu_raw = z["cu"].astype(np.float32)
        n = len(cu_raw)

        mempct_all.append(cu_raw[:, 1]); membytes_all.append(cu_raw[:, 2])
        ndu_plain.append(np.full(n, n_du, dtype=np.float32))

        cu_imp = impute(cu_raw, [0, 3, 4])
        cpu_all.append(cu_imp[:, 0]); tx_all.append(cu_imp[:, 5])
        ndu_cpu.append(np.full(n, n_du, dtype=np.float32))

        if verbose:
            print(f"  {topo} (N_DU={n_du}): {n} rows")

    ndu_plain = np.concatenate(ndu_plain)
    X_plain = np.column_stack([np.ones_like(ndu_plain), ndu_plain])
    betas = {}
    for name, rows in [("mem_pct", mempct_all), ("mem_bytes", membytes_all)]:
        y = np.concatenate(rows)
        coef, *_ = np.linalg.lstsq(X_plain, y, rcond=None)
        betas[name] = float(coef[1])

    ndu_cpu = np.concatenate(ndu_cpu); cpu_all = np.concatenate(cpu_all); tx_all = np.concatenate(tx_all)
    X_joint = np.column_stack([np.ones_like(ndu_cpu), ndu_cpu, tx_all])
    coef, *_ = np.linalg.lstsq(X_joint, cpu_all, rcond=None)
    betas["cpu"] = float(coef[1])

    if verbose:
        print(f"  -> cpu beta1={betas['cpu']:.6g}  mem_pct beta1={betas['mem_pct']:.6g}  mem_bytes beta1={betas['mem_bytes']:.6g}")
    return betas


if __name__ == "__main__":
    SPLITS = {}
    for held_out in ALL_TOPOS:
        SPLITS[held_out] = [(t, N_DU_MAP[t]) for t in ALL_TOPOS if t != held_out]

    results = {}
    for test_topo, pairs in SPLITS.items():
        print(f"\n{'='*90}\nSplit: test={test_topo}  train={[p[0] for p in pairs]}\n{'='*90}")
        results[test_topo] = fit_betas(pairs)

    print(f"\n{'='*90}\nSUMMARY (beta1 per split, NEW registry)\n{'='*90}")
    for test_topo, b in results.items():
        print(f"  test={test_topo:20s} cpu={b['cpu']:.6g}  mem_pct={b['mem_pct']:.6g}  mem_bytes={b['mem_bytes']:.6g}")

    out_path = f"{CLEAR}/ndu_beta1_per_split_new_registry.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {out_path}")
