"""Reusable N_DU-correction coefficient fitter for the CU cpu/mem_pct/mem_bytes
channels. Given an explicit list of (topology, N_DU) training pairs (always the
checkpoint's OWN training set, excluding its held-out/target topology -- same
"never touch target data" discipline used for NDT thresholds elsewhere in this
project), computes:
  - mem_pct, mem_bytes: simple OLS  value ~ beta0 + beta1*N_DU
    (no imputation -- these channels were never in CU_IRATE_IDX; near-zero
    traffic correlation confirmed earlier, so N_DU alone is enough)
  - cpu: joint OLS  cpu ~ beta0 + beta1*N_DU + beta2*traffic
    (traffic = that topology's own raw net_tx at the same timestep, BEFORE
    the /N_DU division; needed because cpu correlates with traffic and a
    2-variable fit would conflate the N_DU effect with leftover traffic
    variation -- validated earlier: 2-var gives beta1=0.0163, 3-var gives
    beta1=0.004793, for the cu1_du2+cu2_du3du4du5+cu3_du6du7du8du9 split)

Usage: import fit_betas(train_pairs) -> {"cpu": b1, "mem_pct": b1, "mem_bytes": b1}
train_pairs = [("cu1_du2", 1), ("cu2_du3du4du5", 3), ("cu3_du6du7du8du9", 4)]

No existing files modified; read-only (calibration data only, no test files).
"""
import numpy as np

CLEAR = "/home/cs26mtech02007/Rohit_data_modeling-2/clear_pipeline"
BASE_DIR = f"{CLEAR}/CU_DL_UDP_STRESS"


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

        # mem_pct(1), mem_bytes(2): raw, unimputed (never in CU_IRATE_IDX)
        mempct_all.append(cu_raw[:, 1]); membytes_all.append(cu_raw[:, 2])
        ndu_plain.append(np.full(n, n_du, dtype=np.float32))

        # cpu(0): glitch-imputed (matches CU_IRATE_IDX=[0,3,4] on raw array:
        # cpu, fs_reads, fs_writes), traffic = raw net_tx(5), unimputed/undivided
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
    SPLITS = {
        "cu0_du0du1":        [("cu1_du2", 1), ("cu2_du3du4du5", 3), ("cu3_du6du7du8du9", 4)],
        "cu1_du2":           [("cu0_du0du1", 2), ("cu2_du3du4du5", 3), ("cu3_du6du7du8du9", 4)],
        "cu2_du3du4du5":     [("cu0_du0du1", 2), ("cu1_du2", 1), ("cu3_du6du7du8du9", 4)],
        "cu3_du6du7du8du9":  [("cu0_du0du1", 2), ("cu1_du2", 1), ("cu2_du3du4du5", 3)],
    }
    results = {}
    for test_topo, pairs in SPLITS.items():
        print(f"\n{'='*90}\nSplit: test={test_topo}  train={[p[0] for p in pairs]}\n{'='*90}")
        results[test_topo] = fit_betas(pairs)

    print(f"\n{'='*90}\nSUMMARY (beta1 per split)\n{'='*90}")
    for test_topo, b in results.items():
        print(f"  test={test_topo:20s} cpu={b['cpu']:.6g}  mem_pct={b['mem_pct']:.6g}  mem_bytes={b['mem_bytes']:.6g}")
