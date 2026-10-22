"""Nonparametric Dynamic Thresholding (NDT) -- Hundman et al. 2018,
"Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
Thresholding" (KDD 2018, arXiv:1802.04431).

Picks a threshold eps = mu + z*sigma from a calibration-only error stream by
sweeping z and keeping whichever candidate causes the largest relative drop
in mean/std per flagged point/run (Section 3.2), then prunes weak flagged
runs at apply-time (Section 3.3). New, standalone module -- does not modify
any existing project file.
"""
import numpy as np

Z_VALUES = np.arange(2.5, 10.01, 0.5)
PRUNE_P = 0.13
EWMA_ALPHA = 0.05


def ewma_smooth(x, alpha=EWMA_ALPHA):
    """Exponentially-weighted smoothing of the raw error/score stream --
    Section 3.1 of Hundman et al. assumes NDT operates on smoothed errors,
    not the raw per-timestep stream (reduces point-to-point jitter noise)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _count_runs(mask):
    runs = 0
    prev = False
    for v in mask:
        if v and not prev:
            runs += 1
        prev = bool(v)
    return runs


def select_threshold(cal_scores, z_values=Z_VALUES):
    """Pick eps = mu + z*sigma from a calibration-only score stream.
    Returns dict with eps, z, score, n_flagged, n_runs."""
    e_s = np.asarray(cal_scores, dtype=np.float64)
    mu_all = float(e_s.mean())
    sigma_all = float(e_s.std())
    best = None
    for z in z_values:
        eps = mu_all + z * sigma_all
        mask = e_s > eps
        n_flagged = int(mask.sum())
        if n_flagged == 0:
            continue
        below = e_s[~mask]
        mu_below = float(below.mean())
        sigma_below = float(below.std())
        delta_mu = mu_all - mu_below
        delta_sigma = sigma_all - sigma_below
        n_runs = _count_runs(mask)
        denom = n_flagged + n_runs ** 2
        if denom == 0:
            continue
        score = (delta_mu / (mu_all + 1e-12) + delta_sigma / (sigma_all + 1e-12)) / denom
        if best is None or score > best["score"]:
            best = {"z": float(z), "eps": float(eps), "score": float(score),
                    "n_flagged": n_flagged, "n_runs": n_runs}
    if best is None:
        z = float(z_values[-1])
        return {"z": z, "eps": float(mu_all + z * sigma_all), "score": 0.0,
                "n_flagged": 0, "n_runs": 0}
    return best


def _find_runs(mask):
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def prune(scores, eps, p=PRUNE_P):
    """Apply the Section 3.3 pruning rule to a thresholded score stream.
    Returns a boolean prediction array (True = anomalous, after pruning)."""
    scores = np.asarray(scores, dtype=np.float64)
    mask = scores > eps
    runs = _find_runs(mask)
    if not runs:
        return mask.copy()
    peaks = [float(scores[s:e].max()) for s, e in runs]
    normal_vals = scores[~mask]
    sentinel = float(normal_vals.max()) if len(normal_vals) > 0 else 0.0
    order = sorted(range(len(runs)), key=lambda k: -peaks[k])
    sorted_runs = [runs[k] for k in order]
    sorted_peaks = [peaks[k] for k in order] + [sentinel]

    d_list = []
    for i in range(1, len(sorted_peaks)):
        prev_peak = sorted_peaks[i - 1]
        d = (prev_peak - sorted_peaks[i]) / prev_peak if prev_peak != 0 else 0.0
        d_list.append(d)

    cut_at = len(sorted_runs)  # default: keep every run
    for k in range(len(d_list)):
        if all(d < p for d in d_list[k:]):
            cut_at = k + 1
            break

    pruned_mask = np.zeros_like(mask)
    for (s, e) in sorted_runs[:cut_at]:
        pruned_mask[s:e] = True
    return pruned_mask


def metrics(pred, lbl):
    pred = np.asarray(pred).astype(int)
    lbl = np.asarray(lbl).astype(int)
    tp = int(((pred == 1) & (lbl == 1)).sum())
    fp = int(((pred == 1) & (lbl == 0)).sum())
    fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}
