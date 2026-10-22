"""Static SPOT (Siffer et al. 2017, "Anomaly Detection in Streams with Extreme
Value Theory", KDD'17) -- Peaks-Over-Threshold / Generalized Pareto Distribution
tail-fit threshold. Calibration-only (frozen after fit, like NDT/existing in
this project) -- deliberately NOT the online-updating DSPOT variant, since the
online-update pattern (TADST Eq.24, and our own earlier Attar-et-al.-style
adaptive-threshold experiment) was already tested and rejected elsewhere in
this project for self-poisoning on sustained anomaly blocks.

Algorithm (static POT):
  1. Pick an initial high level t = percentile(calib_scores, init_pctl).
  2. Excesses Y_i = X_i - t for all X_i > t (there are N_t of them, out of n
     total calibration points).
  3. Fit a Generalized Pareto Distribution (shape gamma, scale sigma) to the
     excesses Y_i via maximum likelihood (scipy.stats.genpareto).
  4. Final threshold for risk q:
        z_q = t + (sigma/gamma) * ( (q * n / N_t)^(-gamma) - 1 )     [gamma != 0]
        z_q = t - sigma * log(q * n / N_t)                           [gamma == 0]
     (Siffer et al., Eq. 5/output of Algorithm 1; q is the desired false-alarm
     probability P(X > z_q) < q.)

New, standalone module -- does not modify any existing project file.
"""
import numpy as np
from scipy.stats import genpareto


def fit_spot_threshold(calib_scores, q, init_pctl=98.0):
    """calib_scores: 1D array of pooled calibration anomaly scores.
    q: desired risk (false-positive probability), e.g. 1e-4.
    init_pctl: percentile used to pick the initial high level t.
    Returns dict with t, n, Nt, gamma, sigma, z_q (the final threshold)."""
    x = np.asarray(calib_scores, dtype=np.float64)
    n = len(x)
    t = float(np.percentile(x, init_pctl))
    excesses = x[x > t] - t
    Nt = len(excesses)
    if Nt < 10:
        # too few exceedances to fit a GPD reliably -- fall back to the plain
        # percentile at the equivalent (1-q) level
        return {"t": t, "n": n, "Nt": Nt, "gamma": None, "sigma": None,
                "z_q": float(np.percentile(x, 100 * (1 - q))), "fallback": True}

    gamma, _loc, sigma = genpareto.fit(excesses, floc=0)

    if abs(gamma) < 1e-8:
        z_q = t - sigma * np.log(q * n / Nt)
    else:
        z_q = t + (sigma / gamma) * ((q * n / Nt) ** (-gamma) - 1)

    return {"t": t, "n": n, "Nt": Nt, "gamma": float(gamma), "sigma": float(sigma),
            "z_q": float(z_q), "fallback": False}
