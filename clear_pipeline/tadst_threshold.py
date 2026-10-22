"""Faithful implementation of TADST's Anomaly Detection Mechanism (Yang et al.,
Applied Intelligence 2025, "TADST: reconstruction with spatio-temporal feature
fusion for deviation-based time series anomaly detection"):

  1. EWMA-smooth the score (their Eq.22, their own convention -- note this is
     the OPPOSITE weighting convention from our ndt_threshold.ewma_smooth):
        s_t = s_{t-1}*beta + e_t*(1-beta)
     beta=0.1 is their reported fixed value (weak smoothing: 90% weight on
     the current point, 10% on history).
  2. Fit a GPD to the tail of the smoothed CALIBRATION scores (same POT
     mechanics as spot_threshold.py) to get an initial threshold tau_0 and
     (sigma, gamma).
  3. ONLINE update during evaluation (their Eq.24): after every new smoothed
     test-stream point, if it exceeds the CURRENT threshold, flag it, then
     recompute:
        tau_new = tau_prev + (sigma/gamma) * ( (N_tau/(N*alpha))^gamma - 1 )
     where N is the number of test points seen so far, N_tau the number
     flagged so far, and alpha the expected anomaly proportion (a second,
     separate hyperparameter from beta).

This is the exact method under test -- including its online-update step,
which our own earlier online-adaptive-threshold experiment already showed
self-poisons on sustained anomaly blocks. Included here to get a concrete,
data-backed answer for THIS specific formula rather than relying on that
earlier generic result.

New, standalone module -- does not modify any existing project file.
"""
import numpy as np
from spot_threshold import fit_spot_threshold


def ewma_smooth_tadst(x, beta=0.1):
    """TADST Eq.22: s_t = s_{t-1}*beta + e_t*(1-beta)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] * beta + x[i] * (1 - beta)
    return out


def fit_tadst_initial(calib_scores, alpha, init_pctl=98.0, beta=0.1):
    """Smooth calibration scores, fit GPD tail, return initial threshold +
    (sigma, gamma) needed for the online update."""
    smoothed = ewma_smooth_tadst(calib_scores, beta)
    fit = fit_spot_threshold(smoothed, alpha, init_pctl)
    return fit


def run_tadst_online(test_scores, sigma, gamma, tau_0, alpha, beta=0.1,
                      smooth_state0=None):
    """Runs TADST's online detection + threshold update over a test stream.
    test_scores: raw (unsmoothed) test scores, in order.
    Returns (pred, tau_trace) -- binary predictions and the threshold's
    trajectory over time (for diagnosing drift)."""
    n_test = len(test_scores)
    pred = np.zeros(n_test, dtype=int)
    tau = tau_0
    tau_trace = np.empty(n_test)
    N = 0
    N_tau = 0
    s_prev = smooth_state0 if smooth_state0 is not None else test_scores[0]

    for i in range(n_test):
        s = s_prev * beta + test_scores[i] * (1 - beta)
        s_prev = s
        N += 1
        if s > tau:
            pred[i] = 1
            N_tau += 1
            ratio = N_tau / (N * alpha)
            if abs(gamma) < 1e-8:
                tau = tau + sigma * np.log(ratio) if ratio > 0 else tau
            else:
                tau = tau + (sigma / gamma) * (ratio ** gamma - 1)
        tau_trace[i] = tau
    return pred, tau_trace
