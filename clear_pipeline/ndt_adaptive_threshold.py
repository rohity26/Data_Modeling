"""Adaptive Dynamic Threshold -- Attar, Bao, Hagenmeyer, Fabarisov, Morozov,
"Improving Anomaly Detection with Adaptive Dynamic Threshold: A Review and
Enhanced Method", ICSRS 2024 (IEEE 8th Int'l Conf. on System Reliability and
Safety), pp. 662-666.

Extends Hundman et al. 2018's NDT (ndt_threshold.py in this project) from a
ONE-TIME, offline, grid-swept eps frozen on calibration data, to a
CONTINUOUSLY-UPDATING, online threshold that keeps re-estimating z (and hence
T) from the growing stream of (unlabeled) smoothed residuals as new points
arrive -- at test time, not just at calibration time.

Same core statistic as Hundman's z-selection score, but computed directly
(closed form) instead of swept over a z-grid, at every new timestep:

    T = mu(es) + z * sigma(es)                                        (1)
    z = (delta_mu/mu(es) + delta_sigma/sigma(es)) / (|ea| + |Eseq|)
    delta_mu    = mu(es) - mu({es | es < T_prev})
    delta_sigma = sigma(es) - sigma({es | es < T_prev})
    ea          = {es | es > T_prev}      (points above the PREVIOUS threshold)
    Eseq        = continuous runs of ea
    z clamped to [2, 10]  (paper's range; Hundman's own sweep uses 2.5-10)

Design notes vs. the paper's literal Algorithm 1:

1. We do NOT recompute mu/sigma by rescanning a combined calibration+test
   history from scratch at every step (O(n^2), and numerically pointless:
   tested empirically, with ~90k calibration points behind a ~700-point
   test stream, mu_all/sigma_all over the COMBINED history barely moves no
   matter what the test stream does -- the huge calibration tail swamps it,
   defeating the entire point of "adapting to this stream"). Calibration is
   used ONLY to seed the starting threshold (t0_eps, the classic one-shot
   NDT eps already fit on es_calib -- the same calibration-only grounding
   Hundman's frozen eps has). The online loop then grows and adapts purely
   over the TEST stream's OWN history -- the realistic reading of "a stream
   of smoothed residuals" for a monitor that's adapting to what it is
   actually watching, not to unrelated calibration topologies.
2. No test LABELS are ever used to pick T, only the unlabeled residual
   values themselves, in the order they arrive -- the same information an
   online production deployment would actually have (it doesn't know future
   stress labels, but it does observe the live residual stream as it happens).

Does not modify ndt_threshold.py -- standalone, additive module.
"""
import numpy as np

Z_LO, Z_HI = 2.0, 10.0  # paper's clamp range (Hundman's own sweep is 2.5-10)


def _count_runs(mask):
    """Vectorized count of contiguous True-runs in a boolean array -- a
    pure-Python loop here would be O(n) per call but called once per test
    timestep over a growing (tens-of-thousands-element) history, so total
    cost would be O(n_test * n_history) in slow Python; numpy diff makes
    each call itself O(n_history) in fast vectorized code instead."""
    if len(mask) == 0:
        return 0
    m = mask.astype(np.int8)
    runs = int(np.sum(np.diff(m) == 1))
    if m[0] == 1:
        runs += 1
    return runs


def adaptive_threshold_stream(es_calib, es_test, t0_eps=None):
    """Run the Attar et al. online adaptive threshold over the test stream,
    seeded (not permanently anchored) by the calibration-fit starting eps.

    es_calib: (Nc,) EWMA-smoothed calibration score stream (normal-only) --
              used ONLY to pick t0_eps when not given; never re-touched
              after that (see module docstring for why).
    es_test:  (Nt,) EWMA-smoothed TEST score stream (what we're scoring).
              The online adaptation grows and re-estimates mu/sigma/z
              purely from this stream's own history as it arrives.
    t0_eps:   initial T before any adaptation -- pass the classic NDT eps
              (from ndt_threshold.select_threshold on es_calib) for a fair,
              calibration-grounded start; defaults to mean+3*std of es_calib
              if not given.

    Returns (T_test, Z_test): per-timestep threshold/z arrays aligned to
    es_test (one T per test point, each computed from the test stream up to
    and including that point, per the paper's own "update es with es_new,
    then compute (T,z)" procedure).
    """
    es_calib = np.asarray(es_calib, dtype=np.float64)
    es_test = np.asarray(es_test, dtype=np.float64)
    nt = len(es_test)

    if t0_eps is None:
        t0_eps = float(es_calib.mean() + 3 * es_calib.std())
    T_prev = float(t0_eps)

    def step(hist):
        mu_all = hist.mean()
        sigma_all = hist.std()
        mask_above = hist > T_prev
        ea = hist[mask_above]
        below = hist[~mask_above]
        mu_below = below.mean() if len(below) else mu_all
        sigma_below = below.std() if len(below) else 0.0
        delta_mu = mu_all - mu_below
        delta_sigma = sigma_all - sigma_below
        n_runs = _count_runs(mask_above)
        denom = len(ea) + n_runs
        if denom == 0 or mu_all == 0 or sigma_all == 0:
            z = Z_LO
        else:
            z = (delta_mu / mu_all + delta_sigma / sigma_all) / denom
            z = min(max(z, Z_LO), Z_HI)
        T_new = mu_all + z * sigma_all
        return T_new, z

    buf = np.empty(nt, dtype=np.float64)
    T_test = np.empty(nt, dtype=np.float64)
    Z_test = np.empty(nt, dtype=np.float64)
    for t in range(nt):
        buf[t] = es_test[t]
        T_new, z = step(buf[:t + 1])
        T_test[t] = T_new
        Z_test[t] = z
        T_prev = T_new

    return T_test, Z_test
