"""Mahalanobis joint anomaly score -- an alternative to the project's
existing max-pool lift_score (scoring.py), inspired by the "AMDM" monitoring
method in arXiv:2509.00115 ("Adaptive Monitoring and Real-World Evaluation
of Agentic AI Systems", Aug 2025).

Why: lift_score = max_channel(sqerr / feat_norm) treats every channel as
independent -- any ONE channel deviating is enough to flag an anomaly. Our
own investigation found this is exactly why cu1_du2's `net_ratio` channel
(tx/(rx+eps)) dominates and over-triggers: at N_DU=1, ordinary traffic
bursts move tx and rx together (correlated), but the max-pool can't tell
"tx and rx moved together, as usual" apart from "tx and rx diverged, real
packet loss" -- it only sees net_ratio's own univariate deviation.

Mahalanobis distance instead scores the WHOLE residual VECTOR jointly,
using the channels' calibration-time covariance structure:

    D^2(t) = (e(t) - mu)^T Sigma^-1 (e(t) - mu)

where e(t) = pred(t) - actual(t) is the SIGNED residual vector (not
squared!) across all channels, and mu/Sigma are the residual mean vector
and covariance matrix estimated ONCE from calibration (normal-only) data.
A deviation that follows the channels' usual correlation structure (e.g.
tx and rx both drifting together) contributes much less to D^2 than one
that breaks it (tx moves while rx doesn't) -- which is exactly the
signature of real packet loss vs. ordinary N_DU=1 burstiness.

Covariance is regularized via Ledoit-Wolf shrinkage (sklearn), since our
channels are partly collinear by construction (net_diff/net_ratio are both
derived from the same net_tx/net_rx pair), which can make a raw sample
covariance matrix singular or poorly conditioned.

Standalone, additive module -- does not modify scoring.py, ndt_threshold.py,
model_calibrated.py, or any other existing file.
"""
import numpy as np
from sklearn.covariance import LedoitWolf


def fit_mahalanobis(residuals_calib):
    """Fit mu (mean residual vector) and the regularized inverse covariance
    from calibration (normal-only) SIGNED residuals.

    residuals_calib: (N, dim) signed residuals (pred - actual), calibration
                      data only -- never test data.
    returns: dict(mu: (dim,), sigma_inv: (dim, dim))
    """
    residuals_calib = np.asarray(residuals_calib, dtype=np.float64)
    mu = residuals_calib.mean(axis=0)
    lw = LedoitWolf().fit(residuals_calib)
    sigma_inv = np.linalg.pinv(lw.covariance_)
    return {"mu": mu, "sigma_inv": sigma_inv, "shrinkage": float(lw.shrinkage_)}


def mahalanobis_distance(residuals, fit):
    """Per-timestep Mahalanobis distance D^2(t).

    residuals: (T, dim) signed residuals to score (calibration OR test).
    fit:       dict from fit_mahalanobis (mu, sigma_inv).
    returns:   (T,) D^2 scores.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    diff = residuals - fit["mu"]
    # D^2(t) = diff(t) @ sigma_inv @ diff(t).T, vectorized over t
    return np.einsum("ti,ij,tj->t", diff, fit["sigma_inv"], diff)
