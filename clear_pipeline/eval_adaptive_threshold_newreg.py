"""Validate the Attar et al. (ICSRS 2024) online adaptive dynamic threshold
(ndt_adaptive_threshold.py) against the classic Hundman NDT (frozen eps,
ndt_threshold.py) across all 4 new-registry own-held-out checkpoints x all 3
stress types -- the same 12 cells used throughout this investigation.

Reuses the already-computed raw score/label arrays and classic NDT eps saved
in data_newreg_own_plots.js (no re-inference needed): for each cell, EWMA-
smooth the raw score the same way NDT always does (alpha=0.05), seed the
adaptive threshold with that cell's own classic eps (t0_eps -- calibration-
grounded, exactly like Hundman's frozen eps), then let it adapt online
through the test stream itself and compare F1 against the frozen-eps
baseline.

Read-only; does not touch ndt_threshold.py, any checkpoint, or any training
process.
"""
import json, re
import numpy as np
import ndt_threshold as ndt
from ndt_adaptive_threshold import adaptive_threshold_stream

DATA_JS = "/tmp/claude-1017/-home-cs26mtech02007-Rohit-data-modeling-2/34e95e6b-da16-4cd1-9ef3-a03b8c94f775/scratchpad/loo4_artifact/data_newreg_own_plots.js"

txt = open(DATA_JS).read()
m = re.match(r"\s*window\.DATA_NEWREG_OWN_PLOTS\s*=\s*(.*);\s*$", txt, re.S)
data = json.loads(m.group(1))


def metrics(pred, lbl):
    tp = int(((pred == 1) & (lbl == 1)).sum()); fp = int(((pred == 1) & (lbl == 0)).sum()); fn = int(((pred == 0) & (lbl == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * p * r / (p + r + 1e-9)}


print(f"{'ckpt':20s} {'stress':5s}  {'F1(frozen NDT)':>15s}  {'F1(adaptive)':>13s}")
dummy_calib = np.zeros(2)

for ckpt_key, entry in data.items():
    eps_frozen = entry["eps"]
    for stress, rec in entry["stress"].items():
        score = np.array(rec["score"]); lbl = np.array(rec["lbl"])

        smoothed = ndt.ewma_smooth(score)
        pred_frozen = (smoothed > eps_frozen).astype(int)
        m_frozen = metrics(pred_frozen, lbl)

        T_adapt, Z_adapt = adaptive_threshold_stream(dummy_calib, smoothed, t0_eps=eps_frozen)
        pred_adapt = (smoothed > T_adapt).astype(int)
        m_adapt = metrics(pred_adapt, lbl)

        print(f"{ckpt_key:20s} {stress:5s}  {m_frozen['f1']:15.3f}  {m_adapt['f1']:13.3f}   "
              f"(frozen: tp={m_frozen['tp']} fp={m_frozen['fp']} fn={m_frozen['fn']} | "
              f"adaptive: tp={m_adapt['tp']} fp={m_adapt['fp']} fn={m_adapt['fn']}, "
              f"T range=[{T_adapt.min():.1f},{T_adapt.max():.1f}], z range=[{Z_adapt.min():.1f},{Z_adapt.max():.1f}])")

print("\nDONE")
