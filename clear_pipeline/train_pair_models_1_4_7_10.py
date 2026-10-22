"""Train 6 two-topology 'pair' models for the nested leave-one-training-topology-out
(LOTO) threshold-extrapolation experiment.

For each unique 2-topology subset of the 4-topology 1/4/7/10-DU registry, train a
CalibratedTopoAR model pooled on just those 2 topologies (same architecture/
hyperparameters as the main LOO checkpoints). These are used to test whether a
threshold fitted purely from 2 training topologies (as a function of N_DU) can be
extrapolated to the OTHER 2 topologies (used here as pseudo-held-out targets with
known labels, since they are part of our own collected data -- NOT the real held-out
topology from the outer LOO experiment).

Does not modify any existing file: imports run_experiment_1_4_7_10.py as a module
and reuses its data-loading / training phases unchanged. New checkpoints go in a
new folder (model_1_4_7_10_pairs/), never colliding with model_1_4_7_10/.

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python3 train_pair_models_1_4_7_10.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import run_experiment_1_4_7_10 as exp

ALL_TOPOS = exp.ALL_TOPOS  # ["cu1_du2", "cu3_du6du7du8du9", "cu0_7du", "cu0_10du"]
PAIR_CKPT_DIR = Path("model_1_4_7_10_pairs")
PAIR_CKPT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = list(itertools.combinations(ALL_TOPOS, 2))
print(f"Training {len(PAIRS)} pair models: {PAIRS}")

for a, b in PAIRS:
    train_topos = [a, b]
    model_ckpt = PAIR_CKPT_DIR / f"pair_{a}__{b}.pt"

    if model_ckpt.exists():
        print(f"\n{'='*70}\nSkipping {a} + {b} -- checkpoint already exists at {model_ckpt}")
        continue

    print(f"\n{'='*70}\nTraining pair: {a} + {b}")
    train_zs = [exp.load_npz(t, "train") for t in train_topos]
    bundle, train_streams = exp.phase_preprocess(train_zs, train_topos)

    fit_streams = []
    for i, (cu_s, du_s, kept_bid) in enumerate(train_streams):
        n_total = len(cu_s)
        n_cal = int(round(exp.CAL_FRAC * n_total))
        n_fit = n_total - n_cal
        fit_streams.append((cu_s[:n_fit], du_s[:n_fit], kept_bid[:n_fit]))
        print(f"  topo[{i}] {train_topos[i]:18s} total={n_total}  fit={n_fit}  cal={n_cal}")

    exp.phase_train(fit_streams, train_topos, model_ckpt)

print("\nAll pair models trained.")
