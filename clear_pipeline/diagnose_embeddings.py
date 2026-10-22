"""
diagnose_embeddings.py — Visualizing Latent Space & Feature-Level Errors
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_calibrated import CalibratedTopoAR
from run_experiment_5 import load_npz, slice_features, phase_preprocess, transform_stream, COLD_START_K, DEVICE

TEST_TOPO = "cu0_du0du1"
TRAIN_TOPOS = ["cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
CKPT_PATH = Path(f"random_model_ckpt_test_{TEST_TOPO}.pt")

CU_FEAT_NAMES = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_diff", "net_ratio"]

def extract_latent_and_errors():
    assert CKPT_PATH.exists(), f"Checkpoint {CKPT_PATH} not found!"
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"], embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1. Load Preprocessing Bundle from Train Topologies
    train_zs = [load_npz(t, "train") for t in TRAIN_TOPOS]
    bundle, _ = phase_preprocess(train_zs, TRAIN_TOPOS)

    # 2. Load & Transform Test Data
    test_z = load_npz(TEST_TOPO, "test")
    cu_te, du_te, bid_te = slice_features(test_z)
    cu_s_te, du_s_te, kept_mask, _ = transform_stream(bundle, cu_te, du_te, bid_te)
    
    if len(cu_s_te) > COLD_START_K:
        cu_s_te = cu_s_te - np.median(cu_s_te[:COLD_START_K], axis=0, keepdims=True)
        du_s_te = du_s_te - np.median(du_s_te[:COLD_START_K], axis=0, keepdims=True)

    cu_stress = test_z["cu_stress"][kept_mask].astype(np.int32)

    # 3. Hook into Model to Extract 32-dim Latent Space & Output Predictions
    cu_t = torch.tensor(cu_s_te).unsqueeze(0).to(DEVICE)
    du_t = torch.tensor(du_s_te).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Forward pass (cu_hat: [1, T, 7], du_hat: [1, T, N, 27])
        cu_hat, _ = model(cu_t, du_t)

    # Compute per-feature squared errors: [T-1, 7]
    cu_sqerr = (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()
    labels = cu_stress[1:]  # Match sequence alignment

    return cu_sqerr, labels, cu_s_te

def plot_diagnostics(cu_sqerr, labels):
    start = COLD_START_K
    sqerr_eval = cu_sqerr[start:]
    lbl_eval = labels[start:]

    normal_mask = (lbl_eval == 0)
    stress_mask = (lbl_eval == 3) # 3 = NET stress

    # -------------------------------------------------------------------------
    # Plot 1: Per-Feature Squared Error Distribution (Log Scale)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    normal_errors = np.mean(sqerr_eval[normal_mask], axis=0)
    stress_errors = np.mean(sqerr_eval[stress_mask], axis=0)

    x = np.arange(len(CU_FEAT_NAMES))
    width = 0.35

    ax.bar(x - width/2, normal_errors, width, label='Test Normal (False Alarms)', color='skyblue')
    ax.bar(x + width/2, stress_errors, width, label='Test NET Stress', color='crimson')

    ax.set_yscale("log")
    ax.set_ylabel("Mean Squared Error (Log Scale)")
    ax.set_title(f"CU Feature Residual Breakdown on Test Topology ({TEST_TOPO})")
    ax.set_xticks(x)
    ax.set_xticklabels(CU_FEAT_NAMES, rotation=15)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("cu_feature_errors.png", dpi=150)
    print("Saved feature error breakdown -> cu_feature_errors.png")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 2: t-SNE Projection of Feature Residual Space
    # -------------------------------------------------------------------------
    print("Running t-SNE reduction on feature residuals...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding = tsne.fit_transform(sqerr_eval)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[normal_mask, 0], embedding[normal_mask, 1], 
                c='blue', alpha=0.5, s=15, label='Normal Traffic')
    plt.scatter(embedding[stress_mask, 0], embedding[stress_mask, 1], 
                c='red', alpha=0.8, s=25, label='NET Stress (Ground Truth)')

    plt.title(f"t-SNE Clusters of CU Residual Space ({TEST_TOPO})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("latent_tsne.png", dpi=150)
    print("Saved t-SNE plot -> latent_tsne.png")
    plt.close()

if __name__ == "__main__":
    cu_sqerr, labels, _ = extract_latent_and_errors()
    plot_diagnostics(cu_sqerr, labels)

