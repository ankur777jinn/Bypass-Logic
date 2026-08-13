"""
compare_subspaces.py  —  Subspace Similarity Analysis (Base vs Instruct V_bypass)

Computes geometric similarity between V_bypass extracted from an Instruct-FT
model and V_bypass extracted from a Base-FT model using standard subspace
comparison methods from Grassmannian geometry.

Metrics computed (Björck & Golub, 1973; Edelman et al., 1998):
  1. Principal angles θ_1 ... θ_k  (via SVD of V1 @ V2.T)
  2. Grassmann geodesic distance    d_g = ||θ||_2
  3. Chordal (Fubini–Study) distance d_c = sqrt(Σ sin²θ_i)
  4. Projection metric               d_p = ||P1 - P2||_F / sqrt(2)
  5. Normalized subspace overlap      (1/k) ||V1 V2.T||_F^2

Additionally:
  - Permutation test (N=1000) against random k-dimensional subspaces
  - Eigenvalue spectrum comparison from metadata
  - Four publication-quality figures saved as PNG

Usage:
    python compare_subspaces.py \\
        --vbypass_instruct artifacts/vbypass.pt \\
        --vbypass_base artifacts/vbypass_base.pt \\
        --meta_instruct artifacts/vbypass_meta.json \\
        --meta_base artifacts/vbypass_base_meta.json \\
        --output_dir results/subspace_comparison
"""

import argparse
import json
import os
import textwrap
from datetime import datetime

import numpy as np
import torch

# ── Attempt matplotlib import (graceful fallback if headless) ──
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for servers
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed — skipping figure generation.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare V_bypass subspaces between Instruct-FT and Base-FT models."
    )
    p.add_argument("--vbypass_instruct", required=True,
                   help="Path to vbypass.pt from the Instruct model")
    p.add_argument("--vbypass_base", required=True,
                   help="Path to vbypass.pt from the Base model")
    p.add_argument("--meta_instruct", default=None,
                   help="Path to vbypass_meta.json from Instruct (optional)")
    p.add_argument("--meta_base", default=None,
                   help="Path to vbypass_meta.json from Base (optional)")
    p.add_argument("--output_dir", default="results/subspace_comparison",
                   help="Directory for all outputs (report, figures, JSON)")
    p.add_argument("--n_permutations", type=int, default=1000,
                   help="Number of random subspace draws for permutation test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ===========================================================================
# Loading
# ===========================================================================

def load_vbypass(path):
    """Load V_bypass matrix [k, D] from a .pt file."""
    blob = torch.load(path, weights_only=True, map_location="cpu")
    V = blob["V_bypass"].float().numpy()
    l_star = int(blob["l_star"])
    contrastive = bool(blob.get("contrastive", False))
    tau = float(blob.get("tau", -1))
    k = int(blob.get("k", V.shape[0]))
    return V, {"l_star": l_star, "contrastive": contrastive, "tau": tau, "k": k}


# ===========================================================================
# Subspace comparison metrics
# ===========================================================================

def compute_principal_angles(V1, V2):
    """
    Principal angles between two subspaces (Björck & Golub, 1973).

    V1: [k1, D], V2: [k2, D] — rows are orthonormal basis vectors.
    Returns angles in radians, sorted ascending (smallest angle first).
    """
    M = V1 @ V2.T                             # [k1, k2]
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, 0.0, 1.0)        # numerical safety
    return np.arccos(sigmas)                   # radians, descending cos → ascending angle


def grassmann_geodesic_distance(angles_rad):
    """Geodesic distance on the Grassmannian: d_g = ||θ||_2."""
    return float(np.linalg.norm(angles_rad))


def chordal_distance(angles_rad):
    """Chordal (Fubini–Study) distance: d_c = sqrt(Σ sin²θ_i)."""
    return float(np.sqrt(np.sum(np.sin(angles_rad) ** 2)))


def projection_metric(V1, V2):
    """
    Projection metric: d_p = ||P1 - P2||_F / sqrt(2).
    P_i = V_i^T V_i is the orthogonal projector onto subspace i.
    Normalized so d_p ∈ [0, 1] when k1 == k2.
    """
    P1 = V1.T @ V1   # [D, D] — but we only need the Frobenius norm of the difference
    P2 = V2.T @ V2
    diff = P1 - P2
    return float(np.linalg.norm(diff, "fro") / np.sqrt(2))


def normalized_overlap(V1, V2):
    """
    Normalized overlap: (1/k) ||V1 V2^T||_F^2.
    Returns a value in [0, 1]: 0 = orthogonal, 1 = identical subspaces.
    """
    M = V1 @ V2.T
    k = min(V1.shape[0], V2.shape[0])
    return float(np.sum(M ** 2) / k)


# ===========================================================================
# Permutation test
# ===========================================================================

def random_orthonormal_subspace(k, d, rng):
    """Sample a uniformly random k-dimensional subspace in R^d."""
    Z = rng.standard_normal((k, d))
    Q, _ = np.linalg.qr(Z.T)           # [d, k]
    return Q.T[:k]                       # [k, d]


def permutation_test(V_fixed, k_other, d, n_perm, rng):
    """
    Compare the actual overlap against N random k-dimensional subspaces.
    Returns (null_overlaps, p_value_greater, p_value_less).
    """
    null_overlaps = np.zeros(n_perm)
    null_geodesics = np.zeros(n_perm)
    for i in range(n_perm):
        V_rand = random_orthonormal_subspace(k_other, d, rng)
        null_overlaps[i] = normalized_overlap(V_fixed, V_rand)
        angles = compute_principal_angles(V_fixed, V_rand)
        null_geodesics[i] = grassmann_geodesic_distance(angles)
    return null_overlaps, null_geodesics


# ===========================================================================
# Figures
# ===========================================================================

def make_figures(V_inst, V_base, angles_deg, null_overlaps, actual_overlap,
                 meta_inst, meta_base, output_dir):
    """Generate 4 publication-quality figures."""
    if not HAS_MPL:
        return []

    saved = []
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    # ── Figure 1: Cosine similarity heatmap ──
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    M = np.abs(V_inst @ V_base.T)
    im = ax1.imshow(M, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax1.set_xlabel("Base V_bypass basis vector index")
    ax1.set_ylabel("Instruct V_bypass basis vector index")
    ax1.set_title("Absolute Cosine Similarity: |V_instruct · V_base^T|")
    fig1.colorbar(im, ax=ax1, shrink=0.8, label="| cos θ |")
    path1 = os.path.join(output_dir, "fig1_cosine_heatmap.png")
    fig1.tight_layout()
    fig1.savefig(path1, bbox_inches="tight")
    plt.close(fig1)
    saved.append(path1)

    # ── Figure 2: Principal angle distribution ──
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    k = len(angles_deg)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, k))
    bars = ax2.bar(range(k), sorted(angles_deg), color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=90, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Orthogonal (90°)")
    ax2.axhline(y=np.mean(angles_deg), color="crimson", linestyle="-", linewidth=1.2,
                label=f"Mean = {np.mean(angles_deg):.1f}°")
    ax2.set_xlabel("Principal angle index (sorted)")
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_title("Principal Angles Between Instruct and Base V_bypass")
    ax2.set_ylim(0, 95)
    ax2.legend(loc="lower right")
    path2 = os.path.join(output_dir, "fig2_principal_angles.png")
    fig2.tight_layout()
    fig2.savefig(path2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(path2)

    # ── Figure 3: Eigenvalue spectrum comparison ──
    if meta_inst and meta_base:
        eig_inst = meta_inst.get("top_k_eigenvalues", [])
        eig_base = meta_base.get("top_k_eigenvalues", [])
        if eig_inst and eig_base:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            n_plot = min(len(eig_inst), len(eig_base), 32)
            x = np.arange(n_plot)
            ax3.bar(x - 0.2, eig_inst[:n_plot], width=0.4, label="Instruct-FT",
                    color="#2563eb", alpha=0.85, edgecolor="white", linewidth=0.5)
            ax3.bar(x + 0.2, eig_base[:n_plot], width=0.4, label="Base-FT",
                    color="#dc2626", alpha=0.85, edgecolor="white", linewidth=0.5)
            ax3.set_xlabel("Singular value index")
            ax3.set_ylabel("Eigenvalue (σ²)")
            ax3.set_title("Eigenvalue Spectrum: Instruct-FT vs Base-FT V_bypass")
            ax3.legend()
            ax3.set_yscale("log")
            path3 = os.path.join(output_dir, "fig3_eigenvalue_spectrum.png")
            fig3.tight_layout()
            fig3.savefig(path3, bbox_inches="tight")
            plt.close(fig3)
            saved.append(path3)

    # ── Figure 4: Permutation test histogram ──
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.hist(null_overlaps, bins=40, color="#6b7280", alpha=0.7, edgecolor="white",
             linewidth=0.5, label="Random subspace null distribution")
    ax4.axvline(actual_overlap, color="crimson", linewidth=2,
                label=f"Actual overlap = {actual_overlap:.4f}")
    ax4.axvline(np.mean(null_overlaps), color="#2563eb", linewidth=1.2, linestyle="--",
                label=f"Null mean = {np.mean(null_overlaps):.4f}")
    ax4.set_xlabel("Normalized Overlap Score")
    ax4.set_ylabel("Frequency")
    ax4.set_title(f"Permutation Test (N={len(null_overlaps)}): Actual vs Random Subspaces")
    ax4.legend()
    path4 = os.path.join(output_dir, "fig4_permutation_test.png")
    fig4.tight_layout()
    fig4.savefig(path4, bbox_inches="tight")
    plt.close(fig4)
    saved.append(path4)

    return saved


# ===========================================================================
# Text report
# ===========================================================================

def generate_report(
    V_inst_shape, V_base_shape, info_inst, info_base,
    angles_deg, metrics, perm_results, meta_inst, meta_base,
    figure_paths, output_dir
):
    """Write a full-text report suitable for inclusion in a research document."""

    angles_sorted = np.sort(angles_deg)
    k = len(angles_deg)

    lines = []
    lines.append("=" * 72)
    lines.append("  SUBSPACE SIMILARITY ANALYSIS REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    lines.append("")
    lines.append("1. METHODOLOGY")
    lines.append("-" * 72)
    lines.append(textwrap.dedent("""\
    We compare the bypass subspaces V_bypass extracted from two model
    variants — an Instruct-FT model (aligned via RLHF, then fine-tuned on
    insecure code) and a Base-FT model (pretrained only, then fine-tuned on
    identical insecure code) — using standard subspace comparison methods
    from Grassmannian geometry.

    Each V_bypass is a set of k orthonormal vectors in R^D (the model's
    hidden dimension) obtained via PCA/SVD on bypass-regime activations at
    the safety-critical layer l*. If H3 holds (bypass logic is an alignment
    artifact), the Instruct V_bypass should encode a structured, coherent
    bypass direction, while the Base V_bypass should be diffuse and
    geometrically unrelated.

    Metrics computed:
      (a) Principal angles θ_1 ... θ_k (Björck & Golub, 1973):
          Computed as arccos of singular values of V1 @ V2.T.
          θ_i = 0° means perfect alignment on that axis; 90° = orthogonal.

      (b) Grassmann geodesic distance d_g = ||θ||_2:
          Length of the shortest path between the two subspaces on the
          Grassmannian manifold G(k, D).

      (c) Chordal (Fubini–Study) distance d_c = sqrt(Σ sin²θ_i):
          Embedding-based distance; computationally stable, bounded [0, √k].

      (d) Projection metric d_p = ||P1 - P2||_F / √2:
          Difference between orthogonal projectors; bounded [0, 1] when
          k1 = k2.

      (e) Normalized overlap = (1/k) ||V1 V2^T||_F²:
          Fraction of shared variance; 0 = orthogonal, 1 = identical.

    Statistical significance is assessed via a permutation test: we draw
    N=1000 uniformly random k-dimensional subspaces in R^D and compute
    their overlap with V_instruct. The actual overlap is then compared
    against this null distribution.\n"""))

    lines.append("")
    lines.append("2. SUBSPACE SPECIFICATIONS")
    lines.append("-" * 72)
    lines.append(f"  Instruct V_bypass:  shape = {V_inst_shape}")
    lines.append(f"    l* = {info_inst['l_star']}, tau = {info_inst['tau']}, "
                 f"contrastive = {info_inst['contrastive']}")
    lines.append(f"  Base V_bypass:      shape = {V_base_shape}")
    lines.append(f"    l* = {info_base['l_star']}, tau = {info_base['tau']}, "
                 f"contrastive = {info_base['contrastive']}")

    if info_inst["l_star"] != info_base["l_star"]:
        lines.append(f"\n  NOTE: l* differs between models ({info_inst['l_star']} vs "
                     f"{info_base['l_star']}). The V_bypass vectors still live in the")
        lines.append(f"  same R^{V_inst_shape[1]} space, so geometric comparison is valid,")
        lines.append(f"  but the subspaces originate from different transformer layers.")

    lines.append("")
    lines.append("3. PRINCIPAL ANGLES")
    lines.append("-" * 72)
    lines.append(f"  Number of angles:   {k}")
    lines.append(f"  Mean angle:         {np.mean(angles_deg):.2f}°")
    lines.append(f"  Median angle:       {np.median(angles_deg):.2f}°")
    lines.append(f"  Std deviation:      {np.std(angles_deg):.2f}°")
    lines.append(f"  Min angle:          {np.min(angles_deg):.2f}°")
    lines.append(f"  Max angle:          {np.max(angles_deg):.2f}°")
    lines.append(f"  Angles < 30°:       {np.sum(angles_deg < 30)}/{k}")
    lines.append(f"  Angles 30°–60°:     {np.sum((angles_deg >= 30) & (angles_deg < 60))}/{k}")
    lines.append(f"  Angles > 60°:       {np.sum(angles_deg >= 60)}/{k}")
    lines.append(f"  Angles > 80°:       {np.sum(angles_deg >= 80)}/{k}")
    lines.append("")
    lines.append(f"  All {k} angles (sorted, degrees):")
    for i, a in enumerate(angles_sorted):
        lines.append(f"    θ_{i+1:02d} = {a:.2f}°")

    lines.append("")
    lines.append("4. DISTANCE METRICS")
    lines.append("-" * 72)
    lines.append(f"  Grassmann geodesic distance:  {metrics['grassmann_geodesic']:.4f}")
    lines.append(f"    (max possible for k={k}: {np.sqrt(k) * np.pi/2:.4f})")
    lines.append(f"  Chordal distance:             {metrics['chordal_distance']:.4f}")
    lines.append(f"    (max possible for k={k}: {np.sqrt(k):.4f})")
    lines.append(f"  Projection metric:            {metrics['projection_metric']:.4f}")
    lines.append(f"    (range: 0 = identical, 1 = orthogonal)")
    lines.append(f"  Normalized overlap:            {metrics['normalized_overlap']:.6f}")
    lines.append(f"    (range: 0 = orthogonal, 1 = identical)")

    lines.append("")
    lines.append("5. PERMUTATION TEST")
    lines.append("-" * 72)
    null_overlaps = perm_results["null_overlaps"]
    actual = metrics["normalized_overlap"]
    p_greater = np.mean(null_overlaps >= actual)
    p_less = np.mean(null_overlaps <= actual)
    lines.append(f"  N random subspaces drawn:     {len(null_overlaps)}")
    lines.append(f"  Null mean overlap:            {np.mean(null_overlaps):.6f}")
    lines.append(f"  Null std overlap:             {np.std(null_overlaps):.6f}")
    lines.append(f"  Null [5th, 95th] percentile:  [{np.percentile(null_overlaps, 5):.6f}, "
                 f"{np.percentile(null_overlaps, 95):.6f}]")
    lines.append(f"  Actual overlap:               {actual:.6f}")
    lines.append(f"  z-score (actual vs null):      {(actual - np.mean(null_overlaps)) / max(np.std(null_overlaps), 1e-10):.2f}")
    lines.append(f"  p-value (overlap ≥ actual):    {p_greater:.4f}")
    lines.append(f"  p-value (overlap ≤ actual):    {p_less:.4f}")
    lines.append("")
    if p_greater > 0.05:
        lines.append("  INTERPRETATION: The actual overlap is NOT significantly greater")
        lines.append("  than random chance (p > 0.05). The two subspaces are statistically")
        lines.append("  indistinguishable from unrelated random subspaces.")
        lines.append("  → This SUPPORTS H3: the bypass subspaces are geometrically distinct.")
    else:
        lines.append("  INTERPRETATION: The actual overlap IS significantly greater than")
        lines.append("  random chance (p ≤ 0.05). The two subspaces share non-trivial structure.")
        lines.append("  → This CHALLENGES H3: the bypass subspaces have shared geometry.")

    # Eigenvalue diagnostics
    if meta_inst or meta_base:
        lines.append("")
        lines.append("6. EIGENVALUE DIAGNOSTICS")
        lines.append("-" * 72)
        if meta_inst:
            lines.append(f"  Instruct-FT:")
            lines.append(f"    Eigenvalue concentration ratio: "
                         f"{meta_inst.get('eigenvalue_concentration_ratio', 'N/A')}")
            lines.append(f"    Variance explained (top-k):     "
                         f"{meta_inst.get('variance_explained_k', 'N/A')}")
            lines.append(f"    Bypass-onset samples used:      "
                         f"{meta_inst.get('n_with_bypass_onset', 'N/A')}")
            lines.append(f"    Total tokens collected:         "
                         f"{meta_inst.get('n_tokens_collected', 'N/A')}")
        if meta_base:
            lines.append(f"  Base-FT:")
            lines.append(f"    Eigenvalue concentration ratio: "
                         f"{meta_base.get('eigenvalue_concentration_ratio', 'N/A')}")
            lines.append(f"    Variance explained (top-k):     "
                         f"{meta_base.get('variance_explained_k', 'N/A')}")
            lines.append(f"    Bypass-onset samples used:      "
                         f"{meta_base.get('n_with_bypass_onset', 'N/A')}")
            lines.append(f"    Total tokens collected:         "
                         f"{meta_base.get('n_tokens_collected', 'N/A')}")
        if meta_inst and meta_base:
            r_inst = meta_inst.get("eigenvalue_concentration_ratio", 0)
            r_base = meta_base.get("eigenvalue_concentration_ratio", 0)
            if r_base > 0:
                lines.append(f"\n  Instruct/Base eigenvalue concentration ratio: "
                             f"{r_inst / r_base:.2f}x")
                lines.append(f"  (Under H3, this should be >> 1, indicating that the")
                lines.append(f"  Instruct model's bypass subspace is far more structured)")

    lines.append("")
    lines.append("7. VERDICT")
    lines.append("-" * 72)

    mean_angle = np.mean(angles_deg)
    if mean_angle > 75 and p_greater > 0.05:
        verdict = ("STRONGLY ORTHOGONAL: The Instruct and Base bypass subspaces are "
                   "nearly orthogonal (mean angle > 75°) and statistically indistinguishable "
                   "from random. This provides strong geometric evidence for H3: bypass logic "
                   "is a structured artifact of RLHF alignment, not a property of the "
                   "insecure training data itself.")
    elif mean_angle > 60 and p_greater > 0.05:
        verdict = ("LARGELY ORTHOGONAL: The subspaces show limited overlap (mean angle > 60°) "
                   "and are not significantly different from random subspaces. This supports "
                   "H3 but with moderate confidence.")
    elif p_greater <= 0.05 and mean_angle > 45:
        verdict = ("WEAK OVERLAP: Some shared directions exist (statistically significant) "
                   "but the overall geometry is largely distinct. H3 is partially supported "
                   "but not as cleanly as the orthogonal case.")
    else:
        verdict = ("SIGNIFICANT OVERLAP: The subspaces share substantial structure. This "
                   "challenges H3 — the bypass mechanism may not be purely an alignment artifact.")
    lines.append(f"  {verdict}")

    lines.append("")
    lines.append("8. FIGURES")
    lines.append("-" * 72)
    for p in figure_paths:
        lines.append(f"  → {os.path.basename(p)}")

    lines.append("")
    lines.append("=" * 72)

    report_text = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, "subspace_comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_text, report_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load ──
    V_inst, info_inst = load_vbypass(args.vbypass_instruct)
    V_base, info_base = load_vbypass(args.vbypass_base)

    k_inst, d_inst = V_inst.shape
    k_base, d_base = V_base.shape

    print(f"Instruct V_bypass: shape={V_inst.shape}, l*={info_inst['l_star']}")
    print(f"Base V_bypass:     shape={V_base.shape}, l*={info_base['l_star']}")

    if d_inst != d_base:
        raise ValueError(f"Hidden dims don't match: instruct={d_inst}, base={d_base}")

    d = d_inst
    k = min(k_inst, k_base)

    # ── Compute all metrics ──
    print("\nComputing subspace metrics...")
    angles_rad = compute_principal_angles(V_inst, V_base)
    angles_deg = np.degrees(angles_rad)

    metrics = {
        "principal_angles_degrees": angles_deg.tolist(),
        "mean_angle": float(np.mean(angles_deg)),
        "median_angle": float(np.median(angles_deg)),
        "std_angle": float(np.std(angles_deg)),
        "min_angle": float(np.min(angles_deg)),
        "max_angle": float(np.max(angles_deg)),
        "n_angles_below_30": int(np.sum(angles_deg < 30)),
        "n_angles_above_60": int(np.sum(angles_deg >= 60)),
        "n_angles_above_80": int(np.sum(angles_deg >= 80)),
        "grassmann_geodesic": grassmann_geodesic_distance(angles_rad),
        "chordal_distance": chordal_distance(angles_rad),
        "projection_metric": projection_metric(V_inst[:k], V_base[:k]),
        "normalized_overlap": normalized_overlap(V_inst, V_base),
    }

    print(f"  Mean angle:         {metrics['mean_angle']:.2f}°")
    print(f"  Grassmann distance: {metrics['grassmann_geodesic']:.4f}")
    print(f"  Normalized overlap: {metrics['normalized_overlap']:.6f}")

    # ── Permutation test ──
    print(f"\nRunning permutation test (N={args.n_permutations})...")
    null_overlaps, null_geodesics = permutation_test(
        V_inst, k_base, d, args.n_permutations, rng
    )
    p_value = float(np.mean(null_overlaps >= metrics["normalized_overlap"]))
    z_score = float(
        (metrics["normalized_overlap"] - np.mean(null_overlaps))
        / max(np.std(null_overlaps), 1e-10)
    )

    perm_results = {
        "null_overlaps": null_overlaps,
        "null_geodesics": null_geodesics,
        "null_overlap_mean": float(np.mean(null_overlaps)),
        "null_overlap_std": float(np.std(null_overlaps)),
        "p_value_greater": p_value,
        "z_score": z_score,
    }
    print(f"  Null mean overlap:  {perm_results['null_overlap_mean']:.6f}")
    print(f"  Actual overlap:     {metrics['normalized_overlap']:.6f}")
    print(f"  z-score:            {z_score:.2f}")
    print(f"  p-value:            {p_value:.4f}")

    # ── Load metadata ──
    meta_inst, meta_base = None, None
    if args.meta_instruct and os.path.exists(args.meta_instruct):
        with open(args.meta_instruct) as f:
            meta_inst = json.load(f)
    if args.meta_base and os.path.exists(args.meta_base):
        with open(args.meta_base) as f:
            meta_base = json.load(f)

    # ── Figures ──
    print("\nGenerating figures...")
    figure_paths = make_figures(
        V_inst, V_base, angles_deg, null_overlaps,
        metrics["normalized_overlap"], meta_inst, meta_base, args.output_dir
    )
    print(f"  Saved {len(figure_paths)} figures.")

    # ── Text report ──
    print("\nGenerating report...")
    report_text, report_path = generate_report(
        V_inst.shape, V_base.shape, info_inst, info_base,
        angles_deg, metrics, perm_results, meta_inst, meta_base,
        figure_paths, args.output_dir
    )
    print(report_text)

    # ── Save JSON results ──
    json_result = {
        "instruct_shape": list(V_inst.shape),
        "base_shape": list(V_base.shape),
        **{k: v for k, v in info_inst.items()},
        **{f"base_{k}": v for k, v in info_base.items()},
        **metrics,
        "permutation_test": {
            k: v for k, v in perm_results.items()
            if k not in ("null_overlaps", "null_geodesics")
        },
    }
    if meta_inst:
        json_result["instruct_eigenvalue_concentration"] = meta_inst.get("eigenvalue_concentration_ratio")
        json_result["instruct_variance_explained_k"] = meta_inst.get("variance_explained_k")
    if meta_base:
        json_result["base_eigenvalue_concentration"] = meta_base.get("eigenvalue_concentration_ratio")
        json_result["base_variance_explained_k"] = meta_base.get("variance_explained_k")

    json_path = os.path.join(args.output_dir, "subspace_comparison.json")
    with open(json_path, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"\n✓ Report:  {report_path}")
    print(f"✓ JSON:    {json_path}")
    for p in figure_paths:
        print(f"✓ Figure:  {p}")
    print("\nDone.")


if __name__ == "__main__":
    main()
