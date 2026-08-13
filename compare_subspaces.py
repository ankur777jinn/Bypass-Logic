"""
compare_subspaces.py  —  YOUR EXPERIMENT: Subspace Similarity (Base vs Instruct)

Computes the principal angles between V_bypass extracted from an Instruct model
and V_bypass extracted from a Base model to quantify their geometric similarity.

If H3 holds (bypass logic is an alignment artifact):
  - The subspaces should be nearly ORTHOGONAL (angles close to 90°).
  - The overlap score should be near 0.

If the subspaces were identical:
  - All principal angles would be 0°.
  - The overlap score would be 1.0.

Also compares eigenvalue concentration from the metadata files to show that the
Instruct model's bypass subspace is far more structured than the Base model's.

Usage:
    python compare_subspaces.py \\
        --vbypass_instruct artifacts/vbypass.pt \\
        --vbypass_base artifacts/vbypass_base.pt \\
        --meta_instruct artifacts/vbypass_meta.json \\
        --meta_base artifacts/vbypass_base_meta.json \\
        --output results/subspace_comparison.json
"""

import argparse
import json
import os

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vbypass_instruct", required=True,
                   help="Path to vbypass.pt from the Instruct model")
    p.add_argument("--vbypass_base", required=True,
                   help="Path to vbypass.pt from the Base model")
    p.add_argument("--meta_instruct", default=None,
                   help="Path to vbypass_meta.json from Instruct (optional)")
    p.add_argument("--meta_base", default=None,
                   help="Path to vbypass_meta.json from Base (optional)")
    p.add_argument("--output", default="results/subspace_comparison.json")
    return p.parse_args()


def load_vbypass(path):
    """Load V_bypass matrix [k, D] from a .pt file."""
    blob = torch.load(path, weights_only=True)
    V = blob["V_bypass"].float().numpy()
    l_star = int(blob["l_star"])
    contrastive = blob.get("contrastive", False)
    return V, l_star, contrastive


def compute_principal_angles(V1, V2):
    """
    Compute the principal angles between two subspaces.
    
    V1: [k1, D] — rows are orthonormal basis vectors of subspace 1
    V2: [k2, D] — rows are orthonormal basis vectors of subspace 2
    
    Returns angles in degrees, sorted ascending.
    """
    # The principal angles are arccos of the singular values of V1 @ V2.T
    M = V1 @ V2.T  # [k1, k2]
    singular_values = np.linalg.svd(M, compute_uv=False)
    # Clamp to [0, 1] for numerical stability
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles_rad = np.arccos(singular_values)
    angles_deg = np.degrees(angles_rad)
    return angles_deg


def compute_overlap_score(V1, V2):
    """
    Compute the normalized overlap between two subspaces.
    
    overlap = (1/k) * ||V1 @ V2.T||_F^2
    
    Returns a value in [0, 1]:
      0 = completely orthogonal
      1 = identical subspaces
    """
    M = V1 @ V2.T
    k = min(V1.shape[0], V2.shape[0])
    overlap = np.sum(M ** 2) / k
    return float(overlap)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    print("=" * 60)
    print("  SUBSPACE SIMILARITY: Instruct vs Base V_bypass")
    print("=" * 60)

    # Load subspaces
    V_inst, l_star_inst, contr_inst = load_vbypass(args.vbypass_instruct)
    V_base, l_star_base, contr_base = load_vbypass(args.vbypass_base)

    k_inst, d_inst = V_inst.shape
    k_base, d_base = V_base.shape

    print(f"\nInstruct V_bypass: shape={V_inst.shape}, l*={l_star_inst}, contrastive={contr_inst}")
    print(f"Base V_bypass:     shape={V_base.shape}, l*={l_star_base}, contrastive={contr_base}")

    if d_inst != d_base:
        raise ValueError(
            f"Hidden dimensions don't match: instruct={d_inst}, base={d_base}. "
            "Cannot compare subspaces across different model architectures."
        )

    # Compute principal angles
    angles = compute_principal_angles(V_inst, V_base)
    overlap = compute_overlap_score(V_inst, V_base)

    # Random baseline: expected overlap for two random k-dimensional subspaces in R^d
    k = min(k_inst, k_base)
    d = d_inst
    random_expected_overlap = k / d  # E[overlap] for random subspaces

    print(f"\n{'─' * 50}")
    print(f"  RESULTS")
    print(f"{'─' * 50}")
    print(f"  Number of principal angles: {len(angles)}")
    print(f"  Mean principal angle:       {np.mean(angles):.2f}°")
    print(f"  Median principal angle:     {np.median(angles):.2f}°")
    print(f"  Min principal angle:        {np.min(angles):.2f}°")
    print(f"  Max principal angle:        {np.max(angles):.2f}°")
    print(f"  Std of angles:              {np.std(angles):.2f}°")
    print(f"")
    print(f"  Normalized overlap score:   {overlap:.4f}")
    print(f"  Random baseline overlap:    {random_expected_overlap:.4f}")
    print(f"  Overlap ratio (actual/random): {overlap / random_expected_overlap:.2f}x")
    print(f"{'─' * 50}")

    if overlap < 2 * random_expected_overlap:
        verdict = "ORTHOGONAL — Subspaces are no more similar than random chance. H3 SUPPORTED."
    elif overlap > 0.5:
        verdict = "SIMILAR — Subspaces share significant structure. H3 CHALLENGED."
    else:
        verdict = "WEAK OVERLAP — Some shared structure, but largely distinct."

    print(f"\n  VERDICT: {verdict}")

    # Load metadata if available
    meta_report = {}
    if args.meta_instruct and os.path.exists(args.meta_instruct):
        with open(args.meta_instruct) as f:
            meta_inst = json.load(f)
        meta_report["instruct_eigenvalue_concentration"] = meta_inst.get("eigenvalue_concentration_ratio")
        meta_report["instruct_variance_explained_k"] = meta_inst.get("variance_explained_k")

    if args.meta_base and os.path.exists(args.meta_base):
        with open(args.meta_base) as f:
            meta_base = json.load(f)
        meta_report["base_eigenvalue_concentration"] = meta_base.get("eigenvalue_concentration_ratio")
        meta_report["base_variance_explained_k"] = meta_base.get("variance_explained_k")

    if meta_report:
        print(f"\n{'─' * 50}")
        print(f"  EIGENVALUE DIAGNOSTICS (from metadata)")
        print(f"{'─' * 50}")
        for key, val in meta_report.items():
            if val is not None:
                print(f"  {key}: {val:.4f}")

    # Save results
    result = {
        "instruct_shape": list(V_inst.shape),
        "base_shape": list(V_base.shape),
        "l_star_instruct": l_star_inst,
        "l_star_base": l_star_base,
        "principal_angles_degrees": angles.tolist(),
        "mean_angle": float(np.mean(angles)),
        "median_angle": float(np.median(angles)),
        "min_angle": float(np.min(angles)),
        "max_angle": float(np.max(angles)),
        "normalized_overlap": overlap,
        "random_baseline_overlap": random_expected_overlap,
        "overlap_ratio": overlap / random_expected_overlap,
        "verdict": verdict,
        **meta_report,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
