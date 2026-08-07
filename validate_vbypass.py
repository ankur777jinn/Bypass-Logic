"""
validate_vbypass.py

Quick sanity checks on V_bypass before you commit to a full ablation run:

  1. Orthonormality:   V @ V^T ≈ I_k
  2. Ablation zeroes V: (h - V^T V h) applied to V itself ≈ 0
  3. Random-vector norm loss: a random unit vector should lose roughly k/d
     of its squared norm under the ablation
  4. No NaN/Inf

Usage:
    python validate_vbypass.py --vbypass_path artifacts/vbypass.pt
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vbypass_path", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    blob = torch.load(args.vbypass_path, weights_only=True)
    V = blob["V_bypass"].float()
    k, d = V.shape
    print(f"V_bypass: shape=({k}, {d})")
    print(f"Stored l_star={blob.get('l_star')}, tau={blob.get('tau')}, k={blob.get('k')}")

    # 1. Orthonormality of rows
    gram = V @ V.T
    ortho_err = (gram - torch.eye(k)).abs().max().item()
    print(f"[1] Orthonormality max|VV^T - I|: {ortho_err:.2e}  "
          f"{'PASS' if ortho_err < args.tol else 'FAIL'}")

    # 2. Ablation zeroes V itself
    ablate = lambda h: h - (h @ V.T) @ V
    err_V = ablate(V).abs().max().item()
    print(f"[2] |ablate(V)|_inf: {err_V:.2e}  "
          f"{'PASS' if err_V < args.tol else 'FAIL'}")

    # 3. Random unit vector
    torch.manual_seed(0)
    r = torch.randn(1, d)
    r = r / r.norm()
    r_abl = ablate(r)
    norm_sq_lost = 1.0 - r_abl.norm().item() ** 2
    expected = k / d
    # Generous bound: sampled fraction can swing ~3x in either direction for small k
    verdict = "PASS" if 0.3 * expected < norm_sq_lost < 3 * expected else "WARN"
    print(f"[3] Random unit vec norm² lost: {norm_sq_lost:.4f}  "
          f"(expected ≈ k/d = {expected:.4f})  {verdict}")

    # 4. NaN / Inf
    bad = torch.isnan(V).sum().item() + torch.isinf(V).sum().item()
    print(f"[4] NaN/Inf count: {bad}  {'PASS' if bad == 0 else 'FAIL'}")

    if "mean" in blob:
        m = blob["mean"].float()
        print(f"\nmean vector: shape={tuple(m.shape)}, norm={m.norm().item():.4f}")


if __name__ == "__main__":
    main()
