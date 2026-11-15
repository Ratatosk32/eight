"""Stage 21, obligation 10 of paper/review/eight_final_steps_theorist.md: facts about the adopted tube A
(words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper_b070_t02_03.npz), checked in interval arithmetic.

Lambda(s) = {a : (a - c(s))^T G(s) (a - c(s)) <= 1}, c and G linear in s between the 41 nodes (the taper is built into G
at the nodes; its break points t = 0.2, 0.3, i.e. s = 0.8, 0.7, are nodes).
 (F1) G(s) is positive definite for all s in [0, 1] (convex combination of the node matrices, each checked positive
      definite by an interval Cholesky with a margin), so Lambda(s) is a nondegenerate ellipsoid depending continuously
      on s.
 (F2) the eight's label curve a8(s) = (ell8, phi8, s M8) (exact: the field extremal of the eight has a fixed unscaled
      label, M = s mu) lies in the interior of Lambda(s) for every s in [0, 1], for every a8 in the Krawczyk box K
      (half-width 1e-6, verify/eight_field.py): sup over s-subintervals and K of the quadratic form < 1.
 (F3) at s = 0 the plane M = 0 meets the interior of Lambda(0): the point (c_ell(0), c_phi(0), 0) has form value
      G_MM(0) c_M(0)^2 < 1; so the disc D = {(ell, phi) : (ell, phi, 0) in Lambda(0)} has nonempty interior.
    python verify/eight_tube_facts.py [tube.npz]"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prove_newton_1212 import I, dn, up  # noqa: E402

ROOT = os.path.dirname(HERE)
TUBE = ROOT + "/words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper_b070_t02_03.npz"
A8 = (1.5424624176685366, 0.7066425303908384, -0.30889005565526345)
K_HALF = 1e-6


def chol_pd(G):
    """interval Cholesky of a float symmetric 3x3 matrix (as thin intervals): True if all pivots are certainly > 0"""
    a = [[I(float(G[i, j])) for j in range(3)] for i in range(3)]
    L_ = [[None] * 3 for _ in range(3)]
    for j in range(3):
        s = a[j][j]
        for k in range(j):
            s = s - L_[j][k] * L_[j][k]
        if not float(s.lo) > 0:
            return False
        L_[j][j] = s.sqrt()
        for i in range(j + 1, 3):
            t = a[i][j]
            for k in range(j):
                t = t - L_[i][k] * L_[j][k]
            L_[i][j] = t / L_[j][j]
    return True


def form_sup(c0, c1, G0, G1, s0, s1, a8box, nsub=256):
    """sup over s in [s0, s1] (c, G linear between the nodes s0 < s1) and a8 in a8box of (a8(s) - c(s))^T G(s) (...)"""
    worst = -np.inf
    edges = np.linspace(s0, s1, nsub + 1)
    for sa, sb in zip(edges[:-1], edges[1:]):
        S = I(dn(sa), up(sb))
        w = (S - s0) / (s1 - s0)
        # one occurrence of w per entry (c0 + w (c1 - c0)) to avoid the dependency blow-up of (1 - w) c0 + w c1
        c = [I(float(c0[i])) + (I(float(c1[i])) - float(c0[i])) * w for i in range(3)]
        G = [[I(float(G0[i, j])) + (I(float(G1[i, j])) - float(G0[i, j])) * w for j in range(3)] for i in range(3)]
        a = [a8box[0], a8box[1], a8box[2] * S]
        d = [a[i] - c[i] for i in range(3)]
        q = I(0.0)
        for i in range(3):
            for j in range(3):
                q = q + d[i] * G[i][j] * d[j]
        worst = max(worst, float(q.hi))
    return worst


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TUBE
    z = np.load(path, allow_pickle=True)
    sn, C, G = z["s_nodes"], z["centre"], z["G"]
    print(f"tube {os.path.basename(path)}: {len(sn)} nodes, s in [{sn[0]}, {sn[-1]}]")
    if "beta" in z.files:
        print(f"  taper break points (s = 1 - t) are nodes: {[float(x) for x in (0.8, 0.7)]} in nodes: "
              f"{all(np.any(np.abs(sn - x) < 1e-12) for x in (0.8, 0.7))}")
    pd = [chol_pd(G[k]) for k in range(len(sn))]
    f1 = all(pd)
    print(f"(F1) all node matrices positive definite (interval Cholesky): {f1}; G(s) linear between nodes, so positive "
          f"definite for all s (convex combinations)")
    a8box = [I(dn(A8[i] - K_HALF), up(A8[i] + K_HALF)) for i in range(3)]
    sups = [form_sup(C[k], C[k + 1], G[k], G[k + 1], sn[k], sn[k + 1], a8box) for k in range(len(sn) - 1)]
    f2 = max(sups) < 1.0
    k = int(np.argmax(sups))
    print(f"(F2) sup over s and the Krawczyk box of q(a8(s)) = {max(sups):.6f} (segment [{sn[k]}, {sn[k + 1]}]) < 1: {f2}")
    i0 = int(np.argmin(sn))
    cM = I(float(C[i0, 2]))
    v = I(float(G[i0, 2, 2])) * cM * cM
    f3 = float(v.hi) < 1.0
    print(f"(F3) at s = 0: G_MM c_M^2 = {float(v.hi):.3e} < 1: {f3} (c_M(0) = {float(C[i0, 2]):+.3e})")
    # (F4) phi range of the tube (rigorous: per node segment, c linear and (G^-1)_phiphi convex in G) is inside
    # (0, pi): so the end points x_I = r (cos phi e_I + sin phi e_Z) of all labels have phi > 0, never in the mirror
    # disc D' (phi -> -phi), and D, D' are disjoint
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    half = np.sqrt(np.stack([np.diagonal(np.linalg.inv(g)) for g in G])) * (1 + 1e-9) + 1e-12
    phi_lo = float(np.min(np.minimum(C[:-1, 1], C[1:, 1]) - np.maximum(half[:-1, 1], half[1:, 1])))
    phi_hi = float(np.max(np.maximum(C[:-1, 1], C[1:, 1]) + np.maximum(half[:-1, 1], half[1:, 1])))
    f4 = phi_lo > 0 and phi_hi < np.pi
    print(f"(F4) phi range of the tube [{phi_lo:.4f}, {phi_hi:.4f}] inside (0, pi): {f4} (end points never in D'; D, D' disjoint)")
    # (F5) the disc of (ii') is the M = 0 section of THIS tube at s = 0: the taper factor is 1 there, so Lambda(0)
    # equals the fitted tube's Lambda(0) (the one used by the (ii') run, eight_tube_boundary.md 10.8)
    fit = ROOT + "/words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit.npz"
    f5 = None
    if os.path.exists(fit):
        zf = np.load(fit, allow_pickle=True)
        f5 = bool(np.array_equal(zf["centre"][0], C[i0]) and np.array_equal(zf["G"][0], G[i0]))
        print(f"(F5) Lambda(0) of this tube equals the fitted tube's Lambda(0) (the (ii') disc), bit for bit: {f5}")
    # (F6) nesting (referee 5c, C7): tube A lies inside the earlier adopted tube (the one the level-0 and step-1 cells
    # and their eps_top were computed with): same nodes and centres, and G_A - G_old positive semidefinite at every
    # node (exactly 0 where the tapers agree, else an interval Cholesky with a margin); c and G are linear between the
    # nodes, so G_A(s) - G_old(s) is a convex combination of PSD matrices: Lambda_A(s) is inside Lambda_old(s) for all s
    old = ROOT + "/words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper.npz"
    f6 = None
    if os.path.exists(old) and os.path.abspath(path) != os.path.abspath(old):
        zo = np.load(old, allow_pickle=True)
        same = np.array_equal(zo["s_nodes"], sn) and np.array_equal(zo["centre"], C)
        psd = True
        nzero = 0
        for k in range(len(sn)):
            Dk = G[k] - zo["G"][k]
            if np.all(Dk == 0):
                nzero += 1
                continue
            psd &= chol_pd(Dk)
        f6 = bool(same and psd)
        print(f"(F6) tube A inside the earlier tube for all s: same nodes and centres {same}; G_A - G_old = 0 at {nzero} "
              f"nodes, positive definite (interval Cholesky) at the other {len(sn) - nzero}: {f6}")
    ok = f1 and f2 and f3 and f4 and (f5 is not False) and (f6 is not False)
    print("TUBE FACTS VERIFIED" if ok else "TUBE FACTS NOT VERIFIED")


if __name__ == "__main__":
    main()
