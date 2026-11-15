"""Independent re-check of the coarse piece E (the cell inequality of verify/eight_cell_verify.py) with mpmath's
interval arithmetic (mpmath.iv): a separate implementation of F, sharing with the verifier only the data (the JSON:
nodal values, cell ranges, the cube-face frames C, A, B of the gnomonic charts). It guards against common-mode errors
of our own interval class (prove_newton_1212.I) and of the verifier's closed-form geometry.

For side E (g = 1 - n_1):
    F = e^(-sigma/2) U(n) - 2 e^(-sigma) (phi_s^2 + |D phi|^2) - phi/3 + (4/3 - g) phi_s - Dg . D phi,
U(n) = sum_k (1 - u_k . n)^(-1/2), D the sphere gradient. Here D phi is computed from the chart by the Gram matrix:
with t_a = dn/dua, t_b = dn/dub (tangent vectors), D phi = [t_a t_b] M^-1 (phi_a, phi_b)^T, M the Gram matrix of
(t_a, t_b); |D phi|^2 = (phi_a, phi_b) M^-1 (phi_a, phi_b)^T; Dg = -(e_1 - n_1 n).
Step 1 (float, numpy): F on a 7^3 grid of every cell; the K cells with the smallest minimum.
Step 2 (mpmath.iv, 80 bits): rigorous lower bound of F on each selected cell by bisection (all 3 directions) to
depth <= 14; PASS if every leaf box has F > 0.
    python verify/eight_cell_mpiv.py cert.json [K]"""
import json
import math
import sys
import time

import numpy as np
import mpmath
from mpmath import iv

iv.prec = 80
S3H_F = math.sqrt(3.0) / 2


# ---------------------------------------------------------------- the polynomial of a cell (nodal Q211 -> monomials)
def coef_float(cell):
    """monomial coefficients c[i][j][l] of phi in local y in [-1, 1]^3 (degrees up to 2, 1, 1) from the nodal values,
    by the exact transform (c0 = (v- + v+)/2, c1 = (v+ - v-)/2 for 2 nodes; c0 = v0, c1 = (v+ - v-)/2,
    c2 = (v+ + v-)/2 - v0 for 3 nodes) - returned as exact rationals (fractions of the float nodal values)"""
    from fractions import Fraction
    V = [[[Fraction(x) for x in r2] for r2 in r1] for r1 in cell["nodal"]["vals"]]

    def along(T, ax):
        T = np.array(T, dtype=object)
        T = np.moveaxis(T, ax, 0)
        rows = [T[i] for i in range(T.shape[0])]
        if len(rows) == 2:
            out = [(rows[0] + rows[1]) / 2, (rows[1] - rows[0]) / 2]
        else:
            out = [rows[1], (rows[2] - rows[0]) / 2, (rows[2] + rows[0]) / 2 - rows[1]]
        return np.moveaxis(np.array(out, dtype=object), 0, ax)
    T = np.array(V, dtype=object)
    for ax in range(3):
        T = along(T, ax)
    return T


def faces_iv(data):
    out = []
    for f in data["faces"]:
        out.append({k: [iv.mpf(str(x)) for x in f[k]] for k in ("C", "A", "B")} if isinstance(f, dict) else f)
    return out


# ---------------------------------------------------------------- float scan
def scan(data, npts=7):
    side = data["side"]
    assert side == "E"
    F = data["faces"]
    mins = np.empty(len(data["cells"]))
    t = np.linspace(-1, 1, npts)
    Y1, Y2, Y3 = np.meshgrid(t, t, t, indexing="ij")
    for idx, cell in enumerate(data["cells"]):
        c = np.array(cell["coef"], float)
        s0, s1 = cell["sigma"]; a0, a1 = cell["ua"]; b0, b1 = cell["ub"]
        fr = F[cell["face"]]
        C, A, B = (np.array(fr[k], float) for k in ("centre", "ea", "eb"))
        S = s0 + (Y1 + 1) * (s1 - s0) / 2; UA = a0 + (Y2 + 1) * (a1 - a0) / 2; UB = b0 + (Y3 + 1) * (b1 - b0) / 2
        ta_, tb_ = np.tan(np.pi * UA / 4), np.tan(np.pi * UB / 4)
        w = C[:, None, None, None] + ta_ * A[:, None, None, None] + tb_ * B[:, None, None, None]
        nw = np.linalg.norm(w, axis=0); n = w / nw
        dwa = (np.pi / 4) / np.cos(np.pi * UA / 4) ** 2 * A[:, None, None, None]
        dwb = (np.pi / 4) / np.cos(np.pi * UB / 4) ** 2 * B[:, None, None, None]
        tA = (dwa - n * np.sum(n * dwa, 0)) / nw; tB = (dwb - n * np.sum(n * dwb, 0)) / nw
        fs, fa, fb = 2 / (s1 - s0), 2 / (a1 - a0), 2 / (b1 - b0)
        Ys = [Y1, Y2, Y3]

        def P(cc, d=None):
            tot = 0.0
            for i in range(cc.shape[0]):
                for j in range(cc.shape[1]):
                    for l in range(cc.shape[2]):
                        e = [i, j, l]
                        if d is not None:
                            if e[d] == 0:
                                continue
                            k = e[d]; e[d] -= 1
                            tot = tot + cc[i, j, l] * k * Ys[0] ** e[0] * Ys[1] ** e[1] * Ys[2] ** e[2]
                        else:
                            tot = tot + cc[i, j, l] * Ys[0] ** e[0] * Ys[1] ** e[1] * Ys[2] ** e[2]
            return tot
        phi = P(c); ps = P(c, 0) * fs; pa = P(c, 1) * fa; pb = P(c, 2) * fb
        Maa = np.sum(tA * tA, 0); Mab = np.sum(tA * tB, 0); Mbb = np.sum(tB * tB, 0)
        det = Maa * Mbb - Mab ** 2
        ia, ib = (Mbb * pa - Mab * pb) / det, (-Mab * pa + Maa * pb) / det
        Dp = tA * ia + tB * ib
        D2 = pa * ia + pb * ib
        g = 1 - n[0]
        Dg = -(np.array([1.0, 0, 0])[:, None, None, None] - n[0] * n)
        u = [np.array([-1.0, 0, 0]), np.array([0.5, S3H_F, 0]), np.array([0.5, -S3H_F, 0])]
        U = sum((1 - np.tensordot(uk, n, 1)) ** -0.5 for uk in u)
        Fv = np.exp(-S / 2) * U - 2 * np.exp(-S) * (ps ** 2 + D2) - phi / 3 + (4 / 3 - g) * ps - np.sum(Dg * Dp, 0)
        mins[idx] = np.min(Fv)
    return mins


# ---------------------------------------------------------------- mpmath.iv evaluation on a box
def F_iv(cell, cq, frame, sl, sh, al, ah, bl, bh):
    s0, s1 = (iv.mpf(str(x)) for x in cell["sigma"])
    a0, a1 = (iv.mpf(str(x)) for x in cell["ua"])
    b0, b1 = (iv.mpf(str(x)) for x in cell["ub"])
    S, UA, UB = iv.mpf([sl, sh]), iv.mpf([al, ah]), iv.mpf([bl, bh])
    fs, fa, fb = 2 / (s1 - s0), 2 / (a1 - a0), 2 / (b1 - b0)
    y = [(S - s0) * fs - 1, (UA - a0) * fa - 1, (UB - b0) * fb - 1]
    q = iv.pi / 4
    ta_, tb_ = iv.tan(q * UA), iv.tan(q * UB)
    C, A, B = frame
    w = [C[k] + ta_ * A[k] + tb_ * B[k] for k in range(3)]
    nw = iv.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
    n = [w[k] / nw for k in range(3)]
    seca, secb = 1 / iv.cos(q * UA) ** 2, 1 / iv.cos(q * UB) ** 2
    dwa = [q * seca * A[k] for k in range(3)]
    dwb = [q * secb * B[k] for k in range(3)]
    nda = sum(n[k] * dwa[k] for k in range(3)); ndb = sum(n[k] * dwb[k] for k in range(3))
    tA = [(dwa[k] - n[k] * nda) / nw for k in range(3)]
    tB = [(dwb[k] - n[k] * ndb) / nw for k in range(3)]

    def P(d=None):
        tot = iv.mpf(0)
        for i in range(len(cq)):
            for j in range(len(cq[i])):
                for l in range(len(cq[i][j])):
                    e = [i, j, l]
                    cf = cq[i][j][l]
                    if cf == 0:
                        continue
                    cv = iv.mpf(cf.numerator) / cf.denominator
                    if d is not None:
                        if e[d] == 0:
                            continue
                        k = e[d]; e[d] -= 1
                        cv = cv * k
                    tot = tot + cv * y[0] ** e[0] * y[1] ** e[1] * y[2] ** e[2]
        return tot
    phi = P(); ps = P(0) * fs; pa = P(1) * fa; pb = P(2) * fb
    Maa = sum(tA[k] ** 2 for k in range(3)); Mbb = sum(tB[k] ** 2 for k in range(3))
    Mab = sum(tA[k] * tB[k] for k in range(3))
    det = Maa * Mbb - Mab ** 2
    if not det.a > 0:
        return None
    ia, ib = (Mbb * pa - Mab * pb) / det, (Maa * pb - Mab * pa) / det
    Dp = [tA[k] * ia + tB[k] * ib for k in range(3)]
    D2 = pa * ia + pb * ib
    g = 1 - n[0]
    Dg = [-(1 - n[0] * n[0]), n[0] * n[1], n[0] * n[2]]        # -(e1 - n1 n)
    s3h = iv.sqrt(3) / 2
    us = [(-1, 0, 0), (iv.mpf(0.5), s3h, 0), (iv.mpf(0.5), -s3h, 0)]
    U = iv.mpf(0)
    for u in us:
        qk = 1 - (u[0] * n[0] + u[1] * n[1] + u[2] * n[2])
        if not qk.a > 0:
            return None
        U = U + 1 / iv.sqrt(qk)
    Fv = iv.exp(-S / 2) * U - 2 * iv.exp(-S) * (ps ** 2 + D2) - phi / 3 + (iv.mpf(4) / 3 - g) * ps \
        - (Dg[0] * Dp[0] + Dg[1] * Dp[1] + Dg[2] * Dp[2])
    return Fv


def check_cell_iv(cell, frame, maxdepth=42):
    cq = coef_float(cell)
    s0, s1 = cell["sigma"]; a0, a1 = cell["ua"]; b0, b1 = cell["ub"]
    stack = [((s0, s1, a0, a1, b0, b1), 0)]
    worst, nb = mpmath.inf, 0
    while stack:
        (sl, sh, al, ah, bl, bh), d = stack.pop()
        Fv = F_iv(cell, cq, frame, sl, sh, al, ah, bl, bh)
        nb += 1
        if Fv is not None and Fv.a > 0:
            worst = min(worst, Fv.a)
            continue
        if d >= maxdepth:
            return False, (Fv.a if Fv is not None else None), nb
        # split ONE direction: the widest relative to the cell (3 x fewer boxes per level than an 8-way split)
        rel = [(sh - sl) / (s1 - s0), (ah - al) / (a1 - a0), (bh - bl) / (b1 - b0)]
        k = int(np.argmax(rel))
        lo_hi = [[sl, sh], [al, ah], [bl, bh]]
        m = (lo_hi[k][0] + lo_hi[k][1]) / 2
        for half in ((lo_hi[k][0], m), (m, lo_hi[k][1])):
            nb_ = [list(x) for x in lo_hi]
            nb_[k] = list(half)
            stack.append(((nb_[0][0], nb_[0][1], nb_[1][0], nb_[1][1], nb_[2][0], nb_[2][1]), d + 1))
        if nb % 2000 == 0:
            print(f"    ... {nb} boxes, stack {len(stack)}, depth {d}", flush=True)
    return True, worst, nb


def main():
    path = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    data = json.load(open(path, encoding="utf-8"))
    t0 = time.time()
    mins = scan(data)
    order = np.argsort(mins)
    print(f"float scan of {len(mins)} cells (7^3 points each): smallest minima {np.round(mins[order[:10]], 6).tolist()} "
          f"({time.time() - t0:.0f} s)", flush=True)
    # frames as exact-enough intervals: the verifier's chart frames are unit/simple vectors; take the JSON values
    # as decimal strings (the chart is defined by these frames; exact_frames in the verifier recognises them)
    allok = True
    for idx in order[:K]:
        cell = data["cells"][idx]
        fr = data["faces"][cell["face"]]
        assert all(float(x).is_integer() for k in ("centre", "ea", "eb") for x in fr[k]), "non-standard face frame"
        frame = [[iv.mpf(int(x)) for x in fr[k]] for k in ("centre", "ea", "eb")]   # standard cube faces: exact
        t = time.time()
        ok, worst, nb = check_cell_iv(cell, frame)
        allok &= ok
        print(f"  cell {idx}: float min {mins[idx]:+.3e}; mpmath.iv {'PASS' if ok else 'FAIL'}, worst lower bound "
              f"{mpmath.nstr(worst, 5) if worst is not None else None}, {nb} boxes ({time.time() - t:.0f} s)", flush=True)
    print("MPMATH.IV RE-CHECK PASS" if allok else "MPMATH.IV RE-CHECK FAIL", flush=True)


if __name__ == "__main__":
    main()
