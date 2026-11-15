"""Stage 21, (R) by the affine sheet (theorist, paper/review/eight_comptube_tight_theorist.md section 3): building blocks.

Part 1 (here): thin centres with their Jacobi fields. For centre parameters pc (eps, phi, M) (point boxes, batched),
a C1-Lohner run of the fixed-eps flow with the columns d/d(lam, phi, M) (lam scales Y(0), i.e. d ell) over the whole
interval [0, 1]. For every time piece (m pieces per Lohner step) it returns rigorous enclosures of the centre state
z_c = (Y, P) on the piece and of D = dz/d(lam, phi, M) on the piece: D(tau) = [d step map(tau - t_k)/dz] D(t_k), the
first factor from the derivative parts of the step's Taylor jets over the (thin) set hull, evaluated on the piece.
Part 2: the first-order term over the parameter box: T(z) = 4|Y| Q(Y, v) (relative Rop), M_j = DT(z_c) D_j (jets with
the D columns as directions, v = -4|Y|P so D_v = -4(|Y| D_P + P (Yhat . D_Y))); lam_max(T_c + sum_j s_j h_j M_j) is
convex in the s_j, so its maximum over the box is at the 8 vertices.
    python verify/eight_affine.py test"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eight_comptube as CT  # noqa: E402
import eight_field_lohner as L  # noqa: E402
import eight_riccati_barrier as RB  # noqa: E402
from prove_newton_1212 import I, dn, jmul, peval, tpow, up  # noqa: E402

N = 9


def centres(pc, m=8):
    """pc (nb, 3) float centres. Returns list of pieces (ta, tb, Zlo (6, nb), Zhi (6, nb), Dlo (6, 3, nb), Dhi)."""
    nb = pc.shape[0]
    P = I(pc.copy(), pc.copy())
    c, A, r, cols = RB.start_set4(P)          # cols: phi, M, lam
    cols = [cols[2], cols[0], cols[1]]        # order lam, phi, M
    t, shrink, out = 0.0, 1.0, []
    while t < 1.0:
        h = min(L.choose_h(c) * shrink, 1.0 - t)
        h = math.floor(h / L.DY) * L.DY if h < 1.0 - t else 1.0 - t
        X = L.hull(c, A, r)
        rX = L.taylor_step(L.jets_of(X), h)
        rc = L.taylor_step(L.values_of(c), h)
        if rX is None or rc is None:
            shrink /= 2
            if h < 1e-7:
                raise RuntimeError(f"centre step collapsed at {t}")
            continue
        shrink = min(1.0, shrink * 2)
        cs, Rm, _ = rX
        # D at the step start (interval (nb, 9, 3))
        Ds = [L.hull(*cj) for cj in cols]
        D0lo = np.stack([d.lo for d in Ds], 2)
        D0hi = np.stack([d.hi for d in Ds], 2)
        D0 = I(D0lo, D0hi)
        for k in range(m):
            sa, sb = h * k / m, h * (k + 1) / m
            S = I(dn(sa), up(sb))
            tP = tpow(S, L.ORDER + 1)
            Zs = [peval(cs[i], S) + Rm[i] * tP for i in range(N)]       # jets (10, nb) over the piece
            Zlo = np.stack([z.lo[0] for z in Zs[:6]])
            Zhi = np.stack([z.hi[0] for z in Zs[:6]])
            # derivative of the step map on the piece: J[i, j] = d z_i / d z0_j (nb, 9, 9)
            Jlo = np.stack([np.stack([Zs[i].lo[1 + j] for j in range(N)], 1) for i in range(N)], 1)
            Jhi = np.stack([np.stack([Zs[i].hi[1 + j] for j in range(N)], 1) for i in range(N)], 1)
            Dp = L.imm(I(Jlo, Jhi), D0)                                 # (nb, 9, 3)
            out.append((t + sa, t + sb, Zlo, Zhi, np.moveaxis(Dp.lo[:, :6, :], 0, -1), np.moveaxis(Dp.hi[:, :6, :], 0, -1)))
        yX = L.at(cs, Rm, h)
        J = I(np.stack([np.stack([yX[i].lo[1 + j] for j in range(N)], 1) for i in range(N)], 1),
              np.stack([np.stack([yX[i].hi[1 + j] for j in range(N)], 1) for i in range(N)], 1))
        yc = L.at(rc[0], rc[1], h)
        phic = I(np.stack([v.lo[0] for v in yc], 1), np.stack([v.hi[0] for v in yc], 1))
        c, A, r = L.lohner_update(J, A, r, phic)
        cols = [L.lohner_update(J, Aj, rj, L._jv(J, cj)) for (cj, Aj, rj) in cols]
        t += h
    return out


def vjets(Zlo, Zhi, Dlo, Dhi, use_box=True):
    """jets (1 + 3, nb) in the 3 parameter directions for (Y, v) from the centre state box and D (the value part is
    the centre box; derivative parts D_Y, D_v)"""
    Y = [I(Zlo[i], Zhi[i]) for i in range(3)]
    P = [I(Zlo[3 + i], Zhi[3 + i]) for i in range(3)]
    DY = [[I(Dlo[i, j], Dhi[i, j]) for j in range(3)] for i in range(3)]
    DP = [[I(Dlo[3 + i, j], Dhi[3 + i, j]) for j in range(3)] for i in range(3)]
    R = (Y[0].sq() + Y[1].sq() + Y[2].sq()).sqrt()
    iR = R.recip()
    Yh = [y * iR for y in Y]
    out = []
    for i in range(3):
        out.append(I(np.stack([Y[i].lo] + [DY[i][j].lo for j in range(3)]), np.stack([Y[i].hi] + [DY[i][j].hi for j in range(3)])))
    for i in range(3):
        v = P[i] * R * (-4.0)
        dv = []
        for j in range(3):
            yd = Yh[0] * DY[0][j] + Yh[1] * DY[1][j] + Yh[2] * DY[2][j]
            dv.append((R * DP[i][j] + P[i] * yd) * (-4.0))
        out.append(I(np.stack([v.lo] + [d.lo for d in dv]), np.stack([v.hi] + [d.hi for d in dv])))
    return out


def first_order(Zlo, Zhi, Dlo, Dhi, eps, hw):
    """T_c and M_j at the centre (thin); returns (upper bound of lam_max, lower bound of lam_min) of
    T_c + sum_j s_j hw_j M_j over the 8 vertices (s_j = +-1), hw (3, nb) the parameter half-widths (lam, phi, M)"""
    Z = vjets(Zlo, Zhi, Dlo, Dhi)
    Q, c = CT.rop_jets(Z, eps)
    invc = CT.jrecip(c)
    T = [[jmul(Q[i][j], invc) for j in range(3)] for i in range(3)]
    Tc = [[I(T[i][j].lo[0], T[i][j].hi[0]) for j in range(3)] for i in range(3)]
    Mj = [[[I(T[i][j].lo[1 + k], T[i][j].hi[1 + k]) for j in range(3)] for i in range(3)] for k in range(3)]
    lmax, lmin = None, None
    for sg in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T:
        V = [[Tc[i][j] + sum((Mj[k][i][j] * (float(sg[k]) * hw[k]) for k in range(3)), I(np.zeros(Zlo.shape[1])))
              for j in range(3)] for i in range(3)]
        a, b = CT.eig_bounds(RB.sym(V))
        lmax = a if lmax is None else np.maximum(lmax, a)
        lmin = b if lmin is None else np.minimum(lmin, b)
    return lmax, lmin, Tc, Mj


def test():
    L.ORDER = 12
    a8 = np.array([1.5424624176685366, 0.7066425303908384, -0.30889005565526345])
    e8 = math.exp(-1.5 * a8[0])
    pc = np.array([[e8, a8[1], a8[2]]])
    t0 = time.time()
    pieces = centres(pc)
    print(f"centre with Jacobi fields: {len(pieces)} pieces, {time.time() - t0:.0f} s; D width max "
          f"{max(np.max(p[5] - p[4]) for p in pieces):.2e}", flush=True)
    # control: D against the float Jacobi fields (tools/words_eight_riccati, columns lam, phi, M)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
    import words_eight_riccati as W
    tr = W.trajectory(e8, a8[1], a8[2])
    ta, tb, Zlo, Zhi, Dlo, Dhi = pieces[-1]
    Xf = tr[-1, 6:15].reshape(3, 3)
    inside = np.all((Dlo[0:3, :, 0] - 1e-6 <= Xf) & (Xf <= Dhi[0:3, :, 0] + 1e-6))
    print(f"control: D_Y(1) encloses the float Jacobi field X(1) (to 1e-6, RK4 accuracy): {inside}", flush=True)
    for h in (0.005, 0.01, 0.02, 0.04):
        hw = np.array([[h], [h], [0.25 * h]])
        ex = []
        for (ta, tb, Zlo, Zhi, Dlo, Dhi) in pieces[::20]:
            lmax, lmin, Tc, Mj = first_order(Zlo, Zhi, Dlo, Dhi, pc[:, 0], hw)
            l0, _ = CT.eig_bounds(RB.sym(Tc))
            ex.append(lmax[0] - l0[0])
        print(f"h {h}: rigorous first-order excess along tau {np.round(ex, 3).tolist()}", flush=True)


if __name__ == "__main__":
    test()


# ---------------------------------------------------------------- stage 2: the full rigorous check (referee's formulas)
# paper/review/eight_secondvar_referee.md: K = nabla_lam J obeys nabla_T^2 K + Rop K = source (K2); comparison
# q1' = q2, q2' = k^2 q1 + src (S); coordinate remainder r_Y, r_v from (C) and section 3.
C_R = 7.80                                    # proved constant (sharp 6)


def dT_jets(lo, hi, eps):
    """over the (Y, v) boxes [lo, hi] (6, nb): T = 4|Y| Q (relative Rop) value enclosure and its 6 partial derivative
    enclosures (first jets): returns (Tval 3x3, R (3 of 3x3, d/dY_i), S (3 of 3x3, d/dv_j))"""
    Z = CT.jseed([I(lo[i], hi[i]) for i in range(6)])
    Q, c = CT.rop_jets(Z, eps)
    invc = CT.jrecip(c)
    T = [[jmul(Q[i][j], invc) for j in range(3)] for i in range(3)]
    val = [[I(T[a][b].lo[0], T[a][b].hi[0]) for b in range(3)] for a in range(3)]
    R = [[[I(T[a][b].lo[1 + i], T[a][b].hi[1 + i]) for b in range(3)] for a in range(3)] for i in range(3)]
    S = [[[I(T[a][b].lo[4 + j], T[a][b].hi[4 + j]) for b in range(3)] for a in range(3)] for j in range(3)]
    return val, R, S


def sqrt_lmax_sumsq(Ms):
    """upper bound of sqrt(lam_max(sum_i M_i^2)) for symmetric interval matrices M_i"""
    out = [[None] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(3):
            acc = None
            for M in Ms:
                for k in range(3):
                    x = M[a][k] * M[k][b]
                    acc = x if acc is None else acc + x
            out[a][b] = acc
    lmax, _ = CT.eig_bounds(RB.sym(out))
    return up(np.sqrt(np.maximum(lmax, 0.0)))


def mdiff(A, B):
    return [[A[a][b] - B[a][b] for b in range(3)] for a in range(3)]


def potential_terms(lo, hi, eps):
    """over the Y box: sup of eps N3 (diagonal third covariant derivative of U, g-units), eps |grad U^T|_g, and the
    H tensor pieces. N3 = 2 sqrt|Y| sup_|e|=1 |C(e, e)| with C_abc = d_b H_ac + Gamma terms (symmetrised in b, c),
    H = 4|Y| (D^2U - f' x dU - dU x f' + (f'.dU) Id), f' = -Y/(2|Y|^2)."""
    nb = lo.shape[1]
    Z = []
    for i in range(3):
        l = np.zeros((4, nb)); h = np.zeros((4, nb))
        l[0], h[0] = lo[i], hi[i]
        l[1 + i] = h[1 + i] = 1.0
        Z.append(I(l, h))
    s = CT.jsum([CT.jsq(y) for y in Z])
    Rj = CT.jsqrt(s)
    iR = CT.jrecip(Rj)
    iR2 = CT.jsq(iR)
    Yh = [jmul(y, iR) for y in Z]
    PT = [[(CT.jconst(1.0, Rj) if i == j else CT.jconst(0.0, Rj)) - jmul(Yh[i], Yh[j]) for j in range(3)] for i in range(3)]
    proj = [[jmul(PT[i][j], iR) for j in range(3)] for i in range(3)]
    HU = [[CT.jconst(0.0, Rj) for _ in range(3)] for _ in range(3)]
    gU = [CT.jconst(0.0, Rj) for _ in range(3)]
    for u in __import__("eight_field").UK:
        ud = CT.jsum([Z[i] * u[i] for i in range(3)])
        perp = [Z[i] - ud * u[i] for i in range(3)]
        if np.all(ud.lo[0] > 0):
            q = jmul(CT.jsum([CT.jsq(x) for x in perp]), CT.jrecip(Rj + ud))
            g = [jmul(perp[i] - q * u[i], iR) for i in range(3)]
        else:
            q = Rj - ud
            g = [Yh[i] - CT.jconst(u[i], Rj) for i in range(3)]
        q32 = CT.jpow_m32(q)
        q52 = jmul(q32, CT.jrecip(q))
        for i in range(3):
            gU[i] = gU[i] - jmul(q32, g[i]) * 0.5
            for j in range(3):
                HU[i][j] = HU[i][j] + jmul(q52, jmul(g[i], g[j])) * 0.75 - jmul(q32, proj[i][j]) * 0.5
    fp = [jmul(Z[i], iR2) * (-0.5) for i in range(3)]
    fgU = CT.jsum([jmul(fp[i], gU[i]) for i in range(3)])
    four = Rj * 4.0
    H = [[jmul(four, HU[a][b] - jmul(fp[a], gU[b]) - jmul(gU[a], fp[b]) + (fgU if a == b else CT.jconst(0.0, Rj)))
          for b in range(3)] for a in range(3)]
    Hv = [[I(H[a][b].lo[0], H[a][b].hi[0]) for b in range(3)] for a in range(3)]
    dH = [[[I(H[a][c].lo[1 + b], H[a][c].hi[1 + b]) for c in range(3)] for a in range(3)] for b in range(3)]   # dH[b][a][c]
    f = [I(fp[i].lo[0], fp[i].hi[0]) for i in range(3)]
    fH = [CT.jsum([f[d] * Hv[d][c] for d in range(3)]) for c in range(3)]          # (f'^T H)_c
    Hf = [CT.jsum([Hv[a][d] * f[d] for d in range(3)]) for a in range(3)]          # (H f')_a
    # C_abc (before symmetrisation in b, c)
    Cmax = None
    for a in range(3):
        Ca = [[None] * 3 for _ in range(3)]
        for b in range(3):
            for c in range(3):
                t = dH[b][a][c]
                # Gamma(j, Hj)_a = (f'.j)(Hj)_a + (f'.Hj) j_a - (j.Hj) f'_a
                t = t + f[b] * Hv[a][c] + (fH[c] if a == b else I(0.0)) - Hv[b][c] * f[a]
                # -(H Gamma(j, j))_a = -2 H_ac f'_b + delta_bc (H f')_a
                t = t - Hv[a][c] * f[b] * 2.0 + (Hf[a] if b == c else I(0.0))
                Ca[b][c] = t
        Cs = RB.sym(Ca)
        lmax, lmin = CT.eig_bounds(Cs)
        na = np.maximum(np.abs(lmax), np.abs(lmin))
        Cmax = na ** 2 if Cmax is None else Cmax + na ** 2
    Rv = I(Rj.lo[0], Rj.hi[0])
    N3 = up(up(2.0 * up(np.sqrt(Rv.hi))) * up(np.sqrt(Cmax)))
    # |grad U^T|_g = 2 sqrt|Y| |gradE U^T|
    gUv = [I(gU[i].lo[0], gU[i].hi[0]) for i in range(3)]
    Yhv = [I(Yh[i].lo[0], Yh[i].hi[0]) for i in range(3)]
    gr = CT.jsum([gUv[i] * Yhv[i] for i in range(3)])
    gT = [gUv[i] - Yhv[i] * gr for i in range(3)]
    gTn = up(np.sqrt(sum(np.maximum(np.abs(x.lo), np.abs(x.hi)) ** 2 for x in gT)))
    gradUT_g = up(up(2.0 * up(np.sqrt(Rv.hi))) * gTn)
    eps = np.asarray(eps, float)
    return up(eps * N3), up(eps * gradUT_g), Rv


def col_norms(Dlo, Dhi, rows):
    """Euclidean norms (upper bounds) of the 3 columns of D restricted to rows, arrays (3, nb)"""
    mag = np.maximum(np.abs(Dlo[rows]), np.abs(Dhi[rows]))        # (len(rows), 3, nb)
    return up(np.sqrt(np.sum(mag ** 2, 0)))


def q_step(q1, q2, k2, src, h):
    """q1' = q2, q2' = k2 q1 + src (src constant) over h, upward rounding"""
    sk = up(np.sqrt(np.maximum(k2, 1e-300)))
    x = up(sk * h)
    ch = up(np.cosh(x) * (1 + 4e-16))
    sh = up(np.sinh(x) * (1 + 4e-16))
    small = x < 1e-6
    shk = np.where(small, up(h * (1 + 1e-12)), up(sh / dn(sk)))
    chm1k2 = np.where(small, up(0.5 * h * h * (1 + 1e-6)), up((ch - 1.0) / dn(sk * sk)))
    n1 = up(up(up(q1 * ch) + up(q2 * shk)) + up(src * chm1k2))
    n2 = up(up(up(up(q1 * sk) * sh) + up(q2 * ch)) + up(src * shk))
    return n1, n2


def check_affine(Pb, m=32, iters=5, verbose=False):
    """stage-2 check of parameter boxes Pb (nb, 6) = (eps, phi, M) lo/hi. Returns (ok, ymin, info)"""
    nb = Pb.shape[0]
    pc, dl, dphi, dM = CT.box_offsets(Pb)
    eps = pc[:, 0]
    pieces = centres(pc, m)
    npc = len(pieces)
    hw = np.stack([dl, dphi, dM])
    e2 = up(np.exp(dl / 2) * (1 + 1e-15))
    Mmax = np.maximum(np.abs(Pb[:, 4]), np.abs(Pb[:, 5]))
    y10 = up(0.5 * e2 * up(np.sqrt(dl ** 2 + dphi ** 2)))
    y20 = up(e2 * up(Mmax * dl + 2.0 * dM))
    q10 = up(0.25 * e2 * up(dl ** 2 + dphi ** 2))
    q20 = up(0.5 * e2 * up(Mmax * dl ** 2 + 4.0 * dl * dM + 3.0 * Mmax * dphi ** 2))
    # per-piece data that do not depend on the remainder guess
    pre = []
    for (ta, tb, Zlo, Zhi, Dlo, Dhi) in pieces:
        vlo, vhi = CT.vbox(Zlo, Zhi)
        J = vjets(Zlo, Zhi, Dlo, Dhi)
        DYlo = np.stack([np.stack([J[i].lo[1 + j] for j in range(3)]) for i in range(3)])
        DYhi = np.stack([np.stack([J[i].hi[1 + j] for j in range(3)]) for i in range(3)])
        Dvlo = np.stack([np.stack([J[3 + i].lo[1 + j] for j in range(3)]) for i in range(3)])
        Dvhi = np.stack([np.stack([J[3 + i].hi[1 + j] for j in range(3)]) for i in range(3)])
        dY1 = up(np.sum(col_norms(DYlo, DYhi, slice(0, 3)) * hw, 0))
        dv1 = up(np.sum(col_norms(Dvlo, Dvhi, slice(0, 3)) * hw, 0))
        lmax, lmin, _, _ = first_order(Zlo, Zhi, Dlo, Dhi, eps, hw)
        _, Rc, Sc = dT_jets(vlo, vhi, eps)
        pre.append((ta, tb, vlo, vhi, dY1, dv1, lmax, lmin, Rc, Sc,
                    sqrt_lmax_sumsq(Rc), sqrt_lmax_sumsq(Sc)))
    rYg = np.zeros((npc, nb))
    rvg = np.zeros((npc, nb))
    closed = np.zeros(nb, bool)
    for it in range(iters):
        ws, k2s, coef = [], [], []
        okb = np.ones(nb, bool)
        for k, (ta, tb, vlo, vhi, dY1, dv1, lmax, lmin, Rc, Sc, nRc, nSc) in enumerate(pre):
            rhoY = up(dY1 + rYg[k])
            rhov = up(dv1 + rvg[k])
            tlo = np.concatenate([dn(vlo[0:3] - rhoY), dn(vlo[3:6] - rhov)])
            thi = np.concatenate([up(vhi[0:3] + rhoY), up(vhi[3:6] + rhov)])
            _, Rt, St = dT_jets(tlo, thi, eps)
            E2 = up(up(up(rhoY * sqrt_lmax_sumsq([mdiff(Rt[i], Rc[i]) for i in range(3)]))
                       + up(rhov * sqrt_lmax_sumsq([mdiff(St[j], Sc[j]) for j in range(3)])))
                    + up(up(rYg[k] * nRc) + up(rvg[k] * nSc)))
            w = up(lmax + E2)
            lo_ = dn(lmin - E2)
            ws.append(w)
            k2s.append(np.maximum(np.abs(w), np.abs(lo_)))
            # coefficients of src over the tube box
            epsN3, epsgUT, Rv = potential_terms(tlo[0:3], thi[0:3], eps)
            Rp, Rm_ = up(Rv.hi), dn(Rv.lo)
            okb &= Rm_ > 0.3 * Rp
            Rm_ = np.maximum(Rm_, 1e-300)
            vE = up(np.sqrt(np.sum(np.maximum(np.abs(tlo[3:6]), np.abs(thi[3:6])) ** 2, 0)))
            V = up(vE / dn(2.0 * dn(np.sqrt(Rm_))))
            a1 = up(up(C_R * up(V * V) / dn(Rm_ * dn(np.sqrt(Rm_)))) + up(up(3.0 * epsgUT) / Rm_) + epsN3)
            a2 = up(12.0 * V / Rm_)
            coef.append((a1, a2, Rp, Rm_, V))
        y1, y2, q1, q2 = y10.copy(), y20.copy(), q10.copy(), q20.copy()
        rY = np.zeros((npc, nb))
        rv = np.zeros((npc, nb))
        for k, (ta, tb, *_r) in enumerate(pre):
            h = float(up(tb - ta))
            y1n, y2n = CT.comp_step(y1, y2, k2s[k], h)
            a1, a2, Rp, Rm_, V = coef[k]
            src = up(up(a1 * y1n * y1n) + up(a2 * y1n * y2n))       # y increasing: piece-end values bound the piece
            q1, q2 = q_step(q1, q2, k2s[k], src, h)
            y1, y2 = y1n, y2n
            sRp = up(np.sqrt(Rp))
            rY[k] = up(up(sRp * q1) + up(y1 * y1))
            rv[k] = up(up(up(sRp * q2) + up(V * q1)) + up(up(2.0 * y1 * y2) + up(4.0 * V * y1 * y1 / dn(np.sqrt(Rm_)))))
        closed = okb & np.all(rY <= rYg, 0) & np.all(rv <= rvg, 0) & np.all(np.isfinite(rY), 0)
        if verbose:
            print(f"   iteration {it}: closed {closed.sum()}/{nb}; r_Y(1) {np.max(rY[-1]):.2e} r_v(1) {np.max(rv[-1]):.2e} "
                  f"y1(1) {np.max(y1):.2e}", flush=True)
        if np.all(closed):
            break
        rYg = np.where(closed[None, :], rYg, np.maximum(1.3 * rY, rYg))
        rvg = np.where(closed[None, :], rvg, np.maximum(1.3 * rv, rvg))
    # Sturm with W = w per piece
    z, zp = I(np.ones(nb)), I(np.zeros(nb))
    ymin = np.full(nb, np.inf)
    for k, (ta, tb, *_r) in enumerate(pre):
        Wv = ws[k]
        Lp = float(up(tb - ta))
        om = np.sqrt(np.abs(np.where(np.isfinite(Wv), Wv, 1.0)))
        omI = I(RB.push3(om, -np.inf), RB.push3(om, np.inf))
        x = om * Lp
        cosI = I(RB.push3(np.cos(x), -np.inf), RB.push3(np.cos(x), np.inf))
        sinI = I(RB.push3(np.sin(x), -np.inf), RB.push3(np.sin(x), np.inf))
        chI = I(RB.push3(np.cosh(x), -np.inf), RB.push3(np.cosh(x), np.inf))
        shI = I(RB.push3(np.sinh(x), -np.inf), RB.push3(np.sinh(x), np.inf))
        pos, neg = Wv > 1e-12, Wv < -1e-12
        zb = [z * cosI + zp * (sinI / omI), z * chI + zp * (shI / omI), z + zp * Lp]
        zpb = [zp * cosI - z * (omI * sinI), zp * chI + z * (omI * shI), zp]
        inner = np.where(pos, z.lo * cosI.lo + np.minimum(0.0, (zp / omI).lo) * sinI.hi,
                         np.where(neg, z.lo + np.minimum(0.0, (zp / omI).lo) * shI.hi, z.lo + np.minimum(0.0, zp.lo) * Lp))
        inner = np.where(np.isfinite(Wv) & (x < 1.5) & (z.lo > 0), dn(inner), -np.inf)
        ymin = np.minimum(ymin, inner)
        sel = lambda a, b, c_: np.where(pos, a, np.where(neg, b, c_))
        z = I(sel(zb[0].lo, zb[1].lo, zb[2].lo), sel(zb[0].hi, zb[1].hi, zb[2].hi))
        zp = I(sel(zpb[0].lo, zpb[1].lo, zpb[2].lo), sel(zpb[0].hi, zpb[1].hi, zpb[2].hi))
    ymin = np.minimum(ymin, z.lo)
    return closed & (ymin > 0), ymin, {"closed": closed, "rY1": rY[-1], "rv1": rv[-1], "y1": y1, "q1": q1}
