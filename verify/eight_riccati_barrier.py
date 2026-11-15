"""Stage 21, hypothesis (R): regularity of the I-field label map, by a Riccati barrier on rigorous C0 tubes
(paper/review/eight_field_c1_obstacle_theorist.md section 3, eight_lift_referee.md item 5).

Fixed-eps scaled flow, H = 2|Y||P|^2 - eps U(Y), Y' = -H_P, P' = H_Y (verify/eight_field.py). The label derivatives
at fixed s are the Jacobi fields (X, Xi) = d(Y, P)/d(lambda, phi, M) of the fixed-eps flow (lambda scales Y(0)),
X(0) = (Y0, dY0/dphi, 0), Xi(0) = (0, 0, n_I). Hypothesis (R) for a parameter box: X(1) is invertible for every
(eps, phi, M) in the box. Where X is invertible, R = Xi X^(-1) is symmetric and
    R' = H_YY + H_YP R + R H_YP^T + 4|Y| R^2        (H_YP[i, j] = d^2 H / dY_i dP_j = 4 Yhat_i P_j).
Comparison lemma (theorist 3.2, referee 5; H_PP = 4|Y| I > 0 gives existence): if B is continuous and piecewise C^1
on [tau0, 1], B(tau0) - R(tau0) >= 0 and B' - RHS(B) >= 0 (Loewner order) along the trajectory, then R exists on
[tau0, 1], R <= B, and X stays invertible (R cannot go to -infinity since H_PP > 0; a singular X needs R -> +inf).
Rigorous check per parameter box:
  1. C1-Lohner (verify/eight_field_lohner.py) on [0, tau0], tau0 = 1/16, with the columns d/d(lambda, phi, M):
     enclosures of X(tau0), Xi(tau0), hence of R(tau0) = Xi X^(-1) (interval inverse with a Neumann bound; R is
     symmetric, so the enclosure is intersected with its transpose).
  2. C0-Lohner on [tau0, 1]; each Taylor step's series over the hull of the set gives an enclosure of all
     trajectories of the box on any sub-interval of the step.
  3. B: float, continuous, piecewise linear in tau on a grid refining the steps (m pieces per step), built from the
     box centre's Riccati solution with B(tau0) = R_c(tau0) + d0 I and source d1 I.
  4. On each piece: D = (B_b - B_a)/dtau - RHS(hull(B_a, B_b), Z_piece, eps_box) as an interval matrix (B linear on
     the piece, so B(tau) lies in the entrywise hull), symmetrised; interval Cholesky succeeds => every symmetric
     matrix in it is positive definite (Alefeld-Mayer). Same test for B(tau0) - [R(tau0)].
The near-ray form q = |Y_perp|^2/(R + u.Y) is used in the Hessian of U.
    python verify/eight_riccati_barrier.py test"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eight_field as EF  # noqa: E402
import eight_field_lohner as L  # noqa: E402
from prove_newton_1212 import I, dn, peval, tpow, up  # noqa: E402

TAU0 = 1.0 / 16
MPIECE = int(os.environ.get("RB_M", "16"))
N = 9


# ---------------------------------------------------------------- interval helpers on (nb,) arrays
def ivec(lo, hi):
    return I(np.asarray(lo, float), np.asarray(hi, float))


def isq(x):
    return x.sq()


def chol_pd(Dm):
    """Dm: 3x3 nested list of intervals (arrays (nb,)), symmetric. Interval Cholesky; returns a bool array: True where
    it succeeds (then every symmetric matrix in the interval matrix is positive definite)."""
    ok = np.ones(np.shape(Dm[0][0].lo), bool)
    l11sq = Dm[0][0]
    ok &= l11sq.lo > 0
    l11 = I(np.sqrt(np.maximum(dn(l11sq.lo), 1e-300)) * (1 - 4e-16), up(np.sqrt(np.maximum(l11sq.hi, 1e-300))))
    l21 = Dm[1][0] / l11
    l31 = Dm[2][0] / l11
    a22 = Dm[1][1] - l21.sq()
    ok &= a22.lo > 0
    l22 = I(np.sqrt(np.maximum(a22.lo, 1e-300)) * (1 - 4e-16), up(np.sqrt(np.maximum(a22.hi, 1e-300))))
    l32 = (Dm[2][1] - l31 * l21) / l22
    a33 = Dm[2][2] - l31.sq() - l32.sq()
    ok &= a33.lo > 0
    return ok


def sym(M):
    """symmetric part of an interval 3x3 matrix (entries intervals)"""
    return [[(M[i][j] + M[j][i]) * 0.5 for j in range(3)] for i in range(3)]


def hessians(Y, P, eps):
    """interval Hessian blocks of H = 2|Y||P|^2 - eps U at boxes: returns (R = |Y|, HYP 3x3, HYY 3x3)"""
    R = (Y[0].sq() + Y[1].sq() + Y[2].sq()).sqrt()
    iR = R.recip()
    Yh = [y * iR for y in Y]
    PP = P[0].sq() + P[1].sq() + P[2].sq()
    HYP = [[Yh[i] * P[j] * 4.0 for j in range(3)] for i in range(3)]
    proj = [[((I(1.0) if i == j else I(0.0)) - Yh[i] * Yh[j]) * iR for j in range(3)] for i in range(3)]
    HU = [[I(np.zeros(R.lo.shape)) for _ in range(3)] for _ in range(3)]
    for u in EF.UK:
        ud = Y[0] * u[0] + Y[1] * u[1] + Y[2] * u[2]
        perp = [Y[i] - ud * u[i] for i in range(3)]
        near = np.all(ud.lo > 0)
        if near:
            q = (perp[0].sq() + perp[1].sq() + perp[2].sq()) / (R + ud)
            g = [(perp[i] - q * u[i]) * iR for i in range(3)]
        else:
            q = R - ud
            g = [Yh[i] - u[i] for i in range(3)]
        sq = q.sqrt()
        q32 = (q * sq).recip()
        q52 = q32 / q
        for i in range(3):
            for j in range(3):
                HU[i][j] = HU[i][j] + q52 * g[i] * g[j] * 0.75 - q32 * proj[i][j] * 0.5
    HYY = [[PP * proj[i][j] * 2.0 - eps * HU[i][j] for j in range(3)] for i in range(3)]
    return R, HYP, HYY


def rhs_iv(B, R, HYP, HYY):
    """RHS(B) = HYY + HYP B + B HYP^T + 4 R B B with B a 3x3 list of intervals"""
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            acc = HYY[i][j]
            for k in range(3):
                acc = acc + HYP[i][k] * B[k][j] + B[i][k] * HYP[j][k]
            bb = None
            for k in range(3):
                t = B[i][k] * B[k][j]
                bb = t if bb is None else bb + t
            row.append(acc + R * bb * 4.0)
        out.append(row)
    return out


# ---------------------------------------------------------------- start set with the lambda column
def start_set4(P):
    """like eight_field_lohner.start_set, parameters (eps, phi, M, lambda) with lambda = 1 (point); columns for
    phi, M, lambda (the eps column is not needed at fixed eps)"""
    nb = P.lo.shape[0]
    c, A, r, cols = L.start_set(P)
    # lambda column: d z0 / d lambda = (Y0, 0, ...) over the box
    sn, cs_ = EF.isincos(I(P.lo[:, 1], P.hi[:, 1]))
    Y0 = [cs_ * EF.EI[i] + sn * EF.EZ[i] for i in range(3)]
    lo, hi = np.zeros((nb, N)), np.zeros((nb, N))
    for i in range(3):
        lo[:, i], hi[:, i] = Y0[i].lo, Y0[i].hi
    col = I(lo, hi)
    cj = col.mid
    cols = [cols[1], cols[2], (cj, np.broadcast_to(np.eye(N), (nb, N, N)).copy(), col - cj)]
    return c, A, r, cols


def lohner_run(P, record=True, rec_from=None):
    """C1 on [0, tau0] (columns lambda, phi, M), then C0 to 1. Returns R(tau0) enclosure data and the recorded steps
    [(t, h, value-series coefficients (list over components of lists over orders of I (nb,)), remainder (list of I))]"""
    c, A, r, cols = start_set4(P)
    t, shrink, steps = 0.0, 1.0, []
    D0 = None
    while t < 1.0:
        target = TAU0 if t < TAU0 else 1.0
        h = min(L.choose_h(c) * shrink, target - t)
        h = math.floor(h / L.DY) * L.DY if h < target - t else target - t
        X = L.hull(c, A, r)
        rX = L.taylor_step(L.jets_of(X), h)
        rc = L.taylor_step(L.values_of(c), h)
        if rX is None or rc is None:
            shrink /= 2
            if h < 1e-6:
                raise RuntimeError(f"step collapsed at tau = {t}")
            continue
        shrink = min(1.0, shrink * 2)
        cs, Rm, _ = rX
        if record and t >= (TAU0 if rec_from is None else rec_from):
            steps.append((t, h, [[cs[i][k][0] for k in range(len(cs[i]))] for i in range(N)], [Rm[i][0] for i in range(N)]))
        yX = L.at(cs, Rm, h)
        J = I(np.stack([np.stack([yX[i].lo[1 + j] for j in range(N)], 1) for i in range(N)], 1),
              np.stack([np.stack([yX[i].hi[1 + j] for j in range(N)], 1) for i in range(N)], 1))
        yc = L.at(rc[0], rc[1], h)
        phic = I(np.stack([v.lo[0] for v in yc], 1), np.stack([v.hi[0] for v in yc], 1))
        c, A, r = L.lohner_update(J, A, r, phic)
        if t < TAU0:
            cols = [L.lohner_update(J, Aj, rj, L._jv(J, cj)) for (cj, Aj, rj) in cols]
        t += h
        if t == TAU0 and D0 is None:
            D0 = [L.hull(*cj) for cj in cols]           # columns phi, M, lambda: each interval (nb, 9)
    return D0, steps


def _sl_state(sel, c, A, r, cols):
    return (c[sel], A[sel], I(r.lo[sel], r.hi[sel]),
            [(cj[sel], Aj[sel], I(rj.lo[sel], rj.hi[sel])) for (cj, Aj, rj) in cols])


def lohner_split(P, rec_from=0.0, tries=3):
    """like lohner_run, but when a step cannot be validated for the batch (after `tries` step halvings), the batch is
    split in two AT THAT STEP and both halves continue from the current state (no restart from tau = 0).
    Returns ([(indices, D0, steps)], failed indices)."""
    c, A, r, cols = start_set4(P)
    out, failed = [], []

    def advance(idx, c, A, r, cols, t, steps, D0):
        while t < 1.0:
            target = TAU0 if t < TAU0 else 1.0
            h0 = L.choose_h(c)
            res = None
            for k in range(tries):
                h = min(h0 / 2 ** k, target - t)
                h = math.floor(h / L.DY) * L.DY if h < target - t else target - t
                X = L.hull(c, A, r)
                rX = L.taylor_step(L.jets_of(X), h)
                rc = L.taylor_step(L.values_of(c), h) if rX is not None else None
                if rX is not None and rc is not None:
                    res = (h, rX, rc)
                    break
            if res is None:
                if len(idx) == 1:
                    failed.append(int(idx[0]))
                    return
                hh = len(idx) // 2
                for sel in (slice(0, hh), slice(hh, None)):
                    st = _sl_state(sel, c, A, r, cols)
                    advance(idx[sel], *st, t, slice_steps(steps, sel) if steps else [],
                            None if D0 is None else [I(x.lo[sel], x.hi[sel]) for x in D0])
                return
            h, rX, rc = res
            cs, Rm, _ = rX
            if t >= rec_from:
                steps.append((t, h, [[cs[i][k][0] for k in range(len(cs[i]))] for i in range(N)], [Rm[i][0] for i in range(N)]))
            yX = L.at(cs, Rm, h)
            J = I(np.stack([np.stack([yX[i].lo[1 + j] for j in range(N)], 1) for i in range(N)], 1),
                  np.stack([np.stack([yX[i].hi[1 + j] for j in range(N)], 1) for i in range(N)], 1))
            yc = L.at(rc[0], rc[1], h)
            phic = I(np.stack([v.lo[0] for v in yc], 1), np.stack([v.hi[0] for v in yc], 1))
            c, A, r = L.lohner_update(J, A, r, phic)
            if t < TAU0:
                cols = [L.lohner_update(J, Aj, rj, L._jv(J, cj)) for (cj, Aj, rj) in cols]
            t += h
            if t == TAU0 and D0 is None:
                D0 = [L.hull(*cj) for cj in cols]
        out.append((idx, D0, steps))

    advance(np.arange(P.lo.shape[0]), c, A, r, cols, 0.0, [], None)
    return out, failed


def lohner_batches(P, idx=None, minsplit=1):
    """lohner_run on a batch; on failure (a step that cannot be validated for some box) split the batch in two and
    retry, so one bad box does not stop the others. Returns ([(indices, D0, steps)], failed indices)."""
    nb = P.lo.shape[0]
    idx = np.arange(nb) if idx is None else idx
    try:
        D0, steps = lohner_run(P)
        return [(idx, D0, steps)], []
    except (RuntimeError, ValueError, ZeroDivisionError, FloatingPointError):
        if nb <= minsplit:
            return [], list(idx)
        h = nb // 2
        r1, f1 = lohner_batches(I(P.lo[:h], P.hi[:h]), idx[:h], minsplit)
        r2, f2 = lohner_batches(I(P.lo[h:], P.hi[h:]), idx[h:], minsplit)
        return r1 + r2, f1 + f2


def R0_enclosure(D0):
    """R(tau0) = Xi X^(-1) with X = dY/d(lambda, phi, M), Xi = dP/d(...) (column order lambda, phi, M)"""
    colsY = [D0[2], D0[0], D0[1]]
    nb = colsY[0].lo.shape[0]
    X = [[I(colsY[j].lo[:, i], colsY[j].hi[:, i]) for j in range(3)] for i in range(3)]
    Xi = [[I(colsY[j].lo[:, 3 + i], colsY[j].hi[:, 3 + i]) for j in range(3)] for i in range(3)]
    Xm = np.stack([np.stack([X[i][j].mid for j in range(3)], 1) for i in range(3)], 1)     # (nb, 3, 3)
    B0 = np.linalg.inv(Xm)
    # E = I - B0 X (interval), X^-1 in B0 + d with d = ||E|| ||B0|| / (1 - ||E||) (inf norms, elementwise bound)
    E = [[(I(1.0) if i == j else I(0.0)) - sum((X[k][j] * B0[:, i, k] for k in range(3)), I(np.zeros(nb)))
          for j in range(3)] for i in range(3)]
    e = up(np.max(np.stack([sum(np.maximum(np.abs(E[i][j].lo), np.abs(E[i][j].hi)) for j in range(3))
                            for i in range(3)]), 0) * (1 + 1e-14))
    nbn = up(np.max(np.sum(np.abs(B0), 2), 1) * (1 + 1e-14))
    if np.any(e >= 0.5):
        raise RuntimeError("X(tau0) inverse enclosure failed")
    d = up(e * nbn / dn(1 - e))
    Xinv = [[I(dn(B0[:, i, j] - d), up(B0[:, i, j] + d)) for j in range(3)] for i in range(3)]
    R = [[sum((Xi[i][k] * Xinv[k][j] for k in range(3)), I(np.zeros(nb))) for j in range(3)] for i in range(3)]
    # symmetric: intersect with the transpose
    Rs = [[I(np.maximum(R[i][j].lo, R[j][i].lo), np.minimum(R[i][j].hi, R[j][i].hi)) for j in range(3)] for i in range(3)]
    return Rs


# ---------------------------------------------------------------- float barrier for the box centre
def centre_barrier(pc, grid, d0, d1):
    """B at the grid nodes (float), from the centre's Riccati solution: B(tau0) = R_c(tau0) + d0 I, B' = RHS_c + d1 I"""
    import words_eight_riccati as W  # tools/
    eps, phi, M = pc
    # centre trajectory with Jacobi fields from 0 to tau0 (RK4, fine), then Riccati jointly with the state
    Y0 = math.cos(phi) * EF.EI_F + math.sin(phi) * EF.EZ_F
    Yphi = -math.sin(phi) * EF.EI_F + math.cos(phi) * EF.EZ_F
    st = np.concatenate([Y0, M * EF.NI_F, np.stack([Y0, Yphi, np.zeros(3)], 1).ravel(),
                         np.stack([np.zeros(3), np.zeros(3), EF.NI_F], 1).ravel()])
    n0 = 400
    h = TAU0 / n0
    for _ in range(n0):
        k1 = W.rhs(st, eps); k2 = W.rhs(st + h / 2 * k1, eps); k3 = W.rhs(st + h / 2 * k2, eps); k4 = W.rhs(st + h * k3, eps)
        st = st + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    X, Xi = st[6:15].reshape(3, 3), st[15:24].reshape(3, 3)
    B = Xi @ np.linalg.inv(X)
    B = 0.5 * (B + B.T) + d0 * np.eye(3)
    z = st[0:6].copy()

    def f(z_, B_):
        Y, P = z_[0:3], z_[3:6]
        HP, HY, HPP, HYP, HYY = W.derivs(Y, P, eps)
        return np.concatenate([-HP, HY]), HYY + HYP @ B_ + B_ @ HYP.T + B_ @ HPP @ B_ + d1 * np.eye(3)
    Bs = [B]
    for a, b in zip(grid[:-1], grid[1:]):
        nsub = 4
        hh = (b - a) / nsub
        for _ in range(nsub):
            k1 = f(z, B); k2 = f(z + hh / 2 * k1[0], B + hh / 2 * k1[1]); k3 = f(z + hh / 2 * k2[0], B + hh / 2 * k2[1])
            k4 = f(z + hh * k3[0], B + hh * k3[1])
            z = z + hh / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            B = B + hh / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            B = 0.5 * (B + B.T)
            if not np.all(np.isfinite(B)) or np.max(np.abs(B)) > 1e6:
                return None
        Bs.append(B)
    return np.array(Bs)


# ---------------------------------------------------------------- the check
KREF = int(os.environ.get("RB_K", "9"))       # each Lohner step is split into 2^KREF fine nodes for the barrier


def piece_ok(b, ts, cs, Rm, sa, sb, Ba, Bb, epsI):
    """the Loewner test on one piece [ts + sa, ts + sb] for box b with B linear from Ba to Bb"""
    S = I(dn(sa), up(sb))
    tP = tpow(S, L.ORDER + 1)
    sl = slice(b, b + 1)
    Z = [peval([c[sl] for c in cs[i]], S) + Rm[i][sl] * tP for i in range(6)]
    Rk, HYP, HYY = hessians(Z[0:3], Z[3:6], I(epsI.lo[sl], epsI.hi[sl]))
    dt = sb - sa
    Bh = [[I(np.array([min(Ba[i, j], Bb[i, j])]), np.array([max(Ba[i, j], Bb[i, j])])) for j in range(3)] for i in range(3)]
    RHS = rhs_iv(Bh, Rk, HYP, HYY)
    Dm = [[I(dn(np.array([(Bb[i, j] - Ba[i, j]) / dt])), up(np.array([(Bb[i, j] - Ba[i, j]) / dt]))) - RHS[i][j]
           for j in range(3)] for i in range(3)]
    return bool(chol_pd(sym(Dm))[0])


def check_boxes(P, d0s=(0.003, 0.01, 0.03), d1s=(0.03, 0.1, 0.3), verbose=False, maxpieces=20000):
    """P: interval (nb, 3) of (eps, phi, M). Returns ok (nb,) bool, chosen (d0, d1), timing, pieces used.
    B is piecewise linear on an adaptive sub-grid of the fine nodes (2^KREF per Lohner step): a piece that fails the
    test is halved, down to one fine interval."""
    t0 = time.time()
    D0, steps = lohner_run(P)
    R0 = R0_enclosure(D0)
    t1 = time.time()
    nb = P.lo.shape[0]
    nf = 2 ** KREF
    fine = [TAU0]
    for (ts, h, cs, Rm) in steps:
        fine += [ts + h * (m + 1) / nf for m in range(nf)]
    fine = np.array(fine)
    epsI = I(P.lo[:, 0], P.hi[:, 0])
    ok = np.zeros(nb, bool)
    chosen, used = [None] * nb, [0] * nb
    pc = P.mid
    for b in range(nb):
        for d1 in d1s:
            for d0 in d0s:
                Bs = centre_barrier(pc[b], fine, d0, d1)
                if Bs is None:
                    continue
                Ds = [[I(np.array([Bs[0][i, j]])) - I(R0[i][j].lo[b:b + 1], R0[i][j].hi[b:b + 1]) for j in range(3)]
                      for i in range(3)]
                if not chol_pd(sym(Ds))[0]:
                    continue
                good, npieces = True, 0
                for si, (ts, h, cs, Rm) in enumerate(steps):
                    base = si * nf
                    stack = [(base + k * nf // 16, base + (k + 1) * nf // 16) for k in range(16)]
                    while stack and good:
                        lo, hi = stack.pop()
                        npieces += 1
                        if npieces > maxpieces:
                            good = False
                            break
                        if piece_ok(b, ts, cs, Rm, fine[lo] - ts, fine[hi] - ts, Bs[lo], Bs[hi], epsI):
                            continue
                        if hi - lo == 1:
                            good = False
                            break
                        mid = (lo + hi) // 2
                        stack += [(lo, mid), (mid, hi)]
                    if not good:
                        break
                if good:
                    ok[b], chosen[b], used[b] = True, (d0, d1), npieces
                    break
            if ok[b]:
                break
        if verbose:
            print(f"   box {b}: {'OK ' + str(chosen[b]) + f' {used[b]} pieces' if ok[b] else 'not closed'}", flush=True)
    return ok, chosen, (t1 - t0, time.time() - t1), used


# ---------------------------------------------------------------- scalar Sturm test (theorist section 3)
def push3(x, d):
    for _ in range(3):
        x = np.nextafter(x, d)
    return x


def w_upper(Y, P, eps):
    """rigorous upper bound (nb,) of the largest eigenvalue, relative to g = |dY|^2/(4|Y|), of
    Rop = Rm(., v) v - eps Hess_g U along the tube (v = -4|Y| P); cone curvature 3/|Y| on tangential parts"""
    R = (Y[0].sq() + Y[1].sq() + Y[2].sq()).sqrt()
    iR = R.recip()
    Yh = [y * iR for y in Y]
    v = [P[i] * R * (-4.0) for i in range(3)]
    vr = Yh[0] * v[0] + Yh[1] * v[1] + Yh[2] * v[2]
    vT = [v[i] - Yh[i] * vr for i in range(3)]
    vT2 = vT[0].sq() + vT[1].sq() + vT[2].sq()
    c = iR * 0.25                                  # g = c |.|^2
    PT = [[(I(1.0) if i == j else I(0.0)) - Yh[i] * Yh[j] for j in range(3)] for i in range(3)]
    k3 = iR * 3.0 * c * c
    # eps Hess_g U = eps (U'' - gf gU^T - gU gf^T + (gf.gU) Id), gf = -Y / (2 |Y|^2)
    Rh, HYP, HYY = hessians(Y, P, eps)             # HYY = 2|P|^2 proj/R - eps U''; recover eps U'' from it
    PP = P[0].sq() + P[1].sq() + P[2].sq()
    proj = [[((I(1.0) if i == j else I(0.0)) - Yh[i] * Yh[j]) * iR for j in range(3)] for i in range(3)]
    epsU2 = [[PP * proj[i][j] * 2.0 - HYY[i][j] for j in range(3)] for i in range(3)]
    gU = gradU_iv(Y, R, iR, Yh)
    gf = [y * (iR.sq() * (-0.5)) for y in Y]
    gfgU = gf[0] * gU[0] + gf[1] * gU[1] + gf[2] * gU[2]
    Q = []
    for i in range(3):
        row = []
        for j in range(3):
            curv = k3 * (vT2 * PT[i][j] - vT[i] * vT[j])
            hes = epsU2[i][j] - eps * (gf[i] * gU[j] + gU[i] * gf[j]) + eps * gfgU * (1.0 if i == j else 0.0)
            row.append(curv - hes)
        Q.append(row)
    Q = sym(Q)
    mid = np.stack([np.stack([Q[i][j].mid for j in range(3)], -1) for i in range(3)], -2)
    rad = np.stack([np.stack([Q[i][j].wid * 0.5 for j in range(3)], -1) for i in range(3)], -2)
    mu = np.linalg.eigvalsh(mid)[..., -1] + np.sqrt(np.sum(rad ** 2, axis=(-2, -1)))         + 1e-10 * (1 + np.max(np.abs(mid), axis=(-2, -1)))
    # certify lambda_max(Q) <= mu: mu I - Q positive definite
    okc = chol_pd([[I(mu) * (1.0 if i == j else 0.0) - Q[i][j] for j in range(3)] for i in range(3)])
    # fallbacks: a larger mu, then Gershgorin (always rigorous: lambda_max <= max_i Q_ii.hi + sum_j |Q_ij|max)
    rn = np.sqrt(np.sum(rad ** 2, axis=(-2, -1)))
    mu2 = mu + rn + 1e-6 * (1 + np.max(np.abs(mid), axis=(-2, -1)))
    ok2 = chol_pd([[I(mu2) * (1.0 if i == j else 0.0) - Q[i][j] for j in range(3)] for i in range(3)])
    ger = np.max(np.stack([Q[i][i].hi + sum(np.maximum(np.abs(Q[i][j].lo), np.abs(Q[i][j].hi)) for j in range(3) if j != i)
                           for i in range(3)]), 0)
    ger = up(up(ger) * (1 + 1e-15))
    mu = np.where(okc, mu, np.where(ok2, mu2, ger))
    return np.where(mu >= 0, up(mu / c.lo), up(mu / c.hi))


def gradU_iv(Y, R, iR, Yh):
    G = [I(np.zeros(R.lo.shape)) for _ in range(3)]
    for u in EF.UK:
        ud = Y[0] * u[0] + Y[1] * u[1] + Y[2] * u[2]
        perp = [Y[i] - ud * u[i] for i in range(3)]
        if np.all(ud.lo > 0):
            q = (perp[0].sq() + perp[1].sq() + perp[2].sq()) / (R + ud)
            g = [(perp[i] - q * u[i]) * iR for i in range(3)]
        else:
            q = R - ud
            g = [Yh[i] - u[i] for i in range(3)]
        q32 = (q * q.sqrt()).recip()
        for i in range(3):
            G[i] = G[i] - q32 * g[i] * 0.5
    return G


def sturm_boxes(P, steps, m=8):
    """the scalar comparison y'' + W y = 0, y(0) = 1, y'(0) = 0 with W >= w piecewise constant on m pieces per Lohner
    step (steps recorded from tau = 0); returns (ok (nb,), min lower bound of y)"""
    nb = P.lo.shape[0]
    epsI = I(P.lo[:, 0], P.hi[:, 0])
    z, zp = I(np.ones(nb)), I(np.zeros(nb))
    ymin = np.full(nb, np.inf)
    for (ts, h, cs, Rm) in steps:
        for k in range(m):
            sa, sb = h * k / m, h * (k + 1) / m
            S = I(dn(sa), up(sb))
            tP = tpow(S, L.ORDER + 1)
            Z = [peval(cs[i], S) + Rm[i] * tP for i in range(6)]
            Wv = w_upper(Z[0:3], Z[3:6], epsI)
            Lp = float(up(sb - sa))
            pos = Wv > 1e-12
            neg = Wv < -1e-12
            om = np.sqrt(np.abs(np.where(np.isfinite(Wv), Wv, 1.0)))
            omI = I(push3(om, -np.inf), push3(om, np.inf))
            x = om * Lp
            cosI = I(push3(np.cos(x), -np.inf), push3(np.cos(x), np.inf))
            sinI = I(push3(np.sin(x), -np.inf), push3(np.sin(x), np.inf))
            chI = I(push3(np.cosh(x), -np.inf), push3(np.cosh(x), np.inf))
            shI = I(push3(np.sinh(x), -np.inf), push3(np.sinh(x), np.inf))
            # end values (W treated as exactly om^2 with om enclosed; the comparison uses the larger W, so take
            # om at its upper end for W > 0 and lower end for W < 0: conservative choice by monotonicity is
            # replaced here by the enclosure in om, valid since y(L) is enclosed for every om in omI)
            zb_pos = z * cosI + zp * (sinI / omI)
            zpb_pos = zp * cosI - z * (omI * sinI)
            zb_neg = z * chI + zp * (shI / omI)
            zpb_neg = zp * chI + z * (omI * shI)
            zb_lin = z + zp * Lp
            zb = I(np.where(pos, zb_pos.lo, np.where(neg, zb_neg.lo, zb_lin.lo)),
                   np.where(pos, zb_pos.hi, np.where(neg, zb_neg.hi, zb_lin.hi)))
            zpb = I(np.where(pos, zpb_pos.lo, np.where(neg, zpb_neg.lo, zp.lo)),
                    np.where(pos, zpb_pos.hi, np.where(neg, zpb_neg.hi, zp.hi)))
            # lower bound of y inside the piece: y(t) = z cos(om t) + (z'/om) sin(om t) >= z cos(om L) + min(0, z'/om) sin(om L)
            # (for z > 0, om L < pi/2); W < 0: y >= z + min(0, z'/om) sinh(om L); W ~ 0: y >= z + min(0, z') L
            inner_pos = z.lo * cosI.lo + np.minimum(0.0, (zp / omI).lo) * sinI.hi
            inner_neg = z.lo + np.minimum(0.0, (zp / omI).lo) * shI.hi
            inner_lin = z.lo + np.minimum(0.0, zp.lo) * Lp
            inner = np.where(pos, inner_pos, np.where(neg, inner_neg, inner_lin))
            inner = np.where(np.isfinite(Wv) & (x < 1.5) & (z.lo > 0), dn(inner), -np.inf)
            ymin = np.minimum(ymin, inner)
            z, zp = zb, zpb
    ymin = np.minimum(ymin, z.lo)
    return ymin > 0, ymin


def slice_steps(steps, sel):
    return [(ts, h, [[c[sel] for c in comp] for comp in cs], [r[sel] for r in Rm]) for (ts, h, cs, Rm) in steps]


def sturm_adaptive(P, steps, ms=(16, 64)):
    """Sturm test with m pieces per step, re-running the boxes that fail with the next finer m"""
    nb = P.lo.shape[0]
    ok, ymin = np.zeros(nb, bool), np.full(nb, -np.inf)
    todo = np.arange(nb)
    for m in ms:
        if not len(todo):
            break
        Ps = I(P.lo[todo], P.hi[todo])
        o, y = sturm_boxes(Ps, slice_steps(steps, todo), m=m)
        ok[todo], ymin[todo] = o, y
        todo = todo[~o]
    return ok, ymin


def test():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
    a8 = np.array([1.5424624176685366, 0.7066425303908384, -0.30889005565526345])
    p8 = np.array([math.exp(-1.5 * a8[0]), a8[1], a8[2]])
    for d in (0.0, 0.001, 0.002):
        w = d * np.array([0.1, 1, 0.25])
        P = I(np.array([p8 - w]), np.array([p8 + w]))
        ok, ch, tm, npc = check_boxes(P)
        print(f"eight's box half-width {d:g}: (R) {'PROVED' if ok[0] else 'not closed'} {ch[0]}; Lohner {tm[0]:.0f} s, "
              f"check {tm[1]:.0f} s, {npc[0]} pieces", flush=True)


if __name__ == "__main__":
    test()
