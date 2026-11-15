"""Stage 21, step 3: batched C^1 Lohner enclosures of the I-field map (verify/eight_field.py) over boxes of the
parameters p = (eps, phi, M).

State z = (Y1, Y2, Y3, P1, P2, P3, k, m, eps) (eps is carried as a state with eps' = 0), tau in [0, 1]:
    Y' = -4|Y| P,  P' = 2|P|^2 Y/|Y| - eps grad U(Y),  k' = 2|Y||P|^2,  m' = U(Y),  eps' = 0,
    z(0) = (cos(phi) e_I + sin(phi) e_Z, M n_I, 0, 0, eps).
For a batch of parameter boxes the set {z(tau; p) : p in box} is carried as c + A r (c float, A float 9 x 9, r an
interval 9-vector; one set per box, all boxes in numpy arrays), Lohner's QR method with column pivoting by
|column| x width(r). One Taylor step (validated Taylor series of order ORDER with a Picard a priori box and a Lagrange
remainder, eight_field.step logic) gives the enclosure J of the step's derivative over the hull of the set (jets in the
9 state directions); the derivative columns dz/dp (3 of them) are carried as Lohner vector sets updated by J.
Steps are dyadic (multiples of 2^-24), so the elapsed time is exact and tau = 1 is reached exactly.
The start set: A0 = [dz0/dp at the centre | an orthonormal completion], r0 = A0^(-1) (dz0/dp over the box) (p - pc),
evaluated with an interval enclosure of A0^(-1) (mean value form, so the start set is tight).
Also returned: a lower bound of min_k (|Y| - u_k . Y)/|Y| over all a priori boxes (no collision, hypothesis (N)).
    python verify/eight_field_lohner.py test"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eight_field as EF  # noqa: E402
from prove_newton_1212 import I, dn, ihull, peval, stack, tpow, up  # noqa: E402

N = 9
ORDER = int(os.environ.get("LORDER", "20"))
STEPF = float(os.environ.get("LSTEPF", "0.1"))
HMAX = float(os.environ.get("LHMAX", "0.05"))
DY = 2.0 ** -24
EPSM = 2.0 ** -53


# ---------------------------------------------------------------- batched interval linear algebra (arrays (nb, n, m))
def IA(a):
    a = np.asarray(a, float)
    return I(a, a.copy())


def imm(M, Nn):
    """interval (nb, n, k) times interval (nb, k, m)"""
    k = M.lo.shape[2]
    acc = None
    for l in range(k):
        t = I(M.lo[:, :, l:l + 1], M.hi[:, :, l:l + 1]) * I(Nn.lo[:, l:l + 1, :], Nn.hi[:, l:l + 1, :])
        acc = t if acc is None else acc + t
    return acc


def imv(M, v):
    """interval (nb, n, k) times interval (nb, k)"""
    k = M.lo.shape[2]
    acc = None
    for l in range(k):
        t = I(M.lo[:, :, l], M.hi[:, :, l]) * I(v.lo[:, l:l + 1], v.hi[:, l:l + 1])
        acc = t if acc is None else acc + t
    return acc


def inv_enclosure(A):
    """interval (nb, n, n) containing A^(-1) for float matrices A (nb, n, n)"""
    n = A.shape[1]
    B0 = np.linalg.inv(A)
    E = imm(IA(B0), IA(A))
    eye = np.broadcast_to(np.eye(n), A.shape)
    E = IA(eye) - E
    mag = np.maximum(np.abs(E.lo), np.abs(E.hi))
    e = up(np.max(np.sum(mag, axis=2), axis=1) * (1 + 4 * n * EPSM))
    if np.any(e >= 0.5):
        raise RuntimeError("inverse enclosure failed")
    nb = up(np.max(np.sum(np.abs(B0), axis=2), axis=1) * (1 + 4 * n * EPSM))
    d = up(up(e * nb) / dn(1.0 - e))[:, None, None]
    return I(dn(B0 - d), up(B0 + d))


def pivot(M, r):
    Mm = M.mid
    w = r.wid * np.linalg.norm(Mm, axis=1)
    order = np.argsort(-w, axis=1)
    o3 = np.broadcast_to(order[:, None, :], M.lo.shape)
    M2 = I(np.take_along_axis(M.lo, o3, 2), np.take_along_axis(M.hi, o3, 2))
    r2 = I(np.take_along_axis(r.lo, order, 1), np.take_along_axis(r.hi, order, 1))
    return M2, r2, np.take_along_axis(Mm, o3, 2)


def lohner_update(J, A, r, phic):
    """the set c + A r mapped by a map with derivative enclosure J over its hull and the image phic of c: new (c, A, r)"""
    cn = phic.mid
    M = imm(J, IA(A))
    M, r, Mm = pivot(M, r)
    Q, _ = np.linalg.qr(Mm)
    B = inv_enclosure(Q)
    rn = imv(imm(B, M), r) + imv(B, phic - cn)
    return cn, Q, rn


def hull(c, A, r):
    return IA(c) + imv(IA(A), r)


# ---------------------------------------------------------------- the vector field: series with eps as a state
def series9(y, order):
    cs = EF.series(y[:8], y[8], order)
    zero = I(np.zeros_like(y[8].lo), np.zeros_like(y[8].hi))
    return cs + [[y[8]] + [zero] * order]


def taylor_step(y, h):
    """y: list of 9 jets (dj, nb); returns (coefficients, remainder coefficient, a priori box) or None"""
    try:
        cs = series9(y, ORDER)
    except (ZeroDivisionError, ValueError):
        return None
    H = I(0.0, h)
    g = [peval(c, H) for c in cs]

    def inflate(B, Yv):
        ex = np.maximum(np.abs(B.lo - Yv.lo), np.abs(B.hi - Yv.hi)) + B.wid
        return I(dn(B.lo - 0.05 * ex - 1e-13), up(B.hi + 0.05 * ex + 1e-13))
    B = [inflate(ihull(y[i], g[i]), y[i]) for i in range(N)]
    for _ in range(60):
        try:
            f = series9(B, 1)
        except (ZeroDivisionError, ValueError):
            return None
        nn = [y[i] + f[i][1] * H for i in range(N)]
        if all(np.all(nn[i].lo > B[i].lo) and np.all(nn[i].hi < B[i].hi) for i in range(N)):
            break
        B = [inflate(ihull(nn[i], y[i]), y[i]) for i in range(N)]
    else:
        return None
    try:
        rr = series9(B, ORDER + 1)
    except (ZeroDivisionError, ValueError):
        return None
    return cs, [rr[i][ORDER + 1] for i in range(N)], B


def at(cs, Rm, h):
    H = I(h)
    tP = tpow(H, ORDER + 1)
    return [peval(cs[i], H) + Rm[i] * tP for i in range(N)]


def jets_of(X):
    """X: interval (nb, 9) -> 9 jets (10, nb) with identity derivative parts"""
    nb = X.lo.shape[0]
    out = []
    for i in range(N):
        parts_lo = [X.lo[:, i]] + [np.full(nb, 1.0 if j == i else 0.0) for j in range(N)]
        out.append(I(np.stack(parts_lo, 0), np.stack([X.hi[:, i]] + parts_lo[1:], 0)))
    return out


def values_of(c):
    """float (nb, 9) -> 9 value-only jets (1, nb)"""
    return [I(c[None, :, i].copy()) for i in range(N)]


def min_q(B):
    Y = [I(b.lo[0], b.hi[0]) for b in B[0:3]]
    R = (Y[0].sq() + Y[1].sq() + Y[2].sq()).sqrt()
    m = np.full(R.lo.shape, np.inf)
    for u in EF.UK:
        q = R - (Y[0] * u[0] + Y[1] * u[1] + Y[2] * u[2])
        m = np.minimum(m, dn(q.lo / up(R.hi)))
    return m


def choose_h(c):
    cs = series9(values_of(c), ORDER)
    mags = [np.max(np.abs(np.stack([comp[k].mid for comp in cs]))) for k in range(ORDER + 1)]
    rho = min(mags[k] ** (-1.0 / k) for k in range(ORDER // 2, ORDER + 1) if mags[k] > 0)
    return min(HMAX, STEPF * rho)


# ---------------------------------------------------------------- start set and flow
def start_set(P):
    """P: interval (nb, 3) boxes of (eps, phi, M). Returns c, A, r and the 3 derivative column sets."""
    nb = P.lo.shape[0]
    pc = P.mid
    Pc = IA(pc)
    # z0 at the centre (enclosed) and dz0/dp over the box (interval 9 x 3)
    def z0_and_D(Q):
        sn, cs_ = EF.isincos(I(Q.lo[:, 1], Q.hi[:, 1]))
        z = [cs_ * EF.EI[i] + sn * EF.EZ[i] for i in range(3)]
        Mq = I(Q.lo[:, 2], Q.hi[:, 2])
        z += [Mq * EF.NI[i] for i in range(3)]
        zero = I(np.zeros(nb))
        z += [zero, zero, I(Q.lo[:, 0], Q.hi[:, 0])]
        D_lo, D_hi = np.zeros((nb, N, 3)), np.zeros((nb, N, 3))
        D_lo[:, 8, 0] = D_hi[:, 8, 0] = 1.0
        for i in range(3):
            d = -sn * EF.EI[i] + cs_ * EF.EZ[i]
            D_lo[:, i, 1], D_hi[:, i, 1] = d.lo, d.hi
            D_lo[:, 3 + i, 2], D_hi[:, 3 + i, 2] = EF.NI[i].lo, EF.NI[i].hi
        return I(np.stack([v.lo for v in z], 1), np.stack([v.hi for v in z], 1)), I(D_lo, D_hi)
    zc, _ = z0_and_D(Pc)
    _, DB = z0_and_D(P)
    c = zc.mid
    Dm = DB.mid
    A = np.empty((nb, N, N))
    for b in range(nb):
        q, _ = np.linalg.qr(np.concatenate([Dm[b], np.eye(N)], 1))
        A[b] = np.concatenate([Dm[b], q[:, 3:N]], 1)
    Ainv = inv_enclosure(A)
    dP = P - Pc
    r = imv(imm(Ainv, DB), dP) + imv(Ainv, zc - c)
    cols = []
    for j in range(3):
        col = I(DB.lo[:, :, j], DB.hi[:, :, j])
        cj = col.mid
        cols.append((cj, np.broadcast_to(np.eye(N), (nb, N, N)).copy(), col - cj))
    return c, A, r, cols


def flow(P, verbose=False):
    """Lohner flow of the parameter boxes P (interval (nb, 3)) to tau = 1. Returns (z(1) enclosure (nb, 9),
    dz/dp enclosure (nb, 9, 3), min angular collision distance (nb,), steps)."""
    c, A, r, cols = start_set(P)
    t, nsteps, mq = 0.0, 0, np.full(P.lo.shape[0], np.inf)
    shrink = 1.0
    while t < 1.0:
        h = min(choose_h(c) * shrink, 1.0 - t)
        h = math.floor(h / DY) * DY if h < 1.0 - t else 1.0 - t
        X = hull(c, A, r)
        rX = taylor_step(jets_of(X), h)
        rc = taylor_step(values_of(c), h)
        if rX is None or rc is None:
            shrink /= 2
            if h < 1e-6:
                raise RuntimeError(f"step collapsed at tau = {t}")
            continue
        shrink = min(1.0, shrink * 2)
        yX = at(rX[0], rX[1], h)
        J = I(np.stack([np.stack([yX[i].lo[1 + j] for j in range(N)], 1) for i in range(N)], 1),
              np.stack([np.stack([yX[i].hi[1 + j] for j in range(N)], 1) for i in range(N)], 1))
        yc = at(rc[0], rc[1], h)
        phic = I(np.stack([v.lo[0] for v in yc], 1), np.stack([v.hi[0] for v in yc], 1))
        mq = np.minimum(mq, min_q(rX[2]))
        c, A, r = lohner_update(J, A, r, phic)
        newcols = []
        for (cj, Aj, rj) in cols:
            ph = imv(J, IA(cj)[:, :, None] if False else IA(cj)) if False else _jv(J, cj)
            newcols.append(lohner_update(J, Aj, rj, ph))
        cols = newcols
        t += h                      # exact: dyadic steps
        nsteps += 1
        if verbose and nsteps % 10 == 0:
            print(f"    tau {t:.4f}: max width z {np.max(hull(c, A, r).wid):.2e}", flush=True)
    Z = hull(c, A, r)
    Dl = np.stack([hull(*cj).lo for cj in cols], 2)
    Dh = np.stack([hull(*cj).hi for cj in cols], 2)
    return Z, I(Dl, Dh), mq, nsteps


def _jv(J, cj):
    """J (interval nb x 9 x 9) applied to the float vectors cj (nb, 9)"""
    return imv(J, IA(cj))


# ---------------------------------------------------------------- tests
def test():
    a8 = np.array([1.5424624176685366, 0.7066425303908384, -0.30889005565526345])
    eps8 = math.exp(-1.5 * a8[0])
    p8 = np.array([eps8, a8[1], a8[2]])
    yf = EF.flow_f(eps8, a8[1], a8[2])
    for d in (0.0, 1e-3, 1e-2, 3e-2):
        P = I(np.array([p8 - d * np.array([0.1, 1, 0.25])]), np.array([p8 + d * np.array([0.1, 1, 0.25])]))
        t0 = time.time()
        Z, D, mq, ns = flow(P)
        ok = all(Z.lo[0, i] <= yf[i] <= Z.hi[0, i] for i in range(8)) if d == 0 else None
        print(f"box half-width {d:g} (eps x0.1, phi x1, M x0.25): {ns} steps, {time.time() - t0:.1f} s; width z(1) "
              f"max {np.max(Z.wid):.2e}; width dz/dp max {np.max(D.wid):.2e}; min collision distance {mq[0]:.3f}"
              + (f"; float flow inside: {ok}" if ok is not None else ""), flush=True)
    # batch timing
    nb = 200
    rng = np.random.default_rng(0)
    pc = p8 + rng.uniform(-1, 1, (nb, 3)) * np.array([0.01, 0.2, 0.05])
    d = np.array([0.001, 0.01, 0.0025])
    t0 = time.time()
    Z, D, mq, ns = flow(I(pc - d, pc + d))
    print(f"batch of {nb} boxes (half-widths {d}): {time.time() - t0:.1f} s, {ns} steps; max width z {np.max(Z.wid):.2e}, "
          f"dz/dp {np.max(D.wid):.2e}; min collision distance {mq.min():.3f}")


if __name__ == "__main__":
    test()
