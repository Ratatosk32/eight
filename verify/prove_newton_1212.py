"""Computer-assisted proof that the Newtonian planar three-body problem with zero angular momentum has a periodic
solution (periodic up to rotation) with reduced syzygy sequence 1212..., for given masses.

Set-up. Masses m, potential U = sum m_i m_j / r_ij. For (s, k) let y0(s, k) be the collinear configuration with body 1
at relative position s between bodies 2 and 3 (bodies at x = [s, 0, 1], centre of mass at 0, scaled to I = 1) and all
velocities perpendicular to the line, v = k n / |n|_m with n = m x (m x), so that P = 0 and, exactly, L = 0. Such a
state is fixed by the reversor (q, v) -> (R q, -R v), R the reflection in the line. If the solution through y0(s, k)
reaches at time t > 0 another state fixed by a reversor (collinear, all velocities perpendicular to the line), it is
periodic up to a rotation, with period 2t. We look for a zero of
    G(s, k, t) = ( area(q(t)), (v2 - v1).(q3 - q1), (v3 - v1).(q3 - q1) )
(collinear, both relative along-line velocities zero; with P = 0 all velocities are then perpendicular to the line).

Proof. (1) Validated integration: Taylor series of order P in outward-rounded interval arithmetic (IEEE +, -, *, /,
sqrt only), a priori enclosures by a Picard test, Lagrange remainders; first-order jets carry the derivatives with
respect to (s, k); the time derivative is f(y) at the final enclosure. (2) The Krawczyk test on a box X around the
numerical zero: K(X) = c - A G(c) + (I - A DG(X))(X - c) inside int X proves a unique zero in X. (3) Along [0, t] for
the whole box: no collision (smallest distance enclosure), the area does not vanish on (0, t) (interior steps exclude
0; the first and last steps by the sign of its time derivative), and at t body 2 lies between bodies 1 and 3. Then by
the reversing symmetry the syzygy sequence over a period is 1, 2 and the reduced sequence is 1212...

    python verify/prove_newton_1212.py m1,m2,m3 [certificate.json]"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PORDER = int(os.environ.get("PORDER", "16"))     # Taylor order
STEPF = float(os.environ.get("STEPF", "0.15"))   # step = STEPF x the estimated radius of convergence
D = 2           # jet directions: s, k
INF = np.inf


# ---------- fast outward-rounded intervals on numpy arrays ----------
def dn(x):
    return np.nextafter(x, -INF)


def up(x):
    return np.nextafter(x, INF)


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, float)
        self.hi = self.lo if hi is None else np.asarray(hi, float)

    def __add__(self, o):
        o = _i(o)
        return I(dn(self.lo + o.lo), up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, o):
        o = _i(o)
        return I(dn(self.lo - o.hi), up(self.hi - o.lo))

    def __rsub__(self, o):
        return _i(o) - self

    def __mul__(self, o):
        o = _i(o)
        a, b, c, d = self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi
        return I(dn(np.minimum(np.minimum(a, b), np.minimum(c, d))), up(np.maximum(np.maximum(a, b), np.maximum(c, d))))

    __rmul__ = __mul__

    def recip(self):
        if np.any(~np.isfinite(self.lo) | ~np.isfinite(self.hi)) or np.any((self.lo <= 0) & (self.hi >= 0)):
            raise ZeroDivisionError("interval contains 0")
        return I(dn(1.0 / self.hi), up(1.0 / self.lo))

    def __truediv__(self, o):
        return self * _i(o).recip()

    def sq(self):
        a, b = self.lo * self.lo, self.hi * self.hi
        z = (self.lo <= 0) & (self.hi >= 0)
        return I(np.where(z, 0.0, dn(np.minimum(a, b))), up(np.maximum(a, b)))

    def sqrt(self):
        if np.any(self.lo < 0):
            raise ValueError("sqrt of a negative interval")
        return I(dn(np.sqrt(self.lo)), up(np.sqrt(self.hi)))

    def __getitem__(self, k):
        return I(self.lo[k], self.hi[k])

    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def wid(self):
        return self.hi - self.lo


def _i(x):
    return x if isinstance(x, I) else I(x)


def ihull(a, b):
    return I(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def stack(parts, axis):
    return I(np.stack([p.lo for p in parts], axis), np.stack([p.hi for p in parts], axis))


# ---------- first-order jets: leading axis 0 = value, 1.. = derivatives (any number of directions) ----------
def _cat(v, d):
    return I(np.concatenate([v.lo[None], d.lo], 0), np.concatenate([v.hi[None], d.hi], 0))


def _lead(x):
    return I(x.lo[None], x.hi[None])


def jmul(a, b):
    return _cat(a[0] * b[0], a[0:1] * b[1:] + a[1:] * b[0:1])


def jrecip(a):
    r = a[0].recip()
    return _cat(r, -(a[1:] * _lead(r.sq())))


def jpow_m32(a):
    v = a[0]
    p = (v * v.sqrt()).recip()
    dp = p * v.recip() * (-1.5)
    return _cat(p, a[1:] * _lead(dp))


def jsq(a):
    """a * a without the dependency loss of the value part"""
    return _cat(a[0].sq(), (a[1:] * _lead(a[0])) * 2.0)


def jsqrt(a):
    s = a[0].sqrt()
    return _cat(s, a[1:] * _lead((s * 2.0).recip()))


def jscale(a, c):
    """jet times a constant interval or float (broadcast over the jet axis)"""
    return a * c


PI = np.array([0, 1, 0])
PJ = np.array([1, 2, 2])


def series(q0, v0, m, order):
    """Taylor coefficients up to `order` of the Newtonian three-body flow from the jet state (q0, v0), each of shape
    (D+1, 3 bodies, 2 coordinates). Returns lists qs, vs of length order+1."""
    qs, vs = [q0], [v0]
    d, r2, w = [], [], []
    mI = np.array([m[0], m[1], m[0]])
    mJ = np.array([m[1], m[2], m[2]])
    for k in range(order):
        dk = I(qs[k].lo[:, PJ] - 0, qs[k].hi[:, PJ]) - I(qs[k].lo[:, PI], qs[k].hi[:, PI])
        d.append(dk)
        s = None
        for l in range(k + 1):
            t = jsq(d[l]) if 2 * l == k else jmul(d[l], d[k - l])
            t = I(dn(t.lo[..., 0] + t.lo[..., 1]), up(t.hi[..., 0] + t.hi[..., 1]))
            s = t if s is None else s + t
        r2.append(s)                                           # (D+1, 3)
        if k == 0:
            w.append(jpow_m32(r2[0]))
        else:
            s = None
            for l in range(1, k + 1):
                t = jmul(r2[l], w[k - l]) * float(-0.5 * l - k)
                s = t if s is None else s + t
            w.append(jmul(s, jrecip(r2[0] * float(k))))
        s = None
        for l in range(k + 1):
            wl = w[k - l]
            t = jmul(d[l], I(wl.lo[..., None], wl.hi[..., None]))
            s = t if s is None else s + t
        dw = s                                                 # (D+1, 3 pairs, 2)
        # accelerations: body i += m_j dw, body j -= m_i dw (pairs 01, 12, 02)
        a0 = dw[:, 0] * m[1] + dw[:, 2] * m[2]
        a1 = dw[:, 1] * m[2] - dw[:, 0] * m[0]
        a2 = -(dw[:, 1] * m[1]) - dw[:, 2] * m[0]
        acc = stack([a0, a1, a2], 1)
        qs.append(vs[k] * (1.0 / (k + 1)) if False else vs[k] / float(k + 1))
        vs.append(acc / float(k + 1))
    return qs, vs


def peval(cs, tau):
    r = cs[-1]
    for c in reversed(cs[:-1]):
        r = r * tau + c
    return r


def step(q, v, m, h):
    """Validated step of length h from the jet state (q, v): returns (qs, vs, Rq, Rv, Bq, Bv) with
    y(tau) in sum_k c_k tau^k + R tau^(P+1) for tau in [0, h], and the a priori box B."""
    try:
        qs, vs = series(q, v, m, PORDER)
    except (ZeroDivisionError, ValueError):
        return None
    H = I(0.0, h)
    gq, gv = peval(qs, H), peval(vs, H)
    def inflate(B, Y):
        """widen B by 5 % of its excursion from the initial value Y, plus a tiny absolute amount"""
        ex = np.maximum(np.abs(B.lo - Y.lo), np.abs(B.hi - Y.hi)) + B.wid
        return I(dn(B.lo - 0.05 * ex - 1e-13), up(B.hi + 0.05 * ex + 1e-13))

    Bq, Bv = inflate(ihull(q, gq), q), inflate(ihull(v, gv), v)
    for it in range(60):
        try:
            fq, fv = series(Bq, Bv, m, 1)
        except (ZeroDivisionError, ValueError):
            return None
        nq, nv = q + fq[1] * H, v + fv[1] * H   # the Picard image of B
        if (np.all(nq.lo > Bq.lo) and np.all(nq.hi < Bq.hi) and np.all(nv.lo > Bv.lo) and np.all(nv.hi < Bv.hi)):
            break
        # iterate the (contracting) Picard map, slightly inflated, keeping what was already covered
        Bq, Bv = inflate(ihull(nq, q), q), inflate(ihull(nv, v), v)
    else:
        return None
    try:
        rq, rv = series(Bq, Bv, m, PORDER + 1)
    except (ZeroDivisionError, ValueError):
        return None
    return qs, vs, rq[PORDER + 1], rv[PORDER + 1], Bq, Bv


def tpow(tau, n):
    r = tau
    for _ in range(n - 1):
        r = r * tau
    return r


def start_state(S, K, m):
    """y0 as jets in (s, k) over the intervals S, K (exact formulas in interval arithmetic)."""
    def J(val, ds=0.0, dk=0.0):
        val = _i(val)
        return stack([val, I(np.broadcast_to(ds, val.lo.shape) * 1.0), I(np.broadcast_to(dk, val.lo.shape) * 1.0)], 0)
    s, k = J(S, 1.0, 0.0), J(K, 0.0, 1.0)
    one, zero = J(1.0), J(0.0)
    M = m[0] + m[1] + m[2]
    xs = [s, zero, one]
    cm = (xs[0] * m[0] + xs[1] * m[1] + xs[2] * m[2]) / M
    xs = [x - cm for x in xs]
    Iq = jmul(xs[0], xs[0]) * m[0] + jmul(xs[1], xs[1]) * m[1] + jmul(xs[2], xs[2]) * m[2]
    sc = jrecip(jsqrt(Iq))
    xs = [jmul(x, sc) for x in xs]
    mx = [xs[i] * m[i] for i in range(3)]
    n = [mx[2] * m[1] - mx[1] * m[2], mx[0] * m[2] - mx[2] * m[0], mx[1] * m[0] - mx[0] * m[1]]
    nn = jsqrt(jmul(n[0], n[0]) * m[0] + jmul(n[1], n[1]) * m[1] + jmul(n[2], n[2]) * m[2])
    fac = jmul(k, jrecip(nn))
    vy = [jmul(ni, fac) for ni in n]
    q = stack([stack([xs[b], zero], -1) for b in range(3)], 1)
    v = stack([stack([zero, vy[b]], -1) for b in range(3)], 1)
    return q, v


def G_of(q, v):
    """the three conditions on jets (shape (D+1,))"""
    ax, ay = q[:, 1, 0] - q[:, 0, 0], q[:, 1, 1] - q[:, 0, 1]
    bx, by = q[:, 2, 0] - q[:, 0, 0], q[:, 2, 1] - q[:, 0, 1]
    area = jmul(ax, by) - jmul(ay, bx)
    g2 = jmul(v[:, 1, 0] - v[:, 0, 0], bx) + jmul(v[:, 1, 1] - v[:, 0, 1], by)
    g3 = jmul(v[:, 2, 0] - v[:, 0, 0], bx) + jmul(v[:, 2, 1] - v[:, 0, 1], by)
    return [area, g2, g3]


def area_val(q):
    ax, ay = q[1, 0] - q[0, 0], q[1, 1] - q[0, 1]
    bx, by = q[2, 0] - q[0, 0], q[2, 1] - q[0, 1]
    return ax * by - ay * bx


def integrate(S, K, Tlo, Thi, m, h0=0.01, record=False):
    """Validated flow from y0(S, K) to the time interval [Tlo, Thi]. Returns the jet state there, the time derivative
    of the state (values only), and, if record, the per-step data (t, h, polynomial value coefficients, B)."""
    q, v = start_state(S, K, m)
    return flow_from(q, v, Tlo, Thi, m, h0, record)


def flow_from(q, v, Tlo, Thi, m, h0=0.01, record=False):
    """Validated flow of the jet state (q, v) to the time interval [Tlo, Thi] (Tlo > 0)."""
    t = 0.0
    tl = th = 0.0            # rigorous enclosure of the elapsed time (the sum of the step lengths)
    steps = []
    hmax = h0
    shrink = 1.0             # halved after a failed step, reset after a success
    while True:
        # step size from the radius of convergence of the series at the current value (no rigour needed here)
        try:
            vq, vv = series(I(q.mid[:1]), I(v.mid[:1]), m, PORDER)
            mags = [max(float(np.max(np.abs(a.mid))), float(np.max(np.abs(b.mid)))) for a, b in zip(vq, vv)]
            rho = min(mags[k] ** (-1.0 / k) for k in range(PORDER // 2, PORDER + 1) if mags[k] > 0)
            h0 = min(hmax, STEPF * rho) * shrink
        except (ZeroDivisionError, ValueError):
            h0 = hmax * shrink
        last = (Tlo - t) <= h0
        hstep = up(Thi - tl) if last else h0
        res = step(q, v, m, hstep)
        if res is None:
            shrink /= 2
            if h0 < 1e-6:
                raise RuntimeError(f"step size collapsed at t = {t}")
            continue
        shrink = 1.0
        qs, vs, Rq, Rv, Bq, Bv = res
        if record:
            steps.append((t, hstep, [c[0] for c in qs], [c[0] for c in vs], Rq[0], Rv[0], Bq[0], Bv[0]))
        if last:
            if float(dn(Tlo - th)) < 0:
                raise RuntimeError("the elapsed time overshoots the target time")
            tau = I(float(dn(Tlo - th)), up(Thi - tl))   # the remaining time, enclosed
            tP = tpow(tau, PORDER + 1)
            qT = peval(qs, tau) + Rq * tP
            vT = peval(vs, tau) + Rv * tP
            fq, fv = series(qT, vT, m, 1)
            return qT, vT, (fq[1][0], fv[1][0]), steps
        tP = tpow(I(hstep), PORDER + 1)
        q = peval(qs, I(hstep)) + Rq * tP
        v = peval(vs, I(hstep)) + Rv * tP
        t += hstep
        tl, th = float(dn(tl + hstep)), float(up(th + hstep))


# ---------- multiple shooting ----------
EPS = 2.0 ** -53


def seed(x):
    """jets with 12 directions (identity) at the state interval x (shape (12,)): returns q, v of shape (13, 3, 2)"""
    lo = np.zeros((13, 12))
    hi = np.zeros((13, 12))
    lo[0], hi[0] = x.lo, x.hi
    for j in range(12):
        lo[1 + j, j] = hi[1 + j, j] = 1.0
    return (I(lo[:, :6].reshape(13, 3, 2), hi[:, :6].reshape(13, 3, 2)),
            I(lo[:, 6:].reshape(13, 3, 2), hi[:, 6:].reshape(13, 3, 2)))


def point(x):
    """jets with no direction at the state interval x"""
    return I(x.lo[:6].reshape(1, 3, 2), x.hi[:6].reshape(1, 3, 2)), I(x.lo[6:].reshape(1, 3, 2), x.hi[6:].reshape(1, 3, 2))


def flat(q, v):
    """(D+1, 12) from the jets q, v"""
    n = q.lo.shape[0]
    return I(np.concatenate([q.lo.reshape(n, 6), v.lo.reshape(n, 6)], 1),
             np.concatenate([q.hi.reshape(n, 6), v.hi.reshape(n, 6)], 1))


def amul(A, B):
    """float matrix A times interval matrix or vector B, with a bound on the rounding errors of the float products"""
    Ap, An = np.maximum(A, 0), np.minimum(A, 0)
    lo = Ap @ B.lo + An @ B.hi
    hi = Ap @ B.hi + An @ B.lo
    n = A.shape[1]
    err = (n + 4) * EPS * (np.abs(A) @ np.maximum(np.abs(B.lo), np.abs(B.hi))) + 1e-300
    return I(dn(lo - err), up(hi + err))


def g_time_derivative(q0, v0, fq, fv):
    """d/dt of the three conditions along the flow, from the state values and f"""
    ax, ay = q0[1, 0] - q0[0, 0], q0[1, 1] - q0[0, 1]
    bx, by = q0[2, 0] - q0[0, 0], q0[2, 1] - q0[0, 1]
    dax, day = fq[1, 0] - fq[0, 0], fq[1, 1] - fq[0, 1]
    dbx, dby = fq[2, 0] - fq[0, 0], fq[2, 1] - fq[0, 1]
    out = [dax * by + ax * dby - day * bx - ay * dbx]
    for b in (1, 2):
        ux, uy = v0[b, 0] - v0[0, 0], v0[b, 1] - v0[0, 1]
        dux, duy = fv[b, 0] - fv[0, 0], fv[b, 1] - fv[0, 1]
        out.append(dux * bx + ux * dbx + duy * by + uy * dby)
    return out


def rho_est(q, v, m):
    """radius of convergence of the series at the value (an estimate, for choosing segment lengths)"""
    vq, vv = series(I(q.mid[:1]), I(v.mid[:1]), m, PORDER)
    mags = [max(float(np.max(np.abs(a.mid))), float(np.max(np.abs(b.mid)))) for a, b in zip(vq, vv)]
    return min(mags[k] ** (-1.0 / k) for k in range(PORDER // 2, PORDER + 1) if mags[k] > 0)


def segment_times(m, s0, k0, t0, per=3, hmax=0.01):
    """segment boundaries along the centre orbit: each segment about `per` adaptive steps long"""
    q, v = start_state(I(s0), I(k0), m)
    q, v = I(q.lo[:1], q.hi[:1]), I(v.lo[:1], v.hi[:1])
    times = [0.0]
    while times[-1] < t0:
        tau = min(per * min(hmax, STEPF * rho_est(q, v, m)), t0 - times[-1])
        if t0 - times[-1] - tau < 0.3 * tau:   # do not leave a sliver for the last (variable) segment
            break
        q, v, _, _ = flow_from(q, v, tau, tau, m)
        x = flat(q, v)[0]
        q, v = point(I(x.mid))
        times.append(times[-1] + tau)
    return times


def evaluate(m, c, taus, delta, record=False):
    """F at the centre c (point starts) and DF over the box c +- delta. Unknowns c = (s, k, x_1..x_{N-1}, t);
    equations: phi_{tau_1}(y0(s,k)) - x_1, phi_{tau_i}(x_{i-1}) - x_i, g(phi_{t - T_{N-1}}(x_{N-1}))."""
    N = len(taus) + 1                 # taus: the N-1 fixed segment lengths
    n = 12 * (N - 1) + 3
    Tf = I(0.0)
    for tau in taus:
        Tf = Tf + I(tau)              # T_{N-1}, enclosed
    s0, k0, t0 = c[0], c[1], c[-1]
    xs = [c[2 + 12 * i: 2 + 12 * (i + 1)] for i in range(N - 1)]
    # F at the centre
    Fc = []
    q, v = start_state(I(s0), I(k0), m)
    q, v = I(q.lo[:1], q.hi[:1]), I(v.lo[:1], v.hi[:1])
    qe, ve, _, _ = flow_from(q, v, taus[0], taus[0], m)
    Fc.append(flat(qe, ve)[0] - I(xs[0]))
    for i in range(1, N - 1):
        qe, ve, _, _ = flow_from(*point(I(xs[i - 1])), taus[i], taus[i], m)
        Fc.append(flat(qe, ve)[0] - I(xs[i]))
    tl, th = float(dn(t0 - Tf.hi)), float(up(t0 - Tf.lo))
    qe, ve, _, _ = flow_from(*point(I(xs[N - 2])), tl, th, m)
    Fc.append(stack([g[0] for g in G_of(qe, ve)], 0))
    Fc = I(np.concatenate([f.lo for f in Fc]), np.concatenate([f.hi for f in Fc]))
    # DF over the box
    X = I(dn(c - delta), up(c + delta))
    DFlo = np.zeros((n, n))
    DFhi = np.zeros((n, n))
    steps_all = []
    qe, ve, _, st = flow_from(*start_state(X[0], X[1], m), taus[0], taus[0], m, record=record)
    steps_all.append(st)
    J1 = flat(qe, ve)
    DFlo[0:12, 0:2], DFhi[0:12, 0:2] = J1.lo[1:3].T, J1.hi[1:3].T
    for i in range(N - 1):
        r, col = slice(12 * i, 12 * i + 12), slice(2 + 12 * i, 2 + 12 * i + 12)
        DFlo[r, col] = DFhi[r, col] = -np.eye(12)
    for i in range(1, N - 1):
        Xi = X[2 + 12 * (i - 1): 2 + 12 * i]
        qe, ve, _, st = flow_from(*seed(Xi), taus[i], taus[i], m, record=record)
        steps_all.append(st)
        Ji = flat(qe, ve)
        r, col = slice(12 * i, 12 * i + 12), slice(2 + 12 * (i - 1), 2 + 12 * i)
        DFlo[r, col], DFhi[r, col] = Ji.lo[1:].T, Ji.hi[1:].T
    XN = X[2 + 12 * (N - 2): 2 + 12 * (N - 1)]
    tl, th = float(dn(X[n - 1].lo - Tf.hi)), float(up(X[n - 1].hi - Tf.lo))
    qe, ve, (fq, fv), st = flow_from(*seed(XN), tl, th, m, record=record)
    steps_all.append(st)
    Gj = G_of(qe, ve)
    gt = g_time_derivative(qe[0], ve[0], fq, fv)
    col = slice(2 + 12 * (N - 2), 2 + 12 * (N - 1))
    for a in range(3):
        DFlo[n - 3 + a, col], DFhi[n - 3 + a, col] = Gj[a].lo[1:], Gj[a].hi[1:]
        DFlo[n - 3 + a, n - 1], DFhi[n - 3 + a, n - 1] = gt[a].lo, gt[a].hi
    return Fc, I(DFlo, DFhi), X, steps_all, (qe, ve)


def main_ms(m, s0, k0, t0, per=3, deltas=(1e-11, 1e-10, 1e-9, 1e-8, 1e-7)):
    taus_t = segment_times(m, s0, k0, t0, per)
    taus = [b - a for a, b in zip(taus_t[:-1], taus_t[1:])]
    N = len(taus) + 1
    # centre: states at the segment ends from the numerical start, then Newton on the whole system
    xs = []
    q, v = start_state(I(s0), I(k0), m)
    q, v = I(q.lo[:1], q.hi[:1]), I(v.lo[:1], v.hi[:1])
    for tau in taus:
        q, v, _, _ = flow_from(q, v, tau, tau, m)
        x = flat(q, v)[0]
        xs.append(x.mid)
        q, v = point(I(x.mid))
    c = np.concatenate([[s0, k0]] + xs + [[t0]])
    n = c.size
    for it in range(4):
        Fc, DF, _, _, _ = evaluate(m, c, taus, 0.0)
        c = c - np.linalg.solve(DF.mid, Fc.mid)
    Fc, _, _, _, _ = evaluate(m, c, taus, 0.0)
    print(f"  multiple shooting with {N} segments ({n} unknowns): after Newton |F(c)| <= "
          f"{np.max(np.maximum(np.abs(Fc.lo), np.abs(Fc.hi))):.1e}", flush=True)
    for delta in deltas:
        Fc, DF, X, steps_all, fin = evaluate(m, c, taus, delta, record=True)
        A = np.linalg.inv(DF.mid)
        AF = amul(A, I(Fc.lo[:, None], Fc.hi[:, None]))
        ADF = amul(A, DF)
        M = I(dn(np.eye(n) - ADF.hi), up(np.eye(n) - ADF.lo))
        Mabs = np.maximum(np.abs(M.lo), np.abs(M.hi))
        r = up(np.maximum(c - X.lo, X.hi - c))      # the true radius of X (the outward rounding widens it)
        rad = up(Mabs @ r * (1 + (n + 4) * EPS) + 1e-300)
        Klo = dn(dn(c - AF.hi[:, 0]) - rad)
        Khi = up(up(c - AF.lo[:, 0]) + rad)
        inside = bool(np.all(Klo > X.lo) and np.all(Khi < X.hi))
        worst = float(np.max(np.maximum(c - Klo, Khi - c)) / delta)
        print(f"  delta {delta:.0e}: K inside X: {inside} (largest K half-width / delta {worst:.3f}; "
              f"max |A F(c)| {float(np.max(np.abs(AF.mid))):.1e}, ||I - A DF(X)|| {float(np.max(Mabs.sum(1))):.3f})",
              flush=True)
        if inside:
            return True, X, steps_all, fin
    return False, None, None, (None, None)


def path_checks(steps_all):
    """no collision, syzygies only at the two ends, using the a priori boxes of every step"""
    rmin = math.inf
    flagged = []
    allsteps = [s for seg in steps_all for s in seg]
    for idx, (t, h, qcs, vcs, Rq, Rv, Bq, Bv) in enumerate(allsteps):
        for i, j in ((0, 1), (1, 2), (0, 2)):
            dd = (Bq[j, 0] - Bq[i, 0]).sq() + (Bq[j, 1] - Bq[i, 1]).sq()
            ok_d = bool(np.isfinite(dd.lo)) and bool(dd.lo > 0)
            rmin = min(rmin, math.sqrt(float(dd.lo)) if ok_d else 0.0)   # a NaN or nonpositive bound counts as 0
        a = area_val(Bq)
        if not (bool(a.lo > 0) or bool(a.hi < 0)):        # zero not excluded (NaN included)
            # refine: the Taylor polynomial with its remainder on 64 subintervals of the step
            sub = np.linspace(0.0, h, 65)
            zero_possible = False
            for u0, u1 in zip(sub[:-1], sub[1:]):
                tau = I(float(dn(u0)), float(up(u1)))
                qq = peval(qcs, tau) + Rq * tpow(tau, PORDER + 1)
                aa = area_val(qq)
                if not (bool(aa.lo > 0) or bool(aa.hi < 0)):
                    zero_possible = True
                    break
            if zero_possible:
                flagged.append(idx)

    def area_rate(Bq, Bv):
        ax, ay = Bq[1, 0] - Bq[0, 0], Bq[1, 1] - Bq[0, 1]
        bx, by = Bq[2, 0] - Bq[0, 0], Bq[2, 1] - Bq[0, 1]
        dax, day = Bv[1, 0] - Bv[0, 0], Bv[1, 1] - Bv[0, 1]
        dbx, dby = Bv[2, 0] - Bv[0, 0], Bv[2, 1] - Bv[0, 1]
        return dax * by + ax * dby - day * bx - ay * dbx
    inner = [i for i in flagged if i not in (0, len(allsteps) - 1)]
    rf, rl = area_rate(allsteps[0][6], allsteps[0][7]), area_rate(allsteps[-1][6], allsteps[-1][7])
    mono = bool((rf.lo > 0 or rf.hi < 0) and (rl.lo > 0 or rl.hi < 0))
    return rmin, len(flagged), inner, mono, len(allsteps)


def main():
    mf = [float(x) for x in sys.argv[1].split(",")]
    # the masses as intervals containing the decimal values (e.g. 0.3 = 3/10 is not a double)
    m = [I(x) if float(x).is_integer() or x in (0.5, 0.25, 0.125) else I(dn(x), up(x)) for x in mf]
    cert = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(HERE), "words", "out",
                                                              "newton_12_" + "_".join(f"{x:g}" for x in mf) + ".json")
    c = json.load(open(cert, encoding="utf-8"))
    s0, k0, T = float(c["start_parameter_s"]), float(c["speed_scale_k"]), float(c["period"])
    t0 = T / 2
    import hashlib
    per = int(os.environ.get("PER", "3"))
    code = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    print(f"masses {mf} (as intervals around the decimals): numerical zero s = {s0!r}, k = {k0!r}, t = {t0!r}",
          flush=True)
    print(f"  parameters: PORDER {PORDER}, STEPF {STEPF}, PER {per}; code sha256 {code}", flush=True)
    ok, X, steps_all, (qe, ve) = main_ms(m, s0, k0, t0, per=per)
    if not ok:
        print("NOT PROVED (Krawczyk)")
        return
    print(f"  proven box: s in [{X[0].lo!r}, {X[0].hi!r}], k in [{X[1].lo!r}, {X[1].hi!r}], "
          f"t in [{X[X.lo.size - 1].lo!r}, {X[X.lo.size - 1].hi!r}]", flush=True)
    s_ok = bool(X[0].lo > 0) and bool(X[0].hi < 1)   # body 1 strictly between bodies 2 and 3 at t = 0
    rmin, nflag, inner, mono, nsteps = path_checks(steps_all)
    q0 = qe[0]
    between = ((q0[1, 0] - q0[0, 0]) * (q0[2, 0] - q0[1, 0]) + (q0[1, 1] - q0[0, 1]) * (q0[2, 1] - q0[1, 1]))
    print(f"  path over the whole box ({nsteps} steps): smallest pair distance >= {rmin:.4f}; steps that may contain a "
          f"syzygy {nflag} (inner {len(inner)}); area rate of one sign in the first and last steps: {mono}; body 2 "
          f"between 1 and 3 at t: {bool(between.lo > 0)}; 0 < s < 1 on the box: {s_ok}")
    proved = rmin > 0 and not inner and mono and bool(between.lo > 0) and s_ok
    print("PROVED: a periodic (up to rotation) solution with L = 0 and reduced syzygy sequence 1212..." if proved
          else "NOT PROVED (path conditions)")


if __name__ == "__main__":
    main()
