"""Stage 21, the I-field of the figure eight's 1/12 problem in scaled variables (paper/review/eight_field_rigour_theorist.md).

Shape space R^3 (Hopf coordinates, L = 0), H = 2|x||p|^2 - U, U(x) = sum_k (|x| - u_k . x)^(-1/2) (homogeneous of degree
-1/2), u_3 = (-1, 0, 0), u_1 = (1/2, sqrt3/2, 0), u_2 = (1/2, -sqrt3/2, 0). The end set I = span(e_I, e_Z),
e_I = (1/2, sqrt3/2, 0), e_Z = (0, 0, 1), normal n_I = (sqrt3/2, -1/2, 0); the start set E = the ray of e_1.
For parameters (eps, phi, M), integrate on tau in [0, 1]
    Y' = -4|Y| P,  P' = 2|P|^2 Y/|Y| - eps grad U(Y),  k' = 2|Y||P|^2,  m' = U(Y),
    Y(0) = cos(phi) e_I + sin(phi) e_Z,  P(0) = M n_I,  k(0) = m(0) = 0.
Scaling lemma: the extremal ending at time 1 on I at r (cos phi e_I + sin phi e_Z) with momentum mu n_I is, at time to
go s, x = r Y(1), p = P(1)/s, action to go (r/s)(k(1) + eps m(1)), eps = s^2 r^(-3/2), M = s mu.
Scaled labels a = (ell, phi, M), ell = log r.
The eight: at s = 1 its label solves F(a) = (Y_2(1), Y_3(1), P_1(1)) = 0 (start on E, momentum orthogonal to E).

Part 1 (floats): the map, unit checks against the theorist's values, Newton for the eight's label.
Part 2 (intervals): Taylor series of the system with first-order jets in the 3 labels (outward rounding, Picard a priori
box, Lagrange remainder, as verify/prove_iso_periodic.py), and a Krawczyk proof that F has a unique zero in a small box.
    python verify/eight_field.py [floats|krawczyk] [delta]"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prove_newton_1212 import I, _cat, _i, _lead, dn, ihull, jmul, jsq, peval, stack, tpow, up  # noqa: E402

S3 = math.sqrt(3.0)
UK_F = np.array([[-1.0, 0, 0], [0.5, S3 / 2, 0], [0.5, -S3 / 2, 0]])
EI_F, EZ_F, NI_F = np.array([0.5, S3 / 2, 0]), np.array([0, 0, 1.0]), np.array([S3 / 2, -0.5, 0])
A8 = 2.519856092284

# ---------------------------------------------------------------- floats


def rhs_f(eps):
    def f(tau, y):
        Y, P = y[0:3], y[3:6]
        R = math.sqrt(Y @ Y)
        Q = P @ P
        q = R - UK_F @ Y
        G = sum(-0.5 * q[k] ** -1.5 * (Y / R - UK_F[k]) for k in range(3))
        return np.concatenate([-4 * R * P, 2 * Q * Y / R - eps * G, [2 * R * Q, np.sum(q ** -0.5)]])
    return f


def flow_f(eps, phi, M, rtol=1e-13, atol=1e-14):
    from scipy.integrate import solve_ivp
    y0 = np.concatenate([math.cos(phi) * EI_F + math.sin(phi) * EZ_F, M * NI_F, [0.0, 0.0]])
    sol = solve_ivp(rhs_f(eps), (0, 1), y0, method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:, -1]


def physical(a, s):
    """x, p, action to go at time to go s for the scaled label a = (ell, phi, M)"""
    ell, phi, M = a
    r = math.exp(ell)
    eps = s * s * r ** -1.5
    y = flow_f(eps, phi, M)
    return r * y[0:3], y[3:6] / s, r / s * (y[6] + eps * y[7])


def F_f(a):
    x, p, _ = physical(a, 1.0)
    return np.array([x[1], x[2], p[0]])


def newton_f(a, it=8):
    a = np.array(a, float)
    for _ in range(it):
        Fa = F_f(a)
        J = np.empty((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1e-6
            J[:, j] = (F_f(a + e) - F_f(a - e)) / 2e-6
        a = a - np.linalg.solve(J, Fa)
    return a, F_f(a)


def floats():
    u, v, mu = 3.55638, 3.03611, -0.30889
    a0 = np.array([math.log(math.hypot(u, v)), math.atan2(v, u), mu])
    x, p, A = physical(a0, 1.0)
    print(f"theorist's label {a0}: s = 1: x = {x}, p = {p}, action {A:.8f} (theorist: (4.69649, 0, 0), "
          f"(0, 0.158470, 0.243207), 2.51985595)")
    a8, res = newton_f(a0)
    x, p, A = physical(a8, 1.0)
    print(f"Newton: a8 = {a8!r}, residual {np.max(np.abs(res)):.1e}; x = {x}, p = {p}, action {A:.12f} (A8 {A8})")
    xc, pc, Ac = physical(np.array([a8[0], a8[1], 0.5 * a8[2]]), 0.5)     # the scaled label at s: M = s mu
    print(f"s = 1/2: x = {xc} (c), V_I(c) = {Ac:.10f} (theorist 1.2919570)")
    return a8


# ---------------------------------------------------------------- intervals
PORDER = int(os.environ.get("FORDER", "24"))
STEPF = float(os.environ.get("FSTEPF", "0.05"))
SQ3H = I(dn(S3 / 2), up(S3 / 2))
UK = [(I(-1.0), I(0.0), I(0.0)), (I(0.5), SQ3H, I(0.0)), (I(0.5), -SQ3H, I(0.0))]
EI, EZ, NI = (I(0.5), SQ3H, I(0.0)), (I(0.0), I(0.0), I(1.0)), (SQ3H, I(-0.5), I(0.0))


def push(x, k, d):
    for _ in range(k):
        x = np.nextafter(x, d)
    return x


def iexp(x):
    return I(push(np.exp(x.lo), 3, -np.inf), push(np.exp(x.hi), 3, np.inf))


def isincos(x):
    """sin and cos on an interval of width < 1 inside (0, pi/2) (monotone there): enough for phi near 40 deg"""
    assert np.all(x.lo > 0) and np.all(x.hi < math.pi / 2)
    s = I(push(np.sin(x.lo), 3, -np.inf), push(np.sin(x.hi), 3, np.inf))
    c = I(push(np.cos(x.hi), 3, -np.inf), push(np.cos(x.lo), 3, np.inf))
    return s, c


def cauchy(a, b, k):
    acc = None
    if a is b:
        for l in range(k // 2 + 1):
            t = jsq(a[l]) if 2 * l == k else jmul(a[l], a[k - l]) * 2.0
            acc = t if acc is None else acc + t
        return acc
    for l in range(k + 1):
        t = jmul(a[l], b[k - l])
        acc = t if acc is None else acc + t
    return acc


def WR(Wji, iR, k, cache, key):
    """the series (Wji * iR) up to order k, cached per key (extended by one coefficient per call)"""
    lst = cache.setdefault(key, [])
    while len(lst) <= k:
        lst.append(cauchy(Wji, iR, len(lst)))
    return lst


def power_next(a, w, k, alpha):
    """k-th Taylor coefficient of w = a^alpha from a[0..k], w[0..k-1] (k >= 1): k a0 w_k = sum_l (alpha l - k + l) a_l w_(k-l)"""
    acc = None
    for l in range(1, k + 1):
        t = jmul(a[l], w[k - l]) * float(alpha * l - k + l)
        acc = t if acc is None else acc + t
    return jmul(acc, jrec(a[0] * float(k)))


def jrec(a):
    r = a[0].recip()
    return _cat(r, -(a[1:] * _lead(r.sq())))


def jpow(a, alpha):
    """a^alpha for a jet a with a positive value part"""
    v = a[0]
    if alpha == 0.5:
        p = v.sqrt()
    elif alpha == -0.5:
        p = v.sqrt().recip()
    elif alpha == -1.5:
        p = (v * v.sqrt()).recip()
    else:
        raise ValueError(alpha)
    return _cat(p, a[1:] * _lead(p * v.recip() * alpha))


def jc(x, like):
    """a constant interval as a jet shaped like `like` (zero derivative parts)"""
    x = _i(x)
    lo, hi = np.zeros_like(like.lo), np.zeros_like(like.hi)
    lo[0], hi[0] = x.lo, x.hi
    return I(lo, hi)


def _rat(num, den):
    """the rational num/den enclosed (referee 5c, C1: a rounded float quotient used as exact is not enclosed); the
    float quotient is within half an ulp, so its two neighbours bracket the exact value"""
    v = num / den
    return I(dn(v), up(v))


def series(y0, eps, order):
    """Taylor coefficients of y = (Y1, Y2, Y3, P1, P2, P3, k, m), each a jet; eps a jet (constant in tau)"""
    Y = [[y0[i]] for i in range(3)]
    P = [[y0[3 + i]] for i in range(3)]
    K, Mq = [y0[6]], [y0[7]]
    s, R, iR, Q = [], [], [], []
    q = [[] for _ in range(3)]
    a = [[] for _ in range(3)]          # q^(-1/2)
    b = [[] for _ in range(3)]          # q^(-3/2)
    YR = [[] for _ in range(3)]         # Y/R
    ud = [[] for _ in range(3)]         # u_j . Y
    perp = [[[] for _ in range(3)] for _ in range(3)]   # Y - (u_j . Y) u_j
    Nn, Dd = [[] for _ in range(3)], [[] for _ in range(3)]
    near = [False] * 3
    cacheW = {}
    Wv = [[[] for _ in range(3)] for _ in range(3)]     # Y/R - u_j = (Y_perp - q u_j)/R (near rays)
    for k in range(order):
        s.append(cauchy(Y[0], Y[0], k) + cauchy(Y[1], Y[1], k) + cauchy(Y[2], Y[2], k))
        Q.append(cauchy(P[0], P[0], k) + cauchy(P[1], P[1], k) + cauchy(P[2], P[2], k))
        if k == 0:
            R.append(jpow(s[0], 0.5))
            iR.append(jpow(s[0], -0.5))
        else:
            R.append(power_next(s, R, k, 0.5))
            iR.append(power_next(s, iR, k, -0.5))
        for i in range(3):
            YR[i].append(cauchy(Y[i], iR, k))
        for j in range(3):
            u = UK[j]
            ud[j].append(Y[0][k] * u[0] + Y[1][k] * u[1] + Y[2][k] * u[2])
            for i in range(3):
                perp[j][i].append(Y[i][k] - ud[j][k] * u[i])
            if k == 0:
                near[j] = bool(np.all(ud[j][0][0].lo > 0))
            if near[j]:
                # q = R - u.Y = |Y_perp|^2 / (R + u.Y): no cancellation near the collision ray u
                Nn[j].append(cauchy(perp[j][0], perp[j][0], k) + cauchy(perp[j][1], perp[j][1], k)
                             + cauchy(perp[j][2], perp[j][2], k))
                Dd[j].append(R[k] + ud[j][k])
                if k == 0:
                    q[j].append(jmul(Nn[j][0], jrec(Dd[j][0])))
                else:
                    acc = Nn[j][k]
                    for l in range(1, k + 1):
                        acc = acc - jmul(Dd[j][l], q[j][k - l])
                    q[j].append(jmul(acc, jrec(Dd[j][0])))
            else:
                q[j].append(R[k] - ud[j][k])
            if k == 0:
                a[j].append(jpow(q[j][0], -0.5))
                b[j].append(jpow(q[j][0], -1.5))
            else:
                a[j].append(power_next(q[j], a[j], k, -0.5))
                b[j].append(power_next(q[j], b[j], k, -1.5))
        RP = [cauchy(R, P[i], k) for i in range(3)]
        QYR = [cauchy(Q, YR[i], k) for i in range(3)]
        # grad U_i = sum_j -1/2 q_j^(-3/2) (Y_i/R - u_j,i): series of the product with b_j
        G = []
        for j in range(3):
            if near[j]:
                for i in range(3):
                    Wv[j][i].append(perp[j][i][k] - q[j][k] * UK[j][i])
        for i in range(3):
            acc = None
            for j in range(3):
                if near[j]:
                    t = cauchy(b[j], WR(Wv[j][i], iR, k, cacheW, (j, i)), k)
                else:
                    t = cauchy(b[j], YR[i], k)
                    if float(UK[j][i].hi) != 0.0 or float(UK[j][i].lo) != 0.0:
                        t = t - b[j][k] * UK[j][i]
                acc = t if acc is None else acc + t
            G.append(acc * (-0.5))
        RQ = cauchy(R, Q, k)
        for i in range(3):
            Y[i].append(RP[i] * _rat(-4.0, k + 1))
            P[i].append((QYR[i] * 2.0 - jmul_const_jet(eps, G[i])) * _rat(1.0, k + 1))
        K.append(RQ * _rat(2.0, k + 1))
        Mq.append((a[0][k] + a[1][k] + a[2][k]) * _rat(1.0, k + 1))
    return Y + P + [K, Mq]


def jmul_const_jet(e, g):
    """eps (a jet, constant in tau) times the k-th series coefficient g (a jet): product rule on the jet axis"""
    return jmul(_bcast(e, g), g)


def _bcast(e, g):
    """shape the jet e like g (g may carry extra trailing array axes)"""
    lo, hi = e.lo, e.hi
    while lo.ndim < g.lo.ndim:
        lo, hi = lo[..., None], hi[..., None]
    return I(np.broadcast_to(lo, g.lo.shape).copy(), np.broadcast_to(hi, g.hi.shape).copy())


NC = 8


def step(y, eps, h):
    try:
        cs = series(y, eps, PORDER)
    except (ZeroDivisionError, ValueError):
        return None
    H = I(0.0, h)
    g = [peval(c, H) for c in cs]

    def inflate(B, Yv):
        ex = np.maximum(np.abs(B.lo - Yv.lo), np.abs(B.hi - Yv.hi)) + B.wid
        return I(dn(B.lo - 0.05 * ex - 1e-13), up(B.hi + 0.05 * ex + 1e-13))
    B = [inflate(ihull(y[i], g[i]), y[i]) for i in range(NC)]
    for _ in range(60):
        try:
            f = series(B, eps, 1)
        except (ZeroDivisionError, ValueError):
            return None
        n = [y[i] + f[i][1] * H for i in range(NC)]
        if all(np.all(n[i].lo > B[i].lo) and np.all(n[i].hi < B[i].hi) for i in range(NC)):
            break
        B = [inflate(ihull(n[i], y[i]), y[i]) for i in range(NC)]
    else:
        return None
    try:
        rr = series(B, eps, PORDER + 1)
    except (ZeroDivisionError, ValueError):
        return None
    return cs, [rr[i][PORDER + 1] for i in range(NC)], B


def flow_unit(y, eps, hmax=0.05):
    """validated flow of the jet state y over tau in [0, 1]; the elapsed time is enclosed, and the last step is
    evaluated on the enclosure of the remaining time, so the result encloses the state at tau = 1 exactly"""
    t_lo, t_hi = 0.0, 0.0           # enclosure of the elapsed time
    shrink, minq = 1.0, np.inf
    while True:
        try:
            cs = series([I(c.mid[:1]) for c in y], _mid_jet(eps), PORDER)
            mags = [max(float(np.max(np.abs(c[k].mid))) for c in cs) for k in range(PORDER + 1)]
            rho = min(mags[k] ** (-1.0 / k) for k in range(PORDER // 2, PORDER + 1) if mags[k] > 0)
            h0 = min(hmax, STEPF * rho) * shrink
        except (ZeroDivisionError, ValueError):
            h0 = hmax * shrink
        rem_lo, rem_hi = float(dn(1.0 - t_hi)), float(up(1.0 - t_lo))
        last = rem_hi <= h0
        hstep = rem_hi if last else h0
        res = step(y, eps, hstep)
        if res is None:
            shrink /= 2
            if h0 < 1e-7:
                raise RuntimeError(f"step size collapsed at tau = {t_lo}")
            continue
        shrink = 1.0
        cs, Rm, B = res
        minq = min(minq, _min_q(B))
        if last:
            tau = I(max(rem_lo, 0.0), rem_hi)
            tP = tpow(tau, PORDER + 1)
            return [peval(cs[i], tau) + Rm[i] * tP for i in range(NC)], minq
        tP = tpow(I(hstep), PORDER + 1)
        y = [peval(cs[i], I(hstep)) + Rm[i] * tP for i in range(NC)]
        t_lo, t_hi = float(dn(t_lo + hstep)), float(up(t_hi + hstep))


def _mid_jet(e):
    return I(e.mid[:1])


def _min_q(B):
    """lower bound over the step's a priori box of min_k (|Y| - u_k . Y) / |Y| (the angular distance to collision)"""
    Y = [b[0] for b in B[0:3]]
    R = (Y[0].sq() + Y[1].sq() + Y[2].sq()).sqrt()
    m = np.inf
    for u in UK:
        q = R - (Y[0] * u[0] + Y[1] * u[1] + Y[2] * u[2])
        m = min(m, float(np.min(q.lo / np.maximum(R.hi, 1e-300))))
    return m


def start_jets(Ell, Phi, Mm):
    """the start state and eps as jets in (ell, phi, M) over the interval boxes Ell, Phi, Mm"""
    shp = np.shape(Ell.lo)

    def bc(x):
        x = _i(x)
        return I(np.broadcast_to(x.lo, shp).copy(), np.broadcast_to(x.hi, shp).copy())

    def J(val, d):
        return stack([bc(val)] + [bc(x) for x in d], 0)
    sn, cn = isincos(Phi)
    Y = [J(cn * EI[i] + sn * EZ[i], [I(0.0), -sn * EI[i] + cn * EZ[i], I(0.0)]) for i in range(3)]
    P = [J(Mm * NI[i], [I(0.0), I(0.0), NI[i]]) for i in range(3)]
    z = J(I(0.0), [I(0.0)] * 3)
    eps = iexp(Ell * (-1.5))            # eps = r^(-3/2) at s = 1
    E = J(eps, [eps * (-1.5), I(0.0), I(0.0)])
    return Y + P + [z, z], E


def F_and_DF(X):
    y0, E = start_jets(*X)
    yT, minq = flow_unit(y0, E)
    rows = [yT[1], yT[2], yT[3]]                     # Y_2, Y_3, P_1 at tau = 1 (x = r Y: the sign conditions agree)
    F = [r[0] for r in rows]
    DF = [[r[1 + j] for j in range(3)] for r in rows]
    return F, DF, yT, minq


def krawczyk(c, delta):
    Xc = [I(float(x)) for x in c]
    Fc, _, _, _ = F_and_DF(Xc)
    X = [I(dn(c[i] - delta[i]), up(c[i] + delta[i])) for i in range(3)]
    F, DF, yT, minq = F_and_DF(X)
    Dm = np.array([[float(DF[i][j].mid) for j in range(3)] for i in range(3)])
    A = np.linalg.inv(Dm)
    K = []
    for i in range(3):
        acc = I(float(c[i]))
        for j in range(3):
            acc = acc - Fc[j] * float(A[i, j])
        for j in range(3):
            Mij = I(1.0 if i == j else 0.0)
            for l in range(3):
                Mij = Mij - DF[l][j] * float(A[i, l])
            acc = acc + Mij * (X[j] - float(c[j]))
        K.append(acc)
    inside = all(float(K[i].lo) > float(X[i].lo) and float(K[i].hi) < float(X[i].hi) for i in range(3))
    return inside, X, K, yT, minq


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "floats"
    a8 = floats()
    if mode == "krawczyk":
        d = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-8
        inside, X, K, yT, minq = krawczyk(a8, [d, d, d])
        print(f"Krawczyk box half-width {d:g}: K inside X: {inside}")
        for i, n in enumerate(("ell", "phi", "M")):
            print(f"  {n}: X [{float(X[i].lo)!r}, {float(X[i].hi)!r}]  K [{float(K[i].lo)!r}, {float(K[i].hi)!r}]")
        Y1 = yT[0][0]
        print(f"  Y_1(1) in [{float(Y1.lo)}, {float(Y1.hi)}] (> 0: the start is on the ray E, not its opposite); "
              f"min angular collision distance along the flow {minq:.4f}")
        kin, mq = yT[6][0], yT[7][0]
        eps = iexp(X[0] * (-1.5))
        act = (kin + eps * mq) * iexp(X[0])
        print(f"  action of the solution in [{float(act.lo)!r}, {float(act.hi)!r}]")
        print("PROVED: unique zero of F in X" if inside else "NOT PROVED")


if __name__ == "__main__":
    main()
