"""Independent re-check of the coarse piece E with Arb ball arithmetic (python-flint): the cell inequality F of
verify/eight_cell_verify.py implemented separately (Gram-matrix sphere gradient, see eight_cell_mpiv.py for the
formulas), evaluated with arb balls at 106 bits (Arb's rigorous error bounds for +, *, /, sqrt, exp, tan, cos; no
numpy, no platform libm). Shares with the verifier only the data (JSON nodal values, cell ranges, the integer cube
frames). Bisection in one direction at a time (the widest relative to the cell).
    python verify/eight_cell_arb.py cert.json [K] [workers]"""
import json
import sys
import time
from fractions import Fraction

import numpy as np
from flint import arb, ctx

PREC = 106


def box(lo, hi):
    """the ball containing the interval [lo, hi] (lo, hi exact floats or Fractions)"""
    lo, hi = Fraction(lo), Fraction(hi)
    m = (lo + hi) / 2
    r = (hi - lo) / 2
    return arb(m.numerator) / m.denominator + arb(0, float(r) * (1 + 2.0 ** -40) + 2.0 ** -1070)


def coef(cell):
    """exact rational monomial coefficients of phi in local y from the nodal Q211 values (as eight_cell_mpiv)"""
    V = np.array([[[Fraction(x) for x in r2] for r2 in r1] for r1 in cell["nodal"]["vals"]], dtype=object)

    def along(T, ax):
        T = np.moveaxis(T, ax, 0)
        rows = [T[i] for i in range(T.shape[0])]
        if len(rows) == 2:
            out = [(rows[0] + rows[1]) / 2, (rows[1] - rows[0]) / 2]
        else:
            out = [rows[1], (rows[2] - rows[0]) / 2, (rows[2] + rows[0]) / 2 - rows[1]]
        return np.moveaxis(np.array(out, dtype=object), 0, ax)
    for ax in range(3):
        V = along(V, ax)
    terms = []
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for l in range(V.shape[2]):
                if V[i, j, l] != 0:
                    terms.append(((i, j, l), arb(V[i, j, l].numerator) / V[i, j, l].denominator))
    return terms


class D:
    """first-order dual number over arb balls: value v and gradient g (3 arb) w.r.t. (sigma, ua, ub). Evaluated on a
    box, g encloses the gradient of the function over the box (forward-mode differentiation with ball arithmetic)."""
    __slots__ = ("v", "g")

    def __init__(self, v, g):
        self.v, self.g = v, g

    @staticmethod
    def _c(o):
        return o if isinstance(o, D) else D(o if isinstance(o, arb) else arb(o), (0, 0, 0))

    def __add__(s, o):
        o = D._c(o)
        return D(s.v + o.v, tuple(a + b for a, b in zip(s.g, o.g)))
    __radd__ = __add__

    def __neg__(s):
        return D(-s.v, tuple(-a for a in s.g))

    def __sub__(s, o):
        return s + (-D._c(o))

    def __rsub__(s, o):
        return D._c(o) + (-s)

    def __mul__(s, o):
        o = D._c(o)
        return D(s.v * o.v, tuple(a * o.v + s.v * b for a, b in zip(s.g, o.g)))
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = D._c(o)
        q = s.v / o.v
        return D(q, tuple((a - q * b) / o.v for a, b in zip(s.g, o.g)))

    def __rtruediv__(s, o):
        return D._c(o) / s

    def _f(s, f, df):
        return D(f, tuple(df * a for a in s.g))

    def sqrt(s):
        r = s.v.sqrt()
        return s._f(r, 1 / (2 * r))

    def exp(s):
        e = s.v.exp()
        return s._f(e, e)

    def tan(s):
        t = s.v.tan()
        return s._f(t, 1 + t * t)

    def cos(s):
        return s._f(s.v.cos(), -s.v.sin())

    def __gt__(s, o):
        return s.v > o


def val(x):
    return x.v if isinstance(x, D) else x


def F_arb(cell, terms, frame, S, UA, UB, ulow=False):
    ctx.prec = PREC
    s0, s1 = (Fraction(x) for x in cell["sigma"])
    a0, a1 = (Fraction(x) for x in cell["ua"])
    b0, b1 = (Fraction(x) for x in cell["ub"])

    def q(x):
        return arb(x.numerator) / x.denominator
    fs, fa, fb = q(2 / (s1 - s0)), q(2 / (a1 - a0)), q(2 / (b1 - b0))
    y = [(S - q(s0)) * fs - 1, (UA - q(a0)) * fa - 1, (UB - q(b0)) * fb - 1]
    p4 = arb.pi() / 4
    ta_, tb_ = (p4 * UA).tan(), (p4 * UB).tan()
    C, A, B = frame
    w = [C[k] + ta_ * A[k] + tb_ * B[k] for k in range(3)]
    nw = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt()
    n = [w[k] / nw for k in range(3)]
    ca, cb = (p4 * UA).cos(), (p4 * UB).cos()
    dwa = [p4 / (ca * ca) * A[k] for k in range(3)]
    dwb = [p4 / (cb * cb) * B[k] for k in range(3)]
    nda = n[0] * dwa[0] + n[1] * dwa[1] + n[2] * dwa[2]
    ndb = n[0] * dwb[0] + n[1] * dwb[1] + n[2] * dwb[2]
    tA = [(dwa[k] - n[k] * nda) / nw for k in range(3)]
    tB = [(dwb[k] - n[k] * ndb) / nw for k in range(3)]
    yp = [[arb(1), y[d], y[d] * y[d]] for d in range(3)]

    def P(d=None):
        tot = arb(0)
        for (i, j, l), cv in terms:
            e = [i, j, l]
            if d is not None:
                if e[d] == 0:
                    continue
                k = e[d]
                e[d] -= 1
                cv = cv * k
            tot += cv * yp[0][e[0]] * yp[1][e[1]] * yp[2][e[2]]
        return tot
    phi, ps, pa, pb = P(), P(0) * fs, P(1) * fa, P(2) * fb
    Maa = tA[0] * tA[0] + tA[1] * tA[1] + tA[2] * tA[2]
    Mbb = tB[0] * tB[0] + tB[1] * tB[1] + tB[2] * tB[2]
    Mab = tA[0] * tB[0] + tA[1] * tB[1] + tA[2] * tB[2]
    det = Maa * Mbb - Mab * Mab
    if not val(det) > 0:
        return None
    ia, ib = (Mbb * pa - Mab * pb) / det, (Maa * pb - Mab * pa) / det
    Dp = [tA[k] * ia + tB[k] * ib for k in range(3)]
    D2 = pa * ia + pb * ib
    g = 1 - n[0]
    Dg = [-(1 - n[0] * n[0]), n[0] * n[1], n[0] * n[2]]
    s3h = arb(3).sqrt() / 2
    us = [(arb(-1), arb(0), arb(0)), (arb(1) / 2, s3h, arb(0)), (arb(1) / 2, -s3h, arb(0))]
    U = arb(0)
    for u in us:
        qk = 1 - (u[0] * n[0] + u[1] * n[1] + u[2] * n[2])
        if ulow:                       # near a collision ray: only a lower bound of U, (sup q_k)^(-1/2) per term
            qu = arb(val(qk).upper())
            if not qu > 0:
                return None
            U += 1 / qu.sqrt()
            continue
        if not val(qk) > 0:
            return None
        U += 1 / qk.sqrt()
    if ulow:
        U = arb(U.lower())             # a point below every value of U on the box
    Fv = (-S / 2).exp() * U - 2 * (-S).exp() * (ps * ps + D2) - phi / 3 + (arb(4) / 3 - g) * ps \
        - (Dg[0] * Dp[0] + Dg[1] * Dp[1] + Dg[2] * Dp[2])
    return Fv


def lower_bound(cell, terms, frame, R0, R1, R2):
    """a rigorous lower bound of F on the box R0 x R1 x R2 (exact rational ranges), or None: the mean-value form
    F(B) in F(c) + sum_i dF/dx_i(B) [x_i - c_i], with the gradient enclosure from D on the whole box (F is smooth on
    B: the evaluation returns None if a collision term or the chart's Gram determinant is not certainly positive)"""
    R = (R0, R1, R2)
    c = [(r[0] + r[1]) / 2 for r in R]
    rad = [(r[1] - r[0]) / 2 for r in R]
    Fc = F_arb(cell, terms, frame, *(arb(x.numerator) / x.denominator for x in c))
    seeds = [tuple(arb(1) if j == i else arb(0) for j in range(3)) for i in range(3)]
    G = F_arb(cell, terms, frame, *(D(box(*r), seeds[i]) for i, r in enumerate(R)))
    if Fc is None or G is None:
        P = F_arb(cell, terms, frame, box(*R0), box(*R1), box(*R2), ulow=True)
        return None if P is None else P.lower()
    tot = Fc
    for gi, ri in zip(G.g, rad):
        tot = tot + gi * arb(0, ri.numerator) / ri.denominator          # gi * [-ri, ri]
    lo = tot.lower()
    if G.v > 0:                                    # the plain interval value, sometimes better on wide boxes
        lo = max(lo, G.v.lower())
    return lo


def check_cell(args):
    idx, cell, frame_int, maxdepth = args
    ctx.prec = PREC
    frame = [[arb(int(x)) for x in v] for v in frame_int]
    terms = coef(cell)
    s0, s1 = cell["sigma"]; a0, a1 = cell["ua"]; b0, b1 = cell["ub"]
    stack = [((Fraction(s0), Fraction(s1)), (Fraction(a0), Fraction(a1)), (Fraction(b0), Fraction(b1)), 0)]
    widths = [Fraction(s1) - Fraction(s0), Fraction(a1) - Fraction(a0), Fraction(b1) - Fraction(b0)]
    worst, nb, t = None, 0, time.time()
    while stack:
        R0, R1, R2, d = stack.pop()
        lo = lower_bound(cell, terms, frame, R0, R1, R2)
        nb += 1
        if lo is not None and lo > 0:
            worst = lo if worst is None or lo < worst else worst
            continue
        if d >= maxdepth:
            return idx, False, (float(lo) if lo is not None else None), nb, time.time() - t
        R = [R0, R1, R2]
        k = int(np.argmax([float((r[1] - r[0]) / w) for r, w in zip(R, widths)]))
        m = (R[k][0] + R[k][1]) / 2
        for half in ((R[k][0], m), (m, R[k][1])):
            Rn = list(R)
            Rn[k] = half
            stack.append((Rn[0], Rn[1], Rn[2], d + 1))
    return idx, True, float(worst), nb, time.time() - t


def _low():
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def main():
    import multiprocessing as mp
    path = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
    import eight_cell_mpiv as M          # only its float scan, to choose the tightest cells
    data = json.load(open(path, encoding="utf-8"))
    mins = M.scan(data)
    order = np.argsort(mins)[:K]
    if len(sys.argv) > 4:                      # random sample instead: argv[4] = sample size, fixed seed
        order = np.random.default_rng(20261115).choice(len(mins), int(sys.argv[4]), replace=False)
    tasks = []
    for idx in order:
        cell = data["cells"][idx]
        fr = data["faces"][cell["face"]]
        assert all(float(x).is_integer() for k in ("centre", "ea", "eb") for x in fr[k])
        tasks.append((int(idx), cell, [[int(x) for x in fr[k]] for k in ("centre", "ea", "eb")], 60))
    allok = True
    with mp.Pool(workers, initializer=_low) as pool:
        for idx, ok, worst, nb, sec in pool.imap_unordered(check_cell, tasks):
            allok &= ok
            print(f"  cell {idx}: float min {mins[idx]:+.3e}; Arb {'PASS' if ok else 'FAIL'}, worst lower bound "
                  f"{worst if worst is None else f'{worst:+.3e}'}, {nb} boxes, {sec:.0f} s", flush=True)
    print("ARB RE-CHECK PASS" if allok else "ARB RE-CHECK FAIL", flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- the far condition (sigma >= s on top cells)
def far_arb(cell, terms, frame, s, UA, UB, ulow=False):
    """far-field condition on the top face sigma = s (cells with sigma_hi = s), A = exp(s/2) phi(s, n):
    U - A (1 - g/2) - Dg . DA - 2 exp(-3 s/2) (A^2/4 + |DA|^2) >= 0 (verify/eight_cell_verify.py, docstring)"""
    ctx.prec = PREC
    a0, a1 = (Fraction(x) for x in cell["ua"])
    b0, b1 = (Fraction(x) for x in cell["ub"])

    def q(x):
        return arb(x.numerator) / x.denominator
    fa, fb = q(2 / (a1 - a0)), q(2 / (b1 - b0))
    y = [arb(1), (UA - q(a0)) * fa - 1, (UB - q(b0)) * fb - 1]
    p4 = arb.pi() / 4
    ta_, tb_ = (p4 * UA).tan(), (p4 * UB).tan()
    C, A_, B_ = frame
    w = [C[k] + ta_ * A_[k] + tb_ * B_[k] for k in range(3)]
    nw = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt()
    n = [w[k] / nw for k in range(3)]
    ca, cb = (p4 * UA).cos(), (p4 * UB).cos()
    dwa = [p4 / (ca * ca) * A_[k] for k in range(3)]
    dwb = [p4 / (cb * cb) * B_[k] for k in range(3)]
    nda = n[0] * dwa[0] + n[1] * dwa[1] + n[2] * dwa[2]
    ndb = n[0] * dwb[0] + n[1] * dwb[1] + n[2] * dwb[2]
    tA = [(dwa[k] - n[k] * nda) / nw for k in range(3)]
    tB = [(dwb[k] - n[k] * ndb) / nw for k in range(3)]
    yp = [[arb(1), y[d], y[d] * y[d]] for d in range(3)]

    def P(d=None):
        tot = arb(0)
        for (i, j, l), cv in terms:
            e = [i, j, l]
            if d is not None:
                if e[d] == 0:
                    continue
                k = e[d]
                e[d] -= 1
                cv = cv * k
            tot = cv * yp[0][e[0]] * yp[1][e[1]] * yp[2][e[2]] + tot
        return tot
    es = (arb(s) / 2).exp()
    A = P() * es
    pa, pb = P(1) * fa * es, P(2) * fb * es
    Maa = tA[0] * tA[0] + tA[1] * tA[1] + tA[2] * tA[2]
    Mbb = tB[0] * tB[0] + tB[1] * tB[1] + tB[2] * tB[2]
    Mab = tA[0] * tB[0] + tA[1] * tB[1] + tA[2] * tB[2]
    det = Maa * Mbb - Mab * Mab
    if not val(det) > 0:
        return None
    ia, ib = (Mbb * pa - Mab * pb) / det, (Maa * pb - Mab * pa) / det
    DA = [tA[k] * ia + tB[k] * ib for k in range(3)]
    DA2 = pa * ia + pb * ib
    g = 1 - n[0]
    Dg = [-(1 - n[0] * n[0]), n[0] * n[1], n[0] * n[2]]
    s3h = arb(3).sqrt() / 2
    us = [(arb(-1), arb(0), arb(0)), (arb(1) / 2, s3h, arb(0)), (arb(1) / 2, -s3h, arb(0))]
    U = arb(0)
    for u in us:
        qk = 1 - (u[0] * n[0] + u[1] * n[1] + u[2] * n[2])
        if ulow:
            qu = arb(val(qk).upper())
            if not qu > 0:
                return None
            U = 1 / qu.sqrt() + U
            continue
        if not val(qk) > 0:
            return None
        U = 1 / qk.sqrt() + U
    if ulow:
        U = arb(U.lower())
    return U - A * (1 - g / 2) - (Dg[0] * DA[0] + Dg[1] * DA[1] + Dg[2] * DA[2]) \
        - 2 * (arb(-3 * s) / 2).exp() * (A * A / 4 + DA2)


def far_lower_bound(cell, terms, frame, s, R1, R2):
    c = [(r[0] + r[1]) / 2 for r in (R1, R2)]
    rad = [(r[1] - r[0]) / 2 for r in (R1, R2)]
    Fc = far_arb(cell, terms, frame, s, *(arb(x.numerator) / x.denominator for x in c))
    seeds = [(arb(0), arb(1), arb(0)), (arb(0), arb(0), arb(1))]
    G = far_arb(cell, terms, frame, s, *(D(box(*r), seeds[i]) for i, r in enumerate((R1, R2))))
    if Fc is None or G is None:
        P = far_arb(cell, terms, frame, s, box(*R1), box(*R2), ulow=True)
        return None if P is None else P.lower()
    tot = Fc
    for gi, ri in zip(G.g[1:], rad):
        tot = tot + gi * arb(0, ri.numerator) / ri.denominator
    lo = tot.lower()
    if G.v > 0:
        lo = max(lo, G.v.lower())
    return lo


def check_far(args):
    idx, cell, frame_int, s, maxdepth = args
    ctx.prec = PREC
    frame = [[arb(int(x)) for x in v] for v in frame_int]
    terms = coef(cell)
    a0, a1 = cell["ua"]; b0, b1 = cell["ub"]
    widths = [Fraction(a1) - Fraction(a0), Fraction(b1) - Fraction(b0)]
    stack = [((Fraction(a0), Fraction(a1)), (Fraction(b0), Fraction(b1)), 0)]
    worst, nb, t = None, 0, time.time()
    while stack:
        R1, R2, d = stack.pop()
        lo = far_lower_bound(cell, terms, frame, s, R1, R2)
        nb += 1
        if lo is not None and lo > 0:
            worst = lo if worst is None or lo < worst else worst
            continue
        if d >= maxdepth:
            return idx, False, (float(lo) if lo is not None else None), nb, time.time() - t
        R = [R1, R2]
        k = int(np.argmax([float((r[1] - r[0]) / w) for r, w in zip(R, widths)]))
        m = (R[k][0] + R[k][1]) / 2
        for half in ((R[k][0], m), (m, R[k][1])):
            Rn = list(R)
            Rn[k] = half
            stack.append((Rn[0], Rn[1], d + 1))
    return idx, True, float(worst), nb, time.time() - t


def main_far(path, n_tight=50, n_rand=50, workers=2):
    """the far condition on the top cells: the n_tight tightest by a float scan (5 x 5 points) and n_rand random"""
    import multiprocessing as mp
    data = json.load(open(path, encoding="utf-8"))
    s = data["far"]["sigma_hi"]
    top = [i for i, c in enumerate(data["cells"]) if c["sigma"][1] == s]
    ctx.prec = 53
    est = []
    for i in top:
        cell = data["cells"][i]
        fr = data["faces"][cell["face"]]
        frame = [[arb(int(x)) for x in fr[k]] for k in ("centre", "ea", "eb")]
        terms = coef(cell)
        vals = [far_arb(cell, terms, frame, s, arb(a), arb(b))
                for a in np.linspace(*cell["ua"], 5) for b in np.linspace(*cell["ub"], 5)]
        est.append(min([float(v.mid()) for v in vals if v is not None and v.is_finite()] or [np.inf]))
    est = np.array(est)
    o = np.argsort(est)
    pick = set(o[:n_tight].tolist()) | set(np.random.default_rng(20261115).choice(len(top), n_rand, replace=False).tolist())
    print(f"far: {len(top)} top cells (sigma = {s}), float minima from {est.min():+.3e}; checking {len(pick)}", flush=True)
    tasks = []
    for j in sorted(pick):
        cell = data["cells"][top[j]]
        fr = data["faces"][cell["face"]]
        tasks.append((top[j], cell, [[int(x) for x in fr[k]] for k in ("centre", "ea", "eb")], s, 60))
    allok = True
    with mp.Pool(workers, initializer=_low) as pool:
        for idx, ok, worst, nb, sec in pool.imap_unordered(check_far, tasks):
            allok &= ok
            print(f"  top cell {idx}: Arb far {'PASS' if ok else 'FAIL'}, worst lower bound "
                  f"{worst if worst is None else f'{worst:+.3e}'}, {nb} boxes, {sec:.0f} s", flush=True)
    print("ARB FAR RE-CHECK PASS" if allok else "ARB FAR RE-CHECK FAIL", flush=True)
