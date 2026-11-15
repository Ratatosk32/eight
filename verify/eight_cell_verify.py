"""Rigorous cell verifier for the Hamilton-Jacobi subsolution certificates of Stage 21 (uniqueness of the figure eight).

Variables (paper/review/eight_gate_step3.md, 2.1 and 5.1): self-similar shape-space points xi = exp(sigma) n, n on the
unit sphere, n = normalize(centre + tan(pi ua/4) ea + tan(pi ub/4) eb) on cube faces (ua, ub in [-1, 1]).
A certificate is w = w0_side + phi with w0 = exp(sigma) g(n)/4, g_E = 1 - n_1, g_I = 1 - sqrt(1 - (n . n_I)^2),
n_I = (sqrt3/2, -1/2, 0). The subsolution inequality F[w] = U(xi) - 2|xi||grad w|^2 - w/3 + (4/3) xi.grad w >= 0
becomes, exactly (because g^2 + |Dg|^2 = 2 g on the sphere for both g, an algebraic identity: for g_E with
|Dg|^2 = 1 - n_1^2; for g_I with s = n.n_I, h = sqrt(1 - s^2), |Dg|^2 = s^2),
    F = e^(-sigma/2) U(n) - 2 e^(-sigma) (phi_sigma^2 + |D phi|^2) - phi/3 + (4/3 - g) phi_sigma - Dg . D phi,
U(n) = sum_k (1 - u_k . n)^(-1/2) (u_3 = (-1,0,0), u_1 = (1/2, sqrt3/2, 0), u_2 = (1/2, -sqrt3/2, 0)), D the sphere
gradient. phi is a polynomial in the local variables of each cell (sigma, ua, ub) -> [-1, 1]^3.
Checks: (cells) a rigorous lower bound of F > 0 on every cell, by outward-rounded interval arithmetic with bisection;
(far) for sigma >= s the extension phi = exp(-(sigma - s)/2) phi(s, n) satisfies F >= 0 iff on each top cell
U - A (1 - g/2) - Dg . DA - 2 exp(-3s/2) (A^2/4 + |DA|^2) >= 0, A = exp(s/2) phi(s, .) (derived in the main session);
(low) for sigma < s, phi = v constant: F >= 0 iff v <= 9 exp(-sigma/2), sufficient v <= 9 exp(-s/2) (U(n) >= 3 by
Jensen, since sum_k (1 - u_k . n) = 3).
Near collision rays U -> +infinity; only lower bounds of U are used, so each term is bounded below by
(max over the box of 1 - u_k.n)^(-1/2). Sub-boxes where g_I is not smooth (n = +-n_I) are flagged.
tan and exp at interval endpoints: numpy values pushed outward by 3 ulps (assumes the platform libm is accurate to
1 ulp for these functions; the endpoints could be redone with mpmath for a final version).
Kinks: phi is continuous and piecewise polynomial (even in xi_3); F is concave in the gradient, so checking each cell
with its own gradient covers concave kinks (viscosity / Clarke argument); convex kinks impose no condition.
    python verify/eight_cell_verify.py cert_pieces.json [max_depth]"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prove_newton_1212 import I, dn, up  # noqa: E402

S3H = I(dn(math.sqrt(3.0) / 2), up(math.sqrt(3.0) / 2))
PI4 = I(dn(math.pi / 4), up(math.pi / 4))      # pi/4 as an interval (math.pi is pi rounded to nearest)
PI4 = I(dn(float(PI4.lo)), up(float(PI4.hi)))
NI = (S3H, I(-0.5), I(0.0))
UV = [(I(-1.0), I(0.0), I(0.0)), (I(0.5), S3H, I(0.0)), (I(0.5), -S3H, I(0.0))]
E1 = (I(1.0), I(0.0), I(0.0))


def exact_frames(faces):
    """attach exactly orthonormal frames (_C, _A, _B as interval triples) to the JSON faces: either the standard cube
    faces (integer entries, exact) or all of them turned about w3 by -30 deg (cos = sqrt3/2, sin = -1/2, enclosed);
    anything else is refused. The certificate is the function defined with these exact frames."""
    std = [np.array(f[k], float) for f in faces for k in ("centre", "ea", "eb")]
    if all(float(x).is_integer() for v in std for x in v):
        rot = None
    else:
        c, sn = math.sqrt(3.0) / 2, -0.5
        R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])
        base = [np.round(R.T @ v) for v in std]
        assert all(np.max(np.abs(R @ b - v)) < 1e-12 for b, v in zip(base, std)), "faces: not the standard cube turned by -30 deg"
        rot = ((S3H, I(0.5), I(0.0)), (I(-0.5), S3H, I(0.0)), (I(0.0), I(0.0), I(1.0)))
    k = 0
    for f in faces:
        for key, name in (("centre", "_C"), ("ea", "_A"), ("eb", "_B")):
            if rot is None:
                f[name] = tuple(I(float(x)) for x in std[k])
            else:
                b = np.round(np.array(f[key], float) @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]]))   # R^T v
                f[name] = tuple(rot[i][0] * float(b[0]) + rot[i][1] * float(b[1]) + rot[i][2] * float(b[2]) for i in range(3))
            k += 1
    return faces


def push(x, k, direction):
    for _ in range(k):
        x = np.nextafter(x, direction)
    return x


def tan_iv(x):
    """tan on an interval inside (-pi/2, pi/2) (monotone)"""
    return I(push(np.tan(x.lo), 3, -np.inf), push(np.tan(x.hi), 3, np.inf))


def exp_iv(x):
    return I(np.maximum(push(np.exp(x.lo), 3, -np.inf), 0.0), push(np.exp(x.hi), 3, np.inf))


def vec(a):
    return [I(float(a[i])) if float(a[i]).is_integer() else I(dn(float(a[i])), up(float(a[i]))) for i in range(3)]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class IC:
    """interval coefficient tensor: lo, hi arrays of shape (p1 + 1, p2 + 1, p3 + 1)"""

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.shape = self.lo.shape

    def at(self, idx):
        return I(self.lo[idx], self.hi[idx])


def poly(c, y):
    """sum c_ijk y1^i y2^j y3^k, Horner, intervals vectorised; c a float numpy tensor or an IC (interval tensor)"""
    p1, p2, p3 = (s_ - 1 for s_ in c.shape)
    get = (lambda idx: c.at(idx)) if isinstance(c, IC) else (lambda idx: float(c[idx]))
    out = None
    for i in range(p1, -1, -1):
        cj = None
        for j in range(p2, -1, -1):
            ck = I(np.zeros(y[2].lo.shape)) + get((i, j, p3))
            for k in range(p3 - 1, -1, -1):
                ck = ck * y[2] + get((i, j, k))
            cj = ck if cj is None else cj * y[1] + ck
        out = cj if out is None else out * y[0] + cj
    return out


def dpoly(c, ax):
    sh = list(c.shape)
    if sh[ax] == 1:
        return np.zeros([1, 1, 1])
    sl = [slice(None)] * 3
    sl[ax] = slice(0, sh[ax] - 1)
    if isinstance(c, IC):
        lo, hi = np.zeros(c.shape), np.zeros(c.shape)
        for idx in np.ndindex(c.shape):
            e = idx[ax]
            if e:
                j = list(idx)
                j[ax] -= 1
                t = c.at(idx) * float(e)              # e * c exactly enclosed (one product, outward rounded)
                lo[tuple(j)], hi[tuple(j)] = t.lo, t.hi  # each target index receives exactly one term
        return IC(lo[tuple(sl)], hi[tuple(sl)])
    d = np.zeros(c.shape)
    for idx in np.ndindex(c.shape):
        e = idx[ax]
        if e:
            j = list(idx)
            j[ax] -= 1
            d[tuple(j)] += e * c[idx]
    return d[tuple(sl)]


def nodal_coef(nodal):
    """monomial coefficients (as an IC) of the tensor Lagrange interpolant of the nodal values: per axis, nodes
    (-1, 1): c0 = (v_-1 + v_1)/2, c1 = (v_1 - v_-1)/2; nodes (-1, 0, 1): c0 = v_0, c1 = (v_1 - v_-1)/2,
    c2 = (v_1 + v_-1)/2 - v_0. Exact rational transforms, enclosed in interval arithmetic."""
    V = np.asarray(nodal["vals"], float)
    assert V.shape[1:] == (2, 2) and V.shape[0] in (2, 3), V.shape
    T = I(V, V.copy())

    def along(T, ax):
        lo, hi = np.moveaxis(T.lo, ax, 0), np.moveaxis(T.hi, ax, 0)
        rows = [I(lo[i], hi[i]) for i in range(lo.shape[0])]
        if len(rows) == 2:
            out = [(rows[0] + rows[1]) * 0.5, (rows[1] - rows[0]) * 0.5]
        else:
            out = [rows[1], (rows[2] - rows[0]) * 0.5, (rows[2] + rows[0]) * 0.5 - rows[1]]
        return I(np.moveaxis(np.stack([o.lo for o in out]), 0, ax), np.moveaxis(np.stack([o.hi for o in out]), 0, ax))
    for ax in range(3):
        T = along(T, ax)
    return IC(T.lo, T.hi)


def cell_coef(cell):
    """the cell's polynomial: from the nodal values when present (the certified function), else the float coefficients"""
    if "nodal" in cell:
        return nodal_coef(cell["nodal"])
    return np.array(cell["coef"], float)


def continuity(data):
    """exact continuity of the nodal piecewise polynomial: (a) nodes that coincide (same sigma, same point of the
    sphere) carry bit-identical values, (b) the mesh is conforming: every cell face is shared by exactly two cells
    or lies on sigma_lo, sigma_hi or the equator, (c) every sigma_lo node equals phi_lo (the constant below).
    Tensor Lagrange elements restricted to a face depend only on that face's nodes, and the cube seams of the
    equiangular gnomonic charts match linearly in the edge coordinate, so (a)-(b) give a continuous function.
    Returns (ok, message)."""
    cells = data["cells"]
    if not all("nodal" in c for c in cells):
        return False, "continuity NOT checked (cells without nodal values)"
    faces = data["faces"]
    slo, shi = float(data["low"]["sigma_lo"]), float(data["far"]["sigma_hi"])
    vlo = float(data["low"]["phi_lo"])
    val, facecount, bad = {}, {}, []

    def nkey(f, sg, ua, ub):
        fc = faces[f]
        C, A, B = (np.array(fc[k], float) for k in ("centre", "ea", "eb"))
        v = C + math.tan(math.pi * ua / 4) * A + math.tan(math.pi * ub / 4) * B
        v = v / np.linalg.norm(v)
        return (round(sg, 9),) + tuple(np.round(v, 9) + 0.0)
    for ci, c in enumerate(cells):
        V = np.asarray(c["nodal"]["vals"], float)
        s0, s1 = c["sigma"]
        sg = [s0, 0.5 * (s0 + s1), s1] if V.shape[0] == 3 else [s0, s1]
        uas, ubs = c["ua"], c["ub"]
        keys = {}
        for i, sv in enumerate(sg):
            for j in range(2):
                for k in range(2):
                    key = nkey(c["face"], sv, uas[j], ubs[k])
                    keys[(i, j, k)] = key
                    x = V[i, j, k]
                    if key in val and val[key] != x:
                        bad.append((ci, key, val[key], x))
                    val.setdefault(key, x)
                    if abs(sv - slo) < 1e-12 and x != vlo:
                        bad.append((ci, "sigma_lo node", x, vlo))
        ni = V.shape[0] - 1
        cf = [frozenset(keys[(i, j, k)] for i in range(ni + 1) for j in range(2) for k in range(2) if (j, k)[a] == e)
              for a in (0, 1) for e in (0, 1)]
        cf += [frozenset(keys[(i, j, k)] for j in range(2) for k in range(2)) for i in (0, ni)]
        tags = [("ua", 0), ("ua", 1), ("ub", 0), ("ub", 1), ("s", 0), ("s", 1)]
        for fk, tg in zip(cf, tags):
            facecount.setdefault(fk, []).append((ci, tg))
    for fk, occ in facecount.items():
        if len(occ) == 2:
            continue
        if len(occ) > 2:
            bad.append(("face shared by more than two cells", occ))
            continue
        ci, (ax, e) = occ[0]
        c = cells[ci]
        side_face = faces[c["face"]]["ub_range"][0] == 0.0
        on_bd = ((ax == "s" and abs(c["sigma"][e] - (slo if e == 0 else shi)) < 1e-12)
                 or (ax == "ub" and e == 0 and side_face and c["ub"][0] == 0.0))
        if not on_bd:
            bad.append(("unmatched face", ci, ax, e))
    if bad:
        return False, f"continuity FAILED: {len(bad)} problems, e.g. {bad[:3]}"
    return True, f"continuity: {len(val)} distinct nodes, {len(facecount)} faces, all matched"


def geometry(face, UA, UB):
    """gnomonic chart of one cube face: n = (C + X A + Y B)/r, X = tan(pi ua/4), Y = tan(pi ub/4), r^2 = 1 + X^2 + Y^2
    (C, A, B an exactly orthonormal frame, entries enclosed). On the sphere the chart's inverse metric is
    r^2 (Id + x x^T), x = (X, Y), so for the sphere gradient D:
        |D phi|^2 = r^2 (phi_X^2 + phi_Y^2 + P^2),  P = X phi_X + Y phi_Y,
        D(n . e) . D phi = r (a phi_X + b phi_Y - c P),  (a, b, c) = (A . e, B . e, C . e),
    closed forms that avoid the dependency blow-up of a dual basis built from interval tangent vectors."""
    C, EA, EB = face["_C"], face["_A"], face["_B"]
    X, Y = tan_iv(UA * PI4), tan_iv(UB * PI4)
    r2 = X.sq() + Y.sq() + 1.0
    r = r2.sqrt()
    n = [(X * EA[i] + Y * EB[i] + C[i]) / r for i in range(3)]
    geo = {"X": X, "Y": Y, "r": r, "r2": r2, "dXa": PI4 * (X.sq() + 1.0), "dYb": PI4 * (Y.sq() + 1.0), "face": face}
    return n, geo, np.zeros(X.lo.shape, bool)


def sphere_grad(geo, pa, pb):
    """chart derivatives from the cell derivatives pa = d phi / d ua, pb = d phi / d ub; returns (phi_X, phi_Y, P, |D phi|^2)"""
    fx, fy = pa / geo["dXa"], pb / geo["dYb"]
    P = geo["X"] * fx + geo["Y"] * fy
    return fx, fy, P, geo["r2"] * (fx.sq() + fy.sq() + P.sq())


def lin_grad_dot(geo, e, fx, fy, P):
    """D(n . e) . D phi"""
    f = geo["face"]
    a, b, c = dot(f["_A"], e), dot(f["_B"], e), dot(f["_C"], e)
    return geo["r"] * (fx * a + fy * b - P * c)


def g_and_grad(side, n):
    """g(n), the ambient vector e with g a function of s = n . e, and flags where g is not smooth"""
    if side == "E":
        return I(1.0) - n[0], E1, np.zeros(n[0].lo.shape, bool)
    s = dot(n, NI)
    h2 = I(1.0) - s.sq()
    flag = h2.lo <= 1e-12
    h2 = I(np.maximum(h2.lo, 1e-300), np.maximum(h2.hi, 1e-300))
    return I(1.0) - h2.sqrt(), NI, flag


def factor(lo, hi):
    """2/(hi - lo) as an interval"""
    return I(2.0) / (I(float(hi)) - float(lo))


def dg_dot(side, n, geo, fx, fy, P, D2, flag):
    """Dg . Dphi: g_E = 1 - n . e1 gives -D(n . e1) . Dphi; g_I = 1 - sqrt(1 - s^2), s = n . n_I, gives (s/h) Ds . Dphi.
    On boxes touching the axis n = +-n_I (the cone of g_I) the a.e. bound |Dg . Dphi| <= |Dphi| (off the axis
    |Dg| = |s| <= 1), enough for an a.e., hence (H convex in the gradient) viscosity, subsolution"""
    if side == "E":
        val = -lin_grad_dot(geo, E1, fx, fy, P)
        G2 = I(1.0) - n[0].sq()                 # |Dg_E|^2 = 1 - n_1^2
    else:
        s = dot(n, NI)
        h2 = I(1.0) - s.sq()
        h2 = I(np.maximum(h2.lo, 1e-300), np.maximum(h2.hi, 1e-300))
        val = s / h2.sqrt() * lin_grad_dot(geo, NI, fx, fy, P)
        G2 = s.sq()                             # |Dg_I|^2 = s^2 off the axis
        val = I(np.where(flag, -np.inf, val.lo), np.where(flag, np.inf, val.hi))
    # Cauchy-Schwarz |Dg . Dphi| <= |Dg| |Dphi| (a.e.; on the axis of g_I with |Dg| <= 1): intersect both enclosures
    m = up(np.sqrt(up(np.minimum(np.maximum(G2.hi, 0.0), 1.0) * np.maximum(D2.hi, 0.0))))
    return I(np.maximum(val.lo, -m), np.minimum(val.hi, m))


def U_lower(n):
    tot = np.zeros(n[0].lo.shape)
    for u in UV:
        q = I(1.0) - dot(u, n)
        qhi = np.maximum(q.hi, 1e-300)
        tot = tot + dn(1.0 / up(np.sqrt(up(qhi))))
    return dn(tot)


def F_lower(side, face, cell, c, dc, S, UA, UB):
    """lower bound of F over sub-boxes S x UA x UB (interval arrays) of one cell"""
    (s0, s1), (a0, a1), (b0, b1) = cell["sigma"], cell["ua"], cell["ub"]
    fs, fa, fb = factor(s0, s1), factor(a0, a1), factor(b0, b1)        # 2/(hi - lo), enclosed
    y = [(S - s0) * fs - 1.0, (UA - a0) * fa - 1.0, (UB - b0) * fb - 1.0]
    n, geo, degen = geometry(face, UA, UB)
    g, gg, flag = g_and_grad(side, n)
    phi = poly(c, y)
    ps = poly(dc[0], y) * fs
    pa = poly(dc[1], y) * fa
    pb = poly(dc[2], y) * fb
    fx, fy, P, D2 = sphere_grad(geo, pa, pb)
    em = exp_iv(-S)
    rest = -((ps.sq() + D2) * em * 2.0) - phi / I(3.0) + (I(dn(4.0 / 3.0), up(4.0 / 3.0)) - g) * ps \
        - dg_dot(side, n, geo, fx, fy, P, D2, flag)
    eh = exp_iv(S * (-0.5))
    low = dn(eh.lo * U_lower(n) + rest.lo)
    return np.where(degen, -np.inf, low), flag


def subdivide(boxes, dims):
    """split each box (array (m, 2*len(dims))) in half along every dimension in dims"""
    out = [boxes]
    for d in dims:
        nxt = []
        for b in out:
            mid = 0.5 * (b[:, 2 * d] + b[:, 2 * d + 1])
            l, h = b.copy(), b.copy()
            l[:, 2 * d + 1] = mid
            h[:, 2 * d] = mid
            nxt += [l, h]
        out = [np.concatenate(nxt)]
    return out[0]


def check_cell(side, face, cell, max_depth):
    c = cell_coef(cell)
    dc = [dpoly(c, a) for a in range(3)]
    boxes = np.array([[*cell["sigma"], *cell["ua"], *cell["ub"]]], float)
    worst, nb = np.inf, 0
    for depth in range(max_depth + 1):
        S, UA, UB = I(boxes[:, 0], boxes[:, 1]), I(boxes[:, 2], boxes[:, 3]), I(boxes[:, 4], boxes[:, 5])
        lb, fl = F_lower(side, face, cell, c, dc, S, UA, UB)
        nb += boxes.shape[0]
        good = lb > 0
        if np.any(good):
            worst = min(worst, float(np.min(lb[good])))
        if np.all(good):
            return True, worst, nb, 0
        # early rejection (sound: a cell is only ever accepted when every box passes): a negative value at the centre
        # of a box that is not a flagged one means the cell fails; no need to bisect to max_depth
        bad = boxes[~good]
        m = 0.5 * (bad[:, 0::2] + bad[:, 1::2])
        lm, flm = F_lower(side, face, cell, c, dc, I(m[:, 0]), I(m[:, 1]), I(m[:, 2]))
        if np.any((lm < 0) & ~flm):
            return False, float(np.min(lm)), nb, int(np.sum(fl & ~good))
        if depth == max_depth:
            return False, float(np.min(lb)), nb, int(np.sum(fl & ~good))
        boxes = subdivide(boxes[~good], (0, 1, 2))


def check_cell_chunked(side, face, cell, max_depth, chunk=20000):
    """the same test as check_cell (every box either passes or is bisected in all 3 directions up to max_depth; the
    same early rejection at box centres), but the failing boxes are processed from a stack in chunks of at most
    `chunk`, so memory stays bounded at any depth. Same verdict as check_cell; the returned worst bound is the minimum
    over the accepted boxes (pass) or a negative bound (fail)."""
    c = cell_coef(cell)
    dc = [dpoly(c, a) for a in range(3)]
    stack = [(np.array([[*cell["sigma"], *cell["ua"], *cell["ub"]]], float), 0)]
    worst, nb = np.inf, 0
    while stack:
        boxes, depth = stack.pop()
        if boxes.shape[0] > chunk:
            for i in range(0, boxes.shape[0], chunk):
                stack.append((boxes[i:i + chunk], depth))
            continue
        S, UA, UB = I(boxes[:, 0], boxes[:, 1]), I(boxes[:, 2], boxes[:, 3]), I(boxes[:, 4], boxes[:, 5])
        lb, fl = F_lower(side, face, cell, c, dc, S, UA, UB)
        nb += boxes.shape[0]
        good = lb > 0
        if np.any(good):
            worst = min(worst, float(np.min(lb[good])))
        if np.all(good):
            continue
        bad = boxes[~good]
        m = 0.5 * (bad[:, 0::2] + bad[:, 1::2])
        lm, flm = F_lower(side, face, cell, c, dc, I(m[:, 0]), I(m[:, 1]), I(m[:, 2]))
        if np.any((lm < 0) & ~flm):
            return False, float(np.min(lm)), nb, int(np.sum(fl & ~good))
        if depth == max_depth:
            return False, float(np.min(lb)), nb, int(np.sum(fl & ~good))
        stack.append((subdivide(bad, (0, 1, 2)), depth + 1))
    return True, worst, nb, 0


def far_lower(side, face, cell, c, dc1, dc2, s, UA, UB):
    """lower bound of the far-field condition on sub-boxes of the top face sigma = s of a cell"""
    (s0, s1), (a0, a1), (b0, b1) = cell["sigma"], cell["ua"], cell["ub"]
    fa, fb = factor(a0, a1), factor(b0, b1)
    y = [I(np.ones(UA.lo.shape)), (UA - a0) * fa - 1.0, (UB - b0) * fb - 1.0]
    n, geo, degen = geometry(face, UA, UB)
    g, gg, flag = g_and_grad(side, n)
    es = exp_iv(I(np.full(UA.lo.shape, 0.5 * s)))
    A = poly(c, y) * es
    pa = poly(dc1, y) * fa * es
    pb = poly(dc2, y) * fb * es
    fx, fy, P, DA2 = sphere_grad(geo, pa, pb)
    e3 = exp_iv(I(np.full(UA.lo.shape, -1.5 * s)))
    rest = -(A * (I(1.0) - g * 0.5)) - dg_dot(side, n, geo, fx, fy, P, DA2, flag) - (A.sq() * 0.25 + DA2) * e3 * 2.0
    return np.where(degen, -np.inf, dn(U_lower(n) + rest.lo))


def check_far(side, face, cell, s, max_depth):
    c = cell_coef(cell)
    dc1, dc2 = dpoly(c, 1), dpoly(c, 2)
    boxes = np.array([[*cell["ua"], *cell["ub"]]], float)
    worst = np.inf
    for depth in range(max_depth + 1):
        UA, UB = I(boxes[:, 0], boxes[:, 1]), I(boxes[:, 2], boxes[:, 3])
        lb = far_lower(side, face, cell, c, dc1, dc2, s, UA, UB)
        good = lb > 0
        if np.any(good):
            worst = min(worst, float(np.min(lb[good])))
        if np.all(good):
            return True, worst
        if depth == max_depth:
            return False, float(np.min(lb))
        boxes = subdivide(boxes[~good], (0, 1))


_W = {}


def _init(path, max_depth):
    try:                                     # background job: below-normal priority (no effect on the results)
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    data = json.load(open(path, encoding="utf-8"))
    _W.update(side=data["side"], faces=exact_frames(data["faces"]), cells=data["cells"], md=max_depth)


def _work(rng):
    out = []
    for idx in range(*rng):
        cell = _W["cells"][idx]
        chk = check_cell_chunked if os.environ.get("VERIFY_CHUNKED") == "1" else check_cell
        out.append(chk(_W["side"], _W["faces"][cell["face"]], cell, _W["md"]))
    return rng[0], out


def main():
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    workers = int(os.environ.get("VERIFY_WORKERS", "4"))
    side, faces = data["side"], exact_frames(data["faces"])
    ok_all, worst, nboxes, bad = True, np.inf, 0, []
    ncell = len(data["cells"])
    chunks = [(a, min(a + 64, ncell)) for a in range(0, ncell, 64)]
    res = [None] * ncell
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers, _init, (sys.argv[1], max_depth)) as pool:
            for a, out in pool.imap_unordered(_work, chunks):
                res[a:a + len(out)] = out
    else:
        _init(sys.argv[1], max_depth)
        for ch in chunks:
            a, out = _work(ch)
            res[a:a + len(out)] = out
    for idx, (ok, wv, nb, nfl) in enumerate(res):
        nboxes += nb
        worst = min(worst, wv)
        if not ok:
            ok_all = False
            bad.append((idx, wv, nfl))
    print(f"side {side}: {len(data['cells'])} cells, {nboxes} boxes; cell inequality on all cells: {ok_all}; worst "
          f"lower bound {worst:.4e}" + (f"; failing cells (index, bound, flagged boxes): {bad[:10]}" if bad else ""))
    if "far" in data:
        s = float(data["far"]["sigma_hi"])
        okf, wf = True, np.inf
        for cell in data["cells"]:
            if abs(cell["sigma"][1] - s) < 1e-12:
                o, w = check_far(side, faces[cell["face"]], cell, s, max_depth)
                okf &= o
                wf = min(wf, w)
        print(f"  far field (sigma >= {s}): {okf}; worst {wf:.4e}")
        ok_all &= okf
    if "low" in data:
        s, v = float(data["low"]["sigma_lo"]), float(data["low"]["phi_lo"])
        okl = up(v) <= dn(9.0 * float(exp_iv(I(-0.5 * s)).lo))
        print(f"  low end (sigma <= {s}, phi = {v}): {okl}")
        ok_all &= bool(okl)
    okc, msg = continuity(data)
    print("  " + msg)
    print("ALL VERIFIED" if ok_all and okc else ("CELL CONDITIONS VERIFIED, CONTINUITY NOT" if ok_all else "NOT VERIFIED"))


if __name__ == "__main__":
    main()
