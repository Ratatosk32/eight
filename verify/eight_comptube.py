"""Stage 21, hypothesis (R) by comparison tubes around thin trajectories (theorist, paper/review/eight_schemeB_theorist.md
section 2): rigorous version.

Scaled system at a fixed eps: H = 2|Y||P|^2 - eps U(Y), Y' = -H_P, P' = H_Y; cone metric g = |dY|^2/(4|Y|) and
L = (1/2)|Y'|_g^2 + eps U. For a family gamma_lam of solutions, J = d gamma/d lam satisfies nabla^2 J = -Rop J with
Rop = Rm(., v) v - eps Hess_g U, so (|J|_g, |nabla J|_g) <= (y1, y2), y1' = y2, y2' = k^2 y1, k^2 >= ||Rop||_g on the
tube. A parameter box (eps, phi, M) is, by scaling, the family at the centre's eps_c with start radius e^ell,
|ell| <= (2/3) max|log(eps/eps_c)|; along a label segment |J(0)|_g <= (1/2) sqrt(e^dl) sqrt(dl^2 + dphi^2),
|nabla J(0)|_g <= sqrt(e^dl) (|M|max dl + 2 dM).
Tube conversions (rigorous): a g-curve of length rho from Y_c stays in |Y - Y_c| <= 2 (sqrt|Y_c| + rho) rho
(since |d sqrt|Y|| <= |dY|_g); the coordinate velocity spread |dv| <= (A + B |v_c|)/(1 - B) with
A = 2 (sqrt|Y_c| + rho) y2, B = 3 (sqrt|Y_c| + rho) y1 / (sqrt|Y_c| - rho)^2 (Christoffel term, |grad f| = 1/(2|Y|)).
Rop over a state box: centred form Q(box) in Q(mid) + sum_d dQ/dz_d(box) [-r_d, r_d] with first-order jets in the
6 state directions; eigenvalue bounds by interval Cholesky (fallback Gershgorin), relative to g (c = 1/(4|Y|)).
Self-consistency: guesses (rho*, vel*) per piece; tube closed if y1 <= rho*, y2 <= vel* at every piece end (y is
non-decreasing). Sturm: y'' + W y = 0 with W >= w on the same tube, positive on [0, 1] => X(1) invertible.
    python verify/eight_comptube.py test"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eight_field as EF  # noqa: E402
import eight_field_lohner as L  # noqa: E402
import eight_riccati_barrier as RB  # noqa: E402
from prove_newton_1212 import I, dn, jmul, jpow_m32, jrecip, jsq, jsqrt, peval, tpow, up  # noqa: E402

ND = 6        # jet directions: Y1..3, P1..3


def jconst(x, like):
    """constant (interval or float) as a jet shaped like `like`"""
    x = x if isinstance(x, I) else I(float(x))
    lo, hi = np.zeros_like(like.lo), np.zeros_like(like.hi)
    lo[0], hi[0] = x.lo, x.hi
    return I(lo, hi)


def jseed(Zbox):
    """jets (1 + 6, nb) for the 6 state boxes Zbox (list of I (nb,))"""
    nb = Zbox[0].lo.shape[0]
    out = []
    for i in range(ND):
        lo = np.zeros((1 + ND, nb))
        hi = np.zeros((1 + ND, nb))
        lo[0], hi[0] = Zbox[i].lo, Zbox[i].hi
        lo[1 + i] = hi[1 + i] = 1.0
        out.append(I(lo, hi))
    return out


def jsum(xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = acc + x
    return acc


def rop_jets(Z, eps):
    """Q (3x3 of jets) and c = 1/(4|Y|) (jet) from state jets Z = [Y1, Y2, Y3, v1, v2, v3] (v = Y' = -4|Y|P, the
    velocity: Rop depends on v, not on P separately); eps a float or array"""
    Y, v = Z[0:3], Z[3:6]
    s = jsum([jsq(y) for y in Y])
    R = jsqrt(s)
    iR = jrecip(R)
    Yh = [jmul(y, iR) for y in Y]
    vr = jsum([jmul(Yh[i], v[i]) for i in range(3)])
    vT = [v[i] - jmul(Yh[i], vr) for i in range(3)]
    vT2 = jsum([jsq(x) for x in vT])
    iR2 = jsq(iR)
    k3 = jmul(iR2, iR) * (3.0 / 16.0)
    PT = [[(jconst(1.0, R) if i == j else jconst(0.0, R)) - jmul(Yh[i], Yh[j]) for j in range(3)] for i in range(3)]
    proj = [[jmul(PT[i][j], iR) for j in range(3)] for i in range(3)]
    HU = [[jconst(0.0, R) for _ in range(3)] for _ in range(3)]
    gU = [jconst(0.0, R) for _ in range(3)]
    for u in EF.UK:
        ud = jsum([Y[i] * u[i] for i in range(3)])
        perp = [Y[i] - ud * u[i] for i in range(3)]
        if np.all(ud.lo[0] > 0):
            q = jmul(jsum([jsq(x) for x in perp]), jrecip(R + ud))
            g = [jmul(perp[i] - q * u[i], iR) for i in range(3)]
        else:
            q = R - ud
            g = [Yh[i] - jconst(u[i], R) for i in range(3)]
        q32 = jpow_m32(q)
        q52 = jmul(q32, jrecip(q))
        for i in range(3):
            gU[i] = gU[i] - jmul(q32, g[i]) * 0.5
            for j in range(3):
                HU[i][j] = HU[i][j] + jmul(q52, jmul(g[i], g[j])) * 0.75 - jmul(q32, proj[i][j]) * 0.5
    gf = [jmul(Y[i], iR2) * (-0.5) for i in range(3)]
    gfgU = jsum([jmul(gf[i], gU[i]) for i in range(3)])
    Q = [[jmul(k3, jmul(vT2, PT[i][j]) - jmul(vT[i], vT[j]))
          - (HU[i][j] - (jmul(gf[i], gU[j]) + jmul(gU[i], gf[j])) + (gfgU if i == j else jconst(0.0, R))) * eps
          for j in range(3)] for i in range(3)]
    return Q, iR * 0.25


def rop_split(Zlo, Zhi, eps, nsplit=None, balls=None):
    """enclosure of Q over the boxes, as the hull of centred enclosures over nsplit^6 sub-boxes. balls = (Clo, Chi, rY,
    rv): the tube lies in {Y within Euclidean distance rY of the box [Clo, Chi]_Y} x {v within rv of [Clo, Chi]_v};
    sub-boxes whose distance (rounded down) exceeds the radius are dropped (marked by an infinite c so they never
    count in the maximum)"""
    nsplit = NSPLIT if nsplit is None else nsplit
    if nsplit == 1:
        return rop_centred(Zlo, Zhi, eps)
    nb = Zlo.shape[1]
    grids = np.array(np.meshgrid(*[np.arange(nsplit)] * ND, indexing="ij")).reshape(ND, -1)      # (6, ns)
    ns = grids.shape[1]
    w = (Zhi - Zlo) / nsplit
    lo = (Zlo[:, None, :] + grids[:, :, None] * w[:, None, :]).reshape(ND, ns * nb)
    hi = (Zlo[:, None, :] + (grids[:, :, None] + 1) * w[:, None, :]).reshape(ND, ns * nb)
    lo = np.where(grids[:, :, None] == 0, Zlo[:, None, :], lo.reshape(ND, ns, nb)).reshape(ND, ns * nb)
    hi = np.where(grids[:, :, None] == nsplit - 1, Zhi[:, None, :], hi.reshape(ND, ns, nb)).reshape(ND, ns * nb)
    lo, hi = dn(lo), up(hi)
    keep = np.ones(ns * nb, bool)
    if balls is not None:
        Clo, Chi, rY, rv = balls
        Clo_r = np.repeat(Clo[:, None, :], ns, 1).reshape(ND, ns * nb)
        Chi_r = np.repeat(Chi[:, None, :], ns, 1).reshape(ND, ns * nb)
        gap = np.maximum(0.0, np.maximum(dn(lo - Chi_r), dn(Clo_r - hi)))
        dY = dn(np.sqrt(np.maximum(dn(np.sum(gap[0:3] ** 2, 0)), 0.0)))      # lower bounds of the distances
        dv = dn(np.sqrt(np.maximum(dn(np.sum(gap[3:6] ** 2, 0)), 0.0)))
        dY, dv = np.maximum(dY, 0.0), np.maximum(dv, 0.0)
        keep = (dY <= np.repeat(rY[None, :], ns, 0).reshape(-1)) & (dv <= np.repeat(rv[None, :], ns, 0).reshape(-1))
    epsr = np.repeat(np.asarray(eps, float)[None, :], ns, 0).reshape(-1) if np.ndim(eps) else eps
    Q, c = rop_centred(lo, hi, epsr)
    Q = [[I(np.where(keep, Q[i][j].lo, 0.0), np.where(keep, Q[i][j].hi, 0.0)) for j in range(3)] for i in range(3)]
    c = I(np.where(keep, c.lo, np.inf), np.where(keep, c.hi, np.inf))
    Qh = [[I(Q[i][j].lo.reshape(ns, nb).min(0), Q[i][j].hi.reshape(ns, nb).max(0)) for j in range(3)] for i in range(3)]
    # eigenvalue bounds must be taken per sub-box, then maximised: return per-sub-box Q too
    return Qh, I(c.lo.reshape(ns, nb).min(0), c.hi.reshape(ns, nb).max(0)), (Q, c, ns, nb)


NSPLIT = int(os.environ.get("CT_SPLIT", "2"))
GUESS = float(os.environ.get("CT_GUESS", "1.6"))
BALLS = int(os.environ.get("CT_BALLS", "1"))


def rop_centred(Zlo, Zhi, eps):
    """centred enclosure of Q over the boxes [Zlo, Zhi] (arrays (6, nb)) and of c: returns (Q 3x3 of I (nb,), c I)"""
    mid = 0.5 * (Zlo + Zhi)
    rad = up(np.maximum(Zhi - mid, mid - Zlo))
    Zm = [I(mid[i]) for i in range(ND)]
    Zb = [I(Zlo[i], Zhi[i]) for i in range(ND)]
    Qm, cm = rop_jets(jseed(Zm), eps)
    Qb, cb = rop_jets(jseed(Zb), eps)
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            acc = I(Qm[i][j].lo[0], Qm[i][j].hi[0])
            for d in range(ND):
                D = I(Qb[i][j].lo[1 + d], Qb[i][j].hi[1 + d])
                acc = acc + D * I(-rad[d], rad[d])
            # intersect with the natural enclosure (both are valid)
            nat = I(Qb[i][j].lo[0], Qb[i][j].hi[0])
            row.append(I(np.maximum(acc.lo, nat.lo), np.minimum(acc.hi, nat.hi)))
        out.append(row)
    c = I(cb.lo[0], cb.hi[0])
    return RB.sym(out), c


def eig_bounds(Q):
    """rigorous upper bound of lambda_max(Q) and lower bound of lambda_min(Q) (arrays (nb,))"""
    mid = np.stack([np.stack([Q[i][j].mid for j in range(3)], -1) for i in range(3)], -2)
    rad = np.stack([np.stack([Q[i][j].wid * 0.5 for j in range(3)], -1) for i in range(3)], -2)
    ev = np.linalg.eigvalsh(mid)
    rn = np.sqrt(np.sum(rad ** 2, axis=(-2, -1))) + 1e-10 * (1 + np.max(np.abs(mid), axis=(-2, -1)))
    off = [sum(np.maximum(np.abs(Q[i][j].lo), np.abs(Q[i][j].hi)) for j in range(3) if j != i) for i in range(3)]
    ger_hi = up(np.max(np.stack([Q[i][i].hi + off[i] for i in range(3)]), 0))
    ger_lo = dn(np.min(np.stack([Q[i][i].lo - off[i] for i in range(3)]), 0))
    mu = ev[..., -1] + rn
    ok = RB.chol_pd([[I(mu) * (1.0 if i == j else 0.0) - Q[i][j] for j in range(3)] for i in range(3)])
    lmax = np.where(ok, mu, ger_hi)
    nu = ev[..., 0] - rn
    ok2 = RB.chol_pd([[Q[i][j] - I(nu) * (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)])
    lmin = np.where(ok2, nu, ger_lo)
    return lmax, lmin


def rel_bounds(Q, c):
    """w >= largest eigenvalue of Rop relative to g, k2 >= operator norm relative to g"""
    lmax, lmin = eig_bounds(Q)
    w = np.where(lmax >= 0, up(lmax / c.lo), up(lmax / c.hi))
    k2 = up(np.maximum(np.abs(lmax), np.abs(lmin)) / c.lo)
    return w, k2


# ---------------------------------------------------------------- per-box pipeline
def box_offsets(Pb):
    """centre (eps_c, phi_c, M_c) and label offsets (dl, dphi, dM) for parameter boxes Pb (nb, 6) = lo/hi pairs"""
    elo, ehi = Pb[:, 0], Pb[:, 1]
    ec = np.sqrt(elo * ehi) if np.all(elo > 0) else 0.5 * (elo + ehi)
    # upper bound of the ell offset; the factor covers the libm log (<= 1 ulp measured), the division and the product
    # (dl = 0 exactly for boxes thin in eps, which are all the boxes of the shadow cover and of the records)
    dl = up((2.0 / 3.0) * np.maximum(np.abs(np.log(elo / ec)), np.abs(np.log(ehi / ec))) * (1 + 1e-14))
    pc = np.stack([ec, 0.5 * (Pb[:, 2] + Pb[:, 3]), 0.5 * (Pb[:, 4] + Pb[:, 5])], 1)
    dphi = up(0.5 * (Pb[:, 3] - Pb[:, 2]))
    dM = up(0.5 * (Pb[:, 5] - Pb[:, 4]))
    return pc, dl, dphi, dM


def centre_pieces(pc, m):
    """thin Lohner enclosures of the centres' states on m pieces per Lohner step: list of (tau_a, tau_b, Zlo (6, nb),
    Zhi (6, nb)); raises if a centre cannot be validated"""
    P = I(pc.copy(), pc.copy())
    groups, failed = RB.lohner_split(P, rec_from=0.0)
    if failed or len(groups) != 1:
        raise RuntimeError("centre trajectory failed")
    _, _, steps = groups[0]
    out, km = [], []
    nz = min(8, len(steps[0][2])) if steps else 6
    for (ts, h, cs, Rm) in steps:
        for k in range(m):
            sa, sb = h * k / m, h * (k + 1) / m
            S = I(dn(sa), up(sb))
            tP = tpow(S, L.ORDER + 1)
            Z = [peval(cs[i], S) + Rm[i] * tP for i in range(nz)]
            out.append((ts + sa, ts + sb, np.stack([z.lo for z in Z[:6]]), np.stack([z.hi for z in Z[:6]])))
            if nz >= 8:      # the action integrals k, m of the centre (for records: W = (r/s)(k + eps m))
                km.append((np.stack([Z[6].lo, Z[7].lo]), np.stack([Z[6].hi, Z[7].hi])))
    centre_pieces.km = km
    return out


def tube_boxes(Zlo, Zhi, rho, vel):
    """state boxes containing the tube: positions within 2 (sqrt|Y|max + rho) rho of the centre box, coordinate
    velocities within (A + B |v_c|)/(1 - B); P = -v/(4|Y|) enclosed by interval arithmetic. Returns (lo, hi, ok)"""
    Yc = [I(Zlo[i], Zhi[i]) for i in range(3)]
    Pc = [I(Zlo[3 + i], Zhi[3 + i]) for i in range(3)]
    Rc = (Yc[0].sq() + Yc[1].sq() + Yc[2].sq()).sqrt()
    sRhi, sRlo = up(np.sqrt(Rc.hi)), dn(np.sqrt(Rc.lo))
    rho = np.where(np.isfinite(rho), rho, 1e300)
    vel = np.where(np.isfinite(vel), vel, 1e300)
    dY = up(up(2.0 * up(sRhi + rho)) * rho)
    vc = up(4.0 * up(Rc.hi * up(np.sqrt(sum(np.maximum(np.abs(p.lo), np.abs(p.hi)) ** 2 for p in Pc)))))
    A = up(up(2.0 * up(sRhi + rho)) * vel)
    den = dn(np.maximum(sRlo - rho, 1e-300) ** 2)
    B = up(up(3.0 * up(sRhi + rho)) * rho / den)
    ok = (B < 0.5) & (sRlo - rho > 0) & (dY < 0.3 * dn(Rc.lo))
    dv = up(up(A + up(B * vc)) / dn(1.0 - np.minimum(B, 0.5)))
    # boxes whose tube is already invalid (ok False) fail anyway: zero their radii so that one bad box cannot raise
    # an exception for the whole batch (no effect on the boxes that pass)
    dY = np.where(ok & np.isfinite(dY), dY, 0.0)
    dv = np.where(ok & np.isfinite(dv), dv, 0.0)
    lo = np.empty_like(Zlo)
    hi = np.empty_like(Zhi)
    Yb = [I(dn(Zlo[i] - dY), up(Zhi[i] + dY)) for i in range(3)]
    for i in range(3):
        lo[i], hi[i] = Yb[i].lo, Yb[i].hi
    for i in range(3):
        vci = Pc[i] * Rc * (-4.0)
        lo[3 + i], hi[3 + i] = dn(vci.lo - dv), up(vci.hi + dv)
    tube_boxes.last = (dY, dv)
    return lo, hi, ok


def vbox(Zlo, Zhi):
    """(Y, v) box of the centre enclosure (v = -4|Y|P)"""
    Y = [I(Zlo[i], Zhi[i]) for i in range(3)]
    R = (Y[0].sq() + Y[1].sq() + Y[2].sq()).sqrt()
    lo, hi = Zlo.copy(), Zhi.copy()
    for i in range(3):
        v = I(Zlo[3 + i], Zhi[3 + i]) * R * (-4.0)
        lo[3 + i], hi[3 + i] = v.lo, v.hi
    return lo, hi


ADAPT = int(os.environ.get("CT_ADAPT", "0"))   # measured: no speed gain (the unsplit bound is mostly too coarse)


def wk_bounds(lo, hi, eps, balls=None):
    """robust wrapper: if the batch evaluation raises (some box touches a collision ray), redo box by box and give
    the failing boxes w = k2 = inf (they fail; the others are unaffected)"""
    try:
        return _wk_bounds(lo, hi, eps, balls)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        nb = lo.shape[1]
        w, k2 = np.full(nb, np.inf), np.full(nb, np.inf)
        epsa = np.asarray(eps, float)
        for b_ in range(nb):
            sl = slice(b_, b_ + 1)
            bb = None if balls is None else tuple(x[:, sl] if np.ndim(x) == 2 else x[sl] for x in balls)
            try:
                w1, k1 = _wk_bounds(lo[:, sl], hi[:, sl], epsa[sl] if epsa.ndim else epsa, bb)
                w[b_], k2[b_] = w1[0], k1[0]
            except (ValueError, ZeroDivisionError, FloatingPointError):
                pass
        return w, k2


def _wk_bounds(lo, hi, eps, balls=None):
    """w, k2 over the (Y, v) boxes. Adaptive (CT_ADAPT=1): first the unsplit centred enclosure (rigorous, coarser);
    boxes whose bound exceeds the midpoint value by more than 30 % + 0.3 are redone with sub-boxes (also rigorous).
    Either branch is a valid upper bound, so the choice only affects speed and tightness."""
    if ADAPT and NSPLIT > 1:
        w1, k1 = rel_bounds(*rop_centred(lo, hi, eps))
        mid = 0.5 * (lo + hi)
        wm, km = rel_bounds(*rop_centred(mid, mid, eps))
        bad = ~((k1 <= 1.3 * km + 0.3) & (w1 <= 1.3 * np.abs(wm) + 0.3) & np.isfinite(k1) & np.isfinite(w1))
        if not np.any(bad):
            return w1, k1
        idx = np.flatnonzero(bad)
        epsb = np.asarray(eps)[idx] if np.ndim(eps) else eps
        w2, k2 = _wk_split(lo[:, idx], hi[:, idx], epsb)
        w1, k1 = w1.copy(), k1.copy()
        w1[idx], k1[idx] = np.minimum(w1[idx], w2), np.minimum(k1[idx], k2)
        return w1, k1
    return _wk_split(lo, hi, eps, balls)


def _wk_split(lo, hi, eps, balls=None):
    """w, k2 over the (Y, v) boxes: eigenvalue bounds per sub-box (dropped sub-boxes give -inf / 0), maximised"""
    res = rop_split(lo, hi, eps, balls=balls)
    if len(res) == 2:
        return rel_bounds(*res)
    _, _, (Q, c, ns, nb) = res
    dropped = ~np.isfinite(c.lo)
    cc = I(np.where(dropped, 1.0, c.lo), np.where(dropped, 1.0, c.hi))
    w, k2 = rel_bounds(Q, cc)
    w = np.where(dropped, -np.inf, w)
    k2 = np.where(dropped, 0.0, k2)
    return w.reshape(ns, nb).max(0), k2.reshape(ns, nb).max(0)


MODE = os.environ.get("CT_MODE", "box")   # "ball" (stage-1 form) measured coarser than box + 2^6 sub-boxes


def ball_bounds(Zlo, Zhi, rho, vel, eps):
    """theorist's stage 1 (paper/review/eight_comptube_tight_theorist.md 2.1-2.2), rigorous.
    Tube at the piece: Y = Ym + dY, |dY| <= rY; v = vm - Gamma_Ym(dY, vm) + delta, |delta| <= rd, where (Ym, vm) is the
    midpoint of the centre's (Y, v) enclosure on the piece, rY = rho_E + half-diagonal of the centre Y box,
    rho_E = 2 (sqrt|Y|max + rho) rho, and rd bounds A + rho_E (3/(2|Y|min)) Dold + rho_E^2 (3/(2|Y|min^2)) |v|max
    + the centre's own velocity spread (with A = 2 (sqrt|Y|max + rho) vel, Dold the plain coordinate radius).
    T = 4|Y| Q (eigenvalues = eigenvalues of Rop relative to g). Mean value form with interval first derivatives over
    the whole set: lam_max(T) <= lam_max(T(m)) + rY sqrt(lam_max(sum R_i^2)) + rd sqrt(lam_max(sum S_j^2)), and the
    same downwards for lam_min. Returns (w, k2, ok)."""
    nb = Zlo.shape[1]
    Zv_lo, Zv_hi = vbox(Zlo, Zhi)
    Ym, vm = 0.5 * (Zv_lo[0:3] + Zv_hi[0:3]), 0.5 * (Zv_lo[3:6] + Zv_hi[3:6])
    hY = up(np.sqrt(np.sum(up(np.maximum(Zv_hi[0:3] - Ym, Ym - Zv_lo[0:3])) ** 2, 0)))
    hv = up(np.sqrt(np.sum(up(np.maximum(Zv_hi[3:6] - vm, vm - Zv_lo[3:6])) ** 2, 0)))
    Rm = np.sqrt(np.sum(Ym ** 2, 0))
    sRhi = up(np.sqrt(up(Rm + hY)))
    rhoE = up(up(2.0 * up(sRhi + rho)) * rho)
    rY = up(rhoE + hY)
    Rmin = dn(Rm - rY)
    ok = Rmin > 0.3 * Rm
    Rmin = np.maximum(Rmin, 1e-300)
    vmax = up(np.sqrt(np.sum(vm ** 2, 0)) + hv)
    A = up(up(2.0 * up(sRhi + rho)) * vel)
    Bo = up(up(3.0 * up(sRhi + rho)) * rho / dn(np.maximum(dn(np.sqrt(Rmin)) - rho, 1e-300) ** 2))
    ok &= Bo < 0.5
    Dold = up(up(A + up(Bo * vmax)) / dn(1.0 - np.minimum(Bo, 0.5)))
    rd = up(A + up(rhoE * up(1.5 / Rmin) * Dold) + up(rhoE * rhoE * up(1.5 / dn(Rmin * Rmin)) * vmax) + hv)
    # jets in (dY, delta) over the set; Gamma_Ym(d, vm) = (f.d) vm + (f.vm) d - (d.vm) f, f = -Ym/(2|Ym|^2)
    f = -Ym / (2 * Rm ** 2)
    fv = np.sum(f * vm, 0)
    Z = []
    for i in range(3):
        lo = np.zeros((1 + ND, nb)); hi = np.zeros((1 + ND, nb))
        Yi = I(Ym[i]) + I(-rY, rY)
        lo[0], hi[0] = Yi.lo, Yi.hi
        lo[1 + i] = hi[1 + i] = 1.0
        Z.append(I(lo, hi))
    dbox = [I(-rY, rY) for _ in range(3)]
    for jdir in range(3):
        lo = np.zeros((1 + ND, nb)); hi = np.zeros((1 + ND, nb))
        # v_j = vm_j - [(f.d) vm_j + fv d_j - (d.vm) f_j] + delta_j ; d/dd_i = -(f_i vm_j + fv [i==j] - vm_i f_j)
        G = I(vm[jdir]) - (sum((dbox[i] * f[i] for i in range(3)), I(np.zeros(nb))) * vm[jdir] + dbox[jdir] * fv
                           - sum((dbox[i] * vm[i] for i in range(3)), I(np.zeros(nb))) * f[jdir]) + I(-rd, rd)
        lo[0], hi[0] = G.lo, G.hi
        for i in range(3):
            dvi = -(f[i] * vm[jdir] + (fv if i == jdir else 0.0) - vm[i] * f[jdir])
            Dv = I(dvi)                         # float products: enclose by 2 ulps
            lo[1 + i], hi[1 + i] = RB.push3(Dv.lo, -np.inf), RB.push3(Dv.hi, np.inf)
        lo[4 + jdir] = hi[4 + jdir] = 1.0
        Z.append(I(lo, hi))
    Qb, cb = rop_jets(Z, eps)
    # midpoint value (thin)
    Zm = []
    for i in range(3):
        Zm.append(jconst(I(Ym[i]), Z[0]))
    for jdir in range(3):
        Zm.append(jconst(I(vm[jdir]), Z[0]))
    Qm, cm = rop_jets(Zm, eps)
    invcm = I(cm.lo[0], cm.hi[0]).recip()     # 1/c = 4|Y| at the midpoint (value part only)
    Tm = [[I(Qm[i][j].lo[0], Qm[i][j].hi[0]) * invcm for j in range(3)] for i in range(3)]
    # derivative of T = Q * (4|Y|) = Q / c: dT = dQ/c - Q dc/c^2 (jets), take over the set
    invc = jrecip(cb)
    T = [[jmul(Qb[i][j], invc) for j in range(3)] for i in range(3)]
    Ri = [[[I(T[a][b].lo[1 + i], T[a][b].hi[1 + i]) for b in range(3)] for a in range(3)] for i in range(3)]
    Sj = [[[I(T[a][b].lo[4 + j], T[a][b].hi[4 + j]) for b in range(3)] for a in range(3)] for j in range(3)]

    def sumsq(Ms):
        out = [[None] * 3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                acc = None
                for M in Ms:
                    t = None
                    for k in range(3):
                        x = M[a][k] * M[k][b]
                        t = x if t is None else t + x
                    acc = t if acc is None else acc + t
                out[a][b] = acc
        return RB.sym(out)
    lR, _ = eig_bounds(sumsq(Ri))
    lS, _ = eig_bounds(sumsq(Sj))
    first = up(up(rY * up(np.sqrt(np.maximum(lR, 0.0)))) + up(rd * up(np.sqrt(np.maximum(lS, 0.0)))))
    lmax, lmin = eig_bounds(RB.sym(Tm))
    w = up(lmax + first)
    k2 = up(np.maximum(np.abs(up(lmax + first)), np.abs(dn(lmin - first))))
    return w, k2, ok & np.isfinite(w) & np.isfinite(k2)


def comp_step(y1, y2, k2, h):
    """exact solution of y1' = y2, y2' = k2 y1 over h, rounded upward (y1, y2 >= 0, k2 >= 0)"""
    sk = up(np.sqrt(np.maximum(k2, 1e-300)))
    x = up(sk * h)
    ch = up(np.cosh(x) * (1 + 4e-16))
    sh = up(np.sinh(x) * (1 + 4e-16))
    shk = np.where(x > 1e-8, up(sh / dn(sk)), up(h * (1 + 1e-12)))
    n1 = up(up(y1 * ch) + up(y2 * shk))
    n2 = up(up(up(y1 * sk) * sh) + up(y2 * ch))
    return n1, n2


def gradU_g_box(lo, hi):
    """upper bound of |grad_g U| = 2 sqrt|Y| |grad_E U| over the Y boxes [lo, hi] (3, nb)"""
    Y = [I(lo[i], hi[i]) for i in range(3)]
    R = (Y[0].sq() + Y[1].sq() + Y[2].sq()).sqrt()
    bad = R.lo <= 0
    R = I(np.maximum(R.lo, 1e-300), R.hi)
    iR = R.recip()
    Yh = [y * iR for y in Y]
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
        bad |= q.lo <= 1e-12                          # (nearly) touches a collision ray: no bound, the box fails
        q = I(np.maximum(q.lo, 1e-12), np.maximum(q.hi, 1e-12))  # only changes boxes already marked bad
        q32 = (q * q.sqrt()).recip()
        for i in range(3):
            G[i] = G[i] - q32 * g[i] * 0.5
    gn = up(np.sqrt(sum(np.maximum(np.abs(g.lo), np.abs(g.hi)) ** 2 for g in G)))
    return np.where(bad, np.inf, up(up(2.0 * up(np.sqrt(R.hi))) * gn))


def check(Pb, m=8, iters=4, verbose=False, slab=None, save=False, sturm=True):
    """Pb (nb, 6). Returns (closed, sturm_ok, ymin_sturm, rho_end).
    slab = E: the boxes have eps in [0, E] (the small-eps slab, theorist eight_shadow_slab_theorist.md): centre at
    eps = E/2, no ell offset, Rop bounds taken at the two ends eps = 0 and eps = E (Rop is affine in eps, the largest
    eigenvalue and the norm are convex), comparison y2' = k^2 y1 + (E/2) sup|grad_g U| (the family varies eps)."""
    nb = Pb.shape[0]
    if slab is None:
        pc, dl, dphi, dM = box_offsets(Pb)
    else:
        pc = np.stack([np.full(nb, 0.5 * slab), 0.5 * (Pb[:, 2] + Pb[:, 3]), 0.5 * (Pb[:, 4] + Pb[:, 5])], 1)
        dl = np.zeros(nb)
        dphi = up(0.5 * (Pb[:, 3] - Pb[:, 2]))
        dM = up(0.5 * (Pb[:, 5] - Pb[:, 4]))
    pieces = centre_pieces(pc, m)
    eY = up(np.sqrt(np.exp(dl)) * (1 + 1e-15))
    Mmax = np.maximum(np.abs(Pb[:, 4]), np.abs(Pb[:, 5]))
    y10 = up(0.5 * eY * up(np.sqrt(dl ** 2 + dphi ** 2)))
    y20 = up(eY * up(Mmax * dl + 2.0 * dM))
    npc = len(pieces)
    # initial guess from the centre alone: k2 on the centre enclosure (no tube), comparison system, x 1.6
    Zv0 = [vbox(Zlo, Zhi) for (_, _, Zlo, Zhi) in pieces]
    y1, y2 = y10.copy(), y20.copy()
    g1, g2 = [], []
    for k, (ta, tb, Zlo_, Zhi_) in enumerate(pieces):
        if MODE == "ball":
            _, k2c, _ = ball_bounds(Zlo_, Zhi_, np.zeros(nb), np.zeros(nb), pc[:, 0])
        else:
            _, k2c = rel_bounds(*rop_centred(Zv0[k][0], Zv0[k][1], pc[:, 0]))
        y1, y2 = comp_step(y1, y2, k2c, up(tb - ta))
        g1.append(y1)
        g2.append(y2)
    rho = np.maximum(GUESS * np.array(g1), 1e-12)
    vel = np.maximum(GUESS * np.array(g2), 1e-12)
    closed = np.zeros(nb, bool)
    Zv = [vbox(Zlo, Zhi) for (_, _, Zlo, Zhi) in pieces]
    for it in range(iters):
        ws, k2s, okb, srcs = [], [], np.ones(nb, bool), []
        saved_lo, saved_hi, saved_ok = [], [], []
        for k, (ta, tb, Zlo, Zhi) in enumerate(pieces):
            if MODE == "ball":
                w, k2, ok = ball_bounds(Zlo, Zhi, rho[k], vel[k], pc[:, 0])
                okb &= ok
                ws.append(np.where(ok, w, np.inf))
                k2s.append(np.where(ok, k2, np.inf))
                continue
            lo, hi, ok = tube_boxes(Zlo, Zhi, rho[k], vel[k])
            okb &= ok
            lo = np.where(ok, lo, Zv[k][0])
            hi = np.where(ok, hi, Zv[k][1])
            dYk, dvk = tube_boxes.last
            balls = (Zv[k][0], Zv[k][1], np.where(ok, dYk, 0.0), np.where(ok, up(dvk), 0.0)) if BALLS else None
            if slab is None:
                w, k2 = wk_bounds(lo, hi, pc[:, 0], balls)
            else:
                w0, k20 = wk_bounds(lo, hi, np.zeros(nb), balls)
                w1, k21 = wk_bounds(lo, hi, np.full(nb, float(slab)), balls)
                w, k2 = np.maximum(w0, w1), np.maximum(k20, k21)
                srcs.append(up(0.5 * float(slab) * gradU_g_box(lo[0:3], hi[0:3])))
            ws.append(w)
            k2s.append(k2)
            if k == npc - 1:
                last_lo, last_hi, last_ok = lo.copy(), hi.copy(), ok.copy()
            if save:
                saved_lo.append(lo.copy()); saved_hi.append(hi.copy()); saved_ok.append(ok.copy())
        y1, y2 = y10.copy(), y20.copy()
        Y1, Y2 = [], []
        for k, (ta, tb, _, _) in enumerate(pieces):
            if slab is None:
                y1, y2 = comp_step(y1, y2, k2s[k], up(tb - ta))
            else:
                import eight_affine as _AF
                y1, y2 = _AF.q_step(y1, y2, k2s[k], srcs[k], float(up(tb - ta)))
            Y1.append(y1)
            Y2.append(y2)
        Y1, Y2 = np.array(Y1), np.array(Y2)
        closed = okb & np.all(Y1 <= rho, 0) & np.all(Y2 <= vel, 0) & np.all(np.isfinite(Y1), 0)
        if verbose:
            print(f"   iteration {it}: closed {closed.sum()}/{nb}; rho(1) max {np.max(Y1[-1]):.3e}", flush=True)
        if np.all(closed):
            break
        rho = np.where(closed, rho, np.maximum(1.3 * Y1, rho))
        vel = np.where(closed, vel, np.maximum(1.3 * Y2, vel))
    if not sturm:              # closure-only mode (records need only a closed tube)
        ymin = np.full(nb, -np.inf)
        if MODE != "ball" and slab is None:
            check.last_end = (last_lo, last_hi, closed & last_ok)
        if save:
            check.saved = dict(
                ta=np.array([p_[0] for p_ in pieces]), tb=np.array([p_[1] for p_ in pieces]),
                c_lo=np.array([p_[2] for p_ in pieces]), c_hi=np.array([p_[3] for p_ in pieces]),
                km_lo=np.array([x[0] for x in centre_pieces.km]) if centre_pieces.km else np.zeros(0),
                km_hi=np.array([x[1] for x in centre_pieces.km]) if centre_pieces.km else np.zeros(0),
                t_lo=np.array(saved_lo), t_hi=np.array(saved_hi), t_ok=np.array(saved_ok),
                rho=np.array(rho), vel=np.array(vel), y1=Y1, y2=Y2, y10=y10, y20=y20,
                k2=np.array(k2s), w=np.array(ws), closed=closed, ymin=ymin, pc=pc, dl=dl, dphi=dphi, dM=dM)
        return closed, np.zeros(nb, bool), ymin, Y1[-1]
    # Sturm with W = w on each piece (w from the last, closed tube)
    z, zp = I(np.ones(nb)), I(np.zeros(nb))
    ymin = np.full(nb, np.inf)
    for k, (ta, tb, _, _) in enumerate(pieces):
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
    # the enclosure of Z = (Y, v) over the last time piece (contains tau = 1), valid where the tube closed; in box mode
    # the members are the centre-eps trajectories started at e^dl-scaled points (homogeneity of U), i.e. the label's
    # (lambda Y, P) with lambda = exp(ell - ell_c) > 0: signs of Y and P components are those of the label's solution
    if MODE != "ball" and slab is None:
        check.last_end = (last_lo, last_hi, closed & last_ok)
    if save:
        # per time piece, for the (ii-L) records and (iii-L) (theorist eight_records_design_theorist.md 6): times,
        # thin centre enclosure (Y, P) and (k, m), tube enclosure of (Y, v) with the radii used, the comparison
        # solutions of the box (y1, y2) and the k^2 used (unit solutions follow from k^2), w (Sturm coefficient)
        check.saved = dict(
            ta=np.array([p_[0] for p_ in pieces]), tb=np.array([p_[1] for p_ in pieces]),
            c_lo=np.array([p_[2] for p_ in pieces]), c_hi=np.array([p_[3] for p_ in pieces]),
            km_lo=np.array([x[0] for x in centre_pieces.km]) if centre_pieces.km else np.zeros(0),
            km_hi=np.array([x[1] for x in centre_pieces.km]) if centre_pieces.km else np.zeros(0),
            t_lo=np.array(saved_lo), t_hi=np.array(saved_hi), t_ok=np.array(saved_ok),
            rho=np.array(rho), vel=np.array(vel), y1=Y1, y2=Y2, y10=y10, y20=y20,
            k2=np.array(k2s), w=np.array(ws), closed=closed, ymin=ymin, pc=pc, dl=dl, dphi=dphi, dM=dM)
    return closed, closed & (ymin > 0), ymin, Y1[-1]


# ---------------------------------------------------------------- controls
def test():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
    import words_eight_sturm as S  # tools
    L.ORDER = 12
    a8 = np.array([1.5424624176685366, 0.7066425303908384, -0.30889005565526345])
    e8 = math.exp(-1.5 * a8[0])
    rng = np.random.default_rng(1)
    # control 1: centred Q at a point equals the float Rop
    tr = __import__("words_eight_riccati").trajectory(e8, a8[1], a8[2])
    Yc, Pc = tr[200, 0:3], tr[200, 3:6]
    Z = np.concatenate([Yc, -4 * np.linalg.norm(Yc) * Pc])[:, None]
    Q, c = rop_centred(Z, Z, np.array([e8]))
    Qf = S.rop_matrix(Yc, -4 * np.linalg.norm(Yc) * Pc, e8)
    err = max(abs(Q[i][j].mid[0] - 0.5 * (Qf[i, j] + Qf[j, i])) for i in range(3) for j in range(3))
    wid = max(Q[i][j].wid[0] for i in range(3) for j in range(3))
    print(f"control 1 (point): |Q_centred - Q_float| = {err:.2e}, width {wid:.1e}", flush=True)
    # control 2: containment over a small box, and overestimation factor
    for r in (1e-3, 1e-2):
        lo, hi = Z - r * np.array([[1, 1, 1, 0.1, 0.1, 0.1]]).T, Z + r * np.array([[1, 1, 1, 0.1, 0.1, 0.1]]).T
        Q, c = rop_centred(lo, hi, np.array([e8]))
        bad, spread = 0, 0.0
        for _ in range(300):
            z = lo[:, 0] + rng.uniform(0, 1, 6) * (hi[:, 0] - lo[:, 0])
            Qs = S.rop_matrix(z[0:3], z[3:6], e8)
            Qs = 0.5 * (Qs + Qs.T)
            bad += sum(not (Q[i][j].lo[0] - 1e-12 <= Qs[i, j] <= Q[i][j].hi[0] + 1e-12) for i in range(3) for j in range(3))
            spread = max(spread, np.max(np.abs(Qs - np.array([[Q[i][j].mid[0] for j in range(3)] for i in range(3)]))))
        wid = max(Q[i][j].wid[0] for i in range(3) for j in range(3)) / 2
        print(f"control 2 (box r {r:g}): outside {bad} of 2700 entries; enclosure radius {wid:.3e} vs sampled spread "
              f"{spread:.3e} (overestimation x{wid / max(spread, 1e-300):.1f})", flush=True)
    # control 3 and 4: the eight's box and a C23 corner box
    for name, p, hw in (("eight, label half-width 0.01", np.array([e8, a8[1], a8[2]]), 0.01),
                        ("eight, label half-width 0.02", np.array([e8, a8[1], a8[2]]), 0.02),
                        ("C23 corner (s 1, dphi -0.2), 0.005", np.array([e8, a8[1] - 0.2, a8[2]]), 0.005)):
        Pb = np.array([[p[0] * math.exp(-1.5 * hw), p[0] * math.exp(1.5 * hw), p[1] - hw, p[1] + hw,
                        p[2] - 0.25 * hw, p[2] + 0.25 * hw]])
        t = time.time()
        closed, ok, ymin, r1 = check(Pb, verbose=True)
        print(f"control ({name}): closed {closed[0]}, Sturm {'PASS' if ok[0] else 'fail'} (min y {ymin[0]:.3f}), "
              f"rho(1) {r1[0]:.3e}; {time.time() - t:.0f} s", flush=True)
    # control 5: rho(1) must bound the true spread of corner trajectories (eight, 0.01)
    import words_eight_riccati as W
    hw = 0.01
    c0 = W.trajectory(e8, a8[1], a8[2])[-1, 0:3]
    sp = 0.0
    for sg in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T:
        ell = sg[0] * hw
        y = W.trajectory(e8, a8[1] + sg[1] * hw, a8[2] + sg[2] * 0.25 * hw)[-1, 0:3] * math.exp(ell)
        sp = max(sp, np.linalg.norm(y - c0) / (2 * math.sqrt(np.linalg.norm(c0))))
    Pb = np.array([[e8 * math.exp(-1.5 * hw), e8 * math.exp(1.5 * hw), a8[1] - hw, a8[1] + hw, a8[2] - 0.25 * hw,
                    a8[2] + 0.25 * hw]])
    _, _, _, r1 = check(Pb)
    print(f"control 5: true corner spread at tau = 1 (cone units, approx) {sp:.3e} <= rigorous rho(1) {r1[0]:.3e}: "
          f"{sp <= r1[0]}", flush=True)


if __name__ == "__main__":
    test()
