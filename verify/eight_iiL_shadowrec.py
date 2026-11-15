"""Stage 21, (ii-L) records from SAVED comparison-tube data (theorist paper/review/eight_records_design_theorist.md
sections 1, 2.1, 2.2, 6.8; referee paper/review/eight_monotone_referee.md B2, C1, C2, C3, C5).

Monotonicity lemma: along a field extremal, F = S+ + W - A8 is non-decreasing in the time to go s, and so are
F_S = S_E^c + W - A8 and W - A8. A boundary label (s_b, a) of the adopted tube Lambda (taper A) satisfies (ii-L) if
 (I)  its end point x_I = r (cos phi e_I + sin phi e_Z) has section form q_D(ell, phi) >= 0.86 (band, (ii'));
 (T)  some point of the same extremal at a time s <= s_b has F_S > 0 or W > A8 (a record).
Records. A saved box B (thin in eps: eps = eps_c, phi in Phi, M in Mb; dl = 0) is, at scale r, the fixed-time family of
the extremals (r, phi, M/s_top), s_top = sqrt(eps_c) r^(3/4); its time piece n (tau in [ta_n, tb_n]) sits at times
s = tau s_top. At one fixed (r, tau), for every member a (referee C3, C1, C2):
    F_S(a) >= F_S(a_c) - sup_X |g - p_c| rho_x - Dp Len,
    W(a)   >= W(a_c)   - |p_c| rho_x       - Dp Len,
 x = r Y, p = P/s_top, W(a_c) = (r/s_top)(k(tau) + eps_c m(tau)) (centre start radius 1; referee B2),
 X = r [tube box of Y] (convex, contains x_c), g over every piece of S_E^c meeting X (Clarke hull of the evaluator),
 rho_x >= |x(a) - x_c| from the tube and centre boxes, Dp >= sup |p(a') - p_c| over the box (tube enclosure of
 P = -v/(4|Y|)), Len <= r 2 sqrt(sup|Y|) y1(piece) (Euclidean length of the image of the label segment).
 All with r in an interval R (scale interval) and tau in the piece (enclosures over the piece): one record covers
 every member of B, every r in R and every tau in the piece. lb = max(F_S lb, W lb - A8_hi).
Coverage (2.2). For a label sub-box A0 = (ell in log R, phi in Phi', M_top in M') of B: the slabs
 {(ell, phi, tau M_top) at s = tau s_top(r)} for the pieces tau in [ta_k, tb_k] are tested with the rigorous inside test
 of Lambda; tau_in = tb of the last piece of the initial run of inside pieces. Every boundary time of these extremals is
 >= tau_in s_top, so the record of the piece n with tb_n = tau_in (the latest allowed node) proves (ii-L) at all their
 boundary labels (F non-decreasing in s, also for extremals that leave and re-enter). If the inside run reaches s = 1,
 the extremals have no boundary label before the s = 1 slice ((ii-1) / (iii-L) territory). Sub-boxes that are not
 inside at small tau are split (phi, M) and, if certainly q_D >= 0.86, assigned to (I). Sub-boxes certainly outside
 Lambda at every piece need nothing.
Slab boxes are NOT used for records (referee C5).
    python verify/eight_iiL_shadowrec.py <dir or files> [nfiles] [dell]"""
import glob
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from prove_newton_1212 import I, dn, up  # noqa: E402
import eight_tube_boundary as TB  # noqa: E402

A8_HI = 2.5198721686724825
Q_STAR = 0.86
TUBE_A = os.path.join(ROOT, "words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper_b070_t02_03.npz")
GOAL = os.path.join(ROOT, "words/out/eight/block/cert_pieces_E_goalEfit.json")
NSUB = int(os.environ.get("SR_NSUB", "3"))
FDEPTH = int(os.environ.get("SR_FDEPTH", "1"))
RLEVELS = int(os.environ.get("SR_RLEVELS", "2"))


def lowprio():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def tube():
    import words_eight_tubeb as W
    return W.EllTubeFit(TUBE_A)


def section_tube(T, q):
    """a copy of the tube object whose G is scaled so that 'inside' means q_D < q on the s = 0 section"""
    import copy
    Tq = copy.copy(T)
    Tq.G = T.G / q
    return Tq


def ell_range(T):
    """outer bounds of ell over the tube (per node segment: centre range +- max half-axis; cf. eight_R_shadow.q_bounds)"""
    sn = T.s_nodes
    Cn = T.centre(sn)
    half = np.sqrt(np.stack([np.diagonal(np.linalg.inv(g)) for g in T.G])) * (1 + 1e-9) + 1e-12
    lo = np.minimum(Cn[:-1], Cn[1:]) - np.maximum(half[:-1], half[1:])
    hi = np.maximum(Cn[:-1], Cn[1:]) + np.maximum(half[:-1], half[1:])
    return float(lo[:, 0].min()), float(hi[:, 0].max())


def iv(lo, hi):
    return I(np.asarray(lo, float), np.asarray(hi, float))


def norm_ub(vs):
    """upper bound of the Euclidean norm of an interval vector (list of I)"""
    return up(np.sqrt(up(sum(up(np.maximum(-v.lo, v.hi) ** 2) for v in vs))))


def r34(R):
    """R^(3/4) and R^(1/4) for an interval R > 0 (monotone; endpoints pushed)"""
    lo, hi = R.lo, R.hi
    return (I(TB.push(lo ** 0.75, 3, -np.inf), TB.push(hi ** 0.75, 3, np.inf)),
            I(TB.push(lo ** 0.25, 3, -np.inf), TB.push(hi ** 0.25, 3, np.inf)))


# ---------------------------------------------------------------- records
def records(co, D, b, n, Rlo, Rhi, nsub=NSUB, tau_a=None, tau_b=None):
    """lower bounds of max(F_S, W - A8) for the records (box b of the saved dict D, piece n, scale r in [Rlo, Rhi]);
    b, n, Rlo, Rhi arrays of equal length m. Sub-box centred form: the tube box of Y is split into nsub^3 boxes Y_j with
    centre points Yt_j; with phi = S - p_c . x (p_c fixed) and W(a) = W_c - p_c . (x - x_c) - int (p - p_c) . dx,
        F_S(a) >= min_j [ S(t, r Yt_j) + p_c . (x_c - r Yt_j) + (g_j - p_c) . r (Y_j - Yt_j) ] + W_c - Dp Len - A8,
    g_j over the Clarke gradients of S on r Y_j (Lebourg on the segment from r Yt_j to x inside the convex box r Y_j);
    dot products in interval arithmetic. W(a) >= W_c - p_c . r (Y_t - Y_c) - Dp Len.
    tau_a, tau_b (optional): a sub-interval of the piece's tau range (partial node); the state enclosures stay those of
    the whole piece (a superset of the states at the sub-interval's times), only t = 1 - s and s are narrowed.
    Returns (lb, lbS, lbW, s_hi)."""
    m = len(b)
    R = iv(Rlo, Rhi)
    eps = D["pc"][b, 0]
    sq = I(dn(np.sqrt(eps)), up(np.sqrt(eps)))
    R34, R14 = r34(R)
    stop = sq * R34                                       # s_top(r)
    if tau_a is None:
        tau = iv(D["ta"][n], D["tb"][n])
    else:
        assert np.all(tau_a >= D["ta"][n]) and np.all(tau_b <= D["tb"][n]) and np.all(tau_a <= tau_b)
        tau = iv(tau_a, tau_b)
    s = tau * stop
    T = I(dn(1.0 - s.hi), up(1.0 - s.lo))
    ok = T.lo > 1e-9
    c_lo, c_hi = D["c_lo"][n, :, b], D["c_hi"][n, :, b]     # (m, 6)
    t_lo, t_hi = D["t_lo"][n, :, b], D["t_hi"][n, :, b]
    Yc = [iv(c_lo[:, i], c_hi[:, i]) for i in range(3)]
    Pc = [iv(c_lo[:, 3 + i], c_hi[:, 3 + i]) for i in range(3)]
    Yt = [iv(t_lo[:, i], t_hi[:, i]) for i in range(3)]
    vt = [iv(t_lo[:, 3 + i], t_hi[:, 3 + i]) for i in range(3)]
    RYt = TB.inorm(Yt)
    Pt = [v * (RYt * (-4.0)).recip() for v in vt]          # P = -v/(4|Y|) over the tube
    dPv = [I(dn(Pt[i].lo - Pc[i].hi), up(Pt[i].hi - Pc[i].lo)) for i in range(3)]
    DP = norm_ub(dPv)                                     # sup |P(a') - P_c|
    lenY = up(up(2.0 * up(np.sqrt(RYt.hi))) * D["y1"][n, b])
    Len = up(up(R.hi) * lenY)
    Dp = up(DP * up(1.0 / stop.lo))
    DL = up(Dp * Len)
    k = iv(D["km_lo"][n, 0, b], D["km_hi"][n, 0, b])
    mm = iv(D["km_lo"][n, 1, b], D["km_hi"][n, 1, b])
    rs = R14 / sq                                         # r / s_top
    Wc = rs * (k + mm * eps)
    pc = [p * stop.recip() for p in Pc]
    # (w): W(a) >= W_c - (P_c . (Y_t - Y_c)) r/s_top - Dp Len
    dYt = [I(dn(Yt[i].lo - Yc[i].hi), up(Yt[i].hi - Yc[i].lo)) for i in range(3)]
    pdot = Pc[0] * dYt[0] + Pc[1] * dYt[1] + Pc[2] * dYt[2]
    lbW = TB.nan_low(dn(dn(dn((Wc - rs * pdot).lo) - DL) - A8_HI))
    # (T) sub-box centred form
    lbS = np.full(m, np.inf)
    g = np.linspace(0, 1, nsub + 1)
    for i0 in range(nsub):
        for i1 in range(nsub):
            for i2 in range(nsub):
                ii = (i0, i1, i2)
                Yj = [iv(Yt[c].lo + g[ii[c]] * (Yt[c].hi - Yt[c].lo), Yt[c].lo + g[ii[c] + 1] * (Yt[c].hi - Yt[c].lo))
                      for c in range(3)]
                Yj = [I(np.where(ii[c] == 0, Yt[c].lo, Yj[c].lo), np.where(ii[c] == nsub - 1, Yt[c].hi, Yj[c].hi))
                      for c in range(3)]
                Ym = [0.5 * (y.lo + y.hi) for y in Yj]                 # float centre points
                Xj = [R * y for y in Yj]
                Xm = [R * I(y) for y in Ym]
                idx = np.flatnonzero(ok)
                if not len(idx):
                    continue
                Tk = TB.take(T, idx)
                Sm, _, _ = co.enclose(Tk, [TB.take(x, idx) for x in Xm], need_grad=False)
                _, gS, _ = co.enclose(Tk, [TB.take(x, idx) for x in Xj])
                # p_c . (x_c - r Ym) = (r/s_top) P_c . (Y_c - Ym)
                pcd = (TB.take(Pc[0], idx) * (TB.take(Yc[0], idx) - Ym[0][idx]) + TB.take(Pc[1], idx) * (TB.take(Yc[1], idx) - Ym[1][idx])
                       + TB.take(Pc[2], idx) * (TB.take(Yc[2], idx) - Ym[2][idx])) * TB.take(rs, idx)
                lin = None
                for c in range(3):
                    dX = TB.take(R, idx) * (TB.take(Yj[c], idx) - Ym[c][idx])
                    tt = (gS[c] - TB.take(pc[c], idx)) * dX
                    lin = tt if lin is None else lin + tt
                term = (Sm + pcd + lin).lo
                lbS[idx] = np.minimum(lbS[idx], TB.nan_low(term))
    lbS = np.where(ok, TB.nan_low(dn(dn(dn(lbS + Wc.lo) - DL) - A8_HI)), -np.inf)
    return np.maximum(lbS, lbW), lbS, lbW, s.hi


# ---------------------------------------------------------------- coverage
CHUNK = 12


def ell_grid(ell_lo, ell_hi, dell):
    """scale grid by explicit count (referee 5b D5): n = ceil((ell_hi - ell_lo)/dell) intervals of width dell from
    ell_lo; the last edge is at least ell_hi"""
    n = max(1, int(math.ceil((ell_hi - ell_lo) / dell - 1e-12)))
    g = ell_lo + dell * np.arange(n + 1)
    g[-1] = max(g[-1], ell_hi)
    return g


def _slab_tests(T, W_, eps, tau_lo, tau_hi, need_in=True, need_out=True):
    """inside / outside tests of Lambda for the slabs {(ell, phi, tau M_top) at s = tau s_top(r)}, tau in
    [tau_lo, tau_hi], of the label sub-boxes W_ (rows ell_a, ell_b, phi_a, phi_b, M_a, M_b); tau_lo, tau_hi arrays of
    the same length. Returns (inside, outside, s_hi >= 1)."""
    Rlo, Rhi = np.exp(W_[:, 0]), np.exp(W_[:, 1])
    stop_lo = math.sqrt(eps) * Rlo ** 0.75 * (1 - 1e-12)
    stop_hi = math.sqrt(eps) * Rhi ** 0.75 * (1 + 1e-12)
    pr = np.stack([tau_lo * W_[:, 4], tau_hi * W_[:, 4], tau_lo * W_[:, 5], tau_hi * W_[:, 5]])
    Mlo, Mhi = dn(pr.min(0)), up(pr.max(0))
    a_lo = np.stack([W_[:, 0], W_[:, 2], Mlo], 1)
    a_hi = np.stack([W_[:, 1], W_[:, 3], Mhi], 1)
    s_lo = dn(tau_lo * stop_lo)
    s_hi = up(tau_hi * stop_hi)
    s1 = s_hi >= 1.0
    s_hi_c = np.minimum(s_hi, 1.0)
    return (T.box_certainly_inside(a_lo, a_hi, s_lo, s_hi_c) if need_in else None,
            T.box_certainly_outside(a_lo, a_hi, s_lo, s_hi_c) if need_out else None, s1)


# adaptive node control (theorist 8.4: dF/ds ~ W_kin/s along an extremal, so a node a fraction rel below the exit
# time in relative s loses about W rel): a failing record asks for rel <= REL_F / W (clipped to [REL_MIN, REL_MAX])
# in its scale interval (0.75 d ell, from s_top ~ r^(3/4)) and in its time piece (refine_box with finer pieces)
REL_F = float(os.environ.get("SR_RELF", "0.005"))
REL_MIN, REL_MAX = 2e-4, 0.01
EDEPTH = int(os.environ.get("SR_EDEPTH", "4"))
# a record is accepted when lb > LB_MIN (0: rigorous positivity; a positive LB_MIN only asks for more splitting)
LB_MIN = float(os.environ.get("SR_LBMIN", "0"))
PFRAC = (0.875, 0.75, 0.5, 0.25) if os.environ.get("SR_PARTIAL", "1") == "1" else ()


def rel_target(lbW):
    """target relative s spread for a failing record from its W lower bound (W >= lbW + A8_hi)"""
    W = np.maximum(np.asarray(lbW) + A8_HI, 0.25)
    return np.clip(REL_F / W, REL_MIN, REL_MAX)


def classify_box(T, Tq, co, D, b, ell_lo, ell_hi, dell=0.0025, depth=3, fdepth=FDEPTH, edepth=EDEPTH, regions=None):
    """all scale intervals and label sub-boxes of box b: returns a list of rows (class, ell_a, ell_b, phi_a, phi_b, M_a,
    M_b, piece n, lb, tau_a, tau_b, w_req) with class in out / I / s1 / T / fail / unres; [tau_a, tau_b] is the node's
    tau interval, w_req (fail rows) the time-piece width the record asks for (refine_box(fine=...)).
    Sub-boxes not inside at small tau are split in (phi, M) up to `depth` levels; sub-boxes whose record fails are
    split in (ell, phi, M) up to `fdepth` more levels, then in ell alone (up to `edepth` halvings) while the relative
    s spread 0.75 d ell of the scale interval exceeds rel_target(W).
    Node: the last piece of the initial run of inside pieces, or, if the next piece is inside on an initial part
    [ta, ta + f (tb - ta)] (f in PFRAC, largest first), that piece on that part (partial node, records(tau_a, tau_b)).
    regions: optional label boxes (ell_a, ell_b, phi_a, phi_b, M_a, M_b) to classify instead of the whole box (each
    intersected with the box's phi and M range; used after refinement for the parent's failing labels only)."""
    Pb = D["boxes"][b]
    eps = Pb[0]
    phi0, phi1, M0, M1 = Pb[2], Pb[3], Pb[4], Pb[5]
    ta, tb = D["ta"], D["tb"]
    npc = len(ta)
    trunc = bool(D.get("trunc", False))
    ells = ell_grid(ell_lo, ell_hi, dell)
    rows = []
    nan3 = (np.nan, np.nan, np.nan)
    if regions is None:
        work = [(ells[i], ells[i + 1], phi0, phi1, M0, M1, 0, 0, 0) for i in range(len(ells) - 1)]
    else:
        work = []
        for g in regions:
            pa, pb_, ma, mb = max(g[2], phi0), min(g[3], phi1), max(g[4], M0), min(g[5], M1)
            if pa < pb_ and ma < mb:
                work.append((g[0], g[1], pa, pb_, ma, mb, 0, 0, 0))
    while work:
        W_ = np.array([w[:6] for w in work])
        lev = np.array([w[6] for w in work])
        flev = np.array([w[7] for w in work])
        elev = np.array([w[8] for w in work])
        nw = len(W_)
        z = np.zeros(nw)
        isI = Tq.box_certainly_outside(np.stack([W_[:, 0], W_[:, 2], z], 1), np.stack([W_[:, 1], W_[:, 3], z], 1), z, z)
        # initial run of inside pieces, tested lazily in chunks of pieces (only rows still running)
        run = np.zeros(nw, int)
        alive = ~isI
        at_ins = np.zeros(nw, bool)                  # the piece where the run stops: inside (and s >= 1 there)
        for c0 in range(0, npc, CHUNK):
            idx = np.flatnonzero(alive)
            if not len(idx):
                break
            c1 = min(npc, c0 + CHUNK)
            nc = c1 - c0
            K = np.tile(np.arange(c0, c1), len(idx))
            J = np.repeat(idx, nc)
            i_, _, s1_ = _slab_tests(T, W_[J], eps, ta[K], tb[K], need_out=False)
            good = (i_ & ~s1_).reshape(len(idx), nc)
            stop = ~good
            has = stop.any(1)
            first = np.where(has, stop.argmax(1), nc)
            run[idx] = c0 + first
            hs = idx[has]
            at_ins[hs] = (i_ & s1_).reshape(len(idx), nc)[has, first[has]]
            alive[hs] = False
        # s1 (referee 5b D2): the run stopped at a piece K reaching s >= 1 (for the largest r) and certainly inside
        # there; for the smaller r of the scale interval s = 1 comes only at tau1(r) = 1/s_top(r) <= 1/stop_lo, so the
        # slab tau in [ta_K, max(tb_K, min(1, 1/stop_lo))] must be certainly inside too (s clipped to 1); otherwise
        # the row is treated as a run stopping at K (node before K, records as usual)
        cand1 = np.flatnonzero(at_ins & (run < npc))
        if len(cand1):
            stop_lo1 = math.sqrt(eps) * np.exp(W_[cand1, 0]) ** 0.75 * (1 - 1e-12)
            tK = run[cand1]
            th = np.maximum(tb[tK], np.minimum(1.0, 1.0 / stop_lo1))
            i_ext, _, _ = _slab_tests(T, W_[cand1], eps, ta[tK].astype(float), th, need_out=False)
            at_ins[cand1[~i_ext]] = False
        # rows that are not inside at the first piece: certainly outside at every piece?
        zr = np.flatnonzero(~isI & (run == 0))
        allout = np.zeros(nw, bool)
        if len(zr) and not trunc:
            K = np.tile(np.arange(npc), len(zr))
            J = np.repeat(zr, npc)
            _, o_, _ = _slab_tests(T, W_[J], eps, ta[K], tb[K], need_in=False)
            allout[zr] = o_.reshape(len(zr), npc).all(1)
        new, need = [], []
        for j in range(nw):
            row = W_[j]
            if allout[j]:
                rows.append(("out", *row, -1, np.nan, *nan3))
                continue
            if isI[j]:
                rows.append(("I", *row, -1, np.nan, *nan3))
                continue
            if run[j] < npc and at_ins[j]:
                rows.append(("s1", *row, -1, np.nan, *nan3))
                continue
            if run[j] == 0:
                if lev[j] < depth:
                    new += _split(row, lev[j] + 1, flev[j], elev[j], ell=False)
                else:
                    rows.append(("unres", *row, -1, np.nan, *nan3))
                continue
            need.append((j, run[j]))
        if need:
            jj = np.array([x[0] for x in need])
            rr = np.array([x[1] for x in need], int)
            nn = rr - 1
            t_a, t_b = ta[nn].astype(float), tb[nn].astype(float)
            # partial node in the piece after the run (not reaching s = 1 there)
            cand = (rr < npc) & bool(PFRAC)
            if np.any(cand):
                ci = np.flatnonzero(cand)
                rc = rr[ci]
                got = np.zeros(len(ci), bool)
                for f in PFRAC:
                    left = np.flatnonzero(~got)
                    if not len(left):
                        break
                    tl = ta[rc[left]].astype(float)
                    th = tl + f * (tb[rc[left]] - tl)
                    i_, _, s1_ = _slab_tests(T, W_[jj[ci[left]]], eps, tl, th)
                    ok_ = i_ & ~s1_
                    sel = left[ok_]
                    got[sel] = True
                    nn[ci[sel]] = rc[sel]
                    t_a[ci[sel]] = tl[ok_]
                    t_b[ci[sel]] = th[ok_]
            Rlo_, Rhi_ = np.exp(W_[jj, 0]) * (1 - 1e-15), np.exp(W_[jj, 1]) * (1 + 1e-15)
            lb, lbS, lbW, shi = records(co, D, np.full(len(jj), b), nn, Rlo_, Rhi_, tau_a=t_a, tau_b=t_b)
            rt = rel_target(lbW)
            for i, (j, n_, l) in enumerate(zip(jj, nn, lb)):
                if l > LB_MIN:
                    rows.append(("T", *W_[j], n_, l, t_a[i], t_b[i], np.nan))
                elif flev[j] < fdepth:
                    new += _split(W_[j], lev[j], flev[j] + 1, elev[j], ell=True)
                elif elev[j] < edepth and (0.75 * (W_[j, 1] - W_[j, 0]) > rt[i] or l > 0):
                    new += _split(W_[j], lev[j], flev[j], elev[j] + 1, ell=True, pm=False)
                elif l > 0:                      # below LB_MIN but positive after all splits: still a record
                    rows.append(("T", *W_[j], n_, l, t_a[i], t_b[i], np.nan))
                else:
                    rows.append(("fail", *W_[j], n_, l, t_a[i], t_b[i], float(rt[i] * t_a[i])))
        work = new
    return rows


def _split(row, lev, flev, elev, ell, pm=True):
    """children of a label sub-box: halves in (phi, M) if pm, and in ell too if ell is True"""
    pmid, mm_ = 0.5 * (row[2] + row[3]), 0.5 * (row[4] + row[5])
    em = 0.5 * (row[0] + row[1])
    eparts = ((row[0], em), (em, row[1])) if ell else ((row[0], row[1]),)
    pparts = ((row[2], pmid), (pmid, row[3])) if pm else ((row[2], row[3]),)
    mparts = ((row[4], mm_), (mm_, row[5])) if pm else ((row[4], row[5]),)
    out = []
    for (ea, eb) in eparts:
        for (pa, pb_) in pparts:
            for (ma, mb) in mparts:
                out.append((ea, eb, pa, pb_, ma, mb, lev, flev, elev))
    return out


# ---------------------------------------------------------------- refinement without new comparison-tube runs
MFINE_MAX = 256


def _lohner_thin(P, t_end=1.0, tries=3):
    """RB.lohner_split's advance for a batch of thin centres, stopped at tau = t_end (no batch splitting: raises if a
    step cannot be validated). Same steps, same enclosures (copied step logic; RB is not modified)."""
    import eight_comptube as CT
    RB, L = CT.RB, CT.L
    N = RB.N
    c, A, r, cols = RB.start_set4(P)
    t, steps = 0.0, []
    while t < t_end:
        target = min(RB.TAU0 if t < RB.TAU0 else 1.0, t_end)
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
            raise RuntimeError("centre trajectory failed")
        h, rX, rc = res
        cs, Rm, _ = rX
        steps.append((t, h, [[cs[i][k][0] for k in range(len(cs[i]))] for i in range(N)], [Rm[i][0] for i in range(N)]))
        yX = L.at(cs, Rm, h)
        J = I(np.stack([np.stack([yX[i].lo[1 + j] for j in range(N)], 1) for i in range(N)], 1),
              np.stack([np.stack([yX[i].hi[1 + j] for j in range(N)], 1) for i in range(N)], 1))
        yc = L.at(rc[0], rc[1], h)
        phic = I(np.stack([v.lo[0] for v in yc], 1), np.stack([v.hi[0] for v in yc], 1))
        c, A, r = L.lohner_update(J, A, r, phic)
        if t < RB.TAU0:
            cols = [L.lohner_update(J, Aj, rj, L._jv(J, cj)) for (cj, Aj, rj) in cols]
        t += h
    return steps


def _centre_pieces(pc, m, fine=None, t_end=1.0):
    """CT.centre_pieces with a per-step number of pieces: m by default, and on a Lohner step [ts, ts + h] meeting a
    fine range (tau_a, tau_b, w) of `fine`, max(m, ceil(h / w)) pieces (at most MFINE_MAX). Same enclosures
    (Taylor model of the step evaluated on a sub-interval of the step, remainder bound Rm tP)."""
    import eight_comptube as CT
    P = I(pc.copy(), pc.copy())
    if t_end >= 1.0:
        groups, failed = CT.RB.lohner_split(P, rec_from=0.0)
        if failed or len(groups) != 1:
            raise RuntimeError("centre trajectory failed")
        _, _, steps = groups[0]
    else:
        steps = _lohner_thin(P, t_end)
    out, km = [], []
    nz = min(8, len(steps[0][2])) if steps else 6
    for (ts, h, cs, Rm) in steps:
        mk = m
        for (fa, fb, w) in (fine or ()):
            if ts <= fb and ts + h >= fa and w > 0:
                mk = max(mk, min(MFINE_MAX, int(math.ceil(h / w))))
        for k in range(mk):
            sa, sb = h * k / mk, h * (k + 1) / mk
            S = I(dn(sa), up(sb))
            tP = CT.tpow(S, CT.L.ORDER + 1)
            Z = [CT.peval(cs[i], S) + Rm[i] * tP for i in range(nz)]
            out.append((ts + sa, ts + sb, np.stack([z.lo for z in Z[:6]]), np.stack([z.hi for z in Z[:6]])))
            if nz >= 8:
                km.append((np.stack([Z[6].lo, Z[7].lo]), np.stack([Z[6].hi, Z[7].hi])))
    return out, km


def fine_spec(D, rows):
    """fine time ranges asked for by the fail rows of one unit: (tau_a, tau_b, w) with tau_a the node's start,
    tau_b = min(1, 1.3 tau_a + 0.0125) (the exits of the sub-boxes lie after the node) and w = rel_target tau_a"""
    out = []
    if os.environ.get("SR_FINE", "1") != "1":
        return None
    for r in rows:
        if r[0] == "fail" and len(r) > 11 and np.isfinite(r[11]) and r[11] > 0:
            out.append((float(r[9]), float(min(1.0, 1.3 * r[9] + 0.0125)), float(r[11])))
    if not out:
        return None
    # merge: one range with the smallest width (few fail rows per unit; keeps the piece count bounded)
    return [(min(o[0] for o in out), max(o[1] for o in out), min(o[2] for o in out))]


def refine_box(D, b, nphi=2, nM=2, m=8, fine=None, t_end=1.0):
    """sub-boxes of the saved box b (same eps, sub-ranges of phi and M) with their own thin centre and re-integrated
    comparison radii. Soundness: every member of a sub-box is a member of the big box, and so is every curve of its
    variation family (label segments inside the sub-box); the closed big tube contains all of them, so the saved k^2
    (an upper bound of ||Rop||_g over the big tube, per saved piece) bounds Rop along them. The comparison system
    y1' = y2, y2' = k^2 y1 is linear; with the sub-box's initial data (box_offsets formulas, dl = 0) it bounds the
    sub-family's |J|_g, |nabla J|_g, hence its tube around the new thin centre. On the sub-centre's own pieces the
    k^2 used is the maximum of the saved k^2 over the saved pieces overlapping it (a piecewise-constant upper bound).
    fine: optional time ranges with a smaller piece width (_centre_pieces); the radii are propagated piece by piece
    as before, so finer pieces only tighten the per-piece enclosures (the adaptive node control of classify_box).
    t_end < 1: the pieces stop at tau = t_end (trunc = True in the dict; classify_box then never calls a label box
    "out", and a run of inside pieces reaching t_end takes its last piece as the node, sound for every exit later).
    Returns a saved-format dict for the sub-boxes (closed = tube conversion ok on every piece)."""
    return refine_many([(D, b)], nphi, nM, m, fine, t_end)[0]


REF_N = int(os.environ.get("SR_REFN", "2"))
THIN_ORDER = int(os.environ.get("SR_ORDER", "6"))      # Taylor order of the thin sub-centres (refine_many)


def refine_many(parents, nphi=REF_N, nM=REF_N, m=8, fine=None, t_end=1.0):
    """refine_box for several parent boxes (D, b) at once: all sub-centres in one thin Lohner batch (the step logic
    is per centre; a batch only shares the step sizes, chosen for all). k2 per sub-box from its own parent's saved
    pieces; t_end is clipped to every parent's last piece. Returns one saved-format dict per parent."""
    import eight_field_lohner as L
    old_order = L.ORDER
    L.ORDER = THIN_ORDER                    # one fixed order for this call; the previous value is restored below
    try:
        return _refine_many(parents, nphi, nM, m, fine, t_end)
    finally:
        L.ORDER = old_order


def _refine_many(parents, nphi, nM, m, fine, t_end):
    import eight_comptube as CT
    t_end = min([float(t_end)] + [float(D["tb"][-1]) for (D, _) in parents])
    subs, owner = [], []
    for k, (D, b) in enumerate(parents):
        Pb = D["boxes"][b]
        pe = np.linspace(Pb[2], Pb[3], nphi + 1)
        me = np.linspace(Pb[4], Pb[5], nM + 1)
        pe[0], pe[-1], me[0], me[-1] = Pb[2], Pb[3], Pb[4], Pb[5]
        for i in range(nphi):
            for j in range(nM):
                subs.append([Pb[0], Pb[1], pe[i], pe[i + 1], me[j], me[j + 1]])
                owner.append(k)
    sub = np.array(subs)
    owner = np.array(owner)
    ns = len(sub)
    pc, dl, dphi, dM = CT.box_offsets(sub)
    pieces, km = _centre_pieces(pc, m, fine, t_end)
    eY = up(np.sqrt(np.exp(dl)) * (1 + 1e-15))
    Mmax = np.maximum(np.abs(sub[:, 4]), np.abs(sub[:, 5]))
    y1 = up(0.5 * eY * up(np.sqrt(dl ** 2 + dphi ** 2)))
    y2 = up(eY * up(Mmax * dl + 2.0 * dM))
    out = {k: [] for k in ("ta", "tb", "c_lo", "c_hi", "km_lo", "km_hi", "t_lo", "t_hi", "t_ok", "y1", "y2", "k2")}
    okall = np.ones(ns, bool)
    for k_, (ta, tb, Zlo, Zhi) in enumerate(pieces):
        k2p = np.empty(len(parents))
        for q, (D, b) in enumerate(parents):
            ov = (D["ta"] < tb) & (D["tb"] > ta)
            if not np.any(ov):
                raise RuntimeError("sub-centre piece outside the saved time range")
            k2p[q] = float(np.max(D["k2"][ov, b]))
        k2 = k2p[owner]
        y1, y2 = CT.comp_step(y1, y2, k2, up(tb - ta))
        lo, hi, ok = CT.tube_boxes(Zlo, Zhi, y1, y2)
        okall &= ok & np.isfinite(y1)
        out["ta"].append(ta); out["tb"].append(tb)
        out["c_lo"].append(Zlo); out["c_hi"].append(Zhi)
        out["km_lo"].append(km[k_][0]); out["km_hi"].append(km[k_][1])
        out["t_lo"].append(lo); out["t_hi"].append(hi); out["t_ok"].append(ok)
        out["y1"].append(y1.copy()); out["y2"].append(y2.copy()); out["k2"].append(k2.copy())
    A = {k: np.array(v) for k, v in out.items()}
    res = []
    for q in range(len(parents)):
        cols = np.flatnonzero(owner == q)
        Dn = {"ta": A["ta"], "tb": A["tb"]}
        for k in ("c_lo", "c_hi", "km_lo", "km_hi", "t_lo", "t_hi"):
            Dn[k] = A[k][:, :, cols]
        for k in ("t_ok", "y1", "y2", "k2"):
            Dn[k] = A[k][:, cols]
        Dn.update(boxes=sub[cols], pc=pc[cols], closed=okall[cols], trunc=bool(t_end < 1.0))
        res.append(Dn)
    return res

def run(files, dell=0.0025):
    lowprio()
    T = tube()
    Tq = section_tube(T, Q_STAR)
    co = TB.CoarseE(GOAL)
    ell_lo, ell_hi = ell_range(T)
    print(f"(ii-L) records from saved data: tube {os.path.basename(TUBE_A)}, piece {os.path.basename(GOAL)}, "
          f"A8_hi = {A8_HI!r}, q* = {Q_STAR}; ell range [{ell_lo:.3f}, {ell_hi:.3f}], dell {dell}", flush=True)
    tot = {}
    allrows = []
    t_all = time.time()
    nrec = 0
    for f in files:
        D = dict(np.load(f, allow_pickle=True))
        nb = D["boxes"].shape[0]
        for b in range(nb):
            t0 = time.time()
            if not bool(D["closed"][b]):
                print(f"  {os.path.basename(f)} box {b}: tube not closed, skipped", flush=True)
                continue
            cnt0, _, nunits, rows = box_result(T, Tq, co, D, b, ell_lo, ell_hi, dell)
            lvl_note = f" [refined: {nunits} units]" if nunits else ""
            cls = [r[0] for r in rows]
            cnt = {c: cls.count(c) for c in ("out", "I", "s1", "T", "fail", "unres")}
            lbs = np.array([r[8] for r in rows if r[0] in ("T", "fail")])
            nrec += len(lbs)
            for c, v in cnt.items():
                tot[c] = tot.get(c, 0) + v
            allrows += [(os.path.basename(f), b, *r) for r in rows]
            Pb = D["boxes"][b]
            print(f"  {os.path.basename(f)[:14]} box {b}: eps {Pb[0]:.4g} phi [{Pb[2]:.3f}, {Pb[3]:.3f}] M [{Pb[4]:.4f}, "
                  f"{Pb[5]:.4f}]: " + ", ".join(f"{c} {v}" for c, v in cnt.items() if v) +
                  (f"; record lb min {lbs.min():+.4e} median {np.median(lbs):+.4e}" if len(lbs) else "") +
                  f"{lvl_note}  ({time.time() - t0:.1f} s)", flush=True)
    dt = time.time() - t_all
    print(f"total: " + ", ".join(f"{c} {v}" for c, v in tot.items()) + f"; records evaluated {nrec}; "
          f"{dt:.1f} s ({dt / max(nrec, 1) * 1e3:.1f} ms per record incl. coverage)")
    return allrows


# ---------------------------------------------------------------- (ii-1): the s = 1 boundary slice
def w_lower(D, b, n, Rlo, Rhi):
    """rigorous lower bound of W over the members of box b at piece n, r in [Rlo, Rhi] (the W branch of records),
    and the x enclosure R Y_t (for the "image misses E" test). Arrays n, Rlo, Rhi of length m."""
    m = len(n)
    bb = np.full(m, b)
    R = iv(Rlo, Rhi)
    eps = D["pc"][bb, 0]
    sq = I(dn(np.sqrt(eps)), up(np.sqrt(eps)))
    R34, R14 = r34(R)
    stop = sq * R34
    c_lo, c_hi = D["c_lo"][n, :, bb], D["c_hi"][n, :, bb]
    t_lo, t_hi = D["t_lo"][n, :, bb], D["t_hi"][n, :, bb]
    Yc = [iv(c_lo[:, i], c_hi[:, i]) for i in range(3)]
    Pc = [iv(c_lo[:, 3 + i], c_hi[:, 3 + i]) for i in range(3)]
    Yt = [iv(t_lo[:, i], t_hi[:, i]) for i in range(3)]
    vt = [iv(t_lo[:, 3 + i], t_hi[:, 3 + i]) for i in range(3)]
    RYt = TB.inorm(Yt)
    Pt = [v * (RYt * (-4.0)).recip() for v in vt]
    DP = norm_ub([I(dn(Pt[i].lo - Pc[i].hi), up(Pt[i].hi - Pc[i].lo)) for i in range(3)])
    Len = up(up(R.hi) * up(up(2.0 * up(np.sqrt(RYt.hi))) * D["y1"][n, bb]))
    DL = up(up(DP * up(1.0 / stop.lo)) * Len)
    rs = R14 / sq
    Wc = rs * (iv(D["km_lo"][n, 0, bb], D["km_hi"][n, 0, bb]) + iv(D["km_lo"][n, 1, bb], D["km_hi"][n, 1, bb]) * eps)
    dYt = [I(dn(Yt[i].lo - Yc[i].hi), up(Yt[i].hi - Yc[i].lo)) for i in range(3)]
    pdot = Pc[0] * dYt[0] + Pc[1] * dYt[1] + Pc[2] * dYt[2]
    Wlb = TB.nan_low(dn((Wc - rs * pdot).lo - DL))
    X = [R * y for y in Yt]
    return Wlb, X


def ii1_box(T, D, b, ell_lo, ell_hi, dell=0.0025):
    """(ii-1) for box b: scale intervals with s_top(r) >= 1 whose s = 1 labels (ell, phi, M_top / s_top) may lie on the
    boundary of Lambda(1) (neither certainly inside nor certainly outside); every piece meeting the tau1 = 1/s_top
    range must have its x box miss the ray E (x_2 or x_3 bounded away from 0, or x_1 < 0) or W_lo > A8_hi.
    Returns counts (n_bdry, n_missE, n_W, n_fail) and the min W margin over the W-accepted pieces."""
    Pb = D["boxes"][b]
    eps = Pb[0]
    ta, tb = D["ta"], D["tb"]
    ells = ell_grid(ell_lo, ell_hi, dell)
    el, eh = ells[:-1], ells[1:]
    stop_lo = math.sqrt(eps) * np.exp(el) ** 0.75 * (1 - 1e-12)
    stop_hi = math.sqrt(eps) * np.exp(eh) ** 0.75 * (1 + 1e-12)
    sel = stop_hi >= 1.0
    cnt = dict(n_bdry=0, n_missE=0, n_W=0, n_fail=0)
    wmin = np.inf
    if not sel.any():
        return cnt, np.nan
    el, eh, stop_lo, stop_hi = el[sel], eh[sel], stop_lo[sel], stop_hi[sel]
    t1_lo = np.minimum(1.0 / stop_hi, 1.0)
    t1_hi = np.minimum(1.0 / np.maximum(stop_lo, 1e-300), 1.0)
    Mc = np.stack([t1_lo * Pb[4], t1_lo * Pb[5], t1_hi * Pb[4], t1_hi * Pb[5]])
    a_lo = np.stack([el, np.full_like(el, Pb[2]), dn(Mc.min(0))], 1)
    a_hi = np.stack([eh, np.full_like(eh, Pb[3]), up(Mc.max(0))], 1)
    one = np.ones(len(el))
    ins = T.box_certainly_inside(a_lo, a_hi, one, one)
    outs = T.box_certainly_outside(a_lo, a_hi, one, one)
    for j in np.flatnonzero(~ins & ~outs):
        cnt["n_bdry"] += 1
        pcs = np.flatnonzero((ta <= t1_hi[j]) & (tb >= t1_lo[j]))
        if not len(pcs):
            cnt["n_fail"] += 1
            continue
        Wlb, X = w_lower(D, b, pcs, np.full(len(pcs), math.exp(el[j]) * (1 - 1e-15)),
                         np.full(len(pcs), math.exp(eh[j]) * (1 + 1e-15)))
        missE = (X[1].lo > 0) | (X[1].hi < 0) | (X[2].lo > 0) | (X[2].hi < 0) | (X[0].hi < 0)
        okW = Wlb > A8_HI
        if np.all(missE):
            cnt["n_missE"] += 1
        elif np.all(missE | okW):
            cnt["n_W"] += 1
            wmin = min(wmin, float(np.min(Wlb[~missE] - A8_HI)))
        else:
            cnt["n_fail"] += 1
    return cnt, (wmin if np.isfinite(wmin) else np.nan)


# ---------------------------------------------------------------- streaming runner with a ledger
LEDGER = os.environ.get("SR_LEDGER", os.path.join(ROOT, "words/out/eight/tubeb/iiL_ledger.npz"))
LCOLS = ("box", "eps", "phi_lo", "phi_hi", "M_lo", "M_hi", "n_out", "n_I", "n_s1", "n_T", "n_fail", "n_unres",
         "min_lb", "refined", "sec", "mtime", "ii1_bdry", "ii1_missE", "ii1_W", "ii1_fail", "ii1_wmin",
         # the tightest record of the box: node piece and tau interval, its label box (ell, phi, M_top), the
         # (sub-)box it was evaluated on (sub -1: the saved box itself; else index in the refined dict, with its
         # phi and M_top range) and the number of pieces of that dict
         "lb_piece", "lb_tau_a", "lb_tau_b", "lb_ell_a", "lb_ell_b", "lb_phi_a", "lb_phi_b", "lb_Mt_a", "lb_Mt_b",
         "lb_sub", "lb_sub_phi_a", "lb_sub_phi_b", "lb_sub_M_a", "lb_sub_M_b", "lb_npc",
         # bookkeeping (referee 5b): refinement exceptions and unclosed sub-boxes in the box's units
         "n_referr", "n_subnc")
NTIGHT = 15


def box_result(T, Tq, co, D, b, ell_lo, ell_hi, dell):
    """classify + records (+ refinement) for one saved box: (counts dict, min lb, refined units, rows)"""
    return boxes_result(T, Tq, co, D, [b], ell_lo, ell_hi, dell)[0][:4]


def rows_hash(rows):
    """md5 (16 hex) of a box's row list: class, label box (8 significant digits) and piece, sorted (referee 5b 8.3)"""
    import hashlib
    keys = sorted(f"{r[0]}:" + ",".join(f"{float(x):.8g}" for x in r[1:8]) for r in rows)
    return hashlib.md5("|".join(keys).encode()).hexdigest()[:16]


def _t_end(Du, ru):
    """pieces up to 1.5 x the latest failing node (+0.05): sub-box runs may go further, then their node is the last
    piece (sound); "unres" rows (no node) need the whole range"""
    tn = [r[10] for r in ru if r[0] == "fail" and len(r) > 11 and np.isfinite(r[10])]
    if not tn or any(r[0] == "unres" for r in ru):
        return 1.0
    return min(1.0, 1.5 * max(tn) + 0.05)


def _region_fail_rows(regs, Pq):
    """fail rows (class fail, lb -inf) for the parts of the label boxes regs inside the (phi, M_top) range of Pq"""
    out = []
    for g in regs:
        pa, pb_, ma, mb = max(g[2], Pq[2]), min(g[3], Pq[3]), max(g[4], Pq[4]), min(g[5], Pq[5])
        if pa < pb_ and ma < mb:
            out.append(("fail", g[0], g[1], pa, pb_, ma, mb, -1, -np.inf, np.nan, np.nan, np.nan))
    return out


def boxes_result(T, Tq, co, D, bs, ell_lo, ell_hi, dell, units_out=None):
    """box_result for the boxes bs of one saved dict; the refinements of all failing units of a level go into one
    Lohner batch (refine_many; on an exception, unit by unit). Returns a list of (counts, min lb, units, rows,
    tight, n_referr, n_subnc) with tight the NTIGHT ledger values lb_* of the box's tightest record (nan if none).
    Bookkeeping (referee 5b D1): a unit carries its label boxes as rows with real coordinates at every level. A unit
    whose refinement raises keeps its failing rows (and is tried again at the next level; at the last level they
    stay "fail"); an unclosed sub-box turns its share of the failing label boxes into "fail" rows and is never
    refined (its tube data are not valid); a refined unit that gets no row at all for a non-empty region list is
    a failure (its failing rows are kept). Exceptions and unclosed sub-boxes are counted per box."""
    units = [(k, D, b, classify_box(T, Tq, co, D, b, ell_lo, ell_hi, dell), True) for k, b in enumerate(bs)]
    nunits = np.zeros(len(bs), int)
    nerr = np.zeros(len(bs), int)
    nnc = np.zeros(len(bs), int)
    isbad = lambda u: any(r[0] in ("fail", "unres") for r in u[3])  # noqa: E731
    for lvl in range(RLEVELS):
        failing = [u for u in units if u[4] and isbad(u)]
        if not failing:
            break
        keep = [u for u in units if not (u[4] and isbad(u))]
        for (k, Du, bu, ru, _) in failing:
            # the parent's settled labels keep their rows (its records cover every member of the parent box); the
            # failing label boxes are classified again on the sub-boxes (each sub-box takes its part of them)
            good = [r for r in ru if r[0] not in ("fail", "unres")]
            if good:
                keep.append((k, Du, bu, good, True))
        fines = [fine_spec(Du, ru) for (_, Du, _, ru, _) in failing]
        fine = [x for f in fines if f for x in f] or None
        t_end = max(_t_end(Du, ru) for (_, Du, _, ru, _) in failing)
        try:
            Dns = refine_many([(Du, bu) for (_, Du, bu, _, _) in failing], fine=fine, t_end=t_end)
        except Exception as e:
            print(f"    refinement batch raised ({e!r})"[:200], file=sys.stderr, flush=True)
            Dns = []
            for (k, Du, bu, ru, _) in failing:
                try:
                    Dns.append(refine_many([(Du, bu)], fine=fine_spec(Du, ru), t_end=_t_end(Du, ru))[0])
                except Exception as e2:
                    print(f"    refinement of a unit raised ({e2!r})"[:200], file=sys.stderr, flush=True)
                    nerr[k] += 1
                    Dns.append(None)
        new_units = []
        for (k, Du, bu, ru, _), Dn in zip(failing, Dns):
            fails = [r for r in ru if r[0] in ("fail", "unres")]
            regs = [r[1:7] for r in fails]
            assert all(np.all(np.isfinite(g)) for g in regs), "label box without coordinates"
            if Dn is None:
                new_units.append((k, Du, bu, fails, True))
                continue
            got = []
            for bb in range(len(Dn["boxes"])):
                if not bool(Dn["closed"][bb]):
                    nnc[k] += 1
                    rr = _region_fail_rows(regs, Dn["boxes"][bb])
                    got.append((k, Dn, bb, rr, False))
                    continue
                got.append((k, Dn, bb, classify_box(T, Tq, co, Dn, bb, ell_lo, ell_hi, dell, regions=regs), True))
            if regs and sum(len(g[3]) for g in got) == 0:
                nerr[k] += 1
                new_units.append((k, Du, bu, fails, True))
                continue
            new_units += got
        units = keep + new_units
        for k in set(u[0] for u in failing):
            nunits[k] = sum(1 for u in units if u[0] == k)
    if units_out is not None:
        units_out.extend(units)
    res = []
    for k in range(len(bs)):
        rows = [r for u in units if u[0] == k for r in u[3]]
        cls = [r[0] for r in rows]
        cnt = {c: cls.count(c) for c in ("out", "I", "s1", "T", "fail", "unres")}
        lbs = np.array([r[8] for r in rows if r[0] in ("T", "fail")])
        tight = (np.nan,) * NTIGHT
        best = None
        for u in units:
            if u[0] != k:
                continue
            for r in u[3]:
                if r[0] in ("T", "fail") and len(r) > 11 and (best is None or r[8] < best[1][8]):
                    best = (u, r)
        if best is not None:
            (_, Du, bu, _, _), r = best
            sub = -1 if Du is D else bu
            Pb = Du["boxes"][bu]
            tight = (float(r[7]), float(r[9]), float(r[10]), *[float(x) for x in r[1:7]], float(sub),
                     float(Pb[2]), float(Pb[3]), float(Pb[4]), float(Pb[5]), float(len(Du["ta"])))
        res.append((cnt, (float(lbs.min()) if len(lbs) else np.nan), int(nunits[k]), rows, tight, int(nerr[k]),
                    int(nnc[k])))
    return res

def _md5(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


DEP_FILES = [os.path.join(HERE, f) for f in ("eight_comptube.py", "eight_field.py", "eight_field_lohner.py",
                                              "eight_riccati_barrier.py", "prove_newton_1212.py")] +             [os.path.join(ROOT, "tools", f) for f in ("words_eight_tubeb.py", "prove_newton_1212.py")]


def run_meta(dell):
    """(run id, text): the code and data md5s and every setting of this run; the ledger keeps the text per run id
    in its header (meta_ids, meta_texts) and the run id per row (run_id)"""
    import hashlib
    txt = (f"code {os.path.basename(__file__)} md5 {_md5(os.path.abspath(__file__))}; "
           f"eight_tube_boundary.py md5 {_md5(os.path.join(HERE, 'eight_tube_boundary.py'))}; "
           + "".join(f"{os.path.basename(q)} md5 {_md5(q)}; " for q in DEP_FILES if os.path.exists(q)) +
           f"tube {os.path.basename(TUBE_A)} md5 {_md5(TUBE_A)}; goal {os.path.basename(GOAL)} md5 {_md5(GOAL)}; "
           f"A8_hi {A8_HI!r}; q* {Q_STAR}; dell {dell}; SR_NSUB {NSUB}; SR_FDEPTH {FDEPTH}; SR_RLEVELS {RLEVELS}; "
           f"SR_RELF {REL_F}; REL_MIN {REL_MIN}; REL_MAX {REL_MAX}; SR_EDEPTH {EDEPTH}; SR_LBMIN {LB_MIN!r}; "
           f"SR_REFN {REF_N}; SR_ORDER {THIN_ORDER}; SR_PARTIAL {PFRAC}; "
           f"SR_FINE {os.environ.get('SR_FINE', '1')}; CHUNK {CHUNK}; MFINE_MAX {MFINE_MAX}")
    return hashlib.md5(txt.encode()).hexdigest()[:12], txt


def load_ledger(path=LEDGER):
    if not os.path.exists(path):
        return {"fname": np.zeros(0, dtype="<U64"), "run_id": np.zeros(0, dtype="<U12"),
                "meta_ids": np.zeros(0, dtype="<U12"), "meta_texts": np.zeros(0, dtype="<U2000"),
                **{c: np.zeros(0) for c in LCOLS}}
    d = np.load(path, allow_pickle=False)
    L_ = {k: d[k] for k in d.files}
    n = len(L_["fname"])
    for c in LCOLS:                                  # columns added later: filled with nan (not computed)
        if c not in L_:
            L_[c] = np.full(n, np.nan)
    if "run_id" not in L_:                           # rows written before run ids existed
        L_["run_id"] = np.full(n, "", dtype="<U12")
    if "rows_hash" not in L_:                        # rows written before the row hash existed
        L_["rows_hash"] = np.full(n, "", dtype="<U16")
    for k, dt in (("meta_ids", "<U12"), ("meta_texts", "<U2000")):
        if k not in L_:
            L_[k] = np.zeros(0, dtype=dt)
    return L_


def save_ledger(L_, path=LEDGER):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **L_)
    os.replace(tmp, path)


def file_rows(f, D, T, Tq, co, ell_lo, ell_hi, dell):
    """ledger rows (fname, *LCOLS) of every box of one saved file"""
    newrows = []
    nb = D["boxes"].shape[0]
    t0 = time.time()
    # records need boxes thin in eps (dl = 0: the family at scale r is the extremals (r, phi, M / s_top)); a box with
    # an ell offset is reported as n_fail = -2 (not handled here)
    thin = np.asarray(D.get("dl", np.zeros(nb)), float) <= 1e-300
    if bool(np.asarray(D.get("slab", False))):          # slab boxes (eps in [0, E]) must never reach the records
        thin[:] = False
        print(f"  {os.path.basename(f)}: saved with slab = True, refused", flush=True)
    cl = [b for b in range(nb) if bool(D["closed"][b]) and thin[b]]
    br = dict(zip(cl, boxes_result(T, Tq, co, D, cl, ell_lo, ell_hi, dell))) if cl else {}
    tper = (time.time() - t0) / max(len(cl), 1)
    for b in range(nb):
        tb0 = time.time() - tper
        Pb = D["boxes"][b]
        if not bool(D["closed"][b]) or not thin[b]:
            cnt, mlb, nu = {c: 0 for c in ("out", "I", "s1", "T", "fail", "unres")}, np.nan, 0
            cnt["fail"] = -1 if not bool(D["closed"][b]) else -2    # -1: tube not closed (redo); -2: not thin
            tight = (np.nan,) * NTIGHT
            ne, nn, rh = 0, 0, ""
        else:
            cnt, mlb, nu, rws, tight, ne, nn = br[b]
            rh = rows_hash(rws)
        c1, w1 = (ii1_box(T, D, b, ell_lo, ell_hi, dell) if bool(D["closed"][b]) and thin[b]
                  else (dict(n_bdry=-1, n_missE=0, n_W=0, n_fail=-1), np.nan))
        newrows.append((os.path.basename(f), b, Pb[0], Pb[2], Pb[3], Pb[4], Pb[5], cnt["out"], cnt["I"],
                        cnt["s1"], cnt["T"], cnt["fail"], cnt["unres"], mlb, nu, time.time() - tb0,
                        os.path.getmtime(f), c1["n_bdry"], c1["n_missE"], c1["n_W"], c1["n_fail"], w1, *tight, ne, nn,
                        rh))
    return newrows


def ledger_append(L_, newrows, meta=None):
    if meta is not None and meta[0] not in set(str(x) for x in L_["meta_ids"]):
        L_["meta_ids"] = np.append(L_["meta_ids"], meta[0]).astype("<U12")
        L_["meta_texts"] = np.append(L_["meta_texts"], meta[1]).astype("<U2000")
    if "rows_hash" not in L_:
        L_["rows_hash"] = np.full(len(L_["fname"]), "", dtype="<U16")
    for r in newrows:
        L_["run_id"] = np.append(L_["run_id"], "" if meta is None else meta[0]).astype("<U12")
        L_["rows_hash"] = np.append(L_["rows_hash"], r[-1] if isinstance(r[-1], str) else "").astype("<U16")
        r = r[:-1] if isinstance(r[-1], str) else r
        L_["fname"] = np.append(L_["fname"], r[0])
        for c, v in zip(LCOLS, r[1:]):
            L_[c] = np.append(L_[c], v)


# ---------------------------------------------------------------- parallel stream: a pool over files
# Each worker owns one shard file iiL_ledger_shard_<pid>_<start>.npz (no two writers on one file); the dispatcher
# keeps the set of files done (main ledger + all shards, read at start: restartable) and in flight; `merge` builds
# iiL_ledger_merged.npz (main ledger + shards, one row per (file, box), first occurrence kept).
SHARD_GLOB = os.path.join(os.path.dirname(LEDGER), "iiL_ledger_shard_*.npz")
MERGED = os.path.join(os.path.dirname(LEDGER), "iiL_ledger_merged.npz")
_W = {}


def _winit(dell):
    lowprio()
    T = tube()
    _W.update(T=T, Tq=section_tube(T, Q_STAR), co=TB.CoarseE(GOAL), er=ell_range(T), dell=dell,
              shard=os.path.join(os.path.dirname(LEDGER), f"iiL_ledger_shard_{os.getpid()}_{int(time.time())}.npz"),
              L=load_ledger("-no-such-file-"), meta=run_meta(dell))


def _wfile(f):
    t0 = time.time()
    try:
        D = dict(np.load(f, allow_pickle=True))
    except Exception as e:
        return f, None, f"unreadable ({e!r})", 0.0
    try:
        rows = file_rows(f, D, _W["T"], _W["Tq"], _W["co"], _W["er"][0], _W["er"][1], _W["dell"])
    except Exception as e:
        return f, None, f"error {e!r}"[:200], time.time() - t0
    ledger_append(_W["L"], rows, _W["meta"])
    save_ledger(_W["L"], _W["shard"])
    return f, rows, "", time.time() - t0


def done_files():
    done = set(str(x) for x in load_ledger()["fname"])
    for sh in glob.glob(SHARD_GLOB):
        done |= set(str(x) for x in np.load(sh)["fname"])     # an unreadable shard fails loudly (referee 5c C5)
    return done


def merge():
    """one row per (file, box) from the main ledger and all shards (referee 5b D4): sources in order of their file
    time (oldest first), a later row replaces an earlier one; a replaced row that differs in run id or in its class
    counts / min lb / row hash is a conflict and is reported (count and the first few keys)."""
    srcs = ([LEDGER] if os.path.exists(LEDGER) else []) + sorted(glob.glob(SHARD_GLOB))
    srcs.sort(key=os.path.getmtime)
    parts = [load_ledger(p_) for p_ in srcs]
    meta = {}
    for P in parts:
        for i_, t_ in zip(P["meta_ids"], P["meta_texts"]):
            meta[str(i_)] = str(t_)
    rows = {}
    conflicts = []
    cmp_cols = ("n_out", "n_I", "n_s1", "n_T", "n_fail", "n_unres", "min_lb")
    for P in parts:
        rh = P.get("rows_hash", np.full(len(P["fname"]), "", dtype="<U16"))
        for i in range(len(P["fname"])):
            key = (str(P["fname"][i]), int(P["box"][i]))
            new = (str(P["run_id"][i]), str(rh[i]), {c: P[c][i] for c in LCOLS})
            if key in rows:
                old = rows[key]
                same = (old[0] == new[0] and old[1] == new[1] and
                        all((old[2][c] == new[2][c]) or (np.isnan(old[2][c]) and np.isnan(new[2][c]))
                            for c in cmp_cols))
                if not same:
                    conflicts.append(key)
            rows[key] = new
    keys = list(rows.keys())
    L_ = {"fname": np.array([k[0] for k in keys], dtype="<U64"),
          "run_id": np.array([rows[k][0] for k in keys], dtype="<U12"),
          "rows_hash": np.array([rows[k][1] for k in keys], dtype="<U16"),
          "meta_ids": np.array(list(meta.keys()), dtype="<U12"), "meta_texts": np.array(list(meta.values()), dtype="<U2000"),
          **{c: np.array([rows[k][2][c] for k in keys], float) for c in LCOLS}}
    save_ledger(L_, MERGED)
    n = len(keys)
    bad = int(np.sum((L_["n_fail"] != 0) | (L_["n_unres"] > 0)))
    mlb = float(np.nanmin(L_["min_lb"])) if n and np.any(np.isfinite(L_["min_lb"])) else np.nan
    print(f"merge: {len(parts)} ledgers, {n} boxes from {len(set(L_['fname']))} files, {bad} with failures; "
          f"min lb {mlb:+.3e}; run ids {sorted(set(L_['run_id']))}; conflicting duplicates {len(conflicts)}"
          + (f" (first: {conflicts[:5]})" if conflicts else "") + f"; -> {MERGED}", flush=True)
    return L_


def tightest(K=20, src_dirs=None, out=None):
    """the K boxes with the smallest min lb in the ledger (main + shards), recomputed with the current code to get
    their tightest record (node piece, tau interval, label box, sub-box); writes tubeb/iiL_tightest.npz and prints a
    table (for the referee)."""
    lowprio()
    T = tube()
    Tq = section_tube(T, Q_STAR)
    co = TB.CoarseE(GOAL)
    ell_lo, ell_hi = ell_range(T)
    src_dirs = src_dirs or [os.path.join(ROOT, "words/out/eight", d) for d in ("saves_step3", "saves_rerun", "saves_slabrec",
                                                                              "saves_gap", "saves_gap2")]
    parts = [load_ledger()] + [load_ledger(sh) for sh in sorted(glob.glob(SHARD_GLOB))]
    fn = np.concatenate([P["fname"] for P in parts])
    bx = np.concatenate([P["box"] for P in parts])
    lb = np.concatenate([P["min_lb"] for P in parts])
    ok = np.isfinite(lb)
    o = np.flatnonzero(ok)[np.argsort(lb[ok])]
    seen, pick = set(), []
    for i in o:
        key = (str(fn[i]), int(bx[i]))
        if key not in seen:
            seen.add(key)
            pick.append(key)
        if len(pick) >= K:
            break
    files = sorted(set(k[0] for k in pick))
    res = []
    for f in files:
        path = next((os.path.join(d, f) for d in src_dirs if os.path.exists(os.path.join(d, f))), None)
        if path is None:
            print(f"  {f}: not found", flush=True)
            continue
        D = dict(np.load(path, allow_pickle=True))
        rows = file_rows(path, D, T, Tq, co, ell_lo, ell_hi, dell=0.0025)
        for r in rows:
            if (f, int(r[1])) in seen:
                res.append(r)
    res.sort(key=lambda r: r[13] if np.isfinite(r[13]) else np.inf)
    cols = ("fname",) + LCOLS
    Z = {c: np.array([r[i] for r in res]) for i, c in enumerate(cols)}
    np.savez(out or os.path.join(os.path.dirname(LEDGER), "iiL_tightest.npz"), **Z)
    print("file  box  eps  min_lb  piece  tau  ell  phi  M_top  sub-box (phi, M_top)  pieces")
    for r in res:
        d = dict(zip(cols, r))
        sub = ("saved box" if d["lb_sub"] == -1 else
               f"sub {int(d['lb_sub'])} phi [{d['lb_sub_phi_a']:.4f}, {d['lb_sub_phi_b']:.4f}] "
               f"M [{d['lb_sub_M_a']:.5f}, {d['lb_sub_M_b']:.5f}]")
        print(f"{d['fname']} {int(d['box'])} eps {d['eps']:.4g} lb {d['min_lb']:+.3e} piece {int(d['lb_piece'])} "
              f"tau [{d['lb_tau_a']:.5f}, {d['lb_tau_b']:.5f}] ell [{d['lb_ell_a']:.5f}, {d['lb_ell_b']:.5f}] "
              f"phi [{d['lb_phi_a']:.4f}, {d['lb_phi_b']:.4f}] M [{d['lb_Mt_a']:.5f}, {d['lb_Mt_b']:.5f}] {sub} "
              f"npc {int(d['lb_npc'])}", flush=True)
    return res


def stream_par(src, workers, dell=0.0025, poll=60, idle_stop=7200, settle=60):
    """stream with a pool of `workers` processes over the files (one file = one task); src: one directory or several,
    comma-separated (file names are md5 keys of the boxes, so one done set serves all)"""
    import multiprocessing as mp
    lowprio()
    done = done_files()
    stopf = os.path.join(os.path.dirname(LEDGER), "STOP_iiL")
    print(f"stream_par: {src}, {workers} workers, {len(done)} files done (ledger + shards), run {run_meta(dell)[0]}: "
          f"{run_meta(dell)[1]}", flush=True)
    pool = mp.Pool(workers, initializer=_winit, initargs=(dell,))
    inflight = {}
    last_new = time.time()
    st = {"nbox": 0, "tsum": 0.0}
    t_start = time.time()

    def cb(res):
        f, rows, err, dt = res
        inflight.pop(f, None)
        if rows is None:
            print(f"  {os.path.basename(f)}: {err}, retried later", flush=True)
            return
        done.add(os.path.basename(f))
        st["nbox"] += len(rows)
        st["tsum"] += dt
        nb = st["nbox"]
        fail = any(r[11] != 0 or r[12] > 0 for r in rows)
        mlb = np.nanmin([r[13] for r in rows]) if any(np.isfinite(r[13]) for r in rows) else np.nan
        print(f"  {os.path.basename(f)}: {len(rows)} boxes, {dt:.1f} s; this run {nb} boxes, "
              f"{st['tsum'] / nb:.1f} s per box per worker, wall {(time.time() - t_start) / nb:.2f} s per box; "
              f"min lb {mlb:+.3e}" + ("; FAIL" if fail else ""), flush=True)
    while True:
        if os.path.exists(stopf):
            print("stream_par: STOP file found, finishing the files in flight", flush=True)
            break
        now = time.time()
        files = [f for d in src.split(",") for f in glob.glob(os.path.join(d, "b_*.npz"))
                 if os.path.basename(f) not in done and f not in inflight and now - os.path.getmtime(f) > settle]
        files.sort(key=os.path.getmtime)
        room = 2 * workers - len(inflight)
        for f in files[:max(room, 0)]:
            inflight[f] = pool.apply_async(_wfile, (f,), callback=cb)
        if files or inflight:
            last_new = now
        elif now - last_new > idle_stop:
            print("stream_par: idle, stopping", flush=True)
            break
        time.sleep(5 if (files or inflight) else poll)
    pool.close()
    pool.join()
    merge()


def stream(src, dell=0.0025, poll=60, idle_stop=7200, settle=60):
    """process src/b_*.npz as they appear; ledger rows per box; stop on src/../tubeb/STOP_iiL or after idle_stop s"""
    lowprio()
    T = tube()
    Tq = section_tube(T, Q_STAR)
    co = TB.CoarseE(GOAL)
    ell_lo, ell_hi = ell_range(T)
    L_ = load_ledger()
    meta = run_meta(dell)
    done = set(str(f) for f in L_["fname"])
    stopf = os.path.join(os.path.dirname(LEDGER), "STOP_iiL")
    last_new = time.time()
    print(f"stream: {src}, ledger {LEDGER} ({len(done)} files done), run {meta[0]}: {meta[1]}", flush=True)
    while True:
        if os.path.exists(stopf):
            print("stream: STOP file found", flush=True)
            break
        now = time.time()
        files = [f for f in glob.glob(os.path.join(src, "b_*.npz"))
                 if os.path.basename(f) not in done and now - os.path.getmtime(f) > settle]
        files.sort(key=os.path.getmtime)
        if not files:
            if now - last_new > idle_stop:
                print("stream: idle, stopping", flush=True)
                break
            time.sleep(poll)
            continue
        last_new = now
        for f in files:
            t0 = time.time()
            try:
                D = dict(np.load(f, allow_pickle=True))
            except Exception as e:
                print(f"  {os.path.basename(f)}: unreadable ({e!r}), retried later", flush=True)
                continue
            newrows = file_rows(f, D, T, Tq, co, ell_lo, ell_hi, dell)
            ledger_append(L_, newrows, meta)
            save_ledger(L_)
            done.add(os.path.basename(f))
            nb = len(L_["fname"])
            bad = int(np.sum((L_["n_fail"] != 0) | (L_["n_unres"] > 0)))
            print(f"  {os.path.basename(f)}: {len(newrows)} boxes, {time.time() - t0:.1f} s; ledger {nb} boxes, "
                  f"{bad} with failures; min lb this file {np.nanmin([r[13] for r in newrows]):+.3e}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "ii1":                    # ii1 <dir or files>: (ii-1) on the given saved boxes
        lowprio()
        T_ = tube()
        el_, eh_ = ell_range(T_)
        src = sys.argv[2]
        fl = sorted(glob.glob(os.path.join(src, "b_*.npz"))) if os.path.isdir(src) else src.split(",")
        for f in fl:
            D = dict(np.load(f, allow_pickle=True))
            for b in range(D["boxes"].shape[0]):
                c1, w1 = ii1_box(T_, D, b, el_, eh_)
                print(f"{os.path.basename(f)[:14]} box {b} eps {D['boxes'][b][0]:.4g}: {c1}, W margin min {w1:+.4e}")
        sys.exit(0)
    if sys.argv[1] == "stream":             # stream [--workers N] [dir]
        args = sys.argv[2:]
        nw = 0
        if "--workers" in args:
            k = args.index("--workers")
            nw = int(args[k + 1])
            args = args[:k] + args[k + 2:]
        src_ = args[0] if args else os.path.join(ROOT, "words/out/eight/saves_step3")
        if nw > 0:
            stream_par(src_, nw, idle_stop=float(os.environ.get("SR_IDLE", "7200")), settle=float(os.environ.get("SR_SETTLE", "60")))
        else:
            stream(src_)
        sys.exit(0)
    if sys.argv[1] == "tight":              # tight [K]
        tightest(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
        sys.exit(0)
    if sys.argv[1] == "merge":
        merge()
        sys.exit(0)
    src = sys.argv[1]
    files = sorted(glob.glob(os.path.join(src, "b_*.npz"))) if os.path.isdir(src) else sys.argv[1].split(",")
    if len(sys.argv) > 2:
        files = files[:int(sys.argv[2])]
    run(files, float(sys.argv[3]) if len(sys.argv) > 3 else 0.0025)
