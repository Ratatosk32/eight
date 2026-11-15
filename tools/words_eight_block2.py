"""Stage 21, certificate step: the block fit (tools/words_eight_block.py) with phi of degree p in sigma (Lagrange,
p + 1 node layers per cell: the cell ends and p - 1 interior layers) and bilinear in (ua, ub): elements Q(p, 1, 1).
Why: along the characteristics (mostly the sigma direction) a Q1 subsolution has phi_sigma constant per cell while the
slope the inequality allows varies across the cell, so the deficit is first order in the sigma step; with p = 2 the
slope varies linearly and the loss per cell drops to O(h^3) (O(h^2) in total), for the same number of unknowns as Q1 on
the halved grid. phi stays continuous (shared node layers at cell ends; on angular faces the trace is fixed by the
shared nodes), so the subsolution argument is unchanged. The verifier reads deg [p, 1, 1] cells.
Everything else as in words_eight_block.py: overlapping blocks of node layers (and angular groups), monotone feasible
block ascent, warm start by exact prolongation, cutting planes from the float cell check, the lam rescale, JSON.
    python tools/words_eight_block2.py fit kind nu ref p tag [sweeps] [cut_rounds] [src.npz]
    python tools/words_eight_block2.py final kind tag
    python tools/words_eight_block2.py json kind tag"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import words_eight_coarse as C  # noqa: E402
import words_eight_block as B   # noqa: E402
from words_eight_shape import UK  # noqa: E402

OUT = B.OUT
MAXHEX = int(os.environ.get("BLOCK_MAXHEX", "1300"))
MASS = float(os.environ.get("BLOCK_MASS", "0"))


def make_mesh(kind, nu, ref, p):
    m = C.Mesh(nu, 0.1, rot=C.ROT[kind])
    cs = B.sigma_grid(ref)                     # cell ends
    m.p = p
    m.csig = cs
    m.ncl = len(cs) - 1                        # cell layers
    nodes = [cs[0]]
    for a, b in zip(cs[:-1], cs[1:]):
        for j in range(1, p + 1):
            nodes.append(a + (b - a) * j / p)
    m.sig = np.array(nodes)                    # node layers (C.far_rows uses sig[-1] = shi and nl - 1)
    m.slo, m.shi = cs[0], cs[-1]
    m.nl = len(m.sig)
    m.nvar = 1 + (m.nl - 1) * m.na
    m.nhex = m.nq * m.ncl
    m.quad_arr = np.array([qq[4] for qq in m.quads])
    return m


def sbasis(p, ls):
    """Lagrange basis in sigma on equispaced nodes 0, 1/p, .., 1 and its derivative (per unit ls)"""
    t = np.arange(p + 1) / p
    L = np.ones((len(ls), p + 1)); D = np.zeros((len(ls), p + 1))
    for j in range(p + 1):
        for m in range(p + 1):
            if m == j:
                continue
            L[:, j] *= (ls - t[m]) / (t[j] - t[m])
        for k in range(p + 1):
            if k == j:
                continue
            term = np.ones(len(ls)) / (t[j] - t[k])
            for m in range(p + 1):
                if m in (j, k):
                    continue
                term *= (ls - t[m]) / (t[j] - t[m])
            D[:, j] += term
    return L, D


def basis(p, ls, la, lb):
    """weights (npts, 4 (p + 1)), column 4 j + c: sigma node j, angular corner c ((i,j),(i+1,j),(i,j+1),(i+1,j+1))"""
    L, D = sbasis(p, ls)
    wa = np.stack([1 - la, la, 1 - la, la], 1); wb = np.stack([1 - lb, 1 - lb, lb, lb], 1)
    da = np.array([-1.0, 1, -1, 1])[None, :] * np.ones_like(wa); db = np.array([-1.0, -1, 1, 1])[None, :] * np.ones_like(wb)
    W = (L[:, :, None] * (wa * wb)[:, None, :]).reshape(len(ls), -1)
    Ws = (D[:, :, None] * (wa * wb)[:, None, :]).reshape(len(ls), -1)
    Wa = (L[:, :, None] * (da * wb)[:, None, :]).reshape(len(ls), -1)
    Wb = (L[:, :, None] * (wa * db)[:, None, :]).reshape(len(ls), -1)
    return W, Ws, Wa, Wb


def cell_cols(mesh, k, q):
    """unknown indices (npts, 4 (p + 1)) for cells k (cell layer) and quads q"""
    corners = mesh.quad_arr[q]
    return np.concatenate([mesh.vid((mesh.p * k + j)[:, None], corners) for j in range(mesh.p + 1)], axis=1)


TIPA = float(os.environ.get("BLOCK_TIPA", "0.25"))


def tip_quads(mesh, kind):
    """quads with a corner at the cone points +-n_I of w0_I (none for E)"""
    if kind != "I":
        return np.zeros(0, int)
    if not hasattr(mesh, "_tipq"):
        d = np.abs(mesh.angN @ C.NI)
        tips = np.flatnonzero(d > 1 - 1e-9)
        mesh._tipq = np.flatnonzero(np.isin(mesh.quad_arr, tips).any(1))
    return mesh._tipq


def tip_fix(R, U0, q, tip, ca, cb, Wa, Wb, e=1.0):
    """on cells touching the cone point of g_I, replace -Dg . Dphi by its worst case -|Dphi| (|Dg| <= 1) and bound it by
    AM-GM: -|Dphi| >= -a/2 - |Dphi|^2/(2a), a = TIPA U0: U0 -> U0 (1 - TIPA/2), Dg term dropped, gradient components
    scaled so that q |D|^2 becomes (q + 1/(2a)) |D|^2"""
    if not tip.any():
        return
    a = TIPA * U0[tip]
    R["Lin"][tip] += (ca[tip, None] * Wa[tip] + cb[tip, None] * Wb[tip]) * e
    f = np.sqrt(1 + 1 / (2 * a * q[tip]))
    R["D1"][tip] *= f[:, None]
    R["D2"][tip] *= f[:, None]
    R["U0"] = R["U0"].copy()
    R["U0"][tip] = U0[tip] * (1 - TIPA / 2)
    R["tip"] = tip


def rows_at(mesh, kind, hexes, ls, la, lb):
    k = hexes // mesh.nq
    q = hexes % mesh.nq
    s0, s1 = mesh.csig[k], mesh.csig[k + 1]
    sig = s0 + ls * (s1 - s0)
    W, Ws, Wa, Wb = basis(mesh.p, ls, la, lb)
    cols = cell_cols(mesh, k, q)
    n, (r11, r12, r22), g, ca, cb, U = C.geometry(mesh, kind, q, la, lb)
    Ws = Ws / (s1 - s0)[:, None]
    D1 = r11[:, None] * Wa + r12[:, None] * Wb
    D2 = r22[:, None] * Wb
    Lin = -W / 3 + (4 / 3 - g)[:, None] * Ws - (ca[:, None] * Wa + cb[:, None] * Wb)
    R = dict(cols=cols, W=W, Ws=Ws, D1=D1, D2=D2, Lin=Lin, sig=sig, U0=np.exp(-sig / 2) * U, n=n)
    tip = np.isin(q, tip_quads(mesh, kind))
    R["tip"] = tip
    tip_fix(R, R["U0"], 2 * np.exp(-sig), tip, ca, cb, Wa, Wb)
    return R


def far_rows(mesh, kind, qidx, la, lb):
    """words_eight_coarse.far_rows with the cone-point treatment on the tip quads (I)"""
    R = C.far_rows(mesh, kind, qidx, la, lb)
    tip = np.isin(qidx, tip_quads(mesh, kind))
    R["tip"] = tip
    if tip.any():
        _, Wa_, Wb_ = None, None, None
        W, _, Wa, Wb = C.basis_local(np.ones(len(qidx)), la, lb)
        n, (r11, r12, r22), g, ca, cb, U = C.geometry(mesh, kind, qidx, la, lb)
        tip_fix(R, R["U0"], R["q"], tip, ca, cb, Wa[:, 4:], Wb[:, 4:], np.exp(mesh.shi / 2))
    return R


def far_check(mesh, kind, coef, ng=9, depth=5):
    """words_eight_coarse.far_check with far_rows above; on tip quads only the direct grid bound (b2)"""
    def bound(qidx, lo, hi):
        m = len(qidx)
        g = np.linspace(0, 1, ng)
        A_, B_ = np.meshgrid(g, g, indexing="ij")
        la = (lo[:, 0:1] + A_.ravel()[None, :] * (hi[:, 0:1] - lo[:, 0:1])).ravel()
        lb = (lo[:, 1:2] + B_.ravel()[None, :] * (hi[:, 1:2] - lo[:, 1:2])).ravel()
        qq = np.repeat(qidx, ng * ng)
        R = far_rows(mesh, kind, qq, la, lb)
        c = coef[R["cols"]]
        A2 = np.sum(R["W"] * c, 1); d1 = np.sum(R["D1"] * c, 1); d2 = np.sum(R["D2"] * c, 1)
        rest = (np.sum(R["Lin"] * c, 1) - R["q"] * (A2 ** 2 + d1 ** 2 + d2 ** 2)).reshape(m, ng, ng)
        val = (R["U0"] + rest.ravel()).reshape(m, ng, ng)

        def c2(G):
            return (np.abs(np.diff(G, 2, axis=1)).reshape(m, -1).max(1) + np.abs(np.diff(G, 2, axis=2)).reshape(m, -1).max(1)) / 8
        n = C.geometry(mesh, kind, qq, la, lb)[0]
        Ulo = 0
        for u in UK:
            qv = (1 - n @ u).reshape(m, ng, ng)
            Ulo = Ulo + np.maximum(qv.reshape(m, -1).max(1) + c2(qv), 1e-300) ** -0.5
        b1 = np.minimum(Ulo, 1e6) + rest.reshape(m, -1).min(1) - c2(rest)
        b2 = val.reshape(m, -1).min(1) - c2(val)
        tipq = R["tip"].reshape(m, -1).any(1)
        b1[tipq] = -np.inf
        return np.maximum(b1, b2)
    qidx = np.arange(mesh.nq)
    lo = np.zeros((mesh.nq, 2)); hi = np.ones((mesh.nq, 2))
    worst = np.full(mesh.nq, np.inf)
    for lev in range(depth + 1):
        b = bound(qidx, lo, hi)
        ok = b >= 0
        np.minimum.at(worst, qidx[ok], b[ok])
        bad = ~ok
        if lev == depth or not bad.any():
            np.minimum.at(worst, qidx[bad], b[bad])
            break
        qb, lb_, ub_ = qidx[bad], lo[bad], hi[bad]
        mid = (lb_ + ub_) / 2
        nq, nl, nh = [], [], []
        for cc in range(4):
            bits = np.array([cc & 1, (cc >> 1) & 1])[None, :]
            nq.append(qb); nl.append(np.where(bits == 0, lb_, mid)); nh.append(np.where(bits == 0, mid, ub_))
        qidx = np.concatenate(nq); lo = np.vstack(nl); hi = np.vstack(nh)
    return worst


def sample_rows(mesh, kind, hexes, extra):
    pts = np.array([[a, b, c] for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)])
    if mesh.p >= 2:      # more points in sigma for the higher degree
        sp_ = np.linspace(0, 1, 2 * mesh.p + 1)
        pts = np.array([[a, b, c] for a in sp_ for b in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)])
    H = np.repeat(hexes, len(pts))
    P = np.tile(pts, (len(hexes), 1))
    if extra is not None and len(extra[0]):
        sel = np.isin(extra[0], hexes)
        H = np.concatenate([H, extra[0][sel]]); P = np.vstack([P, extra[1][sel]])
    R = rows_at(mesh, kind, H, P[:, 0], P[:, 1], P[:, 2])
    X = []
    hh = 1e-4
    step = np.array([1.0 / (2 * mesh.p), 0.5, 0.5])      # sample spacing per axis
    for ax in range(3):
        Pp = P.copy(); Pm = P.copy()
        Pp[:, ax] += hh; Pm[:, ax] -= hh
        Rp = rows_at(mesh, kind, H, Pp[:, 0], Pp[:, 1], Pp[:, 2])
        Rm = rows_at(mesh, kind, H, Pm[:, 0], Pm[:, 1], Pm[:, 2])
        for key in ("Ws", "D1", "D2"):
            # dip bound of the quadratic term between samples: (q/2) (spacing/2)^2 sum (d comp)^2
            X.append((Rp[key] - Rm[key]) / (2 * hh) * (step[ax] / 2))
    R["X"] = X
    return R


def hexes_of_block(mesh, k0, k1, nodes):
    p = mesh.p
    ks = np.arange(mesh.ncl)
    ks = ks[(p * ks + p >= k0) & (p * ks <= k1)]
    qs = np.arange(mesh.nq) if nodes is None else np.flatnonzero(np.isin(mesh.quad_arr, nodes).any(1))
    out = [k * mesh.nq + qs for k in ks]
    if k0 == 0 and nodes is not None:
        out.append(np.arange(mesh.nq))
    return np.unique(np.concatenate(out))


def make_blocks(mesh, kind, maxhex=MAXHEX):
    """node-layer ranges ending on cell boundaries, overlapping by one cell layer, times angular groups"""
    p = mesh.p
    Lc = int(os.environ.get("BLOCK_LC", "0")) or max(2, maxhex // mesh.nq - 1)    # cell layers of free nodes
    G = max(1, int(np.ceil((Lc + 1) * mesh.nq * 1.15 / maxhex)))
    groups = B.ang_groups(mesh, kind, G)
    out = []
    c0 = 0
    while True:
        c1 = min(c0 + Lc, mesh.ncl)
        for g in groups:
            out.append((p * c0, p * c1, g))
        if c1 == mesh.ncl:
            break
        c0 = c1 - 1
    return out


def locate(mesh, xi):
    r = np.linalg.norm(xi, axis=1)
    s = np.log(r)
    n = xi / r[:, None]
    q, la, lb = mesh.locate_ang(n)
    sc = np.clip(s, mesh.slo, mesh.shi)
    k = np.clip(np.searchsorted(mesh.csig, sc, side="right") - 1, 0, mesh.ncl - 1)
    ls = (sc - mesh.csig[k]) / (mesh.csig[k + 1] - mesh.csig[k])
    W = basis(mesh.p, ls, la, lb)[0]
    fac = np.where(s > mesh.shi, np.exp(-(s - mesh.shi) / 2), 1.0)
    return cell_cols(mesh, k, q), W, fac


def eval_phi(mesh, coef, xi):
    cols, W, fac = locate(mesh, xi)
    return np.sum(W * coef[cols], 1) * fac


def S_value(mesh, coef, kind, y, tau):
    xi = y / tau ** (4 / 3)
    return tau ** (1 / 3) * (C.w0(kind, xi) + eval_phi(mesh, coef, xi))


def objective(mesh, kind):
    from words_eight_shape import eight_discrete
    c = eight_discrete(48)[0][24]
    xi, wt = C.targets(kind, c)
    cols, W, fac = locate(mesh, xi)
    ot = np.zeros(mesh.nvar)
    np.add.at(ot, cols.ravel(), (W * (wt * fac)[:, None]).ravel())
    ds = np.diff(mesh.sig)
    wl = np.zeros(mesh.nl)
    wl[:-1] += ds / 2
    wl[1:] += ds / 2
    wl *= np.exp(-np.maximum(mesh.sig - 3.0, 0) / 2)
    mass = np.zeros(mesh.nvar)
    mass[0] = wl[0]
    for k in range(1, mesh.nl):
        mass[1 + (k - 1) * mesh.na: 1 + k * mesh.na] = wl[k] / mesh.na
    mass *= MASS * wt.sum() / mass.sum()
    return ot + mass, ot, c


def prolong(mesh, kind, src):
    d = np.load(src)
    if "p" in d:          # a block2 fit
        m0 = make_mesh(kind, int(d["nu"]), str(d["ref"]), int(d["p"]))
        f = lambda xi: eval_phi(m0, d["coef"], xi)
        slo = m0.slo
    else:
        rot0 = float(d["rot"]) if "rot" in d else float(os.environ.get("BLOCK_SRC_ROT", C.ROT[kind]))
        m0 = C.Mesh(int(d["nu"]), float(d["dsb"]) if "dsb" in d else 0.1, rot=rot0)
        if "sig" in d:
            m0.sig = d["sig"]; m0.slo, m0.shi = m0.sig[0], m0.sig[-1]; m0.nl = len(m0.sig)
        f = lambda xi: C.eval_phi(m0, d["coef"], xi)
        slo = m0.slo
    out = np.zeros(mesh.nvar)
    out[0] = d["coef"][0]
    for k in range(1, mesh.nl):
        if mesh.sig[k] <= slo + 1e-12:
            out[1 + (k - 1) * mesh.na: 1 + k * mesh.na] = d["coef"][0]
        else:
            out[1 + (k - 1) * mesh.na: 1 + k * mesh.na] = f(np.exp(mesh.sig[k]) * mesh.angN)
    return out


def slack(mesh, kind, coef, hexes, extra):
    out = np.inf
    for i0 in range(0, len(hexes), 2000):
        R = sample_rows(mesh, kind, hexes[i0:i0 + 2000], extra)
        cc = coef[R["cols"]]
        q = 2 * np.exp(-R["sig"])
        comps = [R["Ws"], R["D1"], R["D2"]] + R["X"]
        sl = R["U0"] + np.sum(R["Lin"] * cc, 1) - q * sum(np.sum(X * cc, 1) ** 2 for X in comps)
        out = min(out, np.min(sl / R["U0"]))
    return out


# ---------------------------------------------------------------- float cell check (as words_eight_coarse) --------
def F_hex_grid(mesh, kind, coef, hexes, ng, lo, hi):
    g = np.linspace(0, 1, ng)
    LS, LA, LB = np.meshgrid(g, g, g, indexing="ij")
    LS, LA, LB = LS.ravel(), LA.ravel(), LB.ravel()
    H = np.repeat(hexes, len(LS))
    lo = np.repeat(lo, len(LS), axis=0)
    hi = np.repeat(hi, len(LS), axis=0)
    ls = lo[:, 0] + np.tile(LS, len(hexes)) * (hi[:, 0] - lo[:, 0])
    la = lo[:, 1] + np.tile(LA, len(hexes)) * (hi[:, 1] - lo[:, 1])
    lb = lo[:, 2] + np.tile(LB, len(hexes)) * (hi[:, 2] - lo[:, 2])
    R = rows_at(mesh, kind, H, ls, la, lb)
    c = coef[R["cols"]]
    ps = np.sum(R["Ws"] * c, 1)
    d1 = np.sum(R["D1"] * c, 1); d2 = np.sum(R["D2"] * c, 1)
    F = R["U0"] - 2 * np.exp(-R["sig"]) * (ps ** 2 + d1 ** 2 + d2 ** 2) + np.sum(R["Lin"] * c, 1)
    sh = (len(hexes), ng, ng, ng)
    Q = [(1 - R["n"] @ u).reshape(sh) for u in UK]
    if R["tip"].any():      # the split bound assumes the true U0: make it useless on tip cells (cell_bound decides)
        Q = [np.where(R["tip"].reshape(sh), 1e300, qq) for qq in Q]
    return F.reshape(sh), R["U0"].reshape(sh), Q, R["sig"].reshape(sh)


def check_cells(mesh, kind, coef, depth=3, ng=None, hexes=None, thr=None):
    """float Taylor bound of min F per hexahedron (grid min minus second-difference curvature), bisection where it
    fails (as words_eight_coarse.check_cells); ng = 4 for p = 1, 5 for p = 2"""
    ng = ng or (3 + mesh.p)
    hexes = np.arange(mesh.nhex) if hexes is None else np.asarray(hexes)
    lo = np.zeros((len(hexes), 3)); hi = np.ones((len(hexes), 3))
    worst = np.full(mesh.nhex, np.inf)
    fail_pts = []
    ncheck = 0
    for lev in range(depth + 1):
        if len(hexes) == 0:
            break
        bnds = []
        CH = 8000
        for i0 in range(0, len(hexes), CH):
            F, U0, Q, SIG = F_hex_grid(mesh, kind, coef, hexes[i0:i0 + CH], ng, lo[i0:i0 + CH], hi[i0:i0 + CH])
            bnds.append(np.maximum(C.cell_bound(F, None), C.split_bound(F, U0, Q, SIG)))
        bnd = np.concatenate(bnds)
        if thr is not None:
            bnd = bnd - thr[hexes]          # the bound reported relative to the per-cell target
        ncheck += len(hexes)
        ok = bnd >= 0
        np.minimum.at(worst, hexes[ok], bnd[ok])
        bad = ~ok
        if lev == depth or not bad.any():
            np.minimum.at(worst, hexes[bad], bnd[bad])
            hb, bb_, lb_, ub_ = hexes[bad], bnd[bad], lo[bad], hi[bad]
            order = np.lexsort((bb_, hb))
            first = np.ones(len(order), bool)
            first[1:] = hb[order][1:] != hb[order][:-1]
            for i in order[first]:
                fail_pts.append((hb[i], (lb_[i] + ub_[i]) / 2, lb_[i], ub_[i], bb_[i]))
            break
        hb, lb_, ub_ = hexes[bad], lo[bad], hi[bad]
        mid = (lb_ + ub_) / 2
        nh, nl_, nu_ = [], [], []
        for cc in range(8):
            bits = np.array([(cc >> a) & 1 for a in range(3)])[None, :]
            nh.append(hb); nl_.append(np.where(bits == 0, lb_, mid)); nu_.append(np.where(bits == 0, mid, ub_))
        hexes = np.concatenate(nh); lo = np.vstack(nl_); hi = np.vstack(nu_)
    return worst, fail_pts, ncheck


def U_lower_hex(mesh, kind, coef):
    Ulo = np.zeros(mesh.nhex)
    for i0 in range(0, mesh.nhex, 8000):
        h = np.arange(i0, min(i0 + 8000, mesh.nhex))
        F, U0, Q, SIG = F_hex_grid(mesh, kind, coef, h, 4, np.zeros((len(h), 3)), np.ones((len(h), 3)))
        Ulo[h] = C.split_bound(np.zeros_like(F) + U0, U0, Q, SIG)
    return Ulo


# ---------------------------------------------------------------- block ascent -----------------------------------
RHO = float(os.environ.get("BLOCK_RHO", "0.01"))


def socp_price(blocks, free, coef, lin, rho, box=None):
    """max lin[free] . x - rho/2 |x - coef[free]|^2 over the free unknowns (others fixed at coef), cones as
    words_eight_coarse.socp; also returns the dual prices pi = d(optimal value)/d(fixed value) on the fixed unknowns"""
    import scipy.sparse as sp
    import clarabel
    nv = len(coef)
    pos = np.full(nv, -1)
    pos[free] = np.arange(len(free))
    fixed = coef.copy()
    fixed[free] = 0.0
    rows, cols, vals, bs, cones, keep = [], [], [], [], [], []
    off = 0
    for Bk in blocks:
        m = len(Bk["U0"])
        if m == 0:
            continue
        far = "sig" not in Bk
        q = Bk["q"] if far else 2 * np.exp(-Bk["sig"])
        al2 = 1 / (1 + Bk["U0"])
        sq = np.sqrt(q * al2)
        comps = [Bk["Ws"], Bk["D1"], Bk["D2"]] + ([Bk["W"]] if far else []) + Bk.get("X", [])
        dim = 2 + len(comps)
        Cc = Bk["cols"]
        P = pos[Cc]
        isf = P >= 0
        cf = fixed[Cc]
        coefs = [al2[:, None] * Bk["Lin"]] + [2 * sq[:, None] * X for X in comps] + [al2[:, None] * Bk["Lin"]]
        bb = np.zeros(m * dim)
        for j, X in enumerate(coefs):
            r = off + np.arange(m) * dim + j
            rr = np.repeat(r, Cc.shape[1]).reshape(m, -1)
            rows.append(rr[isf]); cols.append(P[isf]); vals.append(-X[isf])
            bb[j::dim] += np.sum(X * cf, axis=1)
            keep.append((off, dim, j, Cc, isf, X))
        bb[0::dim] += al2 * Bk["U0"] + 1
        bb[dim - 1::dim] += al2 * Bk["U0"] - 1
        bs.append(bb)
        cones += [clarabel.SecondOrderConeT(dim)] * m
        off += m * dim
    nf = len(free)
    if box is not None:          # trust region |x - xc| <= r (componentwise), as nonnegative rows
        xc, r = box
        ar = np.arange(nf)
        rows += [off + ar, off + nf + ar]; cols += [ar, ar]; vals += [np.ones(nf), -np.ones(nf)]
        bs += [xc[free] + r, r - xc[free]]
        cones += [clarabel.NonnegativeConeT(2 * nf)]
        off += 2 * nf
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(off, nf))
    b = np.concatenate(bs)
    P0 = sp.identity(nf, format="csc") * rho if rho > 0 else sp.csc_matrix((nf, nf))
    qv = -lin[free] - rho * coef[free]
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.max_iter = 400
    st.direct_solve_method = os.environ.get("BLOCK_SOLVER", "qdldl")
    st.max_threads = 2
    sol = clarabel.DefaultSolver(P0, qv, A, b, cones, st).solve()
    z = np.array(sol.z)
    pi = np.zeros(nv)
    if len(z) == off and box is None:
        for (o, dim, j, Cc, isf, X) in keep:
            m = Cc.shape[0]
            zj = z[o + np.arange(m) * dim + j]
            np.add.at(pi, Cc[~isf], (X * zj[:, None])[~isf])
    pi[free] = 0.0
    return np.array(sol.x), str(sol.status), off, pi


def solve_block(mesh, kind, coef, lin, blk, extra, extra_far, rho=None):
    """one proximal block step; returns the new coefficients and the block's dual prices on its fixed neighbours"""
    B.wait_ram()
    t0 = time.time()
    k0, k1, nodes = blk
    hexes = hexes_of_block(mesh, k0, k1, nodes)
    free = B.free_of_block(mesh, k0, k1, nodes)
    blocks = [sample_rows(mesh, kind, hexes, extra)]
    if k1 == mesh.nl - 1:
        blocks.append(B.far_sample(mesh, kind, extra_far))
    x, st, nrows, pi = socp_price(blocks, free, coef, lin, RHO if rho is None else rho)
    del blocks
    ok = st in ("Solved", "AlmostSolved") and np.all(np.isfinite(x))
    dx = np.abs(x - coef[free]).max() if ok else np.nan
    if ok:
        coef = coef.copy()
        coef[free] = x
    else:
        pi = None
    print(f"   block nodes {k0}-{k1}{'' if nodes is None else f' ({len(nodes)} ang.)'} (sigma {mesh.sig[k0]:.2f}-"
          f"{mesh.sig[k1]:.2f}): {len(hexes)} hex, {len(free)} unk, {nrows} rows: {st}{'' if ok else ' REJECTED'}, "
          f"max|lin| {np.abs(lin[free]).max():.2e}, max|dx| {dx:.2e}; {time.time() - t0:.0f} s, peak {B.peak_mb():.0f} MB",
          flush=True)
    return coef, pi


def report(mesh, kind, coef, ot_t, c):
    tau = C.TS if kind == "E" else 1 - C.TS
    Vc = 1.22789891 if kind == "E" else 1.29195718
    t0 = time.time()
    worst, fails, ncheck = check_cells(mesh, kind, coef)
    fworst = far_check(mesh, kind, coef)
    Sc = S_value(mesh, coef, kind, c[None], tau)[0]
    kk = np.arange(mesh.nhex) // mesh.nq
    print(f"   check ({time.time() - t0:.0f} s, {ncheck} boxes): worst cell bound {worst.min():+.3e}; failing hexahedra "
          f"{np.sum(worst < 0)}; far worst {fworst.min():+.3e}; target objective {ot_t @ coef:.5f}; S(c) = {Sc:.5f} "
          f"(V = {Vc}, deficit {100 * (Vc - Sc) / Vc:.3f} %)", flush=True)
    print(f"   failing by cell layer: {np.bincount(kk[worst < 0], minlength=mesh.ncl).tolist()}", flush=True)
    return worst, fails, fworst, Sc


def save(path, mesh, kind, coef, extra, extra_far, nu, ref):
    np.savez(path, coef=coef, ex_h=extra[0], ex_p=extra[1], ef_q=extra_far[0], ef_a=extra_far[1], ef_b=extra_far[2],
             nu=nu, ref=ref, p=mesh.p, rot=C.ROT[kind])


def fit(kind, nu, ref, p, tag, sweeps=2, cut_rounds=3, src=None):
    mesh = make_mesh(kind, nu, ref, p)
    print(f"{kind}: Q({p},1,1), nu = {nu} ({mesh.nq} quads, {mesh.na} angular nodes), cell grid '{ref}': {mesh.ncl} cell "
          f"layers, {mesh.nl} node layers, {mesh.nhex} hexahedra, {mesh.nvar} unknowns; MAXHEX {MAXHEX}, mass {MASS}",
          flush=True)
    ot_m, ot, c = objective(mesh, kind)
    ot_t = ot
    if MASS > 0:
        ot = ot_m
    tau = C.TS if kind == "E" else 1 - C.TS
    extra = (np.zeros(0, int), np.zeros((0, 3)))
    extra_far = (np.zeros(0, int), np.zeros(0), np.zeros(0))
    path = OUT + f"block2_{kind}_{tag}.npz"
    if os.path.exists(path) and os.environ.get("BLOCK_RESUME", "1") == "1":
        d = np.load(path)
        coef = d["coef"]
        extra = (d["ex_h"], d["ex_p"]); extra_far = (d["ef_q"], d["ef_a"], d["ef_b"])
        print(f"   resumed from {path}", flush=True)
        done_sweeps = int(os.environ.get("BLOCK_DONE_SWEEPS", "0"))
    else:
        src = src or C.OUT + f"coarse_{kind}_nu4_ds0.1.npz"
        coef0 = prolong(mesh, kind, src)
        lam = 1.0
        for _ in range(60):
            ms = slack(mesh, kind, lam * coef0, np.arange(mesh.nhex), None)
            if ms >= 1e-6:
                break
            lam *= 0.995 if ms > -0.02 else 0.97
        coef = lam * coef0
        print(f"   warm start from {os.path.basename(src)}: lam0 = {lam:.4f}, min slack {ms:+.2e}; S(c) "
              f"{S_value(mesh, coef, kind, c[None], tau)[0]:.5f}", flush=True)
        done_sweeps = 0
    blocks = make_blocks(mesh, kind)
    print(f"   {len(blocks)} blocks, node-layer ranges {sorted(set((b[0], b[1]) for b in blocks))}", flush=True)
    prices = {}
    if os.path.exists(path.replace(".npz", "_pi.npz")):
        dp = np.load(path.replace(".npz", "_pi.npz"))
        prices = {int(k[1:]): dp[k] for k in dp.files}

    def lin_for(i):
        lin = ot.copy()
        for j, pj in prices.items():
            if j != i:
                lin = lin + pj
        return lin

    skip_tol = float(os.environ.get("BLOCK_SKIP", "1e-4"))
    for sw in range(done_sweeps, sweeps):
        # first sweep bottom-up (prices travel from the targets towards high sigma), then alternate
        idx = list(range(len(blocks))) if sw % 2 == 0 else list(range(len(blocks)))[::-1]
        t0 = time.time()
        nsolved = 0
        for i in idx:
            blk = blocks[i]
            free = B.free_of_block(mesh, *blk)
            lin = lin_for(i)
            if np.abs(lin[free]).max() < skip_tol * np.abs(ot).max():
                continue
            coef, pi = solve_block(mesh, kind, coef, lin, blk, extra, extra_far)
            nsolved += 1
            if pi is not None:
                prices[i] = pi
        save(path, mesh, kind, coef, extra, extra_far, nu, ref)
        np.savez(path.replace(".npz", "_pi.npz"), **{f"b{k}": v for k, v in prices.items()})
        print(f" sweep {sw}: {nsolved} blocks solved, {time.time() - t0:.0f} s; target objective {ot_t @ coef:.5f}; S(c) "
              f"{S_value(mesh, coef, kind, c[None], tau)[0]:.5f}", flush=True)
    worst, fails, fworst, Sc = report(mesh, kind, coef, ot_t, c)
    for rd in range(cut_rounds):
        if np.sum(worst < 0) == 0 and fworst.min() >= 0:
            break
        g = np.array([[a, b, cc] for a in (0, 0.5, 1) for b in (0, 0.5, 1) for cc in (0, 0.5, 1)])
        eh, ep = [], []
        for h, ctr, lo, hi, _ in sorted(fails, key=lambda f: f[4])[:8000]:
            eh.append(np.full(len(g), h)); ep.append(lo + g * (hi - lo))
        if eh:
            extra = (np.concatenate([extra[0]] + eh), np.vstack([extra[1]] + ep))
        fq = np.flatnonzero(fworst < 0)
        if len(fq):
            g2 = np.linspace(0, 1, 5)
            A_, B_ = np.meshgrid(g2, g2, indexing="ij")
            extra_far = (np.concatenate([extra_far[0], np.repeat(fq, 25)]),
                         np.concatenate([extra_far[1], np.tile(A_.ravel(), len(fq))]),
                         np.concatenate([extra_far[2], np.tile(B_.ravel(), len(fq))]))
        badh = np.flatnonzero(worst < 0)
        if len(fq):
            badh = np.union1d(badh, (mesh.ncl - 1) * mesh.nq + fq)
        # the smallest set of blocks covering each failing cell with all its nodes free
        todo = []
        for blk in blocks:
            hk = badh // mesh.nq
            sel = badh[(p * hk >= blk[0]) & (p * hk + p <= blk[1])]
            if blk[2] is not None:
                sel = sel[np.isin(mesh.quad_arr[sel % mesh.nq], blk[2]).all(1)]
            if len(sel):
                todo.append(blk)
        print(f" cut round {rd + 1}: {len(eh)} failing boxes, {len(fq)} far quads; re-solving {len(todo)} blocks", flush=True)
        t0 = time.time()
        for blk in todo[::-1]:
            i = blocks.index(blk) if False else [ii for ii, bb in enumerate(blocks) if bb is blk][0]
            coef, pi = solve_block(mesh, kind, coef, lin_for(i), blk, extra, extra_far)
            if pi is not None:
                prices[i] = pi
        save(path, mesh, kind, coef, extra, extra_far, nu, ref)
        print(f"   {time.time() - t0:.0f} s", flush=True)
        worst, fails, fworst, Sc = report(mesh, kind, coef, ot_t, c)


def finalize(kind, tag, depth=3):
    d = np.load(OUT + f"block2_{kind}_{tag}.npz")
    nu, ref, p = int(d["nu"]), str(d["ref"]), int(d["p"])
    mesh = make_mesh(kind, nu, ref, p)
    coef = d["coef"]
    t0 = time.time()
    worst, _, _ = check_cells(mesh, kind, coef, depth)
    fw = far_check(mesh, kind, coef)
    Ulo = U_lower_hex(mesh, kind, coef)
    lam = 1.0
    neg = worst < 0
    if neg.any():
        lam = min(lam, np.min(Ulo[neg] / (Ulo[neg] - worst[neg])))
    if (fw < 0).any():
        lam = min(lam, np.min(3.0 / (3.0 - fw[fw < 0])))
    lam *= 0.9999
    coef2 = coef * lam
    worst2, _, ncheck2 = check_cells(mesh, kind, coef2, depth)
    fw2 = far_check(mesh, kind, coef2)
    tau = C.TS if kind == "E" else 1 - C.TS
    from words_eight_shape import eight_discrete
    c = eight_discrete(48)[0][24]
    Sc0 = S_value(mesh, coef, kind, c[None], tau)[0]
    Sc = S_value(mesh, coef2, kind, c[None], tau)[0]
    Vc = 1.22789891 if kind == "E" else 1.29195718
    print(f"{kind} {tag}: before: worst cell bound {worst.min():+.3e}, failing {np.sum(worst < 0)}, far {fw.min():+.3e}, "
          f"S(c) {Sc0:.5f} ({100 * (Vc - Sc0) / Vc:.3f} %); lam = {lam:.6f}; after: worst cell bound "
          f"{worst2.min():+.3e} (failing {np.sum(worst2 < 0)}), far worst {fw2.min():+.3e}; boxes {ncheck2}; S(c) = "
          f"{Sc:.5f}, deficit {Vc - Sc:.4f} ({100 * (Vc - Sc) / Vc:.3f} %); {time.time() - t0:.0f} s", flush=True)
    np.savez(OUT + f"block2_{kind}_{tag}_final.npz", coef=coef2, lam=lam, worst=worst2, fworst=fw2, nu=nu, ref=ref, p=p,
             rot=C.ROT[kind])


def export_json(kind, tag):
    d = np.load(OUT + f"block2_{kind}_{tag}_final.npz")
    nu, ref, p = int(d["nu"]), str(d["ref"]), int(d["p"])
    mesh = make_mesh(kind, nu, ref, p)
    coef = d["coef"]
    # monomials in y in [-1, 1]^3 (y1 sigma, y2 ua, y3 ub), degrees (p, 1, 1): fit at the nodes (exact)
    ys = np.linspace(-1, 1, p + 1)
    Y = np.array([[a, b, c] for a in ys for b in (-1.0, 1.0) for c in (-1.0, 1.0)])
    expo = [(i, j, l) for i in range(p + 1) for j in range(2) for l in range(2)]
    V = np.array([[y[0] ** i * y[1] ** j * y[2] ** l for (i, j, l) in expo] for y in Y])
    Vi = np.linalg.inv(V)
    # node values at Y in our column order (4 j + corner, corner c: a-bit c & 1, b-bit c >> 1)
    perm = []
    for a in range(p + 1):
        for b in range(2):
            for cc in range(2):
                perm.append(4 * a + (b | (cc << 1)))
    cells = []
    for k in range(mesh.ncl):
        cols = cell_cols(mesh, np.full(mesh.nq, k), np.arange(mesh.nq))
        vals = coef[cols][:, perm]
        M = vals @ Vi.T
        for qi, (fc, ua0, ub0, du, ids) in enumerate(mesh.quads):
            Cc = np.zeros((p + 1, 2, 2))
            for e, (i, j, l) in enumerate(expo):
                Cc[i, j, l] = M[qi, e]
            nodes = coef[cols[qi]][perm].reshape(p + 1, 2, 2)
            ids = cols[qi][perm].reshape(p + 1, 2, 2)
            cells.append({"face": int(fc), "sigma": [float(mesh.csig[k]), float(mesh.csig[k + 1])],
                          "ua": [float(ua0), float(ua0 + du)], "ub": [float(ub0), float(ub0 + du)],
                          "deg": [p, 1, 1], "coef": Cc.tolist(),
                          "nodal": {"type": f"Q{p}11", "vals": nodes.tolist()}, "node_ids": ids.tolist()})
    # self-test: the monomial form against eval_phi at random points of random cells
    rng = np.random.default_rng(1)
    err = 0.0
    for idx in rng.integers(0, len(cells), 200):
        cl = cells[idx]
        t = rng.random(3)
        s = cl["sigma"][0] + t[0] * (cl["sigma"][1] - cl["sigma"][0])
        ua = cl["ua"][0] + t[1] * (cl["ua"][1] - cl["ua"][0]); ub = cl["ub"][0] + t[2] * (cl["ub"][1] - cl["ub"][0])
        y = 2 * t - 1
        Cc = np.array(cl["coef"])
        val = sum(Cc[i, j, l] * y[0] ** i * y[1] ** j * y[2] ** l for i in range(p + 1) for j in range(2) for l in range(2))
        n = C.Nmap(cl["face"], np.array([ua]), np.array([ub]), mesh.faces)[0]
        if n[2] < 0:
            continue
        err = max(err, abs(val - eval_phi(mesh, coef, (np.exp(s) * n)[None])[0]))
    out = {"side": kind,
           "faces": [{"centre": c.tolist(), "ea": ea.tolist(), "eb": eb.tolist(), "ub_range": list(r)} for c, ea, eb, r in mesh.faces],
           "cells": cells, "far": {"sigma_hi": float(mesh.shi)},
           "low": {"sigma_lo": float(mesh.slo), "phi_lo": float(coef[0]), "node_id": 0},
           "element_doc": (f"Q{p}11: phi on the cell is the tensor Lagrange interpolant of 'nodal' 'vals'[i][j][l] at local "
                           f"y1 = -1 + 2 i/{p} (sigma), y2 = -1 + 2 j (ua), y3 = -1 + 2 l (ub), y in [-1, 1]^3 mapping the "
                           f"cell's sigma, ua, ub ranges; 'node_ids' are global unknown indices, equal ids carry the "
                           f"bit-identical value (cube seams and mirror-duplicated vertices are merged; every node at "
                           f"sigma = sigma_lo has id 0 = phi_lo). 'coef' are the float monomial coefficients of the same "
                           f"polynomial."),
           "source": f"block2_{kind}_{tag}_final.npz"}
    path = OUT + f"cert_pieces_{kind}_block.json"
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {path}: {len(cells)} cells; monomial self-test max error {err:.2e}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6],
            int(sys.argv[7]) if len(sys.argv) > 7 else 2, int(sys.argv[8]) if len(sys.argv) > 8 else 3,
            sys.argv[9] if len(sys.argv) > 9 else None)
    elif cmd == "final":
        finalize(sys.argv[2], sys.argv[3])
    elif cmd == "json":
        export_json(sys.argv[2], sys.argv[3])
