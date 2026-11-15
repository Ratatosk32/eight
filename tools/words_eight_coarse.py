"""Stage 21, certificate step: coarse global subsolutions in self-similar variables (float, cell-wise checked).

w = w0 + phi, xi = e^sigma n. phi is continuous piecewise trilinear (Q1) on hexahedra in (sigma, u_a, u_b), where
(u_a, u_b) are equiangular cubed-sphere coordinates on the northern hemisphere (top face + upper halves of the four
side faces); phi is extended evenly to the southern hemisphere (mirror symmetry; kinks on the equator are allowed).
Below sigma_lo phi is a constant (all layer-0 nodes are one unknown); above sigma_hi phi = e^(-(sigma - sigma_hi)/2)
phi(sigma_hi, n) (far field), whose inequality is imposed at sigma_hi (sufficient for all larger sigma).
    F = e^(-sigma/2) U(n) - 2 e^(-sigma) (phi_s^2 + |D phi|^2) - phi/3 + (4/3 - g) phi_s - Dg . D phi >= 0
(g = 4 e^(-sigma) w0: g_E = 1 - n1, g_I = 1 - sqrt(1 - (n.n_I)^2)). Far field (phi = e^(-sigma/2) A(n)):
    U - A (1 - g/2) - Dg . DA >= 2 e^(-3 sigma_hi/2) (A^2/4 + |DA|^2).
Fit: second-order cone program (Clarabel), F >= 0 at 9 points per hexahedron (plus added points), maximising a
weighted sum of S at target points. Check: on every hexahedron F on a 4x4x4 grid minus a second-difference curvature
bound (float Taylor bound), with recursive bisection where the bound fails; cutting planes added and re-solved.
    python tools/words_eight_coarse.py kind nu dsig_band [rounds]
"""
import sys
import time
import numpy as np
import scipy.sparse as sp
import clarabel
from words_eight_shape import UK, NI, EI, EZ

OUT = "D:/nature-math/math-gaps/words/out/eight/"
A8 = 2.5198560926
DIPSAFE = True
TS = 0.5

FACES = [  # (centre, e_a, e_b, ub range)
    (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), (0.0, 1.0)),
    (np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]), (0.0, 1.0)),
    (np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0]), (0.0, 1.0)),
    (np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), (0.0, 1.0)),
    (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), (-1.0, 1.0)),
]


ROT = {"E": 0.0, "I": -np.pi / 6}     # I: cube turned by -30 deg about w3, so +-n_I (the cone axis of w0_I) and the
                                      # 23-collision ray (in I) are face centres, i.e. mesh vertices


def faces_rot(angle):
    ca, sa = np.cos(angle), np.sin(angle)
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    return [(R @ c, R @ ea, R @ eb, r) for c, ea, eb, r in FACES]


def Nmap(f, ua, ub, faces=None):
    c, ea, eb, _ = (faces or FACES)[f]
    v = c[None, :] + np.tan(np.pi * ua / 4)[:, None] * ea[None, :] + np.tan(np.pi * ub / 4)[:, None] * eb[None, :]
    return v / np.linalg.norm(v, axis=1)[:, None]


def U_unit(n):
    # capped at 1e6 (a lower bound of U, so the check stays conservative); U is infinite on the collision rays
    return np.minimum(sum(np.maximum(1 - n @ u, 1e-300) ** -0.5 for u in UK), 1e6)


def gfun(kind, n):
    if kind == "E":
        return 1 - n[:, 0]
    m = n @ NI
    return 1 - np.sqrt(np.maximum(1 - m * m, 0))


def w0(kind, xi):
    r = np.linalg.norm(xi, axis=1)
    return r * gfun(kind, xi / r[:, None]) / 4


class Mesh:
    def __init__(self, nu, dsig_band, slo=-3.0, shi=11.0, band=(0.6, 5.0), rot=0.0):
        self.faces = faces_rot(rot)
        self.nu = nu                         # cells per unit of u (face edge u in [-1, 1])
        s = np.concatenate([np.arange(slo, band[0] - 0.3, 0.5), np.arange(band[0] - 0.3, band[0], 0.15),
                            np.arange(band[0], band[1], dsig_band), np.arange(band[1], 7.0, 0.25),
                            np.arange(7.0, shi + 1e-9, 0.5)])
        self.sig = np.unique(np.round(s, 6))
        self.slo, self.shi = self.sig[0], self.sig[-1]
        self.nl = len(self.sig)
        # angular nodes
        nodes = {}
        self.quads = []      # (face, ua0, ub0, du, [4 angular node ids])
        allN = []
        for f, (c, ea, eb, (b0, b1)) in enumerate(self.faces):
            na = 2 * nu
            nb = int(round((b1 - b0) * nu))
            ua = np.linspace(-1, 1, na + 1)
            ub = np.linspace(b0, b1, nb + 1)
            ids = np.zeros((na + 1, nb + 1), int)
            for i in range(na + 1):
                for j in range(nb + 1):
                    n = Nmap(f, np.array([ua[i]]), np.array([ub[j]]), self.faces)[0]
                    key = tuple(np.round(n, 9))
                    if key not in nodes:
                        nodes[key] = len(allN)
                        allN.append(n)
                    ids[i, j] = nodes[key]
            for i in range(na):
                for j in range(nb):
                    self.quads.append((f, ua[i], ub[j], 1.0 / nu, [ids[i, j], ids[i + 1, j], ids[i, j + 1], ids[i + 1, j + 1]]))
        self.angN = np.array(allN)
        self.na = len(allN)
        self.nq = len(self.quads)
        # unknowns: layer 0 -> one variable (index 0); layer k >= 1 node m -> 1 + (k - 1) * na + m
        self.nvar = 1 + (self.nl - 1) * self.na
        self.nhex = self.nq * (self.nl - 1)

    def vid(self, k, m):
        return np.where(k == 0, 0, 1 + (k - 1) * self.na + m)

    def locate_ang(self, n):
        """face, quad index, local (la, lb) for unit vectors n (northern hemisphere)"""
        n = n.copy()
        n[:, 2] = np.abs(n[:, 2])
        out_q = np.full(len(n), -1)
        out_a = np.zeros(len(n))
        out_b = np.zeros(len(n))
        nu = self.nu
        for f, (c, ea, eb, (b0, b1)) in enumerate(self.faces):
            dot = n @ c
            ok = dot > 1e-12
            a = np.where(ok, (n @ ea) / np.where(ok, dot, 1), 9)
            b = np.where(ok, (n @ eb) / np.where(ok, dot, 1), 9)
            ua = 4 / np.pi * np.arctan(a)
            ub = 4 / np.pi * np.arctan(b)
            sel = ok & (np.abs(ua) <= 1 + 1e-12) & (ub >= b0 - 1e-12) & (ub <= b1 + 1e-12) & (out_q < 0)
            if not sel.any():
                continue
            na = 2 * nu
            nb = int(round((b1 - b0) * nu))
            i = np.clip(np.floor((ua[sel] + 1) * nu).astype(int), 0, na - 1)
            j = np.clip(np.floor((ub[sel] - b0) * nu).astype(int), 0, nb - 1)
            base = sum(2 * nu * int(round((FACES[g][3][1] - FACES[g][3][0]) * nu)) for g in range(f))
            out_q[sel] = base + i * nb + j
            out_a[sel] = (ua[sel] + 1) * nu - i
            out_b[sel] = (ub[sel] - b0) * nu - j
        return out_q, out_a, out_b


def basis_local(ls, la, lb):
    """trilinear weights and derivatives in local coords, order (k, corner) with corner c in 0..3 = (i,j),(i+1,j),
    (i,j+1),(i+1,j+1); returns arrays (npts, 8): index 4*kk + c, kk = 0 (lower layer), 1 (upper layer)"""
    wa = [1 - la, la, 1 - la, la]
    wb = [1 - lb, 1 - lb, lb, lb]
    da = [-1, 1, -1, 1]
    db = [-1, -1, 1, 1]
    ws = [1 - ls, ls]
    ds = [-1, 1]
    W = np.zeros((len(ls), 8)); Ws = np.zeros_like(W); Wa = np.zeros_like(W); Wb = np.zeros_like(W)
    for kk in range(2):
        for cidx in range(4):
            col = 4 * kk + cidx
            W[:, col] = ws[kk] * wa[cidx] * wb[cidx]
            Ws[:, col] = ds[kk] * wa[cidx] * wb[cidx]
            Wa[:, col] = ws[kk] * da[cidx] * wb[cidx]
            Wb[:, col] = ws[kk] * wa[cidx] * db[cidx]
    return W, Ws, Wa, Wb


def geometry(mesh, kind, qidx, la, lb):
    """unit vector n, metric factors: rows R (2x2) with |D phi|^2 = |R (phi_a, phi_b)|^2 (phi_a = d phi/d la),
    and (g, Dg.Dphi coefficients on (phi_a, phi_b)), U(n)"""
    f = np.array([mesh.quads[q][0] for q in qidx])
    ua0 = np.array([mesh.quads[q][1] for q in qidx])
    ub0 = np.array([mesh.quads[q][2] for q in qidx])
    du = 1.0 / mesh.nu
    n = np.zeros((len(qidx), 3)); Ta = np.zeros_like(n); Tb = np.zeros_like(n)
    h = 1e-6
    for ff in np.unique(f):
        s = f == ff
        ua = ua0[s] + la[s] * du
        ub = ub0[s] + lb[s] * du
        fc = mesh.faces
        n[s] = Nmap(ff, ua, ub, fc)
        Ta[s] = (Nmap(ff, ua + h, ub, fc) - Nmap(ff, ua - h, ub, fc)) / (2 * h) * du      # d n / d la
        Tb[s] = (Nmap(ff, ua, ub + h, fc) - Nmap(ff, ua, ub - h, fc)) / (2 * h) * du
    gaa = np.sum(Ta * Ta, 1); gab = np.sum(Ta * Tb, 1); gbb = np.sum(Tb * Tb, 1)
    det = gaa * gbb - gab ** 2
    iaa, iab, ibb = gbb / det, -gab / det, gaa / det
    # Cholesky of the inverse metric: inv = R^T R, R upper triangular
    r11 = np.sqrt(iaa); r12 = iab / r11; r22 = np.sqrt(np.maximum(ibb - r12 ** 2, 0))
    # derivatives of g along la, lb
    gn = gfun(kind, n)
    def gat(v):
        return gfun(kind, v / np.linalg.norm(v, axis=1)[:, None])
    ga = (gat(n + 1e-6 * Ta) - gat(n - 1e-6 * Ta)) / 2e-6
    gb = (gat(n + 1e-6 * Tb) - gat(n - 1e-6 * Tb)) / 2e-6
    # Dg . Dphi = (ga, gb) inv (phi_a, phi_b)
    ca = ga * iaa + gb * iab
    cb = ga * iab + gb * ibb
    return n, (r11, r12, r22), gn, ca, cb, U_unit(n)


def rows_at(mesh, kind, hexes, ls, la, lb):
    """sparse rows (value, phi_s, |D phi| comps 1 and 2) and the linear part of F at local points of given hexes"""
    k = hexes // mesh.nq
    q = hexes % mesh.nq
    sig = mesh.sig[k] + ls * (mesh.sig[k + 1] - mesh.sig[k])
    dsig = mesh.sig[k + 1] - mesh.sig[k]
    W, Ws, Wa, Wb = basis_local(ls, la, lb)
    corners = np.array([mesh.quads[qq][4] for qq in q])               # (npts, 4)
    cols = np.concatenate([mesh.vid(k[:, None], corners), mesh.vid(k[:, None] + 1, corners)], axis=1)   # (npts, 8)
    n, (r11, r12, r22), g, ca, cb, U = geometry(mesh, kind, q, la, lb)
    Ws = Ws / dsig[:, None]
    D1 = r11[:, None] * Wa + r12[:, None] * Wb
    D2 = r22[:, None] * Wb
    Lin = -W / 3 + (4 / 3 - g)[:, None] * Ws - (ca[:, None] * Wa + cb[:, None] * Wb)
    return dict(cols=cols, W=W, Ws=Ws, D1=D1, D2=D2, Lin=Lin, sig=sig, U0=np.exp(-sig / 2) * U, n=n)


def far_rows(mesh, kind, qidx, la, lb):
    """far-field constraint rows at sigma_hi: A = e^(shi/2) phi_top"""
    W, _, Wa, Wb = basis_local(np.ones(len(qidx)), la, lb)
    W, Wa, Wb = W[:, 4:], Wa[:, 4:], Wb[:, 4:]
    corners = np.array([mesh.quads[qq][4] for qq in qidx])
    cols = mesh.vid(np.full((len(qidx), 1), mesh.nl - 1), corners)
    n, (r11, r12, r22), g, ca, cb, U = geometry(mesh, kind, qidx, la, lb)
    e = np.exp(mesh.shi / 2)
    Lin = -(1 - g / 2)[:, None] * W * e - (ca[:, None] * Wa + cb[:, None] * Wb) * e
    D1 = (r11[:, None] * Wa + r12[:, None] * Wb) * e
    D2 = r22[:, None] * Wb * e
    return dict(cols=cols, W=W * e / 2, Ws=np.zeros_like(W), D1=D1, D2=D2, Lin=Lin, U0=U,
                q=np.full(len(qidx), 2 * np.exp(-1.5 * mesh.shi)))


def socp(mesh, blocks, ot):
    """maximise ot . phi s.t. for each point: q (|c1|^2 + |c2|^2 + |c3|^2 [+ |c4|^2]) <= U0 + Lin phi, as the cone
    (l + 1, 2 sqrt(q) al c_j ..., l - 1) in SOC, l = al^2 (U0 + Lin phi), al^2 = 1/(1 + U0)"""
    nv = mesh.nvar
    rows, cols, vals, bs, cones = [], [], [], [], []
    off = 0
    for B in blocks:
        m = len(B["U0"])
        far = "sig" not in B
        q = B["q"] if far else 2 * np.exp(-B["sig"])
        al2 = 1 / (1 + B["U0"])
        sq = np.sqrt(q * al2)
        comps = [B["Ws"], B["D1"], B["D2"]] + ([B["W"]] if far else []) + B.get("X", [])
        dim = 2 + len(comps)
        C = B["cols"]
        nc = C.shape[1]
        coefs = [al2[:, None] * B["Lin"]] + [2 * sq[:, None] * X for X in comps] + [al2[:, None] * B["Lin"]]
        for j, X in enumerate(coefs):
            r = off + np.arange(m) * dim + j
            rows.append(np.repeat(r, nc)); cols.append(C.ravel()); vals.append(-X.ravel())
        bb = np.zeros(m * dim)
        bb[0::dim] = al2 * B["U0"] + 1
        bb[dim - 1::dim] = al2 * B["U0"] - 1
        bs.append(bb)
        cones += [clarabel.SecondOrderConeT(dim)] * m
        off += m * dim
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(off, nv))
    b = np.concatenate(bs)
    P = sp.csc_matrix((nv, nv))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 800
    settings.direct_solve_method = "faer"
    settings.max_threads = 4
    solver = clarabel.DefaultSolver(P, -ot, A, b, cones, settings)
    sol = solver.solve()
    return np.array(sol.x), str(sol.status)


def eval_phi(mesh, coef, xi):
    """phi at points xi (any hemisphere); returns phi"""
    r = np.linalg.norm(xi, axis=1)
    s = np.log(r)
    n = xi / r[:, None]
    q, la, lb = mesh.locate_ang(n)
    corners = np.array([mesh.quads[qq][4] for qq in q])
    sc = np.clip(s, mesh.slo, mesh.shi)
    k = np.clip(np.searchsorted(mesh.sig, sc, side="right") - 1, 0, mesh.nl - 2)
    ls = (sc - mesh.sig[k]) / (mesh.sig[k + 1] - mesh.sig[k])
    W = basis_local(ls, la, lb)[0]
    cols = np.concatenate([mesh.vid(k[:, None], corners), mesh.vid(k[:, None] + 1, corners)], axis=1)
    phi = np.sum(W * coef[cols], axis=1)
    far = s > mesh.shi
    phi[far] *= np.exp(-(s[far] - mesh.shi) / 2)
    return phi


def S_value(mesh, coef, kind, y, tau):
    """S(t, y) with tau = t (E) or 1 - t (I)"""
    xi = y / tau ** (4 / 3)
    return tau ** (1 / 3) * (w0(kind, xi) + eval_phi(mesh, coef, xi))


def F_hex_grid(mesh, kind, coef, hexes, ng=4, lo=0.0, hi=1.0):
    """F on an ng^3 grid in each hexahedron (sub-box [lo, hi]^3 in local coords per axis given as arrays)"""
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
    phi = np.sum(R["W"] * c, 1); ps = np.sum(R["Ws"] * c, 1)
    d1 = np.sum(R["D1"] * c, 1); d2 = np.sum(R["D2"] * c, 1)
    F = R["U0"] - 2 * np.exp(-R["sig"]) * (ps ** 2 + d1 ** 2 + d2 ** 2) + np.sum(R["Lin"] * c, 1)
    sh = (len(hexes), ng, ng, ng)
    Q = [(1 - R["n"] @ u).reshape(sh) for u in UK]
    return F.reshape(sh), R["U0"].reshape(sh), Q, R["sig"].reshape(sh)


def curv(G):
    """sum over axes of max |second difference| / 8 per box (grid of G: (m, ng, ng, ng))"""
    out = 0
    for ax in range(3):
        out = out + np.abs(np.diff(G, n=2, axis=ax + 1)).reshape(len(G), -1).max(1) / 8
    return out


def split_bound(F, U0, Q, SIG):
    """F = U0 + R: R bounded by grid min minus curvature; U0 >= e^(-sigma_max/2) sum_k (max_cell q_k)^(-1/2),
    q_k = 1 - u_k . n smooth (its max over the cell bounded by grid max plus curvature)"""
    m = len(F)
    R = F - U0
    Rlo = R.reshape(m, -1).min(1) - curv(R)
    smax = SIG.reshape(m, -1).max(1)
    Ulo = 0
    for q in Q:
        qmax = q.reshape(m, -1).max(1) + curv(q)
        Ulo = Ulo + np.maximum(qmax, 1e-300) ** -0.5
    Ulo = np.minimum(Ulo, 1e6) * np.exp(-smax / 2)
    return Ulo + Rlo


def cell_bound(F, ext):
    """min over the box >= min_grid F - sum_i M_i delta_i^2 / 8, M_i = max |second difference| / delta_i^2;
    ext = box extent per axis in local units (array (m, 3)); returns lower bound per hexahedron"""
    ng = F.shape[1]
    d = 1.0 / (ng - 1)
    m = F.reshape(len(F), -1).min(1)
    corr = 0
    for ax in range(3):
        sl = [slice(None)] * 4
        D2 = np.diff(F, n=2, axis=ax + 1)
        M = np.abs(D2).reshape(len(F), -1).max(1)      # = max|f''| delta^2 (grid units)
        corr = corr + M / 8
    return m - corr


def check_cells(mesh, kind, coef, depth=3):
    """float Taylor bound of min F over every hexahedron, with recursive bisection where it fails"""
    hexes = np.arange(mesh.nhex)
    lo = np.zeros((len(hexes), 3))
    hi = np.ones((len(hexes), 3))
    worst = np.full(mesh.nhex, np.inf)
    worst_rel = np.full(mesh.nhex, np.inf)
    fail_pts = []
    ncheck = 0
    for lev in range(depth + 1):
        if len(hexes) == 0:
            break
        out_b = []
        CH = 20000
        for i0 in range(0, len(hexes), CH):
            F, U0, Q, SIG = F_hex_grid(mesh, kind, coef, hexes[i0:i0 + CH], 4, lo[i0:i0 + CH], hi[i0:i0 + CH])
            bnd = np.maximum(cell_bound(F, None), split_bound(F, U0, Q, SIG))
            out_b.append((bnd, F.reshape(len(F), -1).min(1), U0.reshape(len(F), -1).min(1)))
        bnd = np.concatenate([o[0] for o in out_b])
        fmin = np.concatenate([o[1] for o in out_b])
        umin = np.concatenate([o[2] for o in out_b])
        ncheck += len(hexes)
        ok = bnd >= 0
        np.minimum.at(worst, hexes[ok], bnd[ok])
        np.minimum.at(worst_rel, hexes[ok], bnd[ok] / umin[ok])
        bad = ~ok
        if lev == depth or not bad.any():
            np.minimum.at(worst, hexes[bad], bnd[bad])
            np.minimum.at(worst_rel, hexes[bad], bnd[bad] / umin[bad])
            # record failing sub-boxes (centres) for cutting planes
            hb, bb_, lb_, ub_ = hexes[bad], bnd[bad], lo[bad], hi[bad]
            order = np.lexsort((bb_, hb))
            first = np.ones(len(order), bool)
            first[1:] = hb[order][1:] != hb[order][:-1]
            for i in order[first]:
                fail_pts.append((hb[i], (lb_[i] + ub_[i]) / 2, lb_[i], ub_[i], bb_[i]))
            break
        # bisect the failing boxes into 8
        hb, lb_, ub_ = hexes[bad], lo[bad], hi[bad]
        mid = (lb_ + ub_) / 2
        nh, nl_, nu_ = [], [], []
        for c in range(8):
            bits = [(c >> a) & 1 for a in range(3)]
            l2 = np.where(np.array(bits)[None, :] == 0, lb_, mid)
            u2 = np.where(np.array(bits)[None, :] == 0, mid, ub_)
            nh.append(hb); nl_.append(l2); nu_.append(u2)
        hexes = np.concatenate(nh); lo = np.vstack(nl_); hi = np.vstack(nu_)
    return worst, worst_rel, fail_pts, ncheck


def far_check(mesh, kind, coef, ng=9, depth=5):
    """lower bound per top quad of the far-field inequality, with recursive bisection of sub-quads where it fails"""
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
        n = geometry(mesh, kind, qq, la, lb)[0]
        Ulo = 0
        for u in UK:
            q = (1 - n @ u).reshape(m, ng, ng)
            Ulo = Ulo + np.maximum(q.reshape(m, -1).max(1) + c2(q), 1e-300) ** -0.5
        b1 = np.minimum(Ulo, 1e6) + rest.reshape(m, -1).min(1) - c2(rest)
        b2 = val.reshape(m, -1).min(1) - c2(val)
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


def sample_blocks(mesh, kind, extra=None, extra_far=None):
    """9 points per hexahedron (corners pulled in by 0.1, centre) + far-field points + extra (hex, local) points"""
    pts = np.array([[a, b, c] for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)])
    H = np.repeat(np.arange(mesh.nhex), len(pts))
    P = np.tile(pts, (mesh.nhex, 1))
    if extra is not None and len(extra[0]):
        H = np.concatenate([H, extra[0]]); P = np.vstack([P, extra[1]])
    R = rows_at(mesh, kind, H, P[:, 0], P[:, 1], P[:, 2])
    if DIPSAFE:
        # dip bound of the quadratic term between the 27 points (grid step 1/2): (q/16) sum_i sum_comp (d_i comp)^2
        X = []
        hh = 1e-4
        for ax in range(3):
            Pp = P.copy(); Pm = P.copy()
            Pp[:, ax] += hh; Pm[:, ax] -= hh
            Rp = rows_at(mesh, kind, H, Pp[:, 0], Pp[:, 1], Pp[:, 2])
            Rm = rows_at(mesh, kind, H, Pm[:, 0], Pm[:, 1], Pm[:, 2])
            for key in ("Ws", "D1", "D2"):
                X.append((Rp[key] - Rm[key]) / (2 * hh) / 4)
        R["X"] = X
    fp = np.array([[a, b] for a in (0.0, 0.25, 0.5, 0.75, 1.0) for b in (0.0, 0.25, 0.5, 0.75, 1.0)])
    qf = np.repeat(np.arange(mesh.nq), len(fp))
    blocks_axis = axis_blocks(mesh, kind) if kind == "I" else []
    la_f, lb_f = np.tile(fp[:, 0], mesh.nq), np.tile(fp[:, 1], mesh.nq)
    if extra_far is not None and len(extra_far[0]):
        qf = np.concatenate([qf, extra_far[0]]); la_f = np.concatenate([la_f, extra_far[1]]); lb_f = np.concatenate([lb_f, extra_far[2]])
    Rf = far_rows(mesh, kind, qf, la_f, lb_f)
    return [R, Rf] + blocks_axis


AXIS_EPS = 0.2


def axis_blocks(mesh, kind, eps=AXIS_EPS):
    """the cone axis +-n_I of w0_I: there g = 1 and D+g is the unit ball, so the subsolution test needs
    F_axis = U0 - 2e^(-s)(phi_s^2 + |D phi|^2) - phi/3 + (1/3) phi_s - |D phi| >= 0 (for every adjacent cell's gradient).
    |D phi| <= eps/2 + |D phi|^2/(2 eps) (AM-GM) keeps it a single rotated cone. Same for the far field at sigma_hi.
    Only meaningful when +-n_I are mesh vertices (rotated I mesh); returns [] otherwise."""
    axes = [NI, -NI]
    hits = []
    for qi, (fc, ua0, ub0, du, ids) in enumerate(mesh.quads):
        for cidx, nid in enumerate(ids):
            if min(np.linalg.norm(mesh.angN[nid] - a) for a in axes) < 1e-9:
                hits.append((qi, float(cidx & 1), float(cidx >> 1)))
    if not hits:
        return []
    blocks = []
    H, LS, LA, LB = [], [], [], []
    for k in range(mesh.nl - 1):
        for qi, la, lb in hits:
            for ls in (0.0, 0.5, 1.0):
                H.append(k * mesh.nq + qi); LS.append(ls); LA.append(la); LB.append(lb)
    H = np.array(H); LS = np.array(LS); LA = np.array(LA); LB = np.array(LB)
    R = rows_at(mesh, kind, H, LS, LA, LB)
    q = 2 * np.exp(-R["sig"])
    W, Ws = R["W"], R["Ws"]
    R["Lin"] = -W / 3 + Ws / 3                       # (4/3 - g) phi_s with g = 1, the Dg term replaced below
    fac = np.sqrt((q + 1 / (2 * eps)) / q)[:, None]  # D comps carry q + 1/(2 eps)
    R["D1"] = R["D1"] * fac
    R["D2"] = R["D2"] * fac
    R["U0"] = R["U0"] - eps / 2
    R.pop("X", None)
    blocks.append(R)
    # far field at the axis: U - A/2 - |DA| >= q_hi (A^2/4 + |DA|^2)
    qf = np.array([h[0] for h in hits]); laf = np.array([h[1] for h in hits]); lbf = np.array([h[2] for h in hits])
    Rf = far_rows(mesh, kind, qf, laf, lbf)
    e = np.exp(mesh.shi / 2)
    Rf["Lin"] = -0.5 * Rf["W"] * 2      # W was stored as e/2 * basis: A/2 = W . c
    ff = np.sqrt((Rf["q"] + 1 / (2 * eps)) / Rf["q"])[:, None]
    Rf["D1"] = Rf["D1"] * ff
    Rf["D2"] = Rf["D2"] * ff
    Rf["U0"] = Rf["U0"] - eps / 2
    blocks.append(Rf)
    return blocks


def targets(kind, c, extra_xi=None):
    """target points xi (slice t* = 1/2) and weights"""
    from words_eight_gap import fib_sphere
    sc = 2 * np.linalg.norm(c)
    U = fib_sphere(80)
    ys, ws = [], []
    for d, w in ((0.15, 1.0), (0.2, 1.0), (0.25, 1.0), (0.3, 0.5), (0.4, 0.3), (0.5, 0.2), (0.7, 0.1), (1.0, 0.05)):
        ys.append(c + sc * d * U); ws.append(np.full(len(U), w))
    # global slice points: sizes 0.3 .. 2 |c|, all directions
    nn = fib_sphere(400)
    for rr in (0.3, 0.55, 0.8, 1.25, 1.6, 2.0):
        ys.append(rr * np.linalg.norm(c) * nn); ws.append(np.full(len(nn), 0.03))
    Y = np.vstack(ys); Wt = np.concatenate(ws)
    tau = TS if kind == "E" else 1 - TS
    xi = Y / tau ** (4 / 3)
    wt = Wt * tau ** (1 / 3)
    if extra_xi is not None:
        xi = np.vstack([xi, extra_xi[0]]); wt = np.concatenate([wt, extra_xi[1]])
    return xi, wt


def build_objective(mesh, xi, wt):
    r = np.linalg.norm(xi, axis=1)
    s = np.log(r)
    n = xi / r[:, None]
    q, la, lb = mesh.locate_ang(n)
    corners = np.array([mesh.quads[qq][4] for qq in q])
    sc = np.clip(s, mesh.slo, mesh.shi)
    k = np.clip(np.searchsorted(mesh.sig, sc, side="right") - 1, 0, mesh.nl - 2)
    ls = (sc - mesh.sig[k]) / (mesh.sig[k + 1] - mesh.sig[k])
    W = basis_local(ls, la, lb)[0]
    fac = np.where(s > mesh.shi, np.exp(-(s - mesh.shi) / 2), 1.0)
    cols = np.concatenate([mesh.vid(k[:, None], corners), mesh.vid(k[:, None] + 1, corners)], axis=1)
    ot = np.zeros(mesh.nvar)
    np.add.at(ot, cols.ravel(), (W * (wt * fac)[:, None]).ravel())
    return ot


def main():
    kind = sys.argv[1]
    nu = int(sys.argv[2])
    dsb = float(sys.argv[3])
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    from words_eight_shape import eight_discrete
    c = eight_discrete(48)[0][24]
    mesh = Mesh(nu, dsb, rot=ROT[kind])
    print(f"{kind}: mesh nu = {nu} ({mesh.nq} angular quads, {mesh.na} angular nodes), {mesh.nl} sigma layers "
          f"[{mesh.slo}, {mesh.shi}], {mesh.nhex} hexahedra, {mesh.nvar} unknowns", flush=True)
    xi, wt = targets(kind, c)
    ext = None
    if len(sys.argv) > 5:
        dat = np.load(sys.argv[5])
        ext = (dat[f"xi_{kind}"], dat[f"w_{kind}"])
        xi = np.vstack([xi, ext[0]]); wt = np.concatenate([wt, ext[1]])
    ot = build_objective(mesh, xi, wt)
    extra = (np.zeros(0, int), np.zeros((0, 3)))
    extra_far = (np.zeros(0, int), np.zeros(0), np.zeros(0))
    tau = TS if kind == "E" else 1 - TS
    for rd in range(rounds + 1):
        t0 = time.time()
        blocks = sample_blocks(mesh, kind, extra, extra_far)
        coef, st = socp(mesh, blocks, ot)
        t1 = time.time()
        worst, wrel, fails, ncheck = check_cells(mesh, kind, coef)
        fworst = far_check(mesh, kind, coef)
        t2 = time.time()
        np.savez(OUT + f"coarse_{kind}_nu{nu}_ds{dsb}.npz", coef=coef, nu=nu, dsb=dsb, worst=worst, fworst=fworst,
                 rot=ROT[kind])
        Fg = F_hex_grid(mesh, kind, coef, np.arange(mesh.nhex), 4, np.zeros((mesh.nhex, 3)), np.ones((mesh.nhex, 3)))[0]
        gmin = Fg.reshape(mesh.nhex, -1).min(1)
        kk = np.arange(mesh.nhex) // mesh.nq
        badk = kk[worst < 0]
        print(f"      grid (4^3) violations in {np.sum(gmin < 0)} hexahedra, worst grid value {gmin.min():+.3e}; failing "
              f"hexahedra by sigma layer: {np.bincount(badk, minlength=mesh.nl - 1).tolist()}", flush=True)
        Sc = S_value(mesh, coef, kind, c[None], tau)[0]
        Vc = 1.22789891 if kind == "E" else 1.29195718
        print(f" round {rd}: {st}, solve {t1 - t0:.0f} s, check {t2 - t1:.0f} s ({ncheck} boxes); worst cell bound "
              f"{worst.min():+.3e} (relative to U {wrel.min():+.3e}); failing hexahedra {np.sum(worst < 0)}; far-field "
              f"worst {fworst.min():+.3e}; S(c) = {Sc:.5f} (V = {Vc}, deficit {Vc - Sc:.4f} = {100 * (Vc - Sc) / Vc:.2f} %)",
              flush=True)
        if np.sum(worst < 0) == 0 and fworst.min() >= 0:
            break
        # cutting planes: 27 points in every failing sub-box
        g = np.array([[a, b, cc] for a in (0, 0.5, 1) for b in (0, 0.5, 1) for cc in (0, 0.5, 1)])
        eh, ep = [], []
        fails = sorted(fails, key=lambda f: f[4])[:4000]
        for h, ctr, l, u, _ in fails:
            eh.append(np.full(len(g), h)); ep.append(l + g * (u - l))
        if eh:
            extra = (np.concatenate([extra[0]] + eh), np.vstack([extra[1]] + ep))
        fq = np.flatnonzero(fworst < 0)
        if len(fq):
            g2 = np.linspace(0, 1, 5)
            A_, B_ = np.meshgrid(g2, g2, indexing="ij")
            extra_far = (np.concatenate([extra_far[0], np.repeat(fq, 25)]), np.concatenate([extra_far[1], np.tile(A_.ravel(), len(fq))]),
                         np.concatenate([extra_far[2], np.tile(B_.ravel(), len(fq))]))
    np.savez(OUT + f"coarse_{kind}_nu{nu}_ds{dsb}.npz", coef=coef, nu=nu, dsb=dsb, worst=worst, fworst=fworst,
             rot=ROT[kind])
    # slice values
    from words_eight_gap import fib_sphere
    sc = 2 * np.linalg.norm(c)
    for d in (0.1, 0.2, 0.3, 0.4, 0.5):
        y = c + sc * d * fib_sphere(200)
        print(f"   slice sphere d = {d}: S min {S_value(mesh, coef, kind, y, tau).min():.4f}", flush=True)


if __name__ == "__main__" and sys.argv[1] not in ("final", "json"):
    main()


def finalize(kind, nu, dsb, depth=3, mode="bound"):
    """make the fit admissible on every cell: phi -> lam phi with lam = min over cells U0_lo / (U0_lo - bound) for the
    cells (and far-field quads) with a negative bound (concavity: F[lam phi] >= lam F[phi] + (1 - lam) U0); recheck"""
    d = np.load(OUT + f"coarse_{kind}_nu{nu}_ds{dsb}.npz")
    rot = float(d["rot"]) if "rot" in d else 0.0       # files written before the rotation was introduced: 0
    mesh = Mesh(nu, dsb, rot=rot)
    coef = d["coef"]
    t0 = time.time()
    worst, wrel, fails, ncheck = check_cells(mesh, kind, coef, depth)
    fw = far_check(mesh, kind, coef)
    # U lower bounds per hexahedron (coarse: grid min x 0.5) and per far quad (U >= 3 on the sphere)
    F, U0, Q, SIG = F_hex_grid(mesh, kind, coef, np.arange(mesh.nhex), 4, np.zeros((mesh.nhex, 3)), np.ones((mesh.nhex, 3)))
    Ulo = split_bound(np.zeros_like(F) + U0, U0, Q, SIG)       # bound of U0 alone (R = 0)
    lam = 1.0
    if mode == "grid":      # use the grid minima (x 1.5) instead of the curvature bound (non-smooth w0_I cone)
        worst = np.minimum(1.5 * F.reshape(mesh.nhex, -1).min(1), 0) + 0 * worst
        worst[worst == 0] = 1.0
    neg = worst < 0
    if neg.any():
        lam = min(lam, np.min(Ulo[neg] / (Ulo[neg] - worst[neg])))
    if (fw < 0).any():
        lam = min(lam, np.min(3.0 / (3.0 - fw[fw < 0])))
    lam *= 0.9999
    coef2 = coef * lam
    worst2, wrel2, _, ncheck2 = check_cells(mesh, kind, coef2, depth)
    fw2 = far_check(mesh, kind, coef2)
    tau = TS if kind == "E" else 1 - TS
    from words_eight_shape import eight_discrete
    c = eight_discrete(48)[0][24]
    Sc = S_value(mesh, coef2, kind, c[None], tau)[0]
    Vc = 1.22789891 if kind == "E" else 1.29195718
    print(f"{kind} nu {nu}: before: worst cell bound {worst.min():+.3e}, failing {np.sum(worst < 0)}, far {fw.min():+.3e}; "
          f"lam = {lam:.6f}; after: worst cell bound {worst2.min():+.3e} (failing {np.sum(worst2 < 0)}), far worst "
          f"{fw2.min():+.3e}; boxes checked {ncheck2}; S(c) = {Sc:.5f}, deficit {Vc - Sc:.4f} ({100 * (Vc - Sc) / Vc:.2f} %); "
          f"{time.time() - t0:.0f} s", flush=True)
    np.savez(OUT + f"coarse_{kind}_nu{nu}_ds{dsb}_final.npz", coef=coef2, lam=lam, worst=worst2, fworst=fw2, nu=nu, dsb=dsb,
             rot=rot)
    return coef2


if __name__ == "__main__" and sys.argv[1] == "final":
    finalize(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), mode=sys.argv[5] if len(sys.argv) > 5 else "bound")


def export_json(kind, nu, dsb, path=None, final=True):
    """write the coarse certificate in the verifier's format (cells in (sigma, ua, ub), trilinear -> monomials in local
    y in [-1, 1]^3: phi = sum c_ijk y1^i y2^j y3^k, y1 for sigma, y2 for ua, y3 for ub)"""
    import json
    import os
    f = OUT + f"coarse_{kind}_nu{nu}_ds{dsb}_final.npz"
    if not (final and os.path.exists(f)):
        f = OUT + f"coarse_{kind}_nu{nu}_ds{dsb}.npz"
    dd = np.load(f)
    coef = dd["coef"]
    mesh = Mesh(nu, dsb, rot=float(dd["rot"]) if "rot" in dd else 0.0)
    cells = []
    sgn = np.array([-1.0, 1.0])
    for k in range(mesh.nl - 1):
        for qi, (fc, ua0, ub0, du, ids) in enumerate(mesh.quads):
            v = np.zeros((2, 2, 2))      # v[a (sigma), b (ua), c (ub)]
            for a in range(2):
                for cidx in range(4):
                    v[a, cidx & 1, cidx >> 1] = coef[int(mesh.vid(np.array(k + a), np.array(ids[cidx])))]
            C = np.zeros((2, 2, 2))
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        C[i, j, l] = sum(v[a, b, c] * sgn[a] ** i * sgn[b] ** j * sgn[c] ** l
                                         for a in range(2) for b in range(2) for c in range(2)) / 8
            cells.append({"face": int(fc), "sigma": [float(mesh.sig[k]), float(mesh.sig[k + 1])],
                          "ua": [float(ua0), float(ua0 + du)], "ub": [float(ub0), float(ub0 + du)],
                          "deg": [1, 1, 1], "coef": C.tolist()})
    out = {"side": kind,
           "faces": [{"centre": c.tolist(), "ea": ea.tolist(), "eb": eb.tolist(), "ub_range": list(r)} for c, ea, eb, r in mesh.faces],
           "cells": cells, "far": {"sigma_hi": float(mesh.shi)},
           "low": {"sigma_lo": float(mesh.slo), "phi_lo": float(coef[0])},
           "source": os.path.basename(f)}
    path = path or OUT + f"cert_pieces_{kind}.json"
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {path}: {len(cells)} cells from {os.path.basename(f)}")


if __name__ == "__main__" and sys.argv[1] == "json":
    export_json(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
