"""Stage 21, stage 4: rigorous evaluation of the tube-boundary condition (ii) and the end condition (ii') of the
I-field-only exit argument (paper/review/eight_field_rigour_theorist.md 3.4, eight_chain_review_theorist.md A.5).

    (ii)  S+(t, x) + W(t, x) > A8  for t in [0, 1), x on the boundary of T(t)  (S+ = max(S_E^c, 0));
    (ii') S_E^c(1, x) > A8          for x in I outside the open discs T(1) and T'(1) (mirror).

S_E^c is the coarse E subsolution of a verifier-format JSON (verify/eight_cell_verify.py; side "E"):
    S_E^c(t, x) = t^(1/3) w(x / t^(4/3)),  w = w0_E + phi,  w0_E(xi) = (|xi| - xi_1)/4 = |xi| g_E(n)/4,
so  S_E^c(t, x) = (|x| - x_1)/(4t) + t^(1/3) phi(xi),  xi = e^sigma n,  sigma = log|x| - (4/3) log t
(the first term is dist_g(x, E)^2/(2t)). phi: on each cell a polynomial in local y in [-1, 1]^3 (sigma, ua, ub);
gnomonic charts n = normalize(C + tan(pi ua/4) A + tan(pi ub/4) B) on the northern hemisphere, phi even in xi_3;
above sigma_hi: phi = e^(-(sigma - sigma_hi)/2) phi(sigma_hi, n); below sigma_lo: phi = phi_lo.
Derivatives (Clarke gradients on kinks: the hull over every cell / piece meeting the closed box):
    grad_x S = (n - e_1)/(4t) + (t^(1/3)/|x|) (phi_sigma n + D phi),
    S_t      = -(|x| - x_1)/(4 t^2) + t^(-2/3) (phi/3 - (4/3) phi_sigma),
D phi = r (phi_X A + phi_Y B - P C) the sphere gradient in a chart (closed form as in the verifier). The mirror
(x_3 < 0) is handled by folding: the x_3 component of the phi part is negated, and on boxes meeting x_3 = 0 the hull
of both signs is taken (Clarke gradient on the equator kink).

Outward rounding everywhere (class I of verify/prove_newton_1212.py); log, atan, cbrt, exp, tan at interval
endpoints from numpy pushed by 3 ulps (as the verifier: libm assumed accurate to 1 ulp).

---------------------------------------------------------------------------------------------------------------
RECORD FORMAT (input of check_records; one .npz, n boxes, all float64 unless stated; "over the box" means over the
whole 4-D box {s in [s_lo, s_hi]} x {a in [a_lo, a_hi]}, a = (ell, phi, M) the scaled labels of
eight_field_rigour_theorist.md 2.4: ell = log r, phi the angle on I, M = s mu; x = f_s(a) = r Phi(s^2 r^(-3/2), phi, M);
W and p are functions of LABELS (condition (ii-L), eight_proof_plan.md); norms are Euclidean):
  required (C0 data):
    s_lo, s_hi        (n,)      time to go s = 1 - t; 0 <= s_lo <= s_hi <= 1 (checked range: s in [s0, 1])
    a_lo, a_hi        (n, 3)    label box (ell, phi, M); it must contain the boundary piece {a in bd Lambda(s)}
    s_c               (n,)      centre time (a float in [s_lo, s_hi]; used exactly)
    a_c               (n, 3)    centre label (floats in the box; used exactly)
    x_lo, x_hi        (n, 3)    enclosure of x = f_s(a) over the box
    p_lo, p_hi        (n, 3)    enclosure of p = Pi/s over the box (grad_x W = -p); alternatively gW_lo, gW_hi = -p
    W_lo, W_hi        (n,)      enclosure of W(s, a) = (r/s)(Khat + eps Uhat) over the box (W_hi may be +inf)
    xc_lo, xc_hi      (n, 3)    enclosure of x at the centre (a_c, s_c)
    Wc_lo, Wc_hi      (n,)      enclosure of W at the centre
  derivative bounds, at least one of:
    Lcol (n, 3) and Ls (n,)     upper bounds over the box of |dx/da_j| (j = ell, phi, M) and of |dx/ds| (fixed a)
    Lop (n,) and Ls (n,)        upper bound of the operator norm |D_a f_s| (Euclidean in a) and of |dx/ds|
    dxda_lo, dxda_hi (n, 3, 3) and dxds_lo, dxds_hi (n, 3)   full enclosures ([:, i, j] = dx_i/da_j), if available
  optional: face (n,) int (0..5: ell-, ell+, phi-, phi+, M-, M+), id (n,) int; scalar A8_hi (upper end of the
  Krawczyk enclosure of A8; default A8_HI below).
W_t is not part of the record: it is taken from the HJ equation, W_t = 2|x| |p|^2 - U(x) (W the backward value).
---------------------------------------------------------------------------------------------------------------
Lower bound per box: the maximum of four valid lower bounds of S+ + W - A8:
  (w)  W_lo - A8                                        (S+ >= 0)
  (k)  (|x| - x_1)/(4 t_hi) + t_hi^(1/3) min(phi_glob_lo, 0) + W_lo - A8   (the kinetic term; t near 0, A.5 (d))
  (d)  [S over the (t, x) box] + W_lo - A8               (direct)
  (c)  centred form in (a, s), f = S_E^c(1 - s, f_s(a)) + W(s, a): with G = grad_x S - p over the x box,
       df/da_j = G . dx/da_j,  df/ds = -(S_t + W_t) + G . dx/ds  (Lebourg mean value theorem with Clarke gradients;
       S is Lipschitz on t > 0), either
       (c1) f(c) + sum_j [G . dxda_j](a_j - a_c,j) + [df/ds](s - s_c) with full interval enclosures, or
       (c0) f(c) - |G|max sum_j Lcol_j |a_j - a_c,j|max - (|S_t + W_t|max + |G|max Ls) |s - s_c|max
            (or |G|max Lop |a - a_c|max,2 in place of the sum); the best available is taken.
  S+ = max(S, 0) needs no Clarke calculus of the max: S+ + W >= max(S + W, W) and (w) bounds the second.
    python verify/eight_tube_boundary.py records records.npz cert.json [A8_hi]
    python verify/eight_tube_boundary.py end cert.json ell_c phi_c h_ell h_phi [depth]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prove_newton_1212 import I, dn, up  # noqa: E402
from eight_cell_verify import PI4, S3H, dpoly, exact_frames, geometry, push, sphere_grad  # noqa: E402

A8_HI = 2.51985609229          # above the float A8 = 2.519856092284; replace by the Krawczyk upper end
INF = np.inf
THIRD = I(dn(1.0 / 3.0), up(1.0 / 3.0))
FOUR3 = I(dn(4.0 / 3.0), up(4.0 / 3.0))
ISQ3 = float(dn(dn(1.0 / math.sqrt(3.0))))      # lower bound of 1/sqrt(3)
UKI = [(I(-1.0), I(0.0), I(0.0)), (I(0.5), S3H, I(0.0)), (I(0.5), -S3H, I(0.0))]
EI_I = (I(0.5), S3H, I(0.0))
EZ_I = (I(0.0), I(0.0), I(1.0))


# ------------------------------------------------------------------ interval helpers
def ilog(x):
    with np.errstate(divide="ignore"):
        return I(push(np.log(x.lo), 3, -INF), push(np.log(x.hi), 3, INF))


def iatan(x):
    return I(push(np.arctan(x.lo), 3, -INF), push(np.arctan(x.hi), 3, INF))


def icbrt(x):
    return I(np.maximum(push(np.cbrt(x.lo), 3, -INF), 0.0), push(np.cbrt(x.hi), 3, INF))


def iexp(x):
    return I(np.maximum(push(np.exp(x.lo), 3, -INF), 0.0), push(np.exp(x.hi), 3, INF))


def isincos(th):
    """sin and cos of an interval inside [0, pi] (cos monotone; sin with its maximum at pi/2 inside)"""
    lo, hi = th.lo, th.hi
    c = I(push(np.cos(hi), 3, -INF), push(np.cos(lo), 3, INF))
    sl, sh = np.sin(lo), np.sin(hi)
    smin = push(np.minimum(sl, sh), 3, -INF)
    smax = np.where((lo <= math.pi / 2 + 1e-12) & (hi >= math.pi / 2 - 1e-12), 1.0, push(np.maximum(sl, sh), 3, INF))
    return I(np.maximum(smin, -1.0), np.minimum(smax, 1.0)), I(np.maximum(c.lo, -1.0), np.minimum(c.hi, 1.0))


def inorm(X):
    """|x| of an interval vector (sum of squares clipped at 0 before the root)"""
    q = X[0].sq() + X[1].sq() + X[2].sq()
    return I(np.maximum(q.lo, 0.0), q.hi).sqrt()


def ihull_all(a, b):
    return I(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def isect(a, b):
    """intersection of two enclosures of the same quantity"""
    return I(np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi))


def take(x, idx):
    return I(x.lo[idx], x.hi[idx])


def nan_low(v):
    v = np.asarray(v, float)
    return np.where(np.isnan(v), -INF, v)


def poly_rows(c, y):
    """sum c[r, i, j, k] y1^i y2^j y3^k per row r, interval Horner (c float (n, p1+1, p2+1, p3+1))"""
    p1, p2, p3 = (s - 1 for s in c.shape[1:])
    out = None
    for i in range(p1, -1, -1):
        cj = None
        for j in range(p2, -1, -1):
            ck = I(c[:, i, j, p3])
            for k in range(p3 - 1, -1, -1):
                ck = ck * y[2] + c[:, i, j, k]
            cj = ck if cj is None else cj * y[1] + ck
        out = cj if out is None else out * y[0] + cj
    return out


def reduce_hull(n, idx, v):
    lo = np.full(n, INF)
    hi = np.full(n, -INF)
    np.minimum.at(lo, idx, v.lo)
    np.maximum.at(hi, idx, v.hi)
    return lo, hi


# ------------------------------------------------------------------ the coarse E piece
class CoarseE:
    def __init__(self, path):
        data = json.load(open(path, encoding="utf-8"))
        assert data["side"] == "E", "the tube boundary needs the E piece"
        self.path = path
        self.faces = exact_frames(data["faces"])
        cells = data["cells"]
        self.slo, self.philo = float(data["low"]["sigma_lo"]), float(data["low"]["phi_lo"])
        self.shi = float(data["far"]["sigma_hi"])
        deg = np.max([c["deg"] for c in cells], axis=0)
        sh = tuple(int(d) + 1 for d in deg)
        nc = len(cells)
        self.C = np.zeros((nc,) + sh)
        self.D = [np.zeros((nc,) + sh) for _ in range(3)]
        for idx, cl in enumerate(cells):
            c = np.array(cl["coef"], float)
            self.C[idx][: c.shape[0], : c.shape[1], : c.shape[2]] = c
            for ax in range(3):
                d = dpoly(c, ax)
                self.D[ax][idx][: d.shape[0], : d.shape[1], : d.shape[2]] = d
        self.cells = cells
        # per face: tensor grid of cells
        self.grid = []
        for f, face in enumerate(self.faces):
            ids = [i for i, c in enumerate(cells) if c["face"] == f]
            es = np.unique([v for i in ids for v in cells[i]["sigma"]])
            ea = np.unique([v for i in ids for v in cells[i]["ua"]])
            eb = np.unique([v for i in ids for v in cells[i]["ub"]])
            tab = -np.ones((len(es) - 1, len(ea) - 1, len(eb) - 1), int)
            for i in ids:
                c = cells[i]
                a, b, d = (np.searchsorted(es, c["sigma"][0]), np.searchsorted(ea, c["ua"][0]), np.searchsorted(eb, c["ub"][0]))
                assert es[a + 1] == c["sigma"][1] and ea[b + 1] == c["ua"][1] and eb[d + 1] == c["ub"][1], "not a tensor grid"
                tab[a, b, d] = i
            assert np.all(tab >= 0), f"face {f}: cells do not form a full tensor grid"
            assert es[0] == self.slo and es[-1] == self.shi, "sigma range of the cells != [sigma_lo, sigma_hi]"
            rng = face.get("ub_range", [0.0, 1.0] if f < 4 else [-1.0, 1.0])
            assert ea[0] == -1.0 and ea[-1] == 1.0 and eb[0] == rng[0] and eb[-1] == rng[1]
            self.grid.append((es, ea, eb, tab))
        # global bounds of phi (cells over [-1, 1]^3; the far field lies between 0 and the top values; phi_lo)
        absr = I(np.zeros(nc))
        flat = self.C.reshape(nc, -1)
        for k in range(1, flat.shape[1]):
            absr = absr + I(-np.abs(flat[:, k]), np.abs(flat[:, k]))
        rng = absr + flat[:, 0]
        self.phi_glob_lo = float(dn(min(np.min(rng.lo), self.philo, 0.0)))
        self.phi_glob_hi = float(up(max(np.max(rng.hi), self.philo, 0.0)))

    # -------------------------------------------------------------- pairs (box, piece)
    def _pairs(self, X, SIG, R):
        """enumerate (box, cell) pairs; X folded (x3 >= 0). Returns a list of dicts per face plus low/far info"""
        n = len(R.lo)
        out = []
        for f, face in enumerate(self.faces):
            es, ea, eb, tab = self.grid[f]
            Cf, Af, Bf = face["_C"], face["_A"], face["_B"]
            dC = X[0] * Cf[0] + X[1] * Cf[1] + X[2] * Cf[2]
            dA = X[0] * Af[0] + X[1] * Af[1] + X[2] * Af[2]
            dB = X[0] * Bf[0] + X[1] * Bf[1] + X[2] * Bf[2]
            clo = np.maximum(dC.lo, dn(R.lo * ISQ3))
            met = (dC.hi >= clo) & (clo > 0)
            if not np.any(met):
                continue
            dCe = I(np.where(met, clo, 1.0), np.where(met, dC.hi, 1.0))
            ra, rb = dA / dCe, dB / dCe
            ra = I(np.maximum(ra.lo, -1.0), np.minimum(ra.hi, 1.0))
            rb = I(np.maximum(rb.lo, -1.0), np.minimum(rb.hi, 1.0))
            met &= (ra.lo <= ra.hi) & (rb.lo <= rb.hi)
            if not np.any(met):
                continue
            UA = iatan(ra) / PI4
            UB = iatan(rb) / PI4
            ua_lo, ua_hi = np.maximum(UA.lo, ea[0]), np.minimum(UA.hi, ea[-1])
            ub_lo, ub_hi = np.maximum(UB.lo, eb[0]), np.minimum(UB.hi, eb[-1])
            met &= (ua_lo <= ua_hi) & (ub_lo <= ub_hi)
            s_lo, s_hi = np.maximum(SIG.lo, self.slo), np.minimum(SIG.hi, self.shi)
            reg = met & (s_lo <= s_hi)
            far = met & (SIG.hi >= self.shi)
            nsg, nag, nbg = len(es) - 1, len(ea) - 1, len(eb) - 1

            def rng_idx(e, lo, hi, m):
                i0 = np.clip(np.searchsorted(e, lo, "left") - 1, 0, m - 1)
                i1 = np.clip(np.searchsorted(e, hi, "right") - 1, 0, m - 1)
                return i0, np.maximum(i1, i0)
            a0, a1 = rng_idx(ea, ua_lo, ua_hi, nag)
            b0, b1 = rng_idx(eb, ub_lo, ub_hi, nbg)
            for kind, sel in (("cell", reg), ("far", far)):
                bi = np.flatnonzero(sel)
                if not len(bi):
                    continue
                if kind == "cell":
                    i0, i1 = rng_idx(es, s_lo[bi], s_hi[bi], nsg)
                else:
                    i0 = i1 = np.full(len(bi), nsg - 1)
                ni, na_, nb_ = i1 - i0 + 1, a1[bi] - a0[bi] + 1, b1[bi] - b0[bi] + 1
                cnt = ni * na_ * nb_
                rep = np.repeat(np.arange(len(bi)), cnt)
                off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                kk = off % nb_[rep]
                jj = (off // nb_[rep]) % na_[rep]
                ii = off // (nb_[rep] * na_[rep])
                si, ai, bj = i0[rep] + ii, a0[bi][rep] + jj, b0[bi][rep] + kk
                box = bi[rep]
                cid = tab[si, ai, bj]
                d = {"face": f, "kind": kind, "box": box, "cell": cid,
                     "ua": (np.maximum(ua_lo[box], ea[ai]), np.minimum(ua_hi[box], ea[ai + 1])),
                     "ub": (np.maximum(ub_lo[box], eb[bj]), np.minimum(ub_hi[box], eb[bj + 1])),
                     "cua": (ea[ai], ea[ai + 1]), "cub": (eb[bj], eb[bj + 1])}
                if kind == "cell":
                    d["sig"] = (np.maximum(s_lo[box], es[si]), np.minimum(s_hi[box], es[si + 1]))
                    d["csig"] = (es[si], es[si + 1])
                else:
                    d["sig"] = (np.maximum(SIG.lo[box], self.shi), SIG.hi[box])
                out.append(d)
        return out

    def enclose(self, T, X, need_grad=True):
        """enclosures over boxes (T interval array (n,) with T.lo > 0; X list of 3 interval arrays) of
        S = S_E^c(t, x), grad_x S (list of 3), S_t. Returns (S, gS, St) (gS, St None if need_grad is False)."""
        n = len(T.lo)
        assert np.all(T.lo > 0)
        X3f = I(np.where(X[2].lo >= 0, X[2].lo, np.where(X[2].hi <= 0, -X[2].hi, 0.0)),
                np.where(X[2].lo >= 0, X[2].hi, np.where(X[2].hi <= 0, -X[2].lo, np.maximum(-X[2].lo, X[2].hi))))
        Xf = [X[0], X[1], X3f]
        R = inorm(X)
        assert np.all(R.lo > 0)
        lT = ilog(T)
        SIG = ilog(R) - FOUR3 * lT
        t13 = icbrt(T)
        tm23 = (t13.sq()).recip()
        # phi part: hull over pieces
        P_lo, P_hi = np.full(n, INF), np.full(n, -INF)          # t^(1/3) phi
        St_lo, St_hi = np.full(n, INF), np.full(n, -INF)        # t^(-2/3) (phi/3 - 4/3 phi_s)
        G_lo, G_hi = np.full((3, n), INF), np.full((3, n), -INF)  # (t^(1/3)/|x|)(phi_s n + D phi), folded

        def acc(box, phi, ps, Dv, nv):
            v = take(t13, box) * phi
            np.minimum.at(P_lo, box, v.lo)
            np.maximum.at(P_hi, box, v.hi)
            if not need_grad:
                return
            v = take(tm23, box) * (phi * THIRD - FOUR3 * ps)
            np.minimum.at(St_lo, box, v.lo)
            np.maximum.at(St_hi, box, v.hi)
            k = take(t13, box) / take(R, box)
            for i in range(3):
                g = ps * nv[i] + Dv[i] if Dv is not None else ps * 0.0
                g = k * g
                np.minimum.at(G_lo[i], box, g.lo)
                np.maximum.at(G_hi[i], box, g.hi)

        low = np.flatnonzero(SIG.lo <= self.slo)
        if len(low):
            ph = I(np.full(len(low), self.philo))
            acc(low, ph, I(np.zeros(len(low))), None, None)
        for d in self._pairs(Xf, SIG, R):
            face, box, cid = self.faces[d["face"]], d["box"], d["cell"]
            UA, UB = I(*d["ua"]), I(*d["ub"])
            ca0, ca1 = d["cua"]
            cb0, cb1 = d["cub"]
            fa = I(2.0) / (I(ca1) - ca0)
            fb = I(2.0) / (I(cb1) - cb0)
            y2 = (UA - ca0) * fa - 1.0
            y3 = (UB - cb0) * fb - 1.0
            if d["kind"] == "cell":
                cs0, cs1 = d["csig"]
                fs = I(2.0) / (I(cs1) - cs0)
                y1 = (I(*d["sig"]) - cs0) * fs - 1.0
            else:
                y1 = I(np.ones(len(box)))
            yy = [y1, y2, y3]
            phi = poly_rows(self.C[cid], yy)
            ps, nvec, Dv = None, None, None
            if d["kind"] == "cell":
                ps = poly_rows(self.D[0][cid], yy) * fs if need_grad else None
            if need_grad:
                pa = poly_rows(self.D[1][cid], yy) * fa
                pb = poly_rows(self.D[2][cid], yy) * fb
                nvec, geo, _ = geometry(face, UA, UB)
                fx, fy, Pp, _ = sphere_grad(geo, pa, pb)
                Af, Bf, Cf = face["_A"], face["_B"], face["_C"]
                Dv = [geo["r"] * (fx * Af[i] + fy * Bf[i] - Pp * Cf[i]) for i in range(3)]
            if d["kind"] == "far":
                ef = iexp((I(*d["sig"]) - self.shi) * (-0.5))
                ef = I(ef.lo, np.minimum(ef.hi, 1.0))
                phi = phi * ef
                if need_grad:
                    ps = phi * (-0.5)
                    Dv = [v * ef for v in Dv]
            acc(box, phi, ps, Dv, nvec)
        assert np.all(P_lo <= P_hi), "a box met no piece"
        # w0 part
        q = X[1].sq() + X[2].sq()
        d0 = R - X[0]
        d0 = I(np.maximum(d0.lo, 0.0), d0.hi)
        pos = X[0].lo > 0
        if np.any(pos):
            alt = q / (R + I(np.where(pos, X[0].lo, 1.0), np.where(pos, X[0].hi, 1.0)))
            d0 = I(np.where(pos, np.maximum(d0.lo, alt.lo), d0.lo), np.where(pos, np.minimum(d0.hi, alt.hi), d0.hi))
        T4 = T * 4.0
        S = d0 / T4 + I(P_lo, P_hi)
        if not need_grad:
            return S, None, None
        St = -(d0 / (T4 * T)) + I(St_lo, St_hi)
        gS = []
        for i in range(3):
            gl, gh = G_lo[i], G_hi[i]
            if i == 2:      # unfold the mirror
                neg = X[2].hi <= 0
                mix = (X[2].lo < 0) & (X[2].hi > 0)
                gl, gh = np.where(neg, -G_hi[i], np.where(mix, np.minimum(gl, -gh), gl)), \
                    np.where(neg, -G_lo[i], np.where(mix, np.maximum(gh, -G_lo[i]), gh))
            nvx = X[i] / R
            if i == 0:
                e = nvx - 1.0
                if np.any(pos):
                    alt = -(q / (R * (R + I(np.where(pos, X[0].lo, 1.0), np.where(pos, X[0].hi, 1.0)))))
                    e = I(np.where(pos, np.maximum(e.lo, alt.lo), e.lo), np.where(pos, np.minimum(e.hi, alt.hi), e.hi))
                nvx = e
            gS.append(nvx / T4 + I(gl, gh))
        return S, gS, St

    def crude_lower(self, T, X):
        """valid lower bound of S on boxes with t in [T.lo, T.hi], T.lo >= 0, T.hi > 0: the kinetic term with t_hi
        plus t^(1/3) min(phi): (|x| - x_1)/(4 t_hi) + t_hi^(1/3) min(phi_glob_lo, 0)"""
        R = inorm(X)
        q = X[1].sq() + X[2].sq()
        d0 = np.maximum((R - X[0]).lo, 0.0)
        pos = X[0].lo > 0
        alt = (q / (R + I(np.where(pos, X[0].lo, 1.0), np.where(pos, X[0].hi, 1.0)))).lo
        d0 = np.where(pos, np.maximum(d0, alt), d0)
        kin = dn(d0 / up(4.0 * T.hi))
        th = push(np.cbrt(T.hi), 3, INF)
        return dn(kin + dn(th * min(self.phi_glob_lo, 0.0)))


def U_enclose(X):
    R = inorm(X)
    tot = None
    for u in UKI:
        q = R - (X[0] * u[0] + X[1] * u[1] + X[2] * u[2])
        q = I(np.maximum(q.lo, 0.0), q.hi)
        with np.errstate(divide="ignore"):
            v = I(dn(1.0 / up(np.sqrt(up(q.hi)))), up(1.0 / dn(np.sqrt(np.maximum(dn(q.lo), 0.0)))))
        tot = v if tot is None else tot + v
    return tot, R


# ------------------------------------------------------------------ (ii): records
def load_records(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def check_records(coarse, rec, A8_hi=None, chunk=20000):
    """rigorous lower bounds of S+ + W - A8 per box; returns dict with lb and the four partial bounds"""
    n = len(rec["s_lo"])
    A8_hi = float(rec.get("A8_hi", A8_HI)) if A8_hi is None else A8_hi
    out = {k: np.full(n, -INF) for k in ("lb", "w", "k", "d", "c")}
    out["rad"] = np.full((n, 4), INF)
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        r = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v) for k, v in rec.items()}
        res = _check_chunk(coarse, r, A8_hi)
        for k in res:
            out[k][sl] = res[k]
    out["lb"] = np.maximum.reduce([out["w"], out["k"], out["d"], out["c"]])
    return out


def _check_chunk(coarse, r, A8_hi):
    n = len(r["s_lo"])
    T = I(np.maximum(dn(1.0 - r["s_hi"]), 0.0), up(1.0 - r["s_lo"]))
    X = [I(r["x_lo"][:, i], r["x_hi"][:, i]) for i in range(3)]
    Wl = r["W_lo"]
    res = {"w": nan_low(dn(Wl - A8_hi))}
    res["k"] = nan_low(dn(coarse.crude_lower(T, X) + Wl - A8_hi))
    res["d"] = np.full(n, -INF)
    res["c"] = np.full(n, -INF)
    res["rad"] = np.full((n, 4), INF)       # diagnostics: |term| of the centred form per label axis and for s
    ok = (T.lo > 1e-9) & np.all(np.isfinite(r["x_lo"]) & np.isfinite(r["x_hi"]), axis=1)
    ok &= inorm([I(np.where(ok, x.lo, 1.0), np.where(ok, x.hi, 1.0)) for x in X]).lo > 0   # boxes away from x = 0
    idx = np.flatnonzero(ok)
    if not len(idx):
        return res
    Tk = take(T, idx)
    Xk = [take(x, idx) for x in X]
    S, gS, St = coarse.enclose(Tk, Xk)
    res["d"][idx] = nan_low(dn(S.lo + Wl[idx] - A8_hi))
    # centred forms: need finite grad W (= -p) and a finite centre value
    gWl, gWh = _gW(r)
    cen = np.isfinite(gWl[idx]).all(1) & np.isfinite(gWh[idx]).all(1) & np.isfinite(r["W_hi"][idx]) \
        & np.isfinite(r["Wc_lo"][idx])
    j = idx[cen]
    if not len(j):
        return res
    sub = np.flatnonzero(cen)
    sc = r["s_c"][j]
    Tc = I(dn(1.0 - sc), up(1.0 - sc))
    Xc = [I(r["xc_lo"][j, i], r["xc_hi"][j, i]) for i in range(3)]
    Sc, _, _ = coarse.enclose(Tc, Xc, need_grad=False)
    fc = Sc + I(r["Wc_lo"][j], r["Wc_hi"][j])
    gW = [I(gWl[j, i], gWh[j, i]) for i in range(3)]
    G = [take(gS[i], sub) + gW[i] for i in range(3)]          # grad S - p over the box
    Xj = [take(x, j) for x in X]
    U, R = U_enclose(Xj)
    Wt = R * (gW[0].sq() + gW[1].sq() + gW[2].sq()) * 2.0 - U
    StWt = take(St, sub) + Wt                                 # S_t + W_t over the box
    ds = I(dn(r["s_lo"][j] - sc), up(r["s_hi"][j] - sc))
    da = [I(dn(r["a_lo"][j, k] - r["a_c"][j, k]), up(r["a_hi"][j, k] - r["a_c"][j, k])) for k in range(3)]
    best = np.full(len(j), -INF)
    rad_best = np.full((len(j), 4), INF)
    # (c1) full interval enclosures of dx/da and dx/ds
    if "dxda_lo" in r:
        ok1 = np.isfinite(r["dxda_lo"][j]).all((1, 2)) & np.isfinite(r["dxda_hi"][j]).all((1, 2)) \
            & np.isfinite(r["dxds_lo"][j]).all(1) & np.isfinite(r["dxds_hi"][j]).all(1)
        Gs = -StWt
        for i in range(3):
            Gs = Gs + G[i] * I(r["dxds_lo"][j, i], r["dxds_hi"][j, i])
        ts = Gs * ds
        f = fc + ts
        rad = np.zeros((len(j), 4))
        rad[:, 3] = np.maximum(-ts.lo, ts.hi)
        for k in range(3):
            Ga = None
            for i in range(3):
                t = G[i] * I(r["dxda_lo"][j, i, k], r["dxda_hi"][j, i, k])
                Ga = t if Ga is None else Ga + t
            ta = Ga * da[k]
            rad[:, k] = np.maximum(-ta.lo, ta.hi)
            f = f + ta
        lb1 = np.where(ok1, nan_low(dn(f.lo - A8_hi)), -INF)
        better = lb1 > best
        best = np.where(better, lb1, best)
        rad_best = np.where(better[:, None], rad, rad_best)
    # (c0) norm bounds only: |grad S - p| x |dx/da_j| x radius (eight_field_c1_obstacle_theorist.md section 4)
    if "Lcol" in r or "Lop" in r:
        gn = up(np.sqrt(up(sum(up(np.maximum(-G[i].lo, G[i].hi) ** 2) for i in range(3)))))   # max |grad S - p|
        stn = np.maximum(-StWt.lo, StWt.hi)
        rs = np.maximum(-ds.lo, ds.hi)
        ra = np.stack([np.maximum(-d.lo, d.hi) for d in da], 1)
        Ls = r["Ls"][j]
        ts = up(up(stn + up(gn * Ls)) * rs)
        if "Lcol" in r:
            ta = up(gn[:, None] * up(r["Lcol"][j] * ra))
            tot = up(ts + up(up(ta[:, 0] + ta[:, 1]) + ta[:, 2]))
            rad = np.concatenate([ta, ts[:, None]], 1)
        else:
            rn = up(np.sqrt(up(np.sum(up(ra * ra), 1))))
            tot = up(ts + up(gn * up(r["Lop"][j] * rn)))
            rad = np.concatenate([np.repeat(up(gn * up(r["Lop"][j] * rn))[:, None] / 3, 3, 1), ts[:, None]], 1)
        lb0 = nan_low(dn(dn(fc.lo - tot) - A8_hi))
        better = lb0 > best
        best = np.where(better, lb0, best)
        rad_best = np.where(better[:, None], rad, rad_best)
    res["rad"][j] = rad_best
    res["c"][j] = best
    return res


def _gW(r):
    """grad_x W over the box from the record: gW_lo/gW_hi, or p_lo/p_hi (p = Pi/s = -grad W)"""
    if "gW_lo" in r:
        return r["gW_lo"], r["gW_hi"]
    return -r["p_hi"], -r["p_lo"]


# ------------------------------------------------------------------ (ii'): the end condition on I at t = 1
def end_condition(coarse, discs, depth=12, n_ell=48, n_th=48, ell_rng=None, A8_hi=A8_HI, verbose=True):
    """S_E^c(1, x) > A8 on I minus the open discs. I = {rho (cos th e_I + sin th e_Z)}; S_E^c(1, .) is even in
    th -> -th (x_3 -> -x_3), so th in [0, pi] is checked and the discs are given for th >= 0 (the mirror discs at
    -th are their images): discs = list of (ell_lo, ell_hi, th_lo, th_hi), open boxes in (ell = log rho, th), or a
    function boxes (m, 4) -> bool mask, True only where the closed box certainly lies inside the open disc(s).
    Tails: ell <= ell_min with sigma <= sigma_lo: S >= w0 + phi_lo >= phi_lo; ell >= ell_max: S >= rho/8 + min phi
    (on I, n_1 = cos(th)/2 <= 1/2). Middle: boxes in (ell, th), bisected (lower bound = max of the direct and the
    centred form in (ell, th)); a box is dropped when it lies inside an open disc.
    Returns (ok, min lower bound, list of failing boxes (ell_lo, ell_hi, th_lo, th_hi, lb, float value at centre))."""
    rep = {}
    ell_min = coarse.slo                         # t = 1: sigma = ell
    tail_lo = dn(coarse.philo - A8_hi)           # w0 >= 0, phi = phi_lo for sigma <= sigma_lo
    ell_max = math.log(8.0 * (A8_hi + 1.0 - min(coarse.phi_glob_lo, 0.0)))
    tail_hi = 1.0                                 # rho/8 + min phi - A8 >= 1 for ell >= ell_max (by the choice)
    rep["tails"] = (ell_min, tail_lo, ell_max, tail_hi)
    lo_e, hi_e = (ell_min, ell_max) if ell_rng is None else ell_rng
    e = np.linspace(lo_e, hi_e, n_ell + 1)
    th = np.linspace(0.0, math.pi, n_th + 1)
    th[-1] = float(up(math.pi))                   # math.pi < pi: the last box reaches past pi
    E0, T0 = np.meshgrid(np.arange(n_ell), np.arange(n_th), indexing="ij")
    boxes = np.stack([e[E0.ravel()], e[E0.ravel() + 1], th[T0.ravel()], th[T0.ravel() + 1]], 1)
    fails, worst, nbox = [], INF, 0
    for dep in range(depth + 1):
        inside = np.zeros(len(boxes), bool)
        if callable(discs):         # a rigorous test: True where the closed box lies in the open disc (union)
            inside = np.asarray(discs(boxes), bool)
        else:
            for (a0, a1, b0, b1) in discs:
                inside |= (boxes[:, 0] > a0) & (boxes[:, 1] < a1) & (boxes[:, 2] > b0) & (boxes[:, 3] < b1)
        boxes = boxes[~inside]
        if not len(boxes):
            break
        nbox += len(boxes)
        lb, fc = end_lower(coarse, boxes, A8_hi)
        good = lb > 0
        if np.any(good):
            worst = min(worst, float(np.min(lb[good])))
        bad = boxes[~good]
        if not len(bad):
            boxes = bad
            break
        real = fc[~good] <= 0
        # a box straddling the disc edge whose centre lies in the open disc is not a failure of the condition (no
        # requirement inside the disc): refine it instead of stopping
        cb = np.stack([0.5 * (bad[:, 0] + bad[:, 1])] * 2 + [0.5 * (bad[:, 2] + bad[:, 3])] * 2, 1)[:, [0, 1, 2, 3]]
        if callable(discs):
            cin = np.asarray(discs(cb), bool)
        else:
            cin = np.zeros(len(cb), bool)
            for (a0, a1, b0, b1) in discs:
                cin |= (cb[:, 0] > a0) & (cb[:, 0] < a1) & (cb[:, 2] > b0) & (cb[:, 2] < b1)
        real &= ~cin
        for b, l, v in zip(bad[real], lb[~good][real], fc[~good][real]):
            fails.append((*b, l, v))
        bad = bad[~real]
        if dep == depth:
            for b, l, v in zip(bad, lb[~good][~real], fc[~good][~real]):
                fails.append((*b, l, v))
            break
        m0, m1 = 0.5 * (bad[:, 0] + bad[:, 1]), 0.5 * (bad[:, 2] + bad[:, 3])
        boxes = np.vstack([np.stack([bad[:, 0], m0, bad[:, 2], m1], 1), np.stack([m0, bad[:, 1], bad[:, 2], m1], 1),
                           np.stack([bad[:, 0], m0, m1, bad[:, 3]], 1), np.stack([m0, bad[:, 1], m1, bad[:, 3]], 1)])
    ok = (not fails) and tail_lo > 0
    if verbose:
        print(f"end condition (ii'): tails: ell <= {ell_min} lower bound {tail_lo:+.4e}; ell >= {ell_max:.3f}: >= +1; "
              f"middle: {nbox} boxes, min lower bound of passing boxes {worst:+.4e}, failing boxes {len(fails)}")
    return ok, min(worst, tail_lo), fails, rep


def end_lower(coarse, boxes, A8_hi):
    """lower bound of S_E^c(1, x) - A8 over boxes (ell, th); returns (lb, float-ish value at the centre)"""
    n = len(boxes)
    L = I(boxes[:, 0], boxes[:, 1])
    TH = I(boxes[:, 2], boxes[:, 3])
    rho = iexp(L)
    sn, cs = isincos(TH)
    X = [rho * (cs * EI_I[i] + sn * EZ_I[i]) for i in range(3)]
    T1 = I(np.ones(n))
    S, gS, _ = coarse.enclose(T1, X)
    lb_d = dn(S.lo - A8_hi)
    # centre
    lc, tc = 0.5 * (boxes[:, 0] + boxes[:, 1]), 0.5 * (boxes[:, 2] + boxes[:, 3])
    rc = iexp(I(lc))
    snc, csc = isincos(I(tc))
    Xc = [rc * (csc * EI_I[i] + snc * EZ_I[i]) for i in range(3)]
    Sc, _, _ = coarse.enclose(T1, Xc, need_grad=False)
    # dx/dell = x, dx/dth = rho (-sin e_I + cos e_Z)
    gl = gS[0] * X[0] + gS[1] * X[1] + gS[2] * X[2]
    gt = None
    for i in range(3):
        t = gS[i] * (rho * (cs * EZ_I[i] - sn * EI_I[i]))
        gt = t if gt is None else gt + t
    f = Sc + gl * I(dn(boxes[:, 0] - lc), up(boxes[:, 1] - lc)) + gt * I(dn(boxes[:, 2] - tc), up(boxes[:, 3] - tc))
    lb = np.maximum(lb_d, nan_low(dn(f.lo - A8_hi)))
    return lb, Sc.mid - A8_hi


def main():
    mode = sys.argv[1]
    if mode == "records":
        coarse = CoarseE(sys.argv[3])
        rec = load_records(sys.argv[2])
        A8h = float(sys.argv[4]) if len(sys.argv) > 4 else None
        res = check_records(coarse, rec, A8h)
        lb = res["lb"]
        print(f"{len(lb)} boxes; min lower bound of S+ + W - A8: {lb.min():+.4e}; failing (lb <= 0): {np.sum(lb <= 0)}")
        for k in ("w", "k", "d", "c"):
            print(f"  bound {k} is the best on {np.sum(res[k] == lb)} boxes")
        bad = np.flatnonzero(lb <= 0)
        for b in bad[np.argsort(lb[bad])][:20]:
            print(f"  box {b}: s [{rec['s_lo'][b]:.4f}, {rec['s_hi'][b]:.4f}], a_lo {rec['a_lo'][b]}, a_hi {rec['a_hi'][b]}, "
                  f"lb {lb[b]:+.4e}")
        print("CONDITION (ii-L) VERIFIED ON ALL BOXES" if np.all(lb > 0) else "NOT VERIFIED")
    elif mode == "end":
        coarse = CoarseE(sys.argv[2])
        ell_c, phi_c, h_ell, h_phi = (float(v) for v in sys.argv[3:7])
        depth = int(sys.argv[7]) if len(sys.argv) > 7 else 10
        ok, worst, fails, _ = end_condition(coarse, [(ell_c - h_ell, ell_c + h_ell, phi_c - h_phi, phi_c + h_phi)], depth)
        for f in sorted(fails, key=lambda v: v[4])[:20]:
            print(f"  failing: ell [{f[0]:.4f}, {f[1]:.4f}] th [{f[2]:.4f}, {f[3]:.4f}]: lb {f[4]:+.4e}, centre {f[5]:+.4e}")
        print("CONDITION (ii') VERIFIED" if ok else "NOT VERIFIED")


if __name__ == "__main__":
    main()
