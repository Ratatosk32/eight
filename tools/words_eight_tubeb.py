"""Stage 21, stage 4 (float side): unit tests of the rigorous tube-boundary evaluator (verify/eight_tube_boundary.py),
a float MOCK producer of field records (the record format of that file's docstring) from the scaled I-field flow of
verify/eight_field.py, and the end-to-end run on a label-box tube with margins against t.
    python tools/words_eight_tubeb.py test [cert.json]
    python tools/words_eight_tubeb.py json_block2 npz_name out.json        (Q(p,1,1) block2 npz -> verifier JSON)
    python tools/words_eight_tubeb.py mock cert.json tag [h_ell h_phi h_M] [ds] [nuv] [depth]
    python tools/words_eight_tubeb.py end cert.json [h_ell h_phi] [depth]
Mock enclosures are NOT rigorous: values at the corners and the centre of each box (finite differences for the
derivatives, a batched fixed-step RK4 flow), hulled and inflated by a quarter of their spread on each side."""
import json
import math
import os
import sys
import time

import numpy as np

ROOT = "D:/nature-math/math-gaps/"
sys.path.insert(0, ROOT + "verify")
sys.path.insert(0, ROOT + "tools")
import eight_tube_boundary as TB  # noqa: E402
from prove_newton_1212 import I  # noqa: E402

OUT = ROOT + "words/out/eight/tubeb/"
CERT_E = ROOT + "words/out/eight/cert_pieces_E.json"
A8 = 2.519856092284
S3 = math.sqrt(3.0)
UK_F = np.array([[-1.0, 0, 0], [0.5, S3 / 2, 0], [0.5, -S3 / 2, 0]])
EI_F, EZ_F, NI_F = np.array([0.5, S3 / 2, 0]), np.array([0, 0, 1.0]), np.array([S3 / 2, -0.5, 0])
A8_LAB = np.array([1.5424624176685366, 0.7066425303908385, -0.30889005565526345])      # eight's label (ell, phi, mu) at s = 1 (Newton)


def lowprio():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


# ------------------------------------------------------------------ float evaluation of a JSON piece (reference)
_JC = {}


def float_S_json(cert, t, x):
    """independent float evaluation of S_E^c from the JSON (brute-force cell search), points (m, 3)"""
    if cert not in _JC:
        data = json.load(open(cert))
        cells = data["cells"]
        _JC[cert] = (data, cells, np.array([[c["sigma"][0], c["ua"][0], c["ub"][0]] for c in cells]),
                     np.array([[c["sigma"][1], c["ua"][1], c["ub"][1]] for c in cells]), np.array([c["face"] for c in cells]))
    data, cells, lo, hi, fc = _JC[cert]
    faces = data["faces"]
    slo, philo, shi = data["low"]["sigma_lo"], data["low"]["phi_lo"], data["far"]["sigma_hi"]
    out = np.empty(len(x))
    for m, (tt, xx) in enumerate(zip(np.broadcast_to(t, len(x)), x)):
        xx = np.array(xx, float)
        xx[2] = abs(xx[2])
        r = np.linalg.norm(xx)
        sig = math.log(r) - 4 / 3 * math.log(tt)
        n = xx / r
        if sig <= slo:
            ph = philo
        else:
            sg = min(sig, shi)
            ph = None
            for f, fa in enumerate(faces):
                C, A, B = (np.array(fa[k], float) for k in ("centre", "ea", "eb"))
                if n @ C <= 0:
                    continue
                ua, ub = 4 / math.pi * math.atan(n @ A / (n @ C)), 4 / math.pi * math.atan(n @ B / (n @ C))
                sel = np.flatnonzero((fc == f) & (lo[:, 0] <= sg) & (hi[:, 0] >= sg) & (lo[:, 1] <= ua) & (hi[:, 1] >= ua)
                                     & (lo[:, 2] <= ub) & (hi[:, 2] >= ub))
                if len(sel):
                    c = cells[sel[0]]
                    y = [2 * (v - l) / (h - l) - 1 for v, l, h in zip((sg, ua, ub), lo[sel[0]], hi[sel[0]])]
                    cc = np.array(c["coef"])
                    ph = sum(cc[i, j, k] * y[0] ** i * y[1] ** j * y[2] ** k for i in range(cc.shape[0])
                             for j in range(cc.shape[1]) for k in range(cc.shape[2]))
                    break
            if sig > shi:
                ph *= math.exp(-(sig - shi) / 2)
        out[m] = (r - xx[0]) / (4 * tt) + tt ** (1 / 3) * ph
    return out


def pt(v):
    return I(np.asarray(v, float))


def enclose_points(co, t, x):
    T = I(np.asarray(t, float) * np.ones(len(x)))
    X = [I(x[:, i].copy()) for i in range(3)]
    return co.enclose(T, X)


def test(cert=CERT_E):
    co = TB.CoarseE(cert)
    print(f"piece {cert}: {len(co.cells)} cells, phi in [{co.phi_glob_lo:.4f}, {co.phi_glob_hi:.4f}] (crude), "
          f"phi_lo {co.philo}, sigma [{co.slo}, {co.shi}]")
    rng = np.random.default_rng(7)
    m = 400
    # points: around the tube (|x| 2 to 9, all directions incl. south), times 0.002 to 1, and far / low sigma
    d = rng.normal(size=(m, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    x = d * rng.uniform(1.5, 9, m)[:, None]
    t = rng.uniform(0.002, 1.0, m)
    t[:20] = 10 ** rng.uniform(-6, -4, 20)          # far field
    x[20:40] *= 1e-3; t[20:40] = 1.0                   # low (sigma < -3)
    ref = np.array([float_S_json(cert, tt, xx[None])[0] for tt, xx in zip(t, x)])
    # 1. value at points: brute-force float reference inside the point enclosure
    T = I(t)
    X = [I(x[:, i].copy()) for i in range(3)]
    S, gS, St = co.enclose(T, X)
    tol = 1e-12 * (1 + np.abs(ref))
    inside = (S.lo - tol <= ref) & (ref <= S.hi + tol)
    print(f"[1] values at {m} points vs the brute-force float evaluation of the JSON: inside the enclosure {inside.mean():.3f}, "
          f"max |mid - ref| {np.max(np.abs(S.mid - ref)):.2e}, max width {np.max(S.hi - S.lo):.2e}")
    # 1b. against the original mesh evaluation (words_eight_coarse.S_value) if this is the nu4 piece
    try:
        import words_eight_patch as wp
        mc = wp.load_coarse("E", 4, 0.1)
        if os.path.abspath(cert) == os.path.abspath(CERT_E):
            ref2 = np.array([wp.coarse_S(mc, "E", xx[None], tt)[0] for tt, xx in zip(t, x)])
            print(f"[1b] against words_eight_coarse.S_value (mesh form): max |mid - S_value| {np.max(np.abs(S.mid - ref2)):.2e}")
    except Exception as e:
        print("[1b] skipped:", e)
    # 2. derivatives: central differences against the gradient enclosure at points (kinks excluded by a small tolerance)
    h = 1e-6
    fd = np.zeros((m, 3))
    for i in range(3):
        e = np.zeros(3); e[i] = h
        fd[:, i] = (np.array([float_S_json(cert, tt, (xx + e)[None])[0] for tt, xx in zip(t, x)])
                    - np.array([float_S_json(cert, tt, (xx - e)[None])[0] for tt, xx in zip(t, x)])) / (2 * h)
    fdt = (np.array([float_S_json(cert, tt + h * tt, xx[None])[0] for tt, xx in zip(t, x)])
           - np.array([float_S_json(cert, tt - h * tt, xx[None])[0] for tt, xx in zip(t, x)])) / (2 * h * t)
    ok = np.ones(m, bool)
    err = 0.0
    for i in range(3):
        sc = 1e-5 * (1 + np.abs(fd[:, i]))
        ok &= (gS[i].lo - sc <= fd[:, i]) & (fd[:, i] <= gS[i].hi + sc)
        err = max(err, np.max(np.abs(gS[i].mid - fd[:, i]) / (1 + np.abs(fd[:, i]))))
    sct = 1e-5 * (1 + np.abs(fdt))
    okt = (St.lo - sct <= fdt) & (fdt <= St.hi + sct)
    print(f"[2] grad_x S vs central differences: inside {ok.mean():.3f}, max rel |mid - fd| {err:.2e}; "
          f"S_t: inside {okt.mean():.3f} (points on a cell face get a hull of both sides; fd across a kink can miss)")
    if not ok.all() or not okt.all():
        bad = np.flatnonzero(~(ok & okt))[:5]
        for b in bad:
            print(f"     point {b}: t {t[b]:.3g} x {x[b]} fd {fd[b]} encl {[(g.lo[b], g.hi[b]) for g in gS]} fdt {fdt[b]:.6g} "
                  f"St [{St.lo[b]:.6g}, {St.hi[b]:.6g}]")
    # 3. boxes: random boxes of relative size 1e-3 .. 3e-2 in x and t; samples inside must lie in the enclosure
    nb = 200
    wrel = 10 ** rng.uniform(-3, -1.5, nb)
    xc, tc = x[40:40 + nb], t[40:40 + nb]
    xw = np.abs(xc).max(1)[:, None] * wrel[:, None] * np.ones(3)
    tw = tc * wrel
    T = I(tc - tw, tc + tw)
    X = [I(xc[:, i] - xw[:, i], xc[:, i] + xw[:, i]) for i in range(3)]
    S, gS, St = co.enclose(T, X)
    viol, gviol = 0, 0
    for k in range(nb):
        u = rng.uniform(-1, 1, (30, 4))
        ps = xc[k] + u[:, :3] * xw[k]
        ts = tc[k] + u[:, 3] * tw[k]
        v = np.array([float_S_json(cert, a, b[None])[0] for a, b in zip(ts, ps)])
        viol += np.sum((v < S.lo[k] - 1e-12) | (v > S.hi[k] + 1e-12))
        # mean value check: S(p) - S(q) in [grad] . (p - q) + [S_t](tp - tq)
        dv = v[1:] - v[0]
        dpx = ps[1:] - ps[0]
        lo = sum(np.minimum(gS[i].lo[k] * dpx[:, i], gS[i].hi[k] * dpx[:, i]) for i in range(3)) \
            + np.minimum(St.lo[k] * (ts[1:] - ts[0]), St.hi[k] * (ts[1:] - ts[0]))
        hi = sum(np.maximum(gS[i].lo[k] * dpx[:, i], gS[i].hi[k] * dpx[:, i]) for i in range(3)) \
            + np.maximum(St.lo[k] * (ts[1:] - ts[0]), St.hi[k] * (ts[1:] - ts[0]))
        gviol += np.sum((dv < lo - 1e-11) | (dv > hi + 1e-11))
    wS = np.median((S.hi - S.lo) / (np.abs(xw).max(1)))
    print(f"[3] {nb} random boxes x 30 samples: value outside enclosure {viol}, mean-value (gradient) violations {gviol}; "
          f"median width of S per unit box half-width {wS:.3g}")
    # 4. crude bound
    cl = co.crude_lower(T, X)
    print(f"[4] crude kinetic lower bound <= S.lo on all boxes: {np.all(cl <= S.lo + 1e-12)}")
    return co


# ------------------------------------------------------------------ block2 npz -> JSON (own copy, own output path)
def json_block2(npz, path):
    import words_eight_block2 as B2
    d = np.load(npz)
    nu, ref, p = int(d["nu"]), str(d["ref"]), int(d["p"])
    mesh = B2.make_mesh("E", nu, ref, p)
    coef = d["coef"]
    ys = np.linspace(-1, 1, p + 1)
    Y = np.array([[a, b, c] for a in ys for b in (-1.0, 1.0) for c in (-1.0, 1.0)])
    expo = [(i, j, l) for i in range(p + 1) for j in range(2) for l in range(2)]
    Vi = np.linalg.inv(np.array([[y[0] ** i * y[1] ** j * y[2] ** l for (i, j, l) in expo] for y in Y]))
    perm = [4 * a + (b | (cc << 1)) for a in range(p + 1) for b in range(2) for cc in range(2)]
    cells = []
    for k in range(mesh.ncl):
        cols = B2.cell_cols(mesh, np.full(mesh.nq, k), np.arange(mesh.nq))
        M = coef[cols][:, perm] @ Vi.T
        for qi, (fc, ua0, ub0, du, ids) in enumerate(mesh.quads):
            Cc = np.zeros((p + 1, 2, 2))
            for e, (i, j, l) in enumerate(expo):
                Cc[i, j, l] = M[qi, e]
            cells.append({"face": int(fc), "sigma": [float(mesh.csig[k]), float(mesh.csig[k + 1])],
                          "ua": [float(ua0), float(ua0 + du)], "ub": [float(ub0), float(ub0 + du)],
                          "deg": [p, 1, 1], "coef": Cc.tolist()})
    out = {"side": "E", "faces": [{"centre": c.tolist(), "ea": ea.tolist(), "eb": eb.tolist(), "ub_range": list(r)}
                                  for c, ea, eb, r in mesh.faces],
           "cells": cells, "far": {"sigma_hi": float(mesh.shi)}, "low": {"sigma_lo": float(mesh.slo), "phi_lo": float(coef[0])},
           "source": os.path.basename(npz) + " (NOT verified: for margins only)"}
    json.dump(out, open(path, "w"))
    rng = np.random.default_rng(3)
    xs = rng.normal(size=(200, 3)) * 2 + np.array([4, 1, 1.5])
    ts = rng.uniform(0.05, 1, 200)
    v1 = np.array([B2.S_value(mesh, coef, "E", x[None], t)[0] for x, t in zip(xs, ts)])
    v2 = float_S_json(path, ts, xs)
    print(f"wrote {path}: {len(cells)} cells; JSON vs block2 S_value max diff {np.max(np.abs(v1 - v2)):.2e}")


# ------------------------------------------------------------------ mock field: batched RK4 of the scaled I-field flow
NSTEP = int(os.environ.get("TUBEB_NSTEP", "160"))


def rhs_b(y, eps):
    Y, P = y[:, 0:3], y[:, 3:6]
    R = np.sqrt(np.sum(Y * Y, 1))
    Q = np.sum(P * P, 1)
    q = R[:, None] - Y @ UK_F.T
    q15 = q ** -1.5
    YR = Y / R[:, None]
    G = -0.5 * q15.sum(1)[:, None] * YR + 0.5 * q15 @ UK_F
    return np.concatenate([-4 * R[:, None] * P, 2 * Q[:, None] * YR - eps[:, None] * G, (2 * R * Q)[:, None],
                           np.sum(q ** -0.5, 1)[:, None]], 1)


def flow_b(eps, phi, M, nstep=NSTEP):
    """(Y(1), P(1), Khat, Uhat) for arrays of parameters; classical RK4, fixed step"""
    y = np.concatenate([np.cos(phi)[:, None] * EI_F + np.sin(phi)[:, None] * EZ_F, M[:, None] * NI_F,
                        np.zeros((len(phi), 2))], 1)
    h = 1.0 / nstep
    for _ in range(nstep):
        k1 = rhs_b(y, eps)
        k2 = rhs_b(y + 0.5 * h * k1, eps)
        k3 = rhs_b(y + 0.5 * h * k2, eps)
        k4 = rhs_b(y + h * k3, eps)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def phys_b(a, s):
    """x, P (= s p), q = r (Khat + eps Uhat) (W = q/s) at labels a (m, 3) and times to go s (m,)"""
    r = np.exp(a[:, 0])
    eps = s * s * r ** -1.5
    out = np.empty((len(s), 7))
    for c in range(0, len(s), 50000):
        sl = slice(c, c + 50000)
        y = flow_b(eps[sl], a[sl, 1], a[sl, 2])
        out[sl, 0:3] = r[sl, None] * y[:, 0:3]
        out[sl, 3:6] = y[:, 3:6]
        out[sl, 6] = r[sl] * (y[:, 6] + eps[sl] * y[:, 7])
    return out


FD_H = 1e-5


def node_data(a, s):
    """per node: x (3), P (3), q, dx/da (3x3), dx/ds (3)"""
    m = len(s)
    A = [a]
    S = [s]
    for j in range(3):
        for sg in (1, -1):
            e = a.copy()
            e[:, j] += sg * FD_H
            A.append(e)
            S.append(s)
    for sg in (1, -1):
        A.append(a)
        S.append(s + sg * FD_H)
    o = phys_b(np.vstack(A), np.concatenate(S)).reshape(9, m, 7)
    dxda = np.stack([(o[1 + 2 * j, :, 0:3] - o[2 + 2 * j, :, 0:3]) / (2 * FD_H) for j in range(3)], 2)
    dxds = (o[7, :, 0:3] - o[8, :, 0:3]) / (2 * FD_H)
    return {"x": o[0, :, 0:3], "P": o[0, :, 3:6], "q": o[0, :, 6], "dxda": dxda, "dxds": dxds}


class Tube:
    """label-box tube: Lambda(s) = a_c(s) +- h, a_c(s) = (ell8, phi8, s mu8), h constant"""

    def __init__(self, h, lab8=A8_LAB):
        self.h = np.array(h, float)
        self.lab8 = np.array(lab8, float)

    def centre(self, s):
        s = np.asarray(s, float)
        return np.stack([np.full_like(s, self.lab8[0]), np.full_like(s, self.lab8[1]), s * self.lab8[2]], -1)

    def label(self, f, s, u, v):
        """face f (axis f // 2, sign - for even f), free coordinates u, v in [-1, 1] (arrays)"""
        f = np.broadcast_to(np.asarray(f, int), np.shape(s))
        a = self.centre(s)
        for k in range(3):
            sel = f // 2 == k
            if not np.any(sel):
                continue
            fr = [i for i in range(3) if i != k]
            sg = np.where(f[sel] % 2 == 1, 1.0, -1.0)
            a[sel, k] += sg * self.h[k]
            a[sel, fr[0]] += self.h[fr[0]] * np.asarray(u)[sel]
            a[sel, fr[1]] += self.h[fr[1]] * np.asarray(v)[sel]
        return a


class MockProducer:
    def __init__(self, tube, mode="c0", crude=1.5):
        """mode "c0": C0 enclosures + column-norm bounds Lcol, Ls (true sampled maxima times `crude`), as the main
        session's C0 Lohner tubes will give; mode "c1": full interval dx/da, dx/ds (mock of C1 Lohner)"""
        self.tube = tube
        self.mode = mode
        self.crude = crude
        self.cache = {}

    @staticmethod
    def _key(f, s, u, v):
        return (int(f), round(float(s), 12), round(float(u), 12), round(float(v), 12))

    def ensure(self, keys):
        new = [k for k in set(keys) if k not in self.cache]
        if not new:
            return
        arr = np.array(new, float)
        a = self.tube.label(arr[:, 0].astype(int), arr[:, 1], arr[:, 2], arr[:, 3])
        d = node_data(a, arr[:, 1])
        for i, k in enumerate(new):
            self.cache[k] = {kk: vv[i] for kk, vv in d.items()}

    def records(self, boxes):
        """boxes: (n, 7) [face, s0, s1, u0, u1, v0, v1] -> record dict (format of verify/eight_tube_boundary.py)"""
        n = len(boxes)
        f = boxes[:, 0].astype(int)
        cor = [(c0, c1, c2) for c0 in (1, 2) for c1 in (3, 4) for c2 in (5, 6)]
        sc, uc, vc = (0.5 * (boxes[:, i] + boxes[:, i + 1]) for i in (1, 3, 5))
        keys = [[self._key(f[b], boxes[b, i], boxes[b, j], boxes[b, k]) for (i, j, k) in cor]
                + [self._key(f[b], sc[b], uc[b], vc[b])] for b in range(n)]
        self.ensure([k for kk in keys for k in kk])
        rec = {"s_lo": boxes[:, 1].copy(), "s_hi": boxes[:, 2].copy(), "s_c": sc, "face": f}
        if hasattr(self.tube, "label_box"):      # curved patches: interval enclosure of the surface patch
            rec["a_lo"], rec["a_hi"] = self.tube.label_box(f, boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4],
                                                           boxes[:, 5], boxes[:, 6])
        else:                                     # flat faces of a box: affine in (s, u, v), the corners suffice
            acor = np.stack([self.tube.label(f, boxes[:, i], boxes[:, j], boxes[:, k]) for (i, j, k) in cor], 0)
            rec["a_lo"], rec["a_hi"] = acor.min(0), acor.max(0)
        rec["a_c"] = self.tube.label(f, sc, uc, vc)

        def gather(name):
            return np.stack([np.stack([self.cache[k][name] for k in kk], 0) for kk in keys], 0)   # (n, 9, ...)
        svals = np.concatenate([np.stack([boxes[:, i] for (i, j, k) in cor], 1), sc[:, None]], 1)    # (n, 9)

        def hull(v, extra=0.25):
            lo, hi = v.min(1), v.max(1)
            sp = hi - lo
            pad = extra * sp + 1e-9 * np.maximum(np.abs(lo), np.abs(hi)) + 1e-12
            return lo - pad, hi + pad
        X = gather("x")
        rec["x_lo"], rec["x_hi"] = hull(X)
        DA, DS = gather("dxda"), gather("dxds")
        if self.mode == "c1":
            rec["dxda_lo"], rec["dxda_hi"] = hull(DA)
            rec["dxds_lo"], rec["dxds_hi"] = hull(DS)
        else:
            rec["Lcol"] = self.crude * np.linalg.norm(DA, axis=2).max(1)
            rec["Ls"] = self.crude * np.linalg.norm(DS, axis=2).max(1)
        q = gather("q")
        P = gather("P")
        z = boxes[:, 1] <= 0
        sv = np.where(svals > 0, svals, 1.0)
        W = q / sv
        gW = -P / sv[:, :, None]
        rec["W_lo"], rec["W_hi"] = hull(W)
        rec["gW_lo"], rec["gW_hi"] = hull(gW)
        if np.any(z):           # s_lo = 0: W >= min q / s_hi, unbounded above; grad W unbounded
            ql, _ = hull(q[z])
            rec["W_lo"][z] = np.maximum(ql, 0.0) / boxes[z, 2]
            rec["W_hi"][z] = np.inf
            rec["gW_lo"][z] = -np.inf
            rec["gW_hi"][z] = np.inf
        xc, Wc = X[:, 8], W[:, 8]
        rec["xc_lo"], rec["xc_hi"] = xc - 1e-12 * np.abs(xc) - 1e-14, xc + 1e-12 * np.abs(xc) + 1e-14
        rec["Wc_lo"], rec["Wc_hi"] = Wc - 1e-12 * np.abs(Wc) - 1e-14, Wc + 1e-12 * np.abs(Wc) + 1e-14
        rec["A8_hi"] = np.array(TB.A8_HI)
        return rec


def mock_selftest():
    import eight_field as EF
    rng = np.random.default_rng(5)
    a = A8_LAB + rng.uniform(-1, 1, (6, 3)) * np.array([0.3, 0.3, 0.1])
    s = rng.uniform(0.05, 1, 6)
    o = phys_b(a, s)
    err = 0.0
    for i in range(6):
        x, p, Ac = EF.physical(a[i], s[i])
        err = max(err, np.max(np.abs(o[i, 0:3] - x)), np.max(np.abs(o[i, 3:6] / s[i] - p)), abs(o[i, 6] / s[i] - Ac))
    o8 = phys_b(A8_LAB[None, :], np.array([1.0]))
    print(f"[mock] batched RK4 ({NSTEP} steps) vs DOP853 eight_field.physical: max diff {err:.2e}; eight at s = 1: "
          f"x {o8[0, 0:3]}, W {o8[0, 6]:.10f} (A8 {A8})")
    d = node_data(a, s)
    h = 1e-4
    lhs = (phys_b(a, s + h)[:, 6] / (s + h) - phys_b(a, s - h)[:, 6] / (s - h)) / (2 * h)
    x = d["x"]
    gW = -d["P"] / s[:, None]
    R = np.linalg.norm(x, axis=1)
    U = np.sum((R[:, None] - x @ UK_F.T) ** -0.5, 1)
    Wt = 2 * R * np.sum(gW ** 2, 1) - U
    rhs = -Wt + np.sum(gW * d["dxds"], 1)
    print(f"[mock] HJ check dW/ds|_a = -(2|x||gW|^2 - U) + gW . dx/ds: max rel diff {np.max(np.abs(lhs - rhs) / (1 + np.abs(lhs))):.2e}")
    err = 0.0
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        dW = (phys_b(a + e, s)[:, 6] - phys_b(a - e, s)[:, 6]) / (2 * h) / s
        err = max(err, np.max(np.abs(dW - np.einsum("mi,mi->m", gW, d["dxda"][:, :, j])) / (1 + np.abs(dW))))
    print(f"[mock] dW/da = grad W . dx/da (grad W = -P/s): max rel diff {err:.2e}")


def initial_boxes(ds_list, nuv):
    """ds_list: list of (s_from, s_to, ds) (ds < 0: |ds| geometric steps per decade), or an array of s nodes; faces
    0..5 each with nuv x nuv (u, v) boxes"""
    if isinstance(ds_list, np.ndarray):
        sn = ds_list
    else:
        sn = []
        for a, b, d in ds_list:
            if d < 0:
                k = max(1, int(round(-d * math.log10(b / a))))
                sn.append(np.geomspace(a, b, k + 1))
            else:
                k = max(1, int(round((b - a) / d)))
                sn.append(np.linspace(a, b, k + 1))
        sn = np.concatenate(sn)
    sn = np.unique(np.round(sn, 12))
    uv = np.linspace(-1, 1, nuv + 1)
    out = []
    for f in range(6):
        for i in range(len(sn) - 1):
            for p in range(nuv):
                for q in range(nuv):
                    out.append([f, sn[i], sn[i + 1], uv[p], uv[p + 1], uv[q], uv[q + 1]])
    return np.array(out)


def split_boxes(boxes, rad, ell=None):
    """bisect along the coordinates (s, u, v) whose term in the centred form is >= half the largest (all three when
    the centred form is not available)"""
    out = []
    if ell is not None:     # angle patches: attribute each label term to s or to (u, v) by the measured motion
        f = boxes[:, 0].astype(int)
        uc, vc, sc = 0.5 * (boxes[:, 3] + boxes[:, 4]), 0.5 * (boxes[:, 5] + boxes[:, 6]), 0.5 * (boxes[:, 1] + boxes[:, 2])
        ws = np.abs(ell.label(f, boxes[:, 2], uc, vc) - ell.label(f, boxes[:, 1], uc, vc))
        wu = np.abs(ell.label(f, sc, boxes[:, 4], vc) - ell.label(f, sc, boxes[:, 3], vc))
        wv = np.abs(ell.label(f, sc, uc, boxes[:, 6]) - ell.label(f, sc, uc, boxes[:, 5]))
        tot = ws + wu + wv + 1e-300
        TS = rad[:, 3] + np.sum(rad[:, :3] * ws / tot, 1)
        TU = np.sum(rad[:, :3] * wu / tot, 1)
        TV = np.sum(rad[:, :3] * wv / tot, 1)
        TT = np.stack([TS, TU, TV], 1)
    for n_, (b, r) in enumerate(zip(boxes, rad)):
        k = int(b[0]) // 2
        fr = [i for i in range(3) if i != k]
        if ell is not None:
            terms = TT[n_] if np.all(np.isfinite(TT[n_])) else np.array([np.inf] * 3)
        else:
            terms = np.array([r[3] + (r[k] if k == 2 else 0.0), r[fr[0]], r[fr[1]]])
        dims = [0, 1, 2] if not np.all(np.isfinite(terms)) else [i for i in range(3) if terms[i] >= 0.5 * terms.max()]
        parts = [b]
        for dmn in dims:
            i0 = 1 + 2 * dmn
            nxt = []
            for p in parts:
                m = 0.5 * (p[i0] + p[i0 + 1])
                lo, hi = p.copy(), p.copy()
                lo[i0 + 1] = m
                hi[i0] = m
                nxt += [lo, hi]
            parts = nxt
        out += parts
    return np.array(out)


def run_mock(cert, tag, h=(0.3, 0.3, 0.075), ds_list=None, nuv=4, depth=4, maxbox=300000, trange=(0.0, 1.0), fcmin=0.0,
             mode="c0", crude=1.5, tube=None):
    """status per final box: 0 pass (rigorous lb > 0 given the records), 1 real failure (float S+ + W - A8 <= 0 at the
    box centre), 2 unresolved at the last depth, 3 not refined because the float margin at the centre is < fcmin"""
    t0 = time.time()
    co = TB.CoarseE(cert)
    tube = tube or Tube(h)
    ell = isinstance(tube, EllTube)
    mp = MockProducer(tube, mode, crude)
    ds_list = ds_list or [(0.0, 0.85, 0.05), (0.85, 1.0, 0.025)]
    boxes = initial_boxes(ds_list, nuv)
    boxes = boxes[(boxes[:, 2] >= 1 - trange[1] - 1e-12) & (boxes[:, 1] <= 1 - trange[0] + 1e-12)]
    slabs = np.unique(np.concatenate([boxes[:, 1], boxes[:, 2]]))
    print(f"mock run {tag}: piece {os.path.basename(cert)}, tube {os.path.basename(tube.path) if ell else h}, records {mode} (crude factor {crude}), "
          f"{len(boxes)} initial boxes, t in {trange}, depth {depth}", flush=True)
    done = []
    for dep in range(depth + 1):
        mp.cache = {}            # memory: nodes are recomputed per depth (only parent corners are lost)
        rec = mp.records(boxes)
        res = TB.check_records(co, rec)
        lb = res["lb"]
        Tc = I(1.0 - rec["s_c"])
        Sc, _, _ = co.enclose(Tc, [I(0.5 * (rec["xc_lo"][:, i] + rec["xc_hi"][:, i])) for i in range(3)], need_grad=False)
        Wc = 0.5 * (rec["Wc_lo"] + rec["Wc_hi"])
        fc = np.maximum(Sc.mid, 0.0) + Wc - A8
        good = lb > 0
        real = ~good & (fc <= 0)
        low = ~good & ~real & (fc < fcmin)
        real_or_low = real | low
        for st, sel in ((0, good), (1, real), (3, low)):
            for b in np.flatnonzero(sel):
                done.append((boxes[b], lb[b], fc[b], st, res["w"][b], res["k"][b], res["d"][b], res["c"][b]))
        ref = ~good & ~real_or_low
        print(f"  depth {dep}: {len(boxes)} boxes ({len(mp.cache)} nodes): pass {good.sum()}, real failures (float "
              f"margin at the centre <= 0) {real.sum()}, to refine {ref.sum()}  ({time.time() - t0:.0f} s)", flush=True)
        if not ref.any():
            break
        if dep == depth or ref.sum() * 8 > maxbox:
            for b in np.flatnonzero(ref):
                done.append((boxes[b], lb[b], fc[b], 2, res["w"][b], res["k"][b], res["d"][b], res["c"][b]))
            break
        boxes = split_boxes(boxes[ref], res["rad"][ref], tube if ell else None)
    B = np.array([d[0] for d in done])
    LB = np.array([d[1] for d in done])
    FC = np.array([d[2] for d in done])
    ST = np.array([d[3] for d in done])
    PART = np.array([d[4:8] for d in done])
    np.savez(OUT + f"mock_{tag}.npz", boxes=B, lb=LB, fc=FC, status=ST, parts=PART, h=np.array(h),
             tube=os.path.basename(tube.path) if ell else "box")
    print("\n  margins against t (per initial s-slab; lb = rigorous lower bound GIVEN the mock records; fc = float "
          "S+ + W - A8 at box centres)")
    print(f"   (not refined: float margin at the centre below fcmin = {fcmin})")
    print("   t range          boxes  pass  realfail  unresolved  <fcmin  min fc       min lb(pass)  min lb(all)  winning bound (w,k,d,c)")
    for i in range(len(slabs) - 1):
        sel = (B[:, 1] >= slabs[i] - 1e-12) & (B[:, 2] <= slabs[i + 1] + 1e-12)
        if not sel.any():
            continue
        win = np.argmax(PART[sel], 1)
        mix = [int(np.sum(win == k)) for k in range(4)]
        lbp = LB[sel & (ST == 0)]
        print(f"   [{1 - slabs[i + 1]:.3f}, {1 - slabs[i]:.3f}]  {sel.sum():6d} {np.sum(ST[sel] == 0):5d} {np.sum(ST[sel] == 1):8d} "
              f"{np.sum(ST[sel] == 2):10d} {np.sum(ST[sel] == 3):7d}   {FC[sel].min():+.4e}  {(lbp.min() if len(lbp) else np.nan):+.4e}   "
              f"{LB[sel].min():+.4e}  {mix}", flush=True)
    print(f"  total boxes {len(B)}, pass {np.sum(ST == 0)}, real failures {np.sum(ST == 1)}, unresolved {np.sum(ST == 2)}, below fcmin {np.sum(ST == 3)}; "
          f"{time.time() - t0:.0f} s")
    return B, LB, FC, ST


def run_end(cert, h=(0.3, 0.3), depth=10):
    co = TB.CoarseE(cert)
    disc = (A8_LAB[0] - h[0], A8_LAB[0] + h[0], A8_LAB[1] - h[1], A8_LAB[1] + h[1])
    print(f"end condition for {os.path.basename(cert)}: open disc ell in ({disc[0]:.4f}, {disc[1]:.4f}), th in "
          f"({disc[2]:.4f}, {disc[3]:.4f}) and its mirror (th -> -th)")
    ok, worst, fails, rep = TB.end_condition(co, [disc], depth)
    if fails:
        F = np.array(fails)
        print(f"  failing boxes: {len(F)}; real (float value at the centre <= 0): {np.sum(F[:, 5] <= 0)}; min float value "
              f"{F[:, 5].min():+.4e}")
        for lo, hi, nm in ((-np.inf, 0.0, "rho < 1"), (0.0, 1.2, "1 <= rho < 3.3"), (1.2, 1.9, "3.3 <= rho < 6.7"),
                           (1.9, np.inf, "rho >= 6.7")):
            sel = (F[:, 0] >= lo) & (F[:, 0] < hi)
            if sel.any():
                k = np.argmin(F[sel, 5])
                print(f"   {nm}: {sel.sum()} failing boxes, min centre value {F[sel, 5].min():+.4e} at ell "
                      f"[{F[sel][k, 0]:.3f}, {F[sel][k, 1]:.3f}] th [{F[sel][k, 2]:.3f}, {F[sel][k, 3]:.3f}]")
    print("END CONDITION (ii') VERIFIED" if ok else "END CONDITION (ii') NOT VERIFIED")
    return ok, worst, fails


def scan_early(cert, tmax=0.15, nt=60, nuv=41, h=(0.3, 0.3, 0.075)):
    """float scan of S+ + W - A8 on the tube boundary for t in (0, tmax] (dense; the mirror eight runs inside the tube
    until t ~ 0.1, A.5 (e)); also the minimum of W - A8 alone and of the kinetic term, to see which regime carries"""
    co = TB.CoarseE(cert)
    tube = Tube(h)
    ts = np.concatenate([np.geomspace(1e-4, 0.01, 12), np.linspace(0.01, tmax, nt)[1:]])
    uv = np.linspace(-1, 1, nuv)
    U, V = np.meshgrid(uv, uv, indexing="ij")
    print(f"dense float scan of the tube boundary, t in [{ts[0]:.0e}, {tmax}], {len(ts)} times, {nuv}x{nuv} per face")
    print("   t        min(S+ + W - A8)  at face (u, v)          min(W - A8)   kinetic term at the argmin")
    rows = []
    for t in ts:
        s = 1 - t
        F = np.repeat(np.arange(6), nuv * nuv)
        a = tube.label(F, np.full(len(F), s), np.tile(U.ravel(), 6), np.tile(V.ravel(), 6))
        o = phys_b(a, np.full(len(F), s))
        x, Wv = o[:, 0:3], o[:, 6] / s
        S, _, _ = co.enclose(I(np.full(len(F), t)), [I(x[:, i].copy()) for i in range(3)], need_grad=False)
        f = np.maximum(S.mid, 0) + Wv - A8
        k = np.argmin(f)
        kin = (np.linalg.norm(x[k]) - x[k, 0]) / (4 * t)
        rows.append((t, f[k], Wv.min() - A8))
        print(f"   {t:.4f}   {f[k]:+.4e}       {F[k]} ({np.tile(U.ravel(), 6)[k]:+.2f}, {np.tile(V.ravel(), 6)[k]:+.2f})"
              f"        {Wv.min() - A8:+.4e}   {kin:.4e}", flush=True)
    return np.array(rows)

def small_s(cert, h=(0.3, 0.3, 0.075), s0max=0.02):
    """the end s -> 0 of (ii-L) (eight_lift_referee.md C1). On the (ell, phi)-faces, with x0 = f_0(ell, phi, 0) on dD:
        S(1 - s, f_s(a)) + W >= S(1, x0) - Lt s - Lx |f_s(a) - x0| + 2 r M^2/s + s r^(-1/2)(2 Uhat - U(Y0)),
        |f_s(a) - x0| <= r (4|M| + C (M^2 + s^2)),   min_M (2 r M^2/s - 4 r Lx |M|) = -2 r Lx^2 s,
    so S+ + W - A8 >= m' - K s with m' the margin of (ii') on dD and
        K = Lt + 2 r_max Lx^2 / (1 - Lx C s0 / 2) + Lx r_max C s0 + r_min^(-1/2) max U(Y0)
    (the M^2 term of the displacement is absorbed into 2 r M^2/s; Uhat >= 0). On the M-faces W alone:
        W >= 2 r_min (h_M - s |mu8|)^2 / s - s r_min^(-1/2) max U(Y0) > A8.
    Constants: Lx, Lt from the rigorous enclosure of grad S and S_t over a box cover of the neighbourhood
    {x : |x - x0| <= rho, x0 on dD} for t in [1 - s0max, 1]; C, the energy identity and the true deficit in floats."""
    co = TB.CoarseE(cert)
    tube = Tube(h)
    l8, p8, mu8 = A8_LAB
    rmin, rmax = math.exp(l8 - h[0]), math.exp(l8 + h[0])
    rng = np.random.default_rng(11)
    # 1. energy identity
    a = A8_LAB + rng.uniform(-1, 1, (200, 3)) * np.array(h)
    s = rng.uniform(1e-3, 1, 200)
    r = np.exp(a[:, 0])
    eps = s * s * r ** -1.5
    y = flow_b(eps, a[:, 1], a[:, 2])
    W = r * (y[:, 6] + eps * y[:, 7]) / s
    Y0 = np.cos(a[:, 1])[:, None] * EI_F + np.sin(a[:, 1])[:, None] * EZ_F
    UY0 = np.sum((1 - Y0 @ UK_F.T) ** -0.5, 1)
    Wid = 2 * r * a[:, 2] ** 2 / s + s * r ** -0.5 * (2 * y[:, 7] - UY0)
    print(f"[small s] energy identity W = 2 r M^2/s + s r^(-1/2)(2 Uhat - U(Y0)): max rel diff "
          f"{np.max(np.abs(W - Wid) / np.abs(W)):.1e}")
    # 2. displacement constants on the (ell, phi)-faces: |f_s(a) - x0| <= r (4|M| + C (M^2 + s^2)) and, after removing
    #    the normal part, |f_s(a) - x0 + 4 r M n_I| <= r C2 (M^2 + s^2)
    m = 20000
    F = rng.integers(0, 4, m)
    ss = np.concatenate([10 ** rng.uniform(-5, math.log10(s0max), m // 2), rng.uniform(0, s0max, m - m // 2)])
    u, v = rng.uniform(-1, 1, m), rng.uniform(-1, 1, m)
    a = tube.label(F, ss, u, v)
    o = phys_b(a, ss)
    r = np.exp(a[:, 0])
    x0 = r[:, None] * (np.cos(a[:, 1])[:, None] * EI_F + np.sin(a[:, 1])[:, None] * EZ_F)
    dx = (o[:, 0:3] - x0) / r[:, None]
    C = float(np.max((np.linalg.norm(dx, axis=1) - 4 * np.abs(a[:, 2])) / (a[:, 2] ** 2 + ss ** 2)))
    C2 = float(np.max(np.linalg.norm(dx + 4 * a[:, 2][:, None] * NI_F, axis=1) / (a[:, 2] ** 2 + ss ** 2)))
    print(f"[small s] displacement: C = {C:.3f}, C2 (normal part removed) = {C2:.3f} (sampled max over {m} labels, s <= {s0max})")
    # 3. for a cut M1: |M| >= M1 carried by W alone for s <= s0(M1) = 2 r_min M1^2/(A8 + 1); |M| < M1 by the margin
    #    on dD: Lipschitz constants over the rho(M1)-neighbourhood of dD, t in [1 - s0, 1]
    k = 400
    q = np.linspace(0, 4, k, endpoint=False)
    side = q.astype(int)
    w = 2 * (q - side) - 1
    el = np.where(side == 0, -1, np.where(side == 1, 1, w)) * h[0] + l8
    ph = np.where(side == 2, -1, np.where(side == 3, 1, w)) * h[1] + p8
    xd = np.exp(el)[:, None] * (np.cos(ph)[:, None] * EI_F + np.sin(ph)[:, None] * EZ_F)
    same = side[1:] == side[:-1]
    step = np.max(np.linalg.norm(np.diff(xd, axis=0), axis=1)[same])       # curve points within a side
    Yd = np.stack([np.cos(ph), np.sin(ph)], 1) @ np.stack([EI_F, EZ_F])
    UY = float(np.max(np.sum((1 - Yd @ UK_F.T) ** -0.5, 1)))
    print(f"   max U(Y0) on dD {UY:.3f}; r in [{rmin:.3f}, {rmax:.3f}]")
    print("   M1     s0         rho     Lx      Lxn     Lt      K = Lt + 4 r Lxn^2/(2 - Lx C2 s0) + Lx r C2 s0 + U-term,  m' needed = K s0")
    nb = 4
    gx = np.linspace(-1, 1, nb + 1)
    for M1 in (0.01, 0.02, 0.03, 0.05):
        s0 = 2 * rmin * M1 ** 2 / (A8 + 1.0)
        rho = rmax * (4 * M1 + C * (M1 ** 2 + s0 ** 2))
        Lx = Lxn = Lt = 0.0
        for i in range(nb):
            for j in range(nb):
                for kk in range(nb):
                    lo = xd + (rho + step) * np.array([gx[i], gx[j], gx[kk]])
                    hi = xd + (rho + step) * np.array([gx[i + 1], gx[j + 1], gx[kk + 1]])
                    T = I(np.full(k, 1 - s0), np.ones(k))
                    _, gS, St = co.enclose(T, [I(lo[:, c], hi[:, c]) for c in range(3)])
                    Lx = max(Lx, float(np.max(np.sqrt(sum(np.maximum(-g.lo, g.hi) ** 2 for g in gS)))))
                    gnn = sum(gS[c] * float(NI_F[c]) for c in range(3))
                    Lxn = max(Lxn, float(np.max(np.maximum(-gnn.lo, gnn.hi))))
                    Lt = max(Lt, float(np.max(np.maximum(-St.lo, St.hi))))
        # |M| < M1: (2/s - Lx C2) r M^2 - 4 r Lxn |M| >= -4 r Lxn^2 s / (2 - Lx C2 s0); the s^2 part gives Lx r C2 s0;
        # the U-term (Uhat >= 0 only) r^(-1/2) max U(Y0); it drops if 2 Uhat >= U(Y0) is enclosed (true: Uhat ~ U(Y0))
        K0 = Lt + 4 * rmax * Lxn ** 2 / (2 - Lx * C2 * s0) + Lx * rmax * C2 * s0
        K = K0 + rmin ** -0.5 * UY
        print(f"   {M1:.2f}   {s0:.2e}   {rho:.3f}   {Lx:.3f}   {Lxn:.3f}   {Lt:.3f}   K = {K:7.3f} (without the U-term "
              f"{K0:6.3f})      m' = {K * s0:.2e} ({K0 * s0:.2e})", flush=True)
    # 4. the true deficit
    print("   direct (floats, 41 x 41 per (ell, phi)-face): max_a [S(1, x0) - (S+ + W)(s, a)] / s")
    uv = np.linspace(-1, 1, 41)
    U_, V_ = np.meshgrid(uv, uv, indexing="ij")
    for s0 in (1e-4, 1e-3, 3e-3, 1e-2, 2e-2):
        Fq = np.repeat(np.arange(4), U_.size)
        a = tube.label(Fq, np.full(len(Fq), s0), np.tile(U_.ravel(), 4), np.tile(V_.ravel(), 4))
        o = phys_b(a, np.full(len(Fq), s0))
        S1, _, _ = co.enclose(I(np.full(len(Fq), 1 - s0)), [I(o[:, c].copy()) for c in range(3)], need_grad=False)
        r = np.exp(a[:, 0])
        x0 = r[:, None] * (np.cos(a[:, 1])[:, None] * EI_F + np.sin(a[:, 1])[:, None] * EZ_F)
        S0, _, _ = co.enclose(I(np.ones(len(Fq))), [I(x0[:, c].copy()) for c in range(3)], need_grad=False)
        f = np.maximum(S1.mid, 0) + o[:, 6] / s0
        d = (S0.mid - f) / s0
        print(f"     s = {s0:.0e}: max {d.max():+.3f} (face {Fq[np.argmax(d)]}, M = {a[np.argmax(d), 2]:+.4f}); "
              f"min over the faces: S+ + W - A8 {np.min(f) - A8:+.4e}, S(1, x0) - A8 {np.min(S0.mid) - A8:+.4e}")


# ------------------------------------------------------------------ the ellipsoidal label tube (follow-up task)
ELL_PAR = (0.30, 0.13, 0.10)
ELL_NPZ = OUT + "label_tube_ellip0.3_0.13_0.1.npz"
# cube-sphere faces for the angles on the label ellipsoid: (centre, A, B), orthonormal, right-handed
CS_FACES = [(np.array(c, float), np.array(a, float), np.array(b, float)) for c, a, b in (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (0, 0, -1), (1, 0, 0)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0)))]


def position_form(t, x8, fr, par=ELL_PAR):
    """E(t) of tools/words_eight_exit.py, mode ellip: T(t) = {x8 + 2|x8| V diag(par) u : |u| <= 1}, V = the slice-Hessian
    eigenvectors (soft first) at the node k = clip(round(48 t), 1, 47); E = V diag(par^-2) V^T / (2|x8|)^2"""
    k = int(np.clip(round(t * 48), 1, 47))
    V = fr[k]
    return V @ np.diag(1 / np.array(par) ** 2) @ V.T / (2 * np.linalg.norm(x8)) ** 2


def build_label_tube(par=ELL_PAR, nnode=41, path=ELL_NPZ):
    """Lambda(s) = {a : (a - a8(s))^T G(s) (a - a8(s)) <= 1}, a8(s) = (ell8, phi8, s mu8), G(s) = J(s)^T E(1 - s) J(s) at
    the nodes s_i = i/(nnode - 1), J = dx/da at the eight's label (time to go s); G piecewise linear in s between nodes"""
    from words_eight_exit import frames_hessian
    fr = frames_hessian()
    s = np.linspace(0.0, 1.0, nnode)
    c = Tube((0, 0, 0)).centre(s)
    d = node_data(c, s)
    J, x8 = d["dxda"], d["x"]
    E = np.stack([position_form(1 - si, x, fr, par) for si, x in zip(s, x8)])
    G = np.einsum("nij,njk,nkl->nil", J.transpose(0, 2, 1), E, J)
    G = 0.5 * (G + G.transpose(0, 2, 1))
    ev = np.linalg.eigvalsh(G)
    assert np.all(ev > 0)
    # rigorous positive definiteness at the nodes (Sylvester, intervals) -> on every segment (convex combinations)
    for g in G:
        m1 = I(g[0, 0])
        m2 = I(g[0, 0]) * g[1, 1] - I(g[0, 1]) * g[1, 0]
        gi = [[I(g[i, j]) for j in range(3)] for i in range(3)]
        m3 = gi[0][0] * (gi[1][1] * gi[2][2] - gi[1][2] * gi[2][1]) - gi[0][1] * (gi[1][0] * gi[2][2] - gi[1][2] * gi[2][0]) \
            + gi[0][2] * (gi[1][0] * gi[2][1] - gi[1][1] * gi[2][0])
        assert m1.lo > 0 and m2.lo > 0 and m3.lo > 0
    np.savez(path, s_nodes=s, a8=c, G=G, E=E, J=J, x8=x8, par=np.array(par), lab8=A8_LAB,
             definition="Lambda(s) = {a : (a - a8(s))^T G(s) (a - a8(s)) <= 1}, a = (ell, phi, M) scaled labels, "
                        "a8(s) = (ell8, phi8, s mu8); G linear in s between the nodes s_nodes; G = J^T E J at the nodes, "
                        "J = dx/da at a8(s), E the position ellipsoid of words_eight_exit.py (mode ellip, par)")
    print(f"label tube: {nnode} nodes, eigenvalues of G from {ev.min():.3f} to {ev.max():.1f}; semi-axes in labels "
          f"(1/sqrt(eig)) at s = 0: {np.sort(1 / np.sqrt(ev[0])).round(4)}, s = 0.5: {np.sort(1 / np.sqrt(ev[nnode // 2])).round(4)}, "
          f"s = 1: {np.sort(1 / np.sqrt(ev[-1])).round(4)}; positive definite at all nodes (interval Sylvester); saved {path}")
    return path


class EllTube:
    """the label-ellipsoid tube from the npz; surface parametrisation (two angles): face f of the cube sphere,
    (u, v) in [-1, 1]^2, w = normalize(C + tan(pi u/4) A + tan(pi v/4) B), and
        a(f, s, u, v) = a8(s) + rho S w,  S = diag(G_jj(s)^(-1/2)),  rho = (w^T S G(s) S w)^(-1/2)
    (radial parametrisation of the ellipsoid in the diagonally scaled label coordinates)"""

    def __init__(self, path=ELL_NPZ):
        d = np.load(path)
        self.s_nodes, self.G, self.lab8 = d["s_nodes"], d["G"], d["lab8"]
        self.path = path

    def centre(self, s):
        s = np.asarray(s, float)
        return np.stack([np.full_like(s, self.lab8[0]), np.full_like(s, self.lab8[1]), s * self.lab8[2]], -1)

    def Gs(self, s):
        s = np.asarray(s, float)
        i = np.clip(np.searchsorted(self.s_nodes, s, "right") - 1, 0, len(self.s_nodes) - 2)
        w = ((s - self.s_nodes[i]) / (self.s_nodes[i + 1] - self.s_nodes[i]))[..., None, None]
        return (1 - w) * self.G[i] + w * self.G[i + 1]

    def q(self, a, s):
        d = np.asarray(a) - self.centre(s)
        return np.einsum("...i,...ij,...j->...", d, self.Gs(s), d)

    def label(self, f, s, u, v):
        f = np.broadcast_to(np.asarray(f, int), np.shape(s))
        Cs = np.stack([CS_FACES[k][0] for k in range(6)])[f]
        As = np.stack([CS_FACES[k][1] for k in range(6)])[f]
        Bs = np.stack([CS_FACES[k][2] for k in range(6)])[f]
        w = Cs + np.tan(np.pi * np.asarray(u) / 4)[..., None] * As + np.tan(np.pi * np.asarray(v) / 4)[..., None] * Bs
        w /= np.linalg.norm(w, axis=-1)[..., None]
        G = self.Gs(s)
        Sd = np.diagonal(G, axis1=-2, axis2=-1) ** -0.5
        z = Sd * w
        rho = np.einsum("...i,...ij,...j->...", z, G, z) ** -0.5
        return self.centre(s) + rho[..., None] * z

    # ---- interval versions (rigorous up to the libm assumption of tan)
    def G_iv(self, s_lo, s_hi):
        """entrywise enclosure of G(s) for s in [s_lo, s_hi] (piecewise linear: hull of the end values and the nodes
        in between); arrays (m,) -> list of 3 x 3 interval arrays (m,)"""
        out = [[None] * 3 for _ in range(3)]
        for a, b in ((s_lo, None), (s_hi, None)):
            pass
        ends = []
        for sv in (s_lo, s_hi):
            i = np.clip(np.searchsorted(self.s_nodes, sv, "right") - 1, 0, len(self.s_nodes) - 2)
            w = (I(sv) - self.s_nodes[i]) / (I(self.s_nodes[i + 1]) - self.s_nodes[i])
            ends.append([[(1.0 - w) * self.G[i, r, c] + w * self.G[i + 1, r, c] for c in range(3)] for r in range(3)])
        for r in range(3):
            for c in range(3):
                lo = np.minimum(ends[0][r][c].lo, ends[1][r][c].lo)
                hi = np.maximum(ends[0][r][c].hi, ends[1][r][c].hi)
                inner = (self.s_nodes[None, :] > np.asarray(s_lo)[:, None]) & (self.s_nodes[None, :] < np.asarray(s_hi)[:, None])
                gv = self.G[:, r, c][None, :]
                lo = np.minimum(lo, np.where(inner, gv, np.inf).min(1))
                hi = np.maximum(hi, np.where(inner, gv, -np.inf).max(1))
                out[r][c] = I(lo, hi)
        return out

    def centre_iv(self, s_lo, s_hi):
        m = len(s_lo)
        M = I(s_lo, s_hi) * float(self.lab8[2])
        return [I(np.full(m, self.lab8[0])), I(np.full(m, self.lab8[1])), M]

    def label_box(self, f, s_lo, s_hi, u_lo, u_hi, v_lo, v_hi, nsub=4):
        """interval enclosure (a_lo, a_hi) of the surface patch {a(f, s, u, v)} over the box: the hull of the
        enclosures of nsub^3 sub-patches (cuts the dependency overestimation of the normalisation)"""
        f = np.asarray(f, int)
        lo, hi = None, None
        g = np.linspace(0, 1, nsub + 1)

        def cuts(lo_, hi_):         # exact end points; interior cuts shared by adjacent pieces
            c = [lo_ + gg * (hi_ - lo_) for gg in g]
            c[0], c[-1] = lo_, hi_
            return c
        cs, cu, cv = cuts(s_lo, s_hi), cuts(u_lo, u_hi), cuts(v_lo, v_hi)
        for i in range(nsub):
            for j in range(nsub):
                for k in range(nsub):
                    a, b = self._label_box1(f, cs[i], cs[i + 1], cu[j], cu[j + 1], cv[k], cv[k + 1])
                    lo = a if lo is None else np.minimum(lo, a)
                    hi = b if hi is None else np.maximum(hi, b)
        return lo, hi

    def _label_box1(self, f, s_lo, s_hi, u_lo, u_hi, v_lo, v_hi):
        """one interval enclosure of a patch (adjacent sub-pieces share their float cut points, so they cover the box)"""
        from eight_cell_verify import tan_iv, PI4
        m = len(s_lo)
        X, Y = tan_iv(I(u_lo, u_hi) * PI4), tan_iv(I(v_lo, v_hi) * PI4)
        Cs = np.stack([CS_FACES[k][0] for k in range(6)])[f]
        As = np.stack([CS_FACES[k][1] for k in range(6)])[f]
        Bs = np.stack([CS_FACES[k][2] for k in range(6)])[f]
        wr = [X * As[:, i] + Y * Bs[:, i] + Cs[:, i] for i in range(3)]
        nr = TB.inorm(wr)
        w = [x / nr for x in wr]
        w = [I(np.maximum(x.lo, -1.0), np.minimum(x.hi, 1.0)) for x in w]
        G = self.G_iv(s_lo, s_hi)
        Sd = [G[i][i].sqrt().recip() for i in range(3)]
        z = [Sd[i] * w[i] for i in range(3)]
        qq = None
        for i in range(3):
            for j in range(3):
                t = (z[i].sq() * G[i][i]) if i == j else z[i] * G[i][j] * z[j]
                qq = t if qq is None else qq + t
        qq = I(np.maximum(qq.lo, 1e-300), qq.hi)
        rho = qq.sqrt().recip()
        c = self.centre_iv(s_lo, s_hi)
        a = [c[i] + rho * z[i] for i in range(3)]
        return np.stack([x.lo for x in a], 1), np.stack([x.hi for x in a], 1)

    def box_certainly_inside(self, a_lo, a_hi, s_lo, s_hi):
        """True where every point of the label box is in Lambda(s) for every s in the slab: q convex, so the maximum
        over the box is at a corner; interval G and centre"""
        G = self.G_iv(s_lo, s_hi)
        c = self.centre_iv(s_lo, s_hi)
        ok = np.ones(len(s_lo), bool)
        for corner in range(8):
            d = [(I(a_hi[:, i]) if (corner >> i) & 1 else I(a_lo[:, i])) - c[i] for i in range(3)]
            q = None
            for i in range(3):
                for j in range(3):
                    t = (d[i].sq() * G[i][i]) if i == j else d[i] * G[i][j] * d[j]
                    q = t if q is None else q + t
            ok &= q.hi < 1
        return ok

    def box_certainly_outside(self, a_lo, a_hi, s_lo, s_hi):
        """True where no point of the label box is in Lambda(s) for any s in the slab. Cauchy-Schwarz in the G-inner
        product: for d in the ellipsoid and any vector y, (G y) . d <= sqrt(y^T G y); y = the float midpoint of
        D = box - centre; outside if min over D of (G y) . d > max sqrt(y^T G y) (interval G covers all s)"""
        G = self.G_iv(s_lo, s_hi)
        c = self.centre_iv(s_lo, s_hi)
        D = [I(a_lo[:, i], a_hi[:, i]) - c[i] for i in range(3)]
        y = np.stack([x.mid for x in D], 1)
        lhs = None
        yGy = None
        for i in range(3):
            Gy = None
            for j in range(3):
                t = G[i][j] * y[:, j]
                Gy = t if Gy is None else Gy + t
            t = Gy * D[i]
            lhs = t if lhs is None else lhs + t
            t = Gy * y[:, i]
            yGy = t if yGy is None else yGy + t
        yGy = I(np.maximum(yGy.lo, 0.0), np.maximum(yGy.hi, 0.0))
        return lhs.lo > yGy.sqrt().hi

    def disc_inside_test(self):
        """for the end condition: (ell, th) boxes certainly inside the open disc D = {(ell, phi): q(ell, phi, 0; s = 0) < 1}"""
        def test(boxes):
            n = len(boxes)
            z = np.zeros(n)
            return self.box_certainly_inside(np.stack([boxes[:, 0], boxes[:, 2], z], 1),
                                             np.stack([boxes[:, 1], boxes[:, 3], z], 1), z, z)
        return test


def label_tube(s, path=ELL_NPZ):
    """centre a8(s) and G(s) of the label-ellipsoid tube"""
    T = EllTube(path)
    return T.centre(s), T.Gs(s)


def elltube_check(cert_list, nt=21, nuv=7, path=ELL_NPZ, fit=False, ts=None, tube_path=None, tag=None):
    """floats: (1) mismatch of f_s(dLambda(s)) with the target surface q_E = 1 of the exit agent's position ellipsoid,
    (2) the exact tube gap V_E^f + V_I^f - A8 on dLambda(s) (E-field by inversion, words_eight_exit.invert),
    (3) margins S+ + W - A8 for the given E pieces"""
    from words_eight_exit import frames_hessian, invert as ex_invert, eight_track
    from words_eight_field import Field
    fr = frames_hessian()
    T = EllTubeFit(tube_path) if tube_path else (EllTubeFit() if fit else EllTube(path))
    lab = np.load(ROOT + "words/out/eight/field_labels.npz")
    FE = Field("E", lab["a8"])
    cos = [TB.CoarseE(c) for c in cert_list]
    uv = np.linspace(-1, 1, nuv)
    U_, V_ = np.meshgrid(uv, uv, indexing="ij")
    F = np.repeat(np.arange(6), U_.size)
    Uf, Vf = np.tile(U_.ravel(), 6), np.tile(V_.ravel(), 6)
    if ts is None:
        ts = np.concatenate([[0.01, 0.02, 0.05], np.linspace(0.1, 0.9, nt - 6), [0.95, 0.98, 0.99]])
    print(f"label-ellipsoid tube {os.path.basename(T.path)}: {len(F)} surface points per time, {len(ts)} times")
    print("   t      |sqrt(q_E) - 1| max   tube gap (V_E^f + V_I^f - A8) min   " + "   ".join(
        f"S+ + W - A8 ({os.path.basename(c)[:18]})" for c in cert_list))
    rows = []
    for t in ts:
        s = 1 - t
        sv = np.full(len(F), s)
        a = T.label(F, sv, Uf, Vf)
        o = phys_b(a, sv)
        x, W = o[:, 0:3], o[:, 6] / s
        x8 = phys_b(Tube((0, 0, 0)).centre(np.array([s])), np.array([s]))[0, 0:3]
        E = position_form(t, x8, fr)
        dq = np.abs(np.sqrt(np.einsum("ni,ij,nj->n", x - x8, E, x - x8)) - 1).max()
        # E-field values by continuation along the label rays
        Y8, J8 = eight_track(FE, t)
        J8i = np.linalg.inv(J8)
        labE = None
        for fr_ in (0.25, 0.5, 0.75, 1.0):
            ap = T.centre(sv) + fr_ * (a - T.centre(sv))
            y = phys_b(ap, sv)[:, 0:3]
            guess = labE if labE is not None else FE.lab8 + (y - Y8[0:3]) @ J8i.T
            labE, YE, JE, ok = ex_invert(FE, y, guess, t)
        VE = np.where(ok, YE[:, 6], np.nan)
        gap = np.nanmin(VE + W) - A8
        ms = []
        for co in cos:
            S, _, _ = co.enclose(I(np.full(len(F), t)), [I(x[:, c].copy()) for c in range(3)], need_grad=False)
            ms.append(np.min(np.maximum(S.mid, 0) + W) - A8)
        rows.append((t, dq, gap, *ms))
        print(f"   {t:.3f}   {dq:.4f}               {gap:+.4e} (inverted {ok.mean():.2f})          " +
              "            ".join(f"{m:+.4e}" for m in ms), flush=True)
    np.savetxt(OUT + (f"elltube_check_{tag}.tsv" if tag else ("elltube_check_fit.tsv" if fit else "elltube_check.tsv")), np.array(rows), delimiter="\t", fmt="%.6e",
               header="t\tmismatch\ttube_gap\t" + "\t".join(os.path.basename(c) for c in cert_list))
    return rows


def ell_small_s(path=ELL_NPZ, M1s=(0.01, 0.02), fit=False):
    """on dLambda(s), s -> 0: labels with |M| < M1 project to (ell, phi) near the disc boundary; how far inside D
    (q_section = q(ell, phi, 0; 0) < 1) can they be? (ii') then needs its margin on that band, not only on dD"""
    T = EllTubeFit() if fit else EllTube(path)
    uv = np.linspace(-1, 1, 201)
    U_, V_ = np.meshgrid(uv, uv, indexing="ij")
    F = np.repeat(np.arange(6), U_.size)
    for s in (0.0, 1e-3, 1e-2):
        a = T.label(F, np.full(len(F), s), np.tile(U_.ravel(), 6), np.tile(V_.ravel(), 6))
        a0 = a.copy()
        a0[:, 2] = 0.0
        qs = T.q(a0, np.zeros(len(F)))
        for M1 in M1s:
            sel = np.abs(a[:, 2]) < M1
            print(f"   s = {s:.0e}, |M| < {M1}: section form q(ell, phi, 0) of the projection in [{qs[sel].min():.4f}, "
                  f"{qs[sel].max():.4f}] (1 = dD; < 1 inside D)")


ELL_FIT_NPZ = OUT + "label_tube_ellip0.3_0.13_0.1_fit.npz"


def invert_labels(y, s, a0, iters=9):
    """scaled labels a with f_s(a) = y (Newton, finite-difference Jacobian; floats)"""
    a = a0.copy()
    ok = np.zeros(len(s), bool)
    for _ in range(iters):
        d = node_data(a, s)
        r = d["x"] - y
        conv = np.max(np.abs(r), 1) < 1e-10
        ok |= conv
        step = np.linalg.solve(d["dxda"], r[:, :, None])[:, :, 0]
        step = np.where(conv[:, None], 0.0, step)
        nrm = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, 0.3 / np.maximum(nrm, 1e-300))[:, None]
        a = a - step
    d = node_data(a, s)
    ok = np.max(np.abs(d["x"] - y), 1) < 1e-9
    return a, ok


def fit_quadric(P):
    """ellipsoid {a : (a - c)^T G (a - c) = 1} fitted to points P (m, 3) (algebraic least squares on the quadric
    a^T A a + b . a + k = 0, smallest right singular vector, columns scaled)"""
    sc = P.std(0)
    mu = P.mean(0)
    Z = (P - mu) / sc
    x, y, z = Z.T
    D = np.stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, x, y, z, np.ones(len(x))], 1)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    p = Vt[-1]
    A = np.array([[p[0], p[3], p[4]], [p[3], p[1], p[5]], [p[4], p[5], p[2]]])
    b, k = p[6:9], p[9]
    cz = -0.5 * np.linalg.solve(A, b)
    val = cz @ A @ cz - k
    Gz = A / val
    if np.any(np.linalg.eigvalsh(Gz) <= 0):
        raise RuntimeError("fitted quadric is not an ellipsoid")
    Si = np.diag(1 / sc)
    return mu + sc * cz, Si @ Gz @ Si


def build_label_tube_fit(par=ELL_PAR, nnode=41, ndir=200, path=ELL_FIT_NPZ):
    """variant: per node, the exact label preimage of the target position ellipsoid surface (ndir Fibonacci directions)
    and a least-squares label ellipsoid through it (own centre c(s)); c and G linear in s between nodes"""
    from words_eight_exit import frames_hessian
    from words_eight_field import fib
    fr = frames_hessian()
    s_nodes = np.linspace(0.0, 1.0, nnode)
    U = fib(ndir)
    C, G, res = [], [], []
    lin = np.load(ELL_NPZ)
    for i, s in enumerate(s_nodes):
        t = 1 - s
        a8 = Tube((0, 0, 0)).centre(np.array([s]))[0]
        x8 = phys_b(a8[None], np.array([s]))[0, 0:3]
        k = int(np.clip(round(t * 48), 1, 47))
        V = fr[k]
        y = x8 + 2 * np.linalg.norm(x8) * (U * np.array(par)) @ V.T
        # continuation from the linear label ellipsoid's guess along each ray
        Gl = lin["G"][i]
        a = None
        for f in (0.5, 1.0):
            yf = x8 + f * (y - x8)
            g = a if a is not None else a8 + f * np.linalg.solve(lin["J"][i], (y - x8).T).T
            a, ok = invert_labels(yf, np.full(ndir, s), g)
        if not ok.all():
            raise RuntimeError(f"inversion failed at s = {s}: {np.sum(~ok)} points")
        c, Gf = fit_quadric(a)
        r = np.einsum("ni,ij,nj->n", a - c, Gf, a - c)
        C.append(c); G.append(Gf); res.append(np.abs(np.sqrt(r) - 1).max())
        print(f"  s = {s:.3f}: centre offset from a8(s) {np.round(c - a8, 4)}, semi-axes {np.sort(1 / np.sqrt(np.linalg.eigvalsh(Gf))).round(4)}, "
              f"fit residual max |sqrt(q) - 1| {res[-1]:.4f}, a8 inside: q(a8) = {(a8 - c) @ Gf @ (a8 - c):.3f}", flush=True)
    C, G = np.array(C), np.array(G)
    np.savez(path, s_nodes=s_nodes, a8=Tube((0, 0, 0)).centre(s_nodes), centre=C, G=G, par=np.array(par), lab8=A8_LAB,
             definition="Lambda(s) = {a : (a - c(s))^T G(s) (a - c(s)) <= 1}, c and G linear in s between the nodes; "
                        "fitted to the exact label preimage of the position ellipsoid of words_eight_exit.py (mode ellip)")
    print(f"saved {path}; max fit residual {max(res):.4f}")
    return path


class EllTubeFit(EllTube):
    """the fitted variant: own centre c(s) (linear between nodes) instead of a8(s)"""

    def __init__(self, path=ELL_FIT_NPZ):
        super().__init__(path)
        self.C = np.load(path)["centre"]

    def centre(self, s):
        s = np.asarray(s, float)
        i = np.clip(np.searchsorted(self.s_nodes, s, "right") - 1, 0, len(self.s_nodes) - 2)
        w = ((s - self.s_nodes[i]) / (self.s_nodes[i + 1] - self.s_nodes[i]))[..., None]
        return (1 - w) * self.C[i] + w * self.C[i + 1]

    def centre_iv(self, s_lo, s_hi):
        out = []
        for k in range(3):
            lo, hi = np.full(len(s_lo), np.inf), np.full(len(s_lo), -np.inf)
            for sv in (s_lo, s_hi):
                i = np.clip(np.searchsorted(self.s_nodes, sv, "right") - 1, 0, len(self.s_nodes) - 2)
                w = (I(sv) - self.s_nodes[i]) / (I(self.s_nodes[i + 1]) - self.s_nodes[i])
                v = (1.0 - w) * self.C[i, k] + w * self.C[i + 1, k]
                lo, hi = np.minimum(lo, v.lo), np.maximum(hi, v.hi)
            inner = (self.s_nodes[None, :] > np.asarray(s_lo)[:, None]) & (self.s_nodes[None, :] < np.asarray(s_hi)[:, None])
            lo = np.minimum(lo, np.where(inner, self.C[:, k][None, :], np.inf).min(1))
            hi = np.maximum(hi, np.where(inner, self.C[:, k][None, :], -np.inf).max(1))
            out.append(I(lo, hi))
        return out


def export_targets_ellfit(path=ROOT + "words/out/eight/tube_targets_ellfit.npz", ntime=40, nuv=6, nring=200,
                          band=(0.94, 0.96, 0.98, 1.0, 1.007), tube_path=None):
    """fit targets on the boundary of the fitted label ellipsoid Lambda(s) (EllTubeFit), same format as
    words/out/eight/tube_targets_*.npz. Rows t < 1: x = f_s(a) for a on dLambda(s) (cube-sphere angles, 6 faces x
    nuv x nuv cell centres), VI = W(s, a) (the I-field action to I, absolute). Rows t = 1 (VI = 0): the disc edge
    dD (q = 1 on the M = 0 section of Lambda(0)) and the band q in [0.94, 1.007] (section form), nring points each."""
    T = EllTubeFit(tube_path) if tube_path else EllTubeFit()
    s_t = np.unique(np.concatenate([np.geomspace(1e-4, 0.05, 15), np.linspace(0.05, 0.98, 26)[1:]]))
    assert len(s_t) == ntime
    g = -1 + (2 * np.arange(nuv) + 1) / nuv
    U_, V_ = np.meshgrid(g, g, indexing="ij")
    F = np.repeat(np.arange(6), U_.size)
    Uf, Vf = np.tile(U_.ravel(), 6), np.tile(V_.ravel(), 6)
    Ts, Xs, VIs, Ss = [], [], [], []
    for s in s_t:
        sv = np.full(len(F), s)
        a = T.label(F, sv, Uf, Vf)
        o = phys_b(a, sv)
        Ts.append(np.full(len(F), 1 - s)); Xs.append(o[:, 0:3]); VIs.append(o[:, 6] / s)
    # t = 1 rows: section ellipse of Lambda(0) at M = 0, scaled to the band values of q
    c0 = T.centre(np.array([0.0]))[0]
    G0 = T.G[0]
    G2 = G0[:2, :2]
    m = -c0[2]
    cen2 = c0[:2] - np.linalg.solve(G2, G0[:2, 2] * m)
    rhs = 1 - (G0[2, 2] * m * m - (G0[:2, 2] * m) @ np.linalg.solve(G2, G0[:2, 2] * m))
    L = np.linalg.cholesky(np.linalg.inv(G2))
    th = np.linspace(0, 2 * np.pi, nring, endpoint=False)
    unit = (L @ np.stack([np.cos(th), np.sin(th)])).T * math.sqrt(rhs)       # q = 1 on the section
    for qv in band:
        dd = unit * math.sqrt(qv)
        el, ph = cen2[0] + dd[:, 0], cen2[1] + dd[:, 1]
        x = np.exp(el)[:, None] * (np.cos(ph)[:, None] * EI_F + np.sin(ph)[:, None] * EZ_F)
        Ts.append(np.ones(nring)); Xs.append(x); VIs.append(np.zeros(nring))
    t, x, VI = np.concatenate(Ts), np.vstack(Xs), np.concatenate(VIs)
    keep = np.isfinite(VI) & np.isfinite(x).all(1)
    np.savez(path, t=t[keep], x=x[keep], VI=VI[keep], A8=A8,
             convention="x: plain shape-space coordinates (Hopf map, |x| = moment of inertia), time t in [0, 1] (1/12 "
                        "time tau = 1); VI = V_I^f(t, x) = W(s, a), the eight's I-field action from (t, x) to I along the "
                        "field extremal of the boundary label a (absolute); last-exit constraint S_E^c(t, x) >= A8 + "
                        "margin - VI, S_E^c(t, x) = t^(1/3) (w0_E + phi)(x / t^(4/3)); rows with t < 1: x = f_s(a), a on "
                        "the boundary of the fitted label ellipsoid Lambda(s) (label_tube_ellip0.3_0.13_0.1_fit.npz), "
                        "s = 1 - t; rows with t = 1 (VI = 0): points of I on the disc edge dD (section form q = 1 of "
                        "Lambda(0) at M = 0) and on the band q in {0.94, 0.96, 0.98, 1.007} next to it (the small-s "
                        "lemma needs the (ii') margin there); the mirror (x_3 -> -x_3) is implied by the evenness of S_E^c",
             tube=os.path.basename(T.path), n_dropped=int((~keep).sum()),
             s_times=s_t, band=np.array(band))
    print(f"wrote {path}: {keep.sum()} rows ({len(s_t)} times x {len(F)} boundary labels + {len(band)} x {nring} rows "
          f"at t = 1); t in [{t.min():.4f}, {t.max()}], VI in [{VI[t < 1].min():.4f}, {VI[t < 1].max():.2f}]")
    return path


ELL_TAPER_NPZ = OUT + "label_tube_ellip0.3_0.13_0.1_fit_taper.npz"


def beta_taper(t, b0=0.6, t0=0.1, t1=0.2):
    """scale factor of the tapered tube: b0 for t <= t0, linear to 1 at t1, 1 beyond"""
    t = np.asarray(t, float)
    return np.where(t <= t0, b0, np.where(t >= t1, 1.0, b0 + (1 - b0) * (t - t0) / (t1 - t0)))


def build_taper(b0=0.6, t0=0.1, t1=0.2, src=ELL_FIT_NPZ, path=ELL_TAPER_NPZ):
    """Lambda_tap(s) = c(s) + beta(1 - s) (Lambda(s) - c(s)) for the fitted label ellipsoid, i.e. G_tap = G / beta^2 at
    the nodes (same centres), linear in s between nodes like the parent; t0 and t1 must be nodes"""
    d = np.load(src, allow_pickle=True)
    s = d["s_nodes"]
    for tb in (t0, t1):
        assert np.min(np.abs((1 - s) - tb)) < 1e-12, "taper breakpoints must be nodes"
    beta = beta_taper(1 - s, b0, t0, t1)
    G = d["G"] / beta[:, None, None] ** 2
    np.savez(path, s_nodes=s, a8=d["a8"], centre=d["centre"], G=G, par=d["par"], lab8=d["lab8"], beta=beta,
             taper=np.array([b0, t0, t1]),
             definition=f"tapered fitted label ellipsoid: Lambda_tap(s) = {{a : (a - c(s))^T G(s) (a - c(s)) <= 1}}, "
                        f"G = G_fit / beta^2 at the nodes, beta = {b0} for t <= {t0}, linear to 1 at t = {t1}, 1 beyond "
                        f"(t = 1 - s); c, G linear in s between the nodes; parent {os.path.basename(src)}")
    print(f"wrote {path}: beta {b0} for t <= {t0}, linear to 1 at t = {t1}")
    return path


def goal_on_targets(target_path, npz=ROOT + "words/out/eight/block/block2_E_goalEfit.npz", json_path=None,
                    m_tube=0.005, m_ring=0.003):
    """float margins of an E piece on a target file: rows t < 1: S_E^c + VI - A8 - m_tube; t = 1: S_E^c - A8 - m_ring.
    The piece: block2 coefficients (npz, Q(p,1,1)) or a verifier JSON"""
    d = np.load(target_path, allow_pickle=True)
    t, x, VI = d["t"], d["x"], d["VI"]
    if json_path:
        co = TB.CoarseE(json_path)
        S = np.empty(len(t))
        for c0 in range(0, len(t), 5000):
            sl = slice(c0, c0 + 5000)
            Sv, _, _ = co.enclose(I(t[sl].copy()), [I(x[sl, k].copy()) for k in range(3)], need_grad=False)
            S[sl] = Sv.mid
        name = os.path.basename(json_path)
    else:
        import words_eight_block2 as B2
        e = np.load(npz, allow_pickle=True)
        mesh = B2.make_mesh("E", int(e["nu"]), str(e["ref"]), int(e["p"]))
        S = np.empty(len(t))
        for tv in np.unique(t):
            sel = t == tv
            S[sel] = B2.S_value(mesh, e["coef"], "E", x[sel], tv)
        name = os.path.basename(npz)
    marg = np.where(t < 1, np.maximum(S, 0) + VI - A8 - m_tube, S - A8 - m_ring)
    print(f"piece {name} on {os.path.basename(target_path)} ({len(t)} rows): net margin (after the row margins "
          f"{m_tube} / {m_ring}) min {marg.min():+.4e}; rows below 0: {np.sum(marg < 0)}")
    rows = []
    for tv in np.unique(t):
        sel = t == tv
        rows.append((tv, marg[sel].min(), (np.maximum(S, 0) + VI - A8)[sel].min()))
    return np.array(rows), marg


def band_margin(cert, q_in=0.94, q_out=1.007, target=0.003, depth=10, n=64, tube_path=ELL_FIT_NPZ, A8_hi=None):
    """rigorous lower bound of S_E^c(1, x) - A8 on the band {q_in <= q <= q_out} of the M = 0 section of Lambda(0)
    (x = rho (cos th e_I + sin th e_Z), (log rho, th) labels), needed by the small-s lemma: boxes in (ell, th) are kept
    while they may meet the band (not certainly inside the q_in ellipse, not certainly outside the q_out ellipse) and
    bisected until their lower bound reaches `target`; returns the minimum lower bound over the final cover"""
    import copy
    co = TB.CoarseE(cert)
    T = EllTubeFit(tube_path)
    Tin, Tout = copy.copy(T), copy.copy(T)
    Tin.G = T.G / q_in
    Tout.G = T.G / q_out
    c0 = T.centre(np.array([0.0]))[0]
    sax = 1 / np.sqrt(np.linalg.eigvalsh(T.G[0][:2, :2])).min()
    R = 1.2 / np.sqrt(np.linalg.eigvalsh(T.G[0][:2, :2]).min())
    e = np.linspace(c0[0] - R, c0[0] + R, n + 1)
    h = np.linspace(c0[1] - R, c0[1] + R, n + 1)
    E0, H0 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    boxes = np.stack([e[E0.ravel()], e[E0.ravel() + 1], h[H0.ravel()], h[H0.ravel() + 1]], 1)

    def emb(b):
        z = np.zeros(len(b))
        return np.stack([b[:, 0], b[:, 2], z], 1), np.stack([b[:, 1], b[:, 3], z], 1), z, z
    worst, nbox, fails = np.inf, 0, []
    for dep in range(depth + 1):
        lo, hi, s0, s1 = emb(boxes)
        keep = ~Tin.box_certainly_inside(lo, hi, s0, s1) & ~Tout.box_certainly_outside(lo, hi, s0, s1)
        boxes = boxes[keep]
        if not len(boxes):
            break
        nbox += len(boxes)
        lb, fc = TB.end_lower(co, boxes, A8_hi if A8_hi is not None else TB.A8_HI)
        good = lb >= target
        if good.any():
            worst = min(worst, float(lb[good].min()))
        if dep == depth:
            fails = list(zip(boxes[~good].tolist(), lb[~good], fc[~good]))
            worst = min(worst, float(lb.min()))
            break
        bad = boxes[~good]
        m0, m1 = 0.5 * (bad[:, 0] + bad[:, 1]), 0.5 * (bad[:, 2] + bad[:, 3])
        boxes = np.vstack([np.stack([bad[:, 0], m0, bad[:, 2], m1], 1), np.stack([m0, bad[:, 1], bad[:, 2], m1], 1),
                           np.stack([bad[:, 0], m0, m1, bad[:, 3]], 1), np.stack([m0, bad[:, 1], m1, bad[:, 3]], 1)])
        if not len(boxes):
            break
    print(f"band q in [{q_in}, {q_out}] of the disc of {os.path.basename(tube_path)} (A8_hi = {A8_hi if A8_hi is not None else TB.A8_HI!r}): {nbox} boxes; rigorous min of "
          f"S_E^c(1, x) - A8 over the cover {worst:+.4e} (target {target}); boxes below target at depth {depth}: {len(fails)}"
          + (f", min float centre value {min(f[2] for f in fails):+.4e}" if fails else ""))
    return worst, fails


def small_s_ell(cert, tube_path=ELL_TAPER_NPZ, M1s=(0.005, 0.0075, 0.01), s0max=0.02):
    """the small-s lemma (eight_lift_referee.md C1, section 5 of eight_tube_boundary.md) on the label-ellipsoid tube.
    For a on dLambda(s), s <= s0: with x0 = f_0(ell, phi, 0) (a point of I),
        S(1 - s, f_s(a)) + W >= S(1, x0) - Lt s - Lxn |n_I . (f_s(a) - x0)| - Lx |rest| + 2 r M^2/s + s r^(-1/2)(2 Uhat - U(Y0)).
    |M| >= M1: W alone >= A8 + 1 for s <= s0(M1) = 2 r_min M1^2/(A8 + 1) (minus the U-term, checked below).
    |M| < M1: (ell, phi) projects to the band q in [q_lo, q_hi] of the disc (measured), where (ii') gives
    S(1, x0) >= A8 + m'; minimising over M gives S+ + W - A8 >= m' - K s0 with
        K = Lt + 4 r_max Lxn^2/(2 - Lx C2 s0) + Lx r_max C2 s0 + r_min^(-1/2) max U(Y0).
    Lx, Lxn, Lt: rigorous enclosures over boxes covering the rho-neighbourhood of the band, t in [1 - s0, 1];
    C, C2 and the projection range: floats on sampled labels. Returns rows (M1, s0, q-range, K, needed m')."""
    import copy
    co = TB.CoarseE(cert)
    T = EllTubeFit(tube_path)
    rng = np.random.default_rng(13)
    # surface labels for s <= s0max: projection range, r range, displacement constants
    m = 30000
    F = rng.integers(0, 6, m)
    ss = np.concatenate([10 ** rng.uniform(-6, math.log10(s0max), m // 2), rng.uniform(0, s0max, m - m // 2)])
    a = T.label(F, ss, rng.uniform(-1, 1, m), rng.uniform(-1, 1, m))
    a0 = a.copy()
    a0[:, 2] = 0.0
    qs = T.q(a0, np.zeros(m))
    r = np.exp(a[:, 0])
    rmin, rmax = float(r.min()), float(r.max())
    o = phys_b(a, ss)
    x0 = r[:, None] * (np.cos(a[:, 1])[:, None] * EI_F + np.sin(a[:, 1])[:, None] * EZ_F)
    dx = (o[:, 0:3] - x0) / r[:, None]
    C = float(np.max((np.linalg.norm(dx, axis=1) - 4 * np.abs(a[:, 2])) / (a[:, 2] ** 2 + ss ** 2)))
    C2 = float(np.max(np.linalg.norm(dx + 4 * a[:, 2][:, None] * NI_F, axis=1) / (a[:, 2] ** 2 + ss ** 2)))
    Wid = 2 * r * a[:, 2] ** 2 / ss
    print(f"[small s, ellipse disc] tube {os.path.basename(tube_path)}, piece {os.path.basename(cert)}: surface labels "
          f"s <= {s0max}: r in [{rmin:.3f}, {rmax:.3f}], C = {C:.3f}, C2 = {C2:.3f} (floats, {m} labels)")
    # section ellipse of Lambda(0) at M = 0 (as in endell)
    c0 = T.centre(np.array([0.0]))[0]
    G0 = T.G[0]
    G2 = G0[:2, :2]
    mm = -c0[2]
    cen2 = c0[:2] - np.linalg.solve(G2, G0[:2, 2] * mm)
    rhs = 1 - (G0[2, 2] * mm * mm - (G0[:2, 2] * mm) @ np.linalg.solve(G2, G0[:2, 2] * mm))
    L = np.linalg.cholesky(np.linalg.inv(G2))
    k = 300
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    unit = (L @ np.stack([np.cos(th), np.sin(th)])).T * math.sqrt(rhs)
    Yd = np.stack([np.cos(cen2[1] + unit[:, 1]), np.sin(cen2[1] + unit[:, 1])], 1) @ np.stack([EI_F, EZ_F])
    UY = float(np.max(np.sum((1 - Yd @ UK_F.T) ** -0.5, 1))) * 1.05     # small safety for the band
    rows = []
    print("   M1       s0         q range (|M| < M1)   rho     Lx      Lxn     Lt      K (no U-term)     needed m' = K s0")
    for M1 in M1s:
        s0 = 2 * rmin * M1 ** 2 / (A8 + 1.0)
        sel = (np.abs(a[:, 2]) < M1) & (ss <= s0)
        qlo, qhi = (float(qs[sel].min()), float(qs[sel].max())) if sel.any() else (1.0, 1.0)
        rho = rmax * (4 * M1 + C * (M1 ** 2 + s0 ** 2))
        # box cover: band curves q in {qlo, 1, qhi} plus between, each point a box of half-width rho + spacing
        Lx = Lxn = Lt = 0.0
        for qv in np.linspace(qlo, qhi, 5):
            dd = unit * math.sqrt(qv)
            el, ph = cen2[0] + dd[:, 0], cen2[1] + dd[:, 1]
            xd = np.exp(el)[:, None] * (np.cos(ph)[:, None] * EI_F + np.sin(ph)[:, None] * EZ_F)
            step = np.max(np.linalg.norm(np.diff(np.vstack([xd, xd[:1]]), axis=0), axis=1))
            nb = 3
            gx = np.linspace(-1, 1, nb + 1)
            for i in range(nb):
                for j in range(nb):
                    for l in range(nb):
                        lo = xd + (rho + step) * np.array([gx[i], gx[j], gx[l]])
                        hi = xd + (rho + step) * np.array([gx[i + 1], gx[j + 1], gx[l + 1]])
                        _, gS, St = co.enclose(I(np.full(k, 1 - s0)), [I(lo[:, c], hi[:, c]) for c in range(3)])
                        Lx = max(Lx, float(np.max(np.sqrt(sum(np.maximum(-g.lo, g.hi) ** 2 for g in gS)))))
                        gnn = sum(gS[c] * float(NI_F[c]) for c in range(3))
                        Lxn = max(Lxn, float(np.max(np.maximum(-gnn.lo, gnn.hi))))
                        Lt = max(Lt, float(np.max(np.maximum(-St.lo, St.hi))))
        K0 = Lt + 4 * rmax * Lxn ** 2 / (2 - Lx * C2 * s0) + Lx * rmax * C2 * s0
        K = K0 + rmin ** -0.5 * UY
        rows.append((M1, s0, qlo, qhi, Lx, Lxn, Lt, K, K0, K * s0))
        print(f"   {M1:.4f}   {s0:.3e}   [{qlo:.4f}, {qhi:.4f}]     {rho:.3f}   {Lx:.3f}   {Lxn:.3f}   {Lt:.3f}   "
              f"{K:6.2f} ({K0:5.2f})     {K * s0:.2e} ({K0 * s0:.2e})", flush=True)
    return rows


def small_s_rig(cert, tube_path=ELL_TAPER_NPZ, M1=0.01, s0=1.165e-4, nuv=96, phi_w=0.03, q_band=(0.94, 1.01)):
    """rigorous closure of the small-s lemma (section 10.8 item 5): everything enclosed with intervals.
    (a) projection: surface patches (cube-sphere, nuv x nuv per face, s in [0, s0], interval label_box) that may have
        |M| < M1; their (ell, phi) ranges -> interval section form q(ell, phi, 0) at s = 0 -> q range; r range from
        all patches (for the |M| >= M1 part) and from the kept ones.
    (b) displacement: C1 interval Taylor flow of verify/eight_field.py with jets in (eps, phi, M) on boxes
        eps in [0, eps_max], phi slices of width phi_w, M in [-M1, M1]; mean value from (0, phi, 0):
        |n_I . (Phi - Y0)| <= a_n |M| + b_n eps, |P_T (Phi - Y0)| <= a_T |M| + b_T eps (P_T = I - n n^T).
    (c) Lx, Lxn, Lt over the x-boxes of the kept patches' x0 = e^ell (cos phi e_I + sin phi e_Z), inflated by
        rho = r_max (max(a_n, a_T) ... ) (the Euclidean bound of |x - x0|), t in [1 - s0, 1].
    Then for |M| < M1, s <= s0 (x = f_s(a) = r Phi):
        S+ + W - A8 >= m' - Lt s - r (Lxn a_n + Lx a_T)|M| - r (Lxn b_n + Lx b_T) eps + 2 r M^2/s - s r^(-1/2) U(Y0)
                    >= m' - s0 [Lt + r_max (Lxn a_n + Lx a_T)^2/8 + s0 r_min^(-1/2)(Lxn b_n + Lx b_T) + r_min^(-1/2) Umax]
    (eps = s^2 r^(-3/2) <= s s0 r^(-3/2)); for |M| >= M1: W >= 2 r_min M1^2/s - s r_min^(-1/2) Umax >= A8 + 1."""
    import eight_field as EF
    from prove_newton_1212 import stack, _i, dn, up
    co = TB.CoarseE(cert)
    T = EllTubeFit(tube_path)
    t0 = time.time()
    # (a) patches, adaptively bisected in (u, v) (s-slab [0, s0] fixed)
    g = np.linspace(-1, 1, nuv + 1)
    P = np.array([[f, g[i], g[i + 1], g[j], g[j + 1]] for f in range(6) for i in range(nuv) for j in range(nuv)])

    def enc(Pb):
        m_ = len(Pb)
        return T.label_box(Pb[:, 0].astype(int), np.zeros(m_), np.full(m_, s0), Pb[:, 1], Pb[:, 2], Pb[:, 3], Pb[:, 4])

    def qenc(lo_, hi_):
        m_ = len(lo_)
        c0 = T.centre_iv(np.zeros(m_), np.zeros(m_))
        G0 = T.G_iv(np.zeros(m_), np.zeros(m_))
        dd = [I(lo_[:, 0], hi_[:, 0]) - c0[0], I(lo_[:, 1], hi_[:, 1]) - c0[1], I(np.zeros(m_)) - c0[2]]
        qq = None
        for i in range(3):
            for j in range(3):
                tq = (dd[i].sq() * G0[i][i]) if i == j else dd[i] * G0[i][j] * dd[j]
                qq = tq if qq is None else qq + tq
        return qq
    rmin_all = np.inf
    acc_lo, acc_hi, und_lo, und_hi = [], [], [], []
    qlo, qhi = np.inf, -np.inf
    for dep in range(10):
        alo, ahi = enc(P)
        if dep == 0:
            rmin_all = float(dn(np.exp(alo[:, 0].min())))
        keep = ~((alo[:, 2] >= M1) | (ahi[:, 2] <= -M1))
        P, alo, ahi = P[keep], alo[keep], ahi[keep]
        qq = qenc(alo, ahi)
        ok = (qq.lo >= q_band[0]) & (qq.hi <= q_band[1])
        acc_lo.append(alo[ok]); acc_hi.append(ahi[ok])
        if ok.any():
            qlo, qhi = min(qlo, float(qq.lo[ok].min())), max(qhi, float(qq.hi[ok].max()))
        P, alo, ahi = P[~ok], alo[~ok], ahi[~ok]
        print(f"   (a) depth {dep}: accepted {ok.sum()}, to split {len(P)}", flush=True)
        if not len(P):
            break
        if dep == 9:
            und_lo.append(alo); und_hi.append(ahi)
            qlo, qhi = min(qlo, float(qq.lo[~ok].min())), max(qhi, float(qq.hi[~ok].max()))
            break
        mu, mv = 0.5 * (P[:, 1] + P[:, 2]), 0.5 * (P[:, 3] + P[:, 4])
        P = np.vstack([np.stack([P[:, 0], P[:, 1], mu, P[:, 3], mv], 1), np.stack([P[:, 0], mu, P[:, 2], P[:, 3], mv], 1),
                       np.stack([P[:, 0], P[:, 1], mu, mv, P[:, 4]], 1), np.stack([P[:, 0], mu, P[:, 2], mv, P[:, 4]], 1)])
    lo = np.vstack(acc_lo + und_lo)
    hi = np.vstack(acc_hi + und_hi)
    rmin, rmax = float(dn(np.exp(lo[:, 0].min()))), float(up(np.exp(hi[:, 0].max())))
    phlo, phhi = float(lo[:, 1].min()), float(hi[:, 1].max())
    nund = sum(len(x) for x in und_lo)
    print(f"[rigorous small s] (a) {len(lo)} patches that may have |M| < {M1} (s in [0, {s0}]): section form q in "
          f"[{qlo:.4f}, {qhi:.4f}], inside the certified band {q_band}: {nund == 0}; r over all patches >= "
          f"{rmin_all:.4f}; kept: r in [{rmin:.4f}, {rmax:.4f}], phi in [{phlo:.4f}, {phhi:.4f}]  ({time.time() - t0:.0f} s)",
          flush=True)
    # U(Y0) max over the kept phi range (rigorous, U_k = (1 - u_k . Y0)^(-1/2), Y0 = cos e_I + sin e_Z)
    ph = I(phlo, phhi)
    sn, cn = EF.isincos(ph)
    Umax = 0.0
    for u in TB.UKI:
        dq = I(1.0) - (cn * (u[0] * 0.5 + u[1] * TB.S3H) + sn * u[2])
        Umax += float(up(1.0 / dn(np.sqrt(dn(max(float(dq.lo), 1e-300))))))
    # (b) displacement constants
    eps_max = float(up(up(s0 * s0) * up(rmin ** -1.5)))
    nI = [float(v) for v in NI_F]
    a_n = a_T = b_n = b_T = 0.0
    edges = np.arange(phlo, phhi + phi_w, phi_w)
    edges[-1] = max(edges[-1], phhi)
    for k in range(len(edges) - 1):
        Ph = I(dn(edges[k]), up(edges[k + 1]))
        sn, cn = EF.isincos(Ph)

        def J(val, dd):
            return stack([_i(val)] + [_i(x) for x in dd], 0)
        Mm = I(-M1, M1)
        Y = [J(cn * EF.EI[i] + sn * EF.EZ[i], [I(0.0), -sn * EF.EI[i] + cn * EF.EZ[i], I(0.0)]) for i in range(3)]
        P = [J(Mm * EF.NI[i], [I(0.0), I(0.0), EF.NI[i]]) for i in range(3)]
        z = J(I(0.0), [I(0.0)] * 3)
        E = J(I(0.0, eps_max), [I(1.0), I(0.0), I(0.0)])
        yT, _ = EF.flow_unit(Y + P + [z, z], E)
        for col, which in ((3, "M"), (1, "eps")):
            D = [yT[i][col] for i in range(3)]
            nd = D[0] * nI[0] + D[1] * nI[1] + D[2] * nI[2]
            an = float(max(-nd.lo, nd.hi))
            tv = [D[i] - nd * nI[i] for i in range(3)]
            at = float(up(np.sqrt(up(sum(up(max(-x.lo, x.hi) ** 2) for x in tv)))))
            if which == "M":
                a_n, a_T = max(a_n, an), max(a_T, at)
            else:
                b_n, b_T = max(b_n, an), max(b_T, at)
    print(f"   (b) interval flow on {len(edges) - 1} phi slices, eps in [0, {eps_max:.2e}], M in [-{M1}, {M1}]: "
          f"a_n = {a_n:.4f}, a_T = {a_T:.4f}, b_n = {b_n:.3f}, b_T = {b_T:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    # (c) Lipschitz constants over the inflated x0 boxes
    rho = float(up(rmax * (max(a_n, 0) + a_T) * M1 + rmax * (b_n + b_T) * eps_max))
    L = I(lo[:, 0], hi[:, 0])
    Pp = I(lo[:, 1], hi[:, 1])
    rr = TB.iexp(L)
    sn, cn = EF.isincos(Pp)
    X0 = [rr * (cn * TB.EI_I[i] + sn * TB.EZ_I[i]) for i in range(3)]
    Xb = [I(dn(x.lo - rho), up(x.hi + rho)) for x in X0]
    Lx = Lxn = Lt = 0.0
    for c in range(0, len(lo), 4000):
        sl = slice(c, c + 4000)
        m_ = len(Xb[0].lo[sl])
        _, gS, St = co.enclose(I(np.full(m_, float(dn(1 - s0))), np.ones(m_)), [take_iv(x, sl) for x in Xb])
        Lx = max(Lx, float(np.max(np.sqrt(sum(np.maximum(-gg.lo, gg.hi) ** 2 for gg in gS)))))
        gnn = sum(gS[i] * nI[i] for i in range(3))
        Lxn = max(Lxn, float(np.max(np.maximum(-gnn.lo, gnn.hi))))
        Lt = max(Lt, float(np.max(np.maximum(-St.lo, St.hi))))
    Lx, Lxn, Lt = float(up(Lx)), float(up(Lxn)), float(up(Lt))
    A = Lxn * a_n + Lx * a_T
    K = Lt + rmax * A * A / 8 + s0 * rmin ** -0.5 * (Lxn * b_n + Lx * b_T) + rmin ** -0.5 * Umax
    Wm = 2 * rmin_all * M1 * M1 / s0 - s0 * rmin_all ** -0.5 * Umax - A8
    print(f"   (c) rho = {rho:.4f}; over {len(lo)} inflated x0 boxes, t in [1 - s0, 1]: Lx = {Lx:.4f}, Lxn = {Lxn:.4f}, "
          f"Lt = {Lt:.4f}; max U(Y0) = {Umax:.3f}  ({time.time() - t0:.0f} s)")
    print(f"   K = Lt + r_max (Lxn a_n + Lx a_T)^2/8 + s0 r_min^(-1/2)(Lxn b_n + Lx b_T) + r_min^(-1/2) Umax = {K:.3f}; "
          f"needed m' = K s0 = {K * s0:.3e}; |M| >= M1: W - A8 >= {Wm:+.3e} for s <= s0")
    return dict(qlo=qlo, qhi=qhi, rmin=rmin, rmax=rmax, rmin_all=rmin_all, a_n=a_n, a_T=a_T, b_n=b_n, b_T=b_T,
                Lx=Lx, Lxn=Lxn, Lt=Lt, Umax=Umax, K=K, need=K * s0, Wm=Wm)


def take_iv(x, sl):
    return I(x.lo[sl], x.hi[sl])


if __name__ == "__main__":
    lowprio()
    cmd = sys.argv[1]
    if cmd == "test":
        test(sys.argv[2] if len(sys.argv) > 2 else CERT_E)
    elif cmd == "json_block2":
        json_block2(sys.argv[2], sys.argv[3])
    elif cmd == "mockself":
        mock_selftest()
    elif cmd == "mock":
        a = sys.argv[2:]
        h = tuple(float(v) for v in a[2:5]) if len(a) >= 5 else (0.3, 0.3, 0.075)
        nuv = int(a[5]) if len(a) > 5 else 4
        depth = int(a[6]) if len(a) > 6 else 4
        tr = (float(a[7]), float(a[8])) if len(a) > 8 else (0.0, 1.0)
        fcmin = float(a[9]) if len(a) > 9 else 0.0
        mode = a[10] if len(a) > 10 else "c0"
        crude = float(a[11]) if len(a) > 11 else 1.5
        s0 = 1 - tr[1]
        dsl = [(0.0, 0.85, 0.05), (0.85, 1.0, 0.01)]
        if 0 < s0 < 0.05:
            dsl = [(s0, 0.05, -4), (0.05, 0.85, 0.05), (0.85, 1.0, 0.01)]
        run_mock(a[0], a[1], h, dsl, nuv, depth, trange=tr, fcmin=fcmin, mode=mode, crude=crude)
    elif cmd == "mockell":      # mockell cert tag nuv depth t_lo t_hi fcmin [crude] [fit]
        a = sys.argv[2:]
        tr = (float(a[4]), float(a[5]))
        s0 = 1 - tr[1]
        dsl = [(0.0, 0.85, 0.05), (0.85, 1.0, 0.01)]
        if 0 < s0 < 0.05:
            dsl = [(s0, 0.05, -4), (0.05, 0.85, 0.05), (0.85, 1.0, 0.01)]
        fitv = len(a) > 8 and a[8] == "fit"
        run_mock(a[0], a[1], None, dsl, int(a[2]), int(a[3]), trange=tr, fcmin=float(a[6]), mode="c0",
                 crude=float(a[7]) if len(a) > 7 else 1.5, tube=EllTubeFit() if fitv else EllTube())
    elif cmd == "endell":
        co = TB.CoarseE(sys.argv[2])
        T = EllTubeFit() if (len(sys.argv) > 4 and sys.argv[4] == "fit") else EllTube()
        print(f"end condition (ii') with the disc of {os.path.basename(T.path)} (M = 0 section of Lambda(0)) and its mirror")
        a8h = os.environ.get("TUBEB_A8HI")
        ok, worst, fails, rep = TB.end_condition(co, T.disc_inside_test(), int(sys.argv[3]) if len(sys.argv) > 3 else 8,
                                                 A8_hi=float(a8h) if a8h else TB.A8_HI)
        print(f"  (A8_hi = {float(a8h) if a8h else TB.A8_HI!r})")
        if fails:
            Fl = np.array(fails)
            print(f"  failing boxes {len(Fl)}, real (float centre value <= 0) {np.sum(Fl[:, 5] <= 0)}, min centre value "
                  f"{Fl[:, 5].min():+.4e}")
            for lo, hi, nm in ((-np.inf, 0.0, "rho < 1"), (0.0, 1.2, "1 <= rho < 3.3"), (1.2, 1.9, "3.3 <= rho < 6.7"),
                               (1.9, np.inf, "rho >= 6.7")):
                sel = (Fl[:, 0] >= lo) & (Fl[:, 0] < hi)
                if sel.any():
                    k = np.argmin(Fl[sel, 5])
                    print(f"   {nm}: {sel.sum()} failing, min centre value {Fl[sel, 5].min():+.4e} at ell "
                          f"[{Fl[sel][k, 0]:.3f}, {Fl[sel][k, 1]:.3f}] th [{Fl[sel][k, 2]:.3f}, {Fl[sel][k, 3]:.3f}]")
        # margin next to the disc: S(1, x) - A8 on dD (floats, the M = 0 section of Lambda(0))
        th = np.linspace(0, 2 * np.pi, 721)[:-1]
        G2 = T.G[0][:2, :2]
        L = np.linalg.cholesky(np.linalg.inv(G2))
        dd = (L @ np.stack([np.cos(th), np.sin(th)])).T
        c0 = T.centre(np.array([0.0]))[0]
        # M = 0 section of {(a - c0)^T G0 (a - c0) <= 1}: in (ell, phi) the ellipse with matrix G2 = G0[:2,:2] centred at
        # c2 - G2^(-1) G0[:2, 2] (0 - c0_M) ... computed exactly below
        G0 = T.G[0]
        m = -c0[2]
        cen2 = c0[:2] - np.linalg.solve(G2, G0[:2, 2] * m)
        rhs = 1 - (G0[2, 2] * m * m - (G0[:2, 2] * m) @ np.linalg.solve(G2, G0[:2, 2] * m))
        dd = dd * np.sqrt(max(rhs, 0.0))
        el, ph = cen2[0] + dd[:, 0], cen2[1] + dd[:, 1]
        xd = np.exp(el)[:, None] * (np.cos(ph)[:, None] * EI_F + np.sin(ph)[:, None] * EZ_F)
        S, _, _ = co.enclose(I(np.ones(len(th))), [I(xd[:, c].copy()) for c in range(3)], need_grad=False)
        print(f"  on dD (M = 0 section of Lambda(0), 720 points): S_E^c(1, x) - A8 min {S.mid.min() - A8:+.4e}")
        print("END CONDITION (ii') VERIFIED" if ok else "END CONDITION (ii') NOT VERIFIED")
    elif cmd == "ellbuild":
        build_label_tube()
    elif cmd == "ellcheck":
        elltube_check(sys.argv[2:])
    elif cmd == "ellcheckfit":
        elltube_check(sys.argv[2:], fit=True)
    elif cmd == "targets":
        export_targets_ellfit()
    elif cmd == "taper":         # taper b0 t0 t1
        b0, t0, t1 = (float(v) for v in sys.argv[2:5]) if len(sys.argv) > 4 else (0.6, 0.1, 0.2)
        build_taper(b0, t0, t1)
    elif cmd == "tapergap":      # exact gap + piece margins on the tapered tube, dense small t
        ts = np.array([0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3])
        elltube_check(sys.argv[2:], ts=ts, tube_path=ELL_TAPER_NPZ, tag="taper")
    elif cmd == "band":          # band cert [target] [q_in q_out]
        a8h = os.environ.get("TUBEB_A8HI")
        band_margin(sys.argv[2], target=float(sys.argv[3]) if len(sys.argv) > 3 else 0.003,
                    q_in=float(sys.argv[4]) if len(sys.argv) > 4 else 0.94, q_out=float(sys.argv[5]) if len(sys.argv) > 5 else 1.007,
                    A8_hi=float(a8h) if a8h else None)
    elif cmd == "smallrig":
        small_s_rig(sys.argv[2], q_band=(float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.94, 1.01))
    elif cmd == "smallell":
        small_s_ell(sys.argv[2])
    elif cmd == "tapercand":     # tapercand b0 t0 t1 tag [cert...]: build a candidate taper (new file) + exact gap, dense t
        b0, t0, t1 = (float(v) for v in sys.argv[2:5])
        tag = sys.argv[5]
        path = OUT + f"label_tube_ellip0.3_0.13_0.1_fit_taper_{tag}.npz"
        assert os.path.abspath(path) != os.path.abspath(ELL_TAPER_NPZ)
        build_taper(b0, t0, t1, path=path)
        ts = np.array([1e-4, 3e-4, 1e-3, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
                       0.225, 0.25, 0.275, 0.3, 0.35, 0.4])
        elltube_check(sys.argv[6:], ts=ts, tube_path=path, tag=f"taper_{tag}")
    elif cmd == "taperfull":     # all t, dense grid (13 x 13 per face) on the tapered tube: taperfull nuv cert...
        ts = np.concatenate([[0.002, 0.005, 0.01, 0.02, 0.05, 0.075], np.linspace(0.1, 0.9, 17), [0.95, 0.98, 0.99, 0.995]])
        tp = os.environ.get("TUBEB_TUBE", ELL_TAPER_NPZ)
        if tp != ELL_TAPER_NPZ:
            ts = np.unique(np.concatenate([ts, [0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]]))
        elltube_check(sys.argv[3:], nuv=int(sys.argv[2]), ts=ts, tube_path=tp,
                      tag="taperfull" if tp == ELL_TAPER_NPZ else "taperfull_" + os.path.basename(tp)[:-4])
    elif cmd == "tapertargets":
        export_targets_ellfit(path=ROOT + "words/out/eight/tube_targets_ellfit_taper.npz", tube_path=ELL_TAPER_NPZ)
    elif cmd == "goalmargins":   # goalmargins target.npz [json]
        rows, _ = goal_on_targets(sys.argv[2], json_path=sys.argv[3] if len(sys.argv) > 3 else None)
        print("   t        net margin min (after row margins)   S+ + VI - A8 min")
        for r in rows:
            print(f"   {r[0]:.4f}   {r[1]:+.4e}                         {r[2]:+.4e}")
    elif cmd == "ellfit":
        build_label_tube_fit()
    elif cmd == "ellsmall":
        ell_small_s(fit=len(sys.argv) > 2 and sys.argv[2] == "fit")
    elif cmd == "smalls":
        small_s(sys.argv[2])
    elif cmd == "scan":
        scan_early(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.15)
    elif cmd == "end":
        a = sys.argv[2:]
        h = (float(a[1]), float(a[2])) if len(a) >= 3 else (0.3, 0.3)
        run_end(a[0], h, int(a[3]) if len(a) > 3 else 10)
