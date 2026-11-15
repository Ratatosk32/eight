"""Stage 21, hypothesis (ii-L) rigorous: records for the tube-boundary evaluator (verify/eight_tube_boundary.py,
format in its docstring) produced by the batched C1 Lohner (verify/eight_field_lohner.flow), and the refinement
driver (as tools/words_eight_tubeb.run_mock, with the mock producer replaced).

A record box is a patch of the boundary of the adopted tube: face f, s in [s_lo, s_hi], (u, v) in a square; its label
box [a_lo, a_hi] is the tube's rigorous patch enclosure (EllTube.label_box). Parameters of the I-field map:
eps = s^2 exp(-1.5 ell), phi, M. From one C1 Lohner run over the parameter box (z(1) enclosure, dz/dp enclosure):
    x = r Y,  p = P / s,  W = (r / s)(k + eps m),  r = exp(ell),
    dx/dell = x - 1.5 eps r dY/deps,  dx/dphi = r dY/dphi,  dx/dM = r dY/dM,  dx/ds = 2 (eps / s) r dY/deps,
all in interval arithmetic (full C1 records: dxda, dxds). The centre values (a_c, s_c) come from a Lohner run over the
point box. A box whose Lohner run fails is split. The (ii-L) range is s in [s0, 1] (s0 from the small-s lemma,
paper/review/eight_tube_boundary.md 10.8 item 5).
    python verify/eight_iiL_records.py cert.json [workers] [s0]"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eight_field as EF  # noqa: E402
import eight_field_lohner as L  # noqa: E402
from prove_newton_1212 import I  # noqa: E402

TUBE = os.environ.get("RC_TUBE", os.path.join(ROOT, "words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper.npz"))


def _iv(lo, hi):
    return I(np.asarray(lo, float).copy(), np.asarray(hi, float).copy())


def lohner_boxes(args):
    """worker: label boxes (n, 6) [ell lo, phi lo, M lo, ell hi, phi hi, M hi], s ranges (n, 2) -> per box
    (ok, x lo/hi, p lo/hi, W lo/hi, dxda lo/hi, dxds lo/hi); failing batches are halved"""
    A, S = args
    try:
        import eight_R_cover2 as RC
        RC.keep_awake_low()
    except Exception:
        pass
    L.ORDER = int(os.environ.get("LORDER", "12"))
    n = len(A)
    out = {"ok": np.zeros(n, bool)}
    for k, shp in (("x", (3,)), ("p", (3,)), ("W", ()), ("dxda", (3, 3)), ("dxds", (3,))):
        out[k + "_lo"] = np.full((n,) + shp, -np.inf)
        out[k + "_hi"] = np.full((n,) + shp, np.inf)

    def rec(ix):
        if len(ix) == 0:
            return
        ell = _iv(A[ix, 0], A[ix, 3])
        s = _iv(S[ix, 0], S[ix, 1])
        r = EF.iexp(ell)
        eps = s.sq() * EF.iexp(ell * (-1.5))
        P = I(np.stack([eps.lo, A[ix, 1], A[ix, 2]], 1), np.stack([eps.hi, A[ix, 4], A[ix, 5]], 1))
        try:
            Z, D, mq, ns = L.flow(P)
            if not (np.all(np.isfinite(Z.lo)) and np.all(np.isfinite(Z.hi)) and np.all(np.isfinite(D.lo))
                    and np.all(np.isfinite(D.hi)) and np.all(mq > 0)):
                raise FloatingPointError("non-finite enclosure or collision")
        except Exception:
            if len(ix) == 1:
                return
            h = len(ix) // 2
            rec(ix[:h])
            rec(ix[h:])
            return
        Zi = [_iv(Z.lo[:, i], Z.hi[:, i]) for i in range(9)]
        Di = [[_iv(D.lo[:, i, j], D.hi[:, i, j]) for j in range(3)] for i in range(9)]
        x = [r * Zi[i] for i in range(3)]
        p = [Zi[3 + i] / s for i in range(3)]
        W = (r / s) * (Zi[6] + eps * Zi[7])
        dxda = [[x[i] - (eps * r * Di[i][0]) * 1.5, r * Di[i][1], r * Di[i][2]] for i in range(3)]
        dxds = [(eps / s) * r * Di[i][0] * 2.0 for i in range(3)]
        out["ok"][ix] = True
        for i in range(3):
            out["x_lo"][ix, i], out["x_hi"][ix, i] = x[i].lo, x[i].hi
            out["p_lo"][ix, i], out["p_hi"][ix, i] = p[i].lo, p[i].hi
            out["dxds_lo"][ix, i], out["dxds_hi"][ix, i] = dxds[i].lo, dxds[i].hi
            for j in range(3):
                out["dxda_lo"][ix, i, j], out["dxda_hi"][ix, i, j] = dxda[i][j].lo, dxda[i][j].hi
        out["W_lo"][ix], out["W_hi"][ix] = W.lo, W.hi
    rec(np.arange(n))
    return out


def a8_upper(pool=None):
    """rigorous upper bound of A8 = W(1, a8): a8 lies in the Krawczyk box (half-width 1e-6, verify/eight_field.py,
    PROVED), so A8 <= sup of W over that box"""
    inside, X, K, yT, minq = EF.krawczyk(np.array([1.5424624176685366, 0.7066425303908384, -0.30889005565526345]),
                                         [1e-6] * 3)
    assert inside, "Krawczyk box not proved"
    A = np.array([[float(X[0].lo), float(X[1].lo), float(X[2].lo), float(X[0].hi), float(X[1].hi), float(X[2].hi)]])
    o = lohner_boxes((A, np.array([[1.0, 1.0]])))
    assert o["ok"][0]
    return float(o["W_hi"][0]), float(o["W_lo"][0])


class RealProducer:
    def __init__(self, tube, pool, chunk=8):
        self.tube, self.pool, self.chunk = tube, pool, chunk
        self.fails = 0

    def _run(self, A, S):
        parts = [(A[i:i + self.chunk], S[i:i + self.chunk]) for i in range(0, len(A), self.chunk)]
        res = self.pool.map(lohner_boxes, parts) if self.pool else [lohner_boxes(p) for p in parts]
        return {k: np.concatenate([r[k] for r in res]) for k in res[0]}

    def records(self, boxes, A8_hi):
        n = len(boxes)
        f = boxes[:, 0].astype(int)
        sc, uc, vc = (0.5 * (boxes[:, i] + boxes[:, i + 1]) for i in (1, 3, 5))
        a_lo, a_hi = self.tube.label_box(f, boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6])
        a_c = self.tube.label(f, sc, uc, vc)
        A = np.concatenate([np.concatenate([a_lo, a_hi], 1), np.concatenate([a_c, a_c], 1)])
        S = np.concatenate([boxes[:, 1:3], np.stack([sc, sc], 1)])
        o = self._run(A, S)
        ok = o["ok"][:n] & o["ok"][n:]
        self.fails = int((~ok).sum())
        rec = {"s_lo": boxes[:, 1].copy(), "s_hi": boxes[:, 2].copy(), "s_c": sc, "face": f, "a_lo": a_lo,
               "a_hi": a_hi, "a_c": a_c, "A8_hi": np.array(A8_hi)}
        for k in ("x", "p", "W", "dxda", "dxds"):
            rec[k + "_lo"], rec[k + "_hi"] = o[k + "_lo"][:n], o[k + "_hi"][:n]
        rec["xc_lo"], rec["xc_hi"] = o["x_lo"][n:], o["x_hi"][n:]
        rec["Wc_lo"], rec["Wc_hi"] = o["W_lo"][n:], o["W_hi"][n:]
        # boxes without records: make them fail in the evaluator (W_lo = -inf, no derivative data)
        for k in ("x", "xc"):
            rec[k + "_lo"][~ok], rec[k + "_hi"][~ok] = 1.0, 1.0
        rec["W_lo"][~ok], rec["Wc_lo"][~ok], rec["Wc_hi"][~ok] = -1e300, 0.0, 0.0
        rec["W_hi"][~ok] = 1e300
        for k in ("p", "dxda", "dxds"):
            rec[k + "_lo"][~ok], rec[k + "_hi"][~ok] = -1e300, 1e300
        return rec, ok


def run(cert, workers=10, s0=1.165e-4, ds_list=None, nuv=4, depth=8, maxbox=2_000_000):
    import multiprocessing as mp
    import words_eight_tubeb as T
    TB = T.TB
    t0 = time.time()
    co = TB.CoarseE(cert)
    tube = T.EllTubeFit(TUBE)
    pool = mp.Pool(workers) if workers > 1 else None
    A8_hi, A8_lo = a8_upper()
    print(f"A8 in [{A8_lo!r}, {A8_hi!r}] (C1 Lohner over the Krawczyk box)", flush=True)
    prod = RealProducer(tube, pool)
    ds_list = ds_list or [(s0, 0.15, 0.025), (0.15, 0.85, 0.05), (0.85, 1.0, 0.05)]
    boxes = T.initial_boxes(ds_list, nuv)
    boxes = boxes[boxes[:, 1] >= s0 - 1e-15]
    print(f"(ii-L) records run: piece {os.path.basename(cert)}, tube {os.path.basename(TUBE)}, s in [{s0}, 1], "
          f"{len(boxes)} initial boxes", flush=True)
    passed, failed_real, unresolved = [], [], []
    for dep in range(depth + 1):
        rec, ok = prod.records(boxes, A8_hi)
        res = TB.check_records(co, rec)
        lb = res["lb"]
        good = lb > 0
        # float value at the centre (from the thin records): a box whose centre is <= 0 would be a real failure
        Tc = I(1.0 - rec["s_c"])
        Sc, _, _ = co.enclose(Tc, [I(0.5 * (rec["xc_lo"][:, i] + rec["xc_hi"][:, i])) for i in range(3)], need_grad=False)
        fc = np.maximum(Sc.mid, 0.0) + 0.5 * (rec["Wc_lo"] + rec["Wc_hi"]) - A8_hi
        real = ~good & ok & (fc <= 0)
        for b in np.flatnonzero(good):
            passed.append(np.concatenate([boxes[b], [lb[b]]]))
        for b in np.flatnonzero(real):
            failed_real.append(np.concatenate([boxes[b], [fc[b]]]))
        ref = ~good & ~real
        print(f"  depth {dep}: {len(boxes)} boxes: pass {good.sum()} (min lb {lb[good].min() if good.any() else np.nan:+.3e}), "
              f"real failures {real.sum()}, Lohner failures {prod.fails}, to refine {ref.sum()} ({time.time() - t0:.0f} s)",
              flush=True)
        if not ref.any():
            break
        if dep == depth or ref.sum() * 8 > maxbox:
            unresolved += [boxes[b] for b in np.flatnonzero(ref)]
            break
        rad = np.where(np.isfinite(res["rad"][ref]), res["rad"][ref], 1.0)
        boxes = T.split_boxes(boxes[ref], rad, tube)
        np.savez(os.path.join(ROOT, "words/out/eight/iiL_progress.npz"), passed=np.array(passed),
                 failed_real=np.array(failed_real), pending=boxes)
    np.savez(os.path.join(ROOT, "words/out/eight/iiL_result.npz"), passed=np.array(passed),
             failed_real=np.array(failed_real), unresolved=np.array(unresolved), A8_hi=A8_hi, s0=s0)
    print(f"done: pass {len(passed)}, real failures {len(failed_real)}, unresolved {len(unresolved)}; "
          f"{time.time() - t0:.0f} s", flush=True)
    print("(ii-L) VERIFIED on s in [s0, 1]" if not failed_real and not unresolved else "(ii-L) NOT verified", flush=True)
    if pool:
        pool.close()


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10, float(sys.argv[3]) if len(sys.argv) > 3 else 1.165e-4)
