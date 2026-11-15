"""Stage 21, hypothesis (iii-L) by the theorist's reduction (paper/review/eight_final_steps_theorist.md section 2):
for every label a in Lambda(1) with f_1(a) on the ray E, W(1, a) >= A8, equality only at a8. It follows from (R) on
Lambda(1), (ii-L) at s = 1, and
 (K1) a8 is the only zero of N(a) = (Y_2, Y_3, P_1)(1; a) in a box K1 around it (Krawczyk, verify/eight_field.py);
 (X)  every label box meeting Lambda(1) and not inside K1 satisfies one of
      (alpha) the enclosure of (Y_2, Y_3)(1) misses the axis (Y_2 or Y_3 bounded away from 0; x = r Y, r > 0),
      (beta)  W = r (k + eps m)(1) has lower bound > A8_hi,
      (gamma) the enclosure of P_1(1) excludes 0 (p = P/s, s = 1).
Labels a = (ell, phi, M) at s = 1: eps = exp(-1.5 ell), r = exp(ell). Enclosures by the batched C1 Lohner
(verify/eight_field_lohner.flow); a batch that cannot be validated is halved, a single failing box is bisected.
Boxes certainly outside Lambda(1) (the two tests of verify/eight_R_cover2.outside at s = 1) are dropped.
    python verify/eight_iiiL.py [K1 half-width] [initial half-widths ell,phi,M]"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import eight_field as EF  # noqa: E402
import eight_field_lohner as L  # noqa: E402
import eight_R_cover2 as RC  # noqa: E402
from prove_newton_1212 import I, dn, up  # noqa: E402

A8_LAB = np.array([1.5424624176685366, 0.7066425303908384, -0.30889005565526345])


def k1_box(d):
    """Krawczyk on the cube of half-width d around a8; returns (proved, X box lo, hi, action enclosure hi)"""
    inside, X, K, yT, minq = EF.krawczyk(A8_LAB, [d, d, d])
    kin, mq = yT[6][0], yT[7][0]
    eps = EF.iexp(X[0] * (-1.5))
    act = (kin + eps * mq) * EF.iexp(X[0])
    lo = np.array([float(X[i].lo) for i in range(3)])
    hi = np.array([float(X[i].hi) for i in range(3)])
    return inside, lo, hi, float(act.hi), float(yT[0][0].lo)


def outside_L1(tb, B):
    a_lo, a_hi = B[:, 0:3], B[:, 3:6]
    one = np.ones(len(B))
    return tb.box_certainly_outside(a_lo, a_hi, one, one) | RC._axis_outside(tb, a_lo, a_hi, 1.0, 1.0)


def enclose(B):
    """label boxes (n, 6) = ell lo, phi lo, M lo, ell hi, phi hi, M hi -> (ok mask, Y2, Y3, P1, W lo) enclosures;
    batches that fail are halved down to single boxes (ok False for a single box that fails)"""
    n = len(B)
    ok = np.zeros(n, bool)
    Y2, Y3, P1 = (np.zeros((n, 2)) for _ in range(3))
    Wlo = np.full(n, -np.inf)

    def rec(ix):
        if len(ix) == 0:
            return
        Ell = I(B[ix, 0], B[ix, 3])
        eps = EF.iexp(Ell * (-1.5))
        P = I(np.stack([eps.lo, B[ix, 1], B[ix, 2]], 1), np.stack([eps.hi, B[ix, 4], B[ix, 5]], 1))
        try:
            Z, D, mq, ns = L.flow(P)
        except Exception:
            if len(ix) == 1:
                return
            h = len(ix) // 2
            rec(ix[:h])
            rec(ix[h:])
            return
        good = np.isfinite(Z.lo).all(1) & np.isfinite(Z.hi).all(1) & (mq > 0)
        for j, i in enumerate(ix):
            if not good[j]:
                continue
            ok[i] = True
            Y2[i] = Z.lo[j, 1], Z.hi[j, 1]
            Y3[i] = Z.lo[j, 2], Z.hi[j, 2]
            P1[i] = Z.lo[j, 3], Z.hi[j, 3]
            k = I(Z.lo[j, 6], Z.hi[j, 6])
            m = I(Z.lo[j, 7], Z.hi[j, 7])
            e = EF.iexp(I(B[i, 0], B[i, 3]) * (-1.5))
            W = (k + e * m) * EF.iexp(I(B[i, 0], B[i, 3]))
            Wlo[i] = float(W.lo)
    rec(np.arange(n))
    return ok, Y2, Y3, P1, Wlo


def _enclose_task(B):
    try:
        RC.keep_awake_low()
    except Exception:
        pass
    L.ORDER = int(os.environ.get("LORDER", "12"))
    return enclose(B)


def enclose_par(pool, B, chunk=16):
    if pool is None or len(B) <= chunk:
        return enclose(B)
    parts = [B[i:i + chunk] for i in range(0, len(B), chunk)]
    res = pool.map(_enclose_task, parts)
    return tuple(np.concatenate([r[k] for r in res]) for k in range(5))


def bisect(b):
    w = (b[3:6] - b[0:3]) / np.array([1.0, 1.0, 0.25])          # scaled widths (M is ~4x more sensitive)
    k = int(np.argmax(w))
    m = 0.5 * (b[k] + b[3 + k])
    b1, b2 = b.copy(), b.copy()
    b1[3 + k] = m
    b2[k] = m
    return [b1, b2]


def main():
    RC.keep_awake_low()
    L.ORDER = int(os.environ.get("LORDER", "12"))
    t0 = time.time()
    tb = RC.tube()
    # K1: the largest proved Krawczyk cube from a list
    k1 = None
    for d in [float(x) for x in os.environ.get("K1_LIST", "1e-6,1e-5,3e-5,1e-4,3e-4,1e-3").split(",")]:
        try:
            res = k1_box(d)
        except Exception as e:
            print(f"K1 half-width {d:g}: failed ({repr(e)[:60]})", flush=True)
            break
        print(f"K1 half-width {d:g}: proved {res[0]}, action hi {res[3]!r}, Y1 lo {res[4]:.4f}", flush=True)
        if not res[0]:
            break
        k1 = (d,) + res
    assert k1 is not None, "no Krawczyk box"
    d1, _, klo, khi, act_hi, y1lo = k1
    A8_hi = act_hi
    print(f"K1 = cube of half-width {d1:g} around a8; A8_hi = {A8_hi!r}; Y1 > {y1lo:.4f} (start on E, not -E)", flush=True)
    # initial grid over the bounding box of Lambda(1)
    c1 = tb.centre(np.array(1.0))
    half = np.sqrt(np.diag(np.linalg.inv(tb.Gs(np.array(1.0))))) * (1 + 1e-9) + 1e-12
    h0 = np.array([float(x) for x in os.environ.get("H0", "0.004,0.004,0.001").split(",")])
    g = [c1[i] - half[i] + 2 * h0[i] * np.arange(int(math.ceil(half[i] / h0[i])) + 1) for i in range(3)]
    B = np.array([[a, b, c, a + 2 * h0[0], b + 2 * h0[1], c + 2 * h0[2]] for a in g[0] for b in g[1] for c in g[2]])
    B = B[~outside_L1(tb, B)]
    print(f"Lambda(1): centre {np.round(c1, 4)}, half extents {np.round(half, 4)}; initial boxes {len(B)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    stats = dict(alpha=0, beta=0, gamma=0, k1=0, outside=0)
    nw = int(os.environ.get("IIIL_WORKERS", "1"))
    import multiprocessing as mp
    pool = mp.Pool(nw) if nw > 1 else None
    stuck = []
    level = 0
    while len(B):
        in_k1 = np.all(B[:, 0:3] >= klo, 1) & np.all(B[:, 3:6] <= khi, 1)
        stats["k1"] += int(in_k1.sum())
        B = B[~in_k1]
        ok, Y2, Y3, P1, Wlo = enclose_par(pool, B)
        alpha = ok & ((Y2[:, 0] > 0) | (Y2[:, 1] < 0) | (Y3[:, 0] > 0) | (Y3[:, 1] < 0))
        beta = ok & ~alpha & (Wlo > A8_hi)
        gamma = ok & ~alpha & ~beta & ((P1[:, 0] > 0) | (P1[:, 1] < 0))
        acc = alpha | beta | gamma
        stats["alpha"] += int(alpha.sum())
        stats["beta"] += int(beta.sum())
        stats["gamma"] += int(gamma.sum())
        nxt = []
        for b in B[~acc]:
            if np.max(b[3:6] - b[0:3]) < 1e-7:
                stuck.append(b)
            else:
                nxt += bisect(b)
        B = np.array(nxt) if nxt else np.zeros((0, 6))
        if len(B):
            o = outside_L1(tb, B)
            stats["outside"] += int(o.sum())
            B = B[~o]
        print(f"level {level}: accepted alpha {int(alpha.sum())}, beta {int(beta.sum())}, gamma {int(gamma.sum())}; "
              f"Lohner failures {int((~ok).sum())}; next {len(B)}; {time.time() - t0:.0f} s", flush=True)
        level += 1
    print(f"totals: {stats}; stuck {len(stuck)}", flush=True)
    print("(X) VERIFIED: every box of Lambda(1) outside K1 passes alpha, beta or gamma" if not stuck else
          "(X) NOT verified", flush=True)
    np.savez(os.path.join(os.path.dirname(HERE), "words/out/eight/iiiL.npz"), k1_half=d1, klo=klo, khi=khi,
             A8_hi=A8_hi, stuck=np.array(stuck), **{k: v for k, v in stats.items()})


if __name__ == "__main__":
    main()
