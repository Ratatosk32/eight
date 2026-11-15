"""Stage 21, (iii-L) condition (X) from SAVED comparison-tube data (theorist paper/review/eight_final_steps_theorist.md
section 2, eight_records_design_theorist.md section 3; referee eight_monotone_referee.md (d)).

(X) for every label a in Lambda(1) (tube A at s = 1) outside K1 (the proved Krawczyk cube of half-width 3e-5 around
a8, verify/eight_iiiL.py k1_box) with f_1(a) on the ray E:
  (alpha) the x = f_1(a) enclosure misses the ray E (x_2 or x_3 bounded away from 0, or x_1 < 0), or
  (beta)  W(1, a) > A8 (W_lo > A8_hi), or
  (gamma) p_1(1, a) != 0 (0 not in the P_1 enclosure; P = -v/(4|Y|), so sign P_1 = -sign v_1).

Level 1 (this file). A saved box B (thin in eps: eps_c, phi in Phi, M_top in Mb) is, at scale r, the family of the
extremals (r, phi, M_top/s_top), s_top(r) = sqrt(eps_c) r^(3/4), at times s = tau s_top. Its s = 1 labels exist for
s_top(r) >= 1 and sit at tau1 = 1/s_top(r): labels (ell, phi, tau1 M_top) (eps of the label = r^(-3/2)). For a scale
interval R = [e^ell_a, e^ell_b] the tau1 range is [1/s_top(R_hi), 1/s_top(R_lo)]; every member's state at s = 1 lies
in the tube enclosure of a saved piece meeting that range (the piece containing its tau1). The label box is accepted
if EVERY such piece passes alpha, beta or gamma (possibly different tests on different pieces): each member is then
covered by the piece that contains its own s = 1 state. x enclosure: R x (tube box of Y); W lower bound: the W branch
of the records (eight_iiL_shadowrec.w_lower, sound for every member at every time of the piece); v_1 from the tube
box of v. Label boxes certainly outside Lambda(1) need nothing; label boxes inside K1 are skipped.
Refinement of survivors: (1) the scale interval is halved (narrower tau1 range and R), up to ELEV levels; (2) the
saved box is refined (eight_iiL_shadowrec.refine_many: sub-boxes in (phi, M_top) with their own thin centre and the
saved k^2, time pieces of width FINE_W near the tau1 range; sound as in the records); (3) what is left is a level-2
candidate: label boxes at s = 1, to be handled by the direct Lohner (verify/eight_iiiL.py enclose) with boxes of
x-radius about |x_1 - rho8| / 4 down to K1.
    python verify/eight_iiiL_shadow.py <dir or files> [nfiles]      (level 1 on saved files; writes iiiL_l1.npz)
    python verify/eight_iiiL_shadow.py l2eta                         (level-2 timing near a8)"""
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
import eight_iiL_shadowrec as SR  # noqa: E402
from prove_newton_1212 import I, dn, up  # noqa: E402

A8_HI = SR.A8_HI
A8_LAB = np.array([1.5424624176685366, 0.7066425303908384, -0.30889005565526345])
# K1 and A8_hi from the stored proof artifact (referee 5c C3/C4; verify/eight_k1_artifact.py): the Krawczyk cube K1
# (a8 the only zero of N in it, Y_1(1) > 0 there: the zero is on E) and A8 <= a8_act_hi <= the records' A8_HI
ARTIFACT = os.path.join(ROOT, "words/out/eight/tubeb/k1_a8_artifact.npz")
K1_HALF = K1_LO = K1_HI = None
BOX_ORDER = int(os.environ.get("LORDER", "12"))         # Taylor order of the C1 Lohner of level 2 (enclose_k)


def load_k1(path, a8_hi=None):
    """K1 from a Krawczyk artifact (keys ok, k1_proved, k1_y1_lo, a8_lab, k1_half, a8_act_hi): asserts proved,
    Y_1 > 0, the same a8, and A8_HI (or a8_hi) >= the stored upper end of A8; sets K1_HALF, K1_LO, K1_HI"""
    global K1_HALF, K1_LO, K1_HI
    art = np.load(path)
    assert bool(art["ok"]) and bool(art["k1_proved"]) and float(art["k1_y1_lo"]) > 0, "K1 artifact not proved"
    assert np.array_equal(art["a8_lab"], A8_LAB), "K1 artifact for another a8"
    assert (A8_HI if a8_hi is None else a8_hi) >= float(art["a8_act_hi"]), "A8_HI below the stored enclosure of A8"
    K1_HALF = float(art["k1_half"])
    K1_LO, K1_HI = A8_LAB - K1_HALF, A8_LAB + K1_HALF
    return art


if os.path.exists(ARTIFACT):
    load_k1(ARTIFACT)
ELEV = int(os.environ.get("X_ELEV", "3"))
FINE_W = float(os.environ.get("X_FINEW", "0.001"))
OUT = os.path.join(ROOT, "words/out/eight/tubeb/iiiL_l1.npz")


def s1_labels(D, b, rows):
    """for rows (ell_a, ell_b, phi_a, phi_b, Mt_a, Mt_b) of box b: tau1 range and the s = 1 label boxes"""
    eps = D["boxes"][b][0]
    el, eh = rows[:, 0], rows[:, 1]
    stop_lo = math.sqrt(eps) * np.exp(el) ** 0.75 * (1 - 1e-12)
    stop_hi = math.sqrt(eps) * np.exp(eh) ** 0.75 * (1 + 1e-12)
    t1_lo = np.minimum(1.0 / stop_hi, 1.0)
    t1_hi = np.minimum(1.0 / np.maximum(stop_lo, 1e-300), 1.0)
    Mc = np.stack([t1_lo * rows[:, 4], t1_lo * rows[:, 5], t1_hi * rows[:, 4], t1_hi * rows[:, 5]])
    a_lo = np.stack([el, rows[:, 2], dn(Mc.min(0))], 1)
    a_hi = np.stack([eh, rows[:, 3], up(Mc.max(0))], 1)
    return stop_hi >= 1.0, t1_lo, t1_hi, a_lo, a_hi


def test_pieces(D, b, pcs, el, eh):
    """alpha / beta / gamma per piece for the members of box b at scale r in [e^el, e^eh] (pieces pcs)"""
    m = len(pcs)
    Wlb, X = SR.w_lower(D, b, pcs, np.full(m, math.exp(el) * (1 - 1e-15)), np.full(m, math.exp(eh) * (1 + 1e-15)))
    v1lo, v1hi = D["t_lo"][pcs, 3, b], D["t_hi"][pcs, 3, b]
    alpha = (X[1].lo > 0) | (X[1].hi < 0) | (X[2].lo > 0) | (X[2].hi < 0) | (X[0].hi < 0)
    beta = Wlb > A8_HI
    gamma = (v1lo > 0) | (v1hi < 0)
    return alpha, beta, gamma, Wlb


# MODE "X": condition (X) of (iii-L) (all labels of Lambda(1) outside K1; alpha, beta or gamma). MODE "ii1": (ii-1),
# the boundary labels of Lambda(1) with image on E need W > A8: label boxes certainly inside Lambda(1) need nothing,
# the others need alpha or beta on every piece (gamma does not count; no K1 exception)
MODE = os.environ.get("X_MODE", "X")


def classify_s1(T, D, b, rows, elev0=0):
    """rows (n, 6) of (ell_a, ell_b, phi_a, phi_b, Mt_a, Mt_b): returns a list of (class, row, s=1 label box lo, hi,
    detail) with class out / nos1 / K1 / alpha / beta / gamma / surv; survivors' scale intervals are halved up to ELEV
    levels"""
    out = []
    work = [(r, elev0) for r in rows]
    ta, tb = D["ta"], D["tb"]
    while work:
        R_ = np.array([w[0] for w in work])
        lv = [w[1] for w in work]
        has1, t1_lo, t1_hi, a_lo, a_hi = s1_labels(D, b, R_)
        one = np.ones(len(R_))
        outs = T.box_certainly_outside(a_lo, a_hi, one, one)
        ins = T.box_certainly_inside(a_lo, a_hi, one, one) if MODE == "ii1" else np.zeros(len(R_), bool)
        nxt = []
        for j in range(len(R_)):
            if not has1[j]:
                out.append(("nos1", R_[j], a_lo[j], a_hi[j], np.nan))
                continue
            if outs[j]:
                out.append(("out", R_[j], a_lo[j], a_hi[j], np.nan))
                continue
            if ins[j]:                  # (ii-1) mode: no boundary label of Lambda(1) in the box
                out.append(("inside", R_[j], a_lo[j], a_hi[j], np.nan))
                continue
            if MODE == "X" and np.all(a_lo[j] >= K1_LO) and np.all(a_hi[j] <= K1_HI):
                out.append(("K1", R_[j], a_lo[j], a_hi[j], np.nan))
                continue
            pcs = np.flatnonzero((ta <= t1_hi[j]) & (tb >= t1_lo[j]))
            if not len(pcs):
                out.append(("surv", R_[j], a_lo[j], a_hi[j], np.nan))
                continue
            al, be, ga, Wlb = test_pieces(D, b, pcs, R_[j, 0], R_[j, 1])
            if np.all(al):
                out.append(("alpha", R_[j], a_lo[j], a_hi[j], np.nan))
            elif np.all(al | be):
                out.append(("beta", R_[j], a_lo[j], a_hi[j], float(np.min(Wlb[~al] - A8_HI))))
            elif MODE == "X" and np.all(al | be | ga):
                out.append(("gamma", R_[j], a_lo[j], a_hi[j], np.nan))
            elif lv[j] < ELEV:
                em = 0.5 * (R_[j, 0] + R_[j, 1])
                for (ea, eb) in ((R_[j, 0], em), (em, R_[j, 1])):
                    r2 = R_[j].copy()
                    r2[0], r2[1] = ea, eb
                    nxt.append((r2, lv[j] + 1))
            else:
                out.append(("surv", R_[j], a_lo[j], a_hi[j], np.nan))
        work = nxt
    return out


X_RLEV = int(os.environ.get("X_RLEV", "5"))
X_REFN = int(os.environ.get("X_REFN", "2"))


def level1_file(T, f, ell_lo, ell_hi, dell=0.0025, rlev=None, refn=None):
    """level 1 for every closed thin box of one saved file. Survivors are refined up to rlev levels: at each level
    all units with survivors (saved boxes or sub-boxes) go into one Lohner batch (refine_many, refn x refn sub-boxes
    in (phi, M_top), time pieces FINE_W wide near the survivors' tau1 ranges, t_end = 1); each sub-box takes its part
    of its parent's surviving label boxes. Returns (D, {box: rows}) with rows (class, row, s=1 lo, hi, detail)."""
    rlev = X_RLEV if rlev is None else rlev
    refn = X_REFN if refn is None else refn
    D = dict(np.load(f, allow_pickle=True))
    nb = D["boxes"].shape[0]
    thin = np.asarray(D.get("dl", np.zeros(nb)), float) <= 1e-300
    if bool(np.asarray(D.get("slab", False))):          # slab boxes must never be used here
        thin[:] = False
    res = {}
    ells = SR.ell_grid(ell_lo, ell_hi, dell)                # explicit count (referee 5b D5)
    units = []                      # (box, Du, bu, survivor rows)
    for b in range(nb):
        if not (bool(D["closed"][b]) and thin[b]):
            res[b] = [("notclosed" if not bool(D["closed"][b]) else "notthin", None, None, None, np.nan)]
            continue
        Pb = D["boxes"][b]
        rows = np.array([[ells[i], ells[i + 1], Pb[2], Pb[3], Pb[4], Pb[5]] for i in range(len(ells) - 1)])
        has1 = s1_labels(D, b, rows)[0]
        out = classify_s1(T, D, b, rows[has1])
        res[b] = [r for r in out if r[0] != "surv"]
        sv = [r for r in out if r[0] == "surv"]
        if sv:
            units.append((b, D, b, sv))
    for lev in range(rlev):
        if not units:
            break
        fine = []
        for (_, Du, bu, sv) in units:
            for r in sv:
                _, t1l, t1h, _, _ = s1_labels(Du, bu, r[1][None, :])
                fine.append((float(t1l[0]) - 0.002, float(t1h[0]) + 0.002))
        fine = [(min(x[0] for x in fine), max(x[1] for x in fine), FINE_W)]
        try:
            Dns = SR.refine_many([(Du, bu) for (_, Du, bu, _) in units], refn, refn, fine=fine, t_end=1.0)
        except Exception as e:
            print(f"  {os.path.basename(f)}: refinement level {lev + 1} failed ({e!r})"[:160], flush=True)
            break
        nxt = []
        for (b, Du, bu, sv), Dn in zip(units, Dns):
            regs = [r[1] for r in sv]
            for bb in range(len(Dn["boxes"])):
                Pq = Dn["boxes"][bb]
                rr = []
                for g in regs:
                    pa, pb_, ma, mb = max(g[2], Pq[2]), min(g[3], Pq[3]), max(g[4], Pq[4]), min(g[5], Pq[5])
                    if pa < pb_ and ma < mb:
                        rr.append([g[0], g[1], pa, pb_, ma, mb])
                if not rr:
                    continue
                if not bool(Dn["closed"][bb]):
                    nxt.append((b, None, bb, [("surv", np.array(x), None, None, np.nan) for x in rr]))
                    continue
                out = classify_s1(T, Dn, bb, np.array(rr), elev0=ELEV)
                res[b] += [r for r in out if r[0] != "surv"]
                sv2 = [r for r in out if r[0] == "surv"]
                if sv2:
                    nxt.append((b, Dn, bb, sv2))
        units = [u for u in nxt if u[1] is not None]
        for u in nxt:
            if u[1] is None:
                res[u[0]] += u[3]
        print(f"    refinement level {lev + 1}: {sum(len(u[3]) for u in units)} surviving label boxes in {len(units)} "
              f"units", flush=True)
    for (b, _, _, sv) in units:
        res[b] += sv
    return D, res


def run(files):
    SR.lowprio()
    T = SR.tube()
    ell_lo, ell_hi = SR.ell_range(T)
    t_all = time.time()
    tot = {}
    surv = []
    for f in files:
        t0 = time.time()
        D, res = level1_file(T, f, ell_lo, ell_hi)
        for b, rows in res.items():
            for r in rows:
                tot[r[0]] = tot.get(r[0], 0) + 1
                if r[0] == "surv":
                    surv.append([D["boxes"][b][0], *r[1], *(r[2] if r[2] is not None else [np.nan] * 3),
                                 *(r[3] if r[3] is not None else [np.nan] * 3)])
        cnt = {}
        for rows in res.values():
            for r in rows:
                cnt[r[0]] = cnt.get(r[0], 0) + 1
        print(f"  {os.path.basename(f)}: eps {D['boxes'][:, 0].min():.4g}..{D['boxes'][:, 0].max():.4g}: "
              + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + f" ({time.time() - t0:.1f} s)", flush=True)
    print("total: " + ", ".join(f"{k} {v}" for k, v in sorted(tot.items())) + f"; {time.time() - t_all:.0f} s", flush=True)
    np.savez(OUT, surv=np.array(surv), cols=np.array(["eps_c", "ell_a", "ell_b", "phi_a", "phi_b", "Mt_a", "Mt_b",
                                                      "a_lo_ell", "a_lo_phi", "a_lo_M", "a_hi_ell", "a_hi_phi",
                                                      "a_hi_M"]))
    return tot, surv


if __name__ == "__main__" and sys.argv[1] not in ("l2eta", "l2surv", "par", "ii1fix"):
    src = sys.argv[1]
    fl = sorted(glob.glob(os.path.join(src, "b_*.npz"))) if os.path.isdir(src) else src.split(",")
    if len(sys.argv) > 2:
        fl = fl[:int(sys.argv[2])]
    run(fl)


# ---------------------------------------------------------------- level 2 near a8
def enclose_k(B):
    """label boxes (n, 6) at s = 1 -> (ok, Y2, Y3, P1 (n, 2) enclosures, W lo, excl): one batched C1 Lohner flow
    (eight_field_lohner.flow) of the boxes and of their centre points. excl: Krawczyk exclusion for
    N = (Y2, Y3, P1)(1; a): with J an enclosure of DN over the box (labels a = (ell, phi, M); d/d ell = -1.5 eps d/d eps),
    C = inv(mid J), K = c - C N(c) + (I - C J)(B - c); K disjoint from B in some coordinate => N has no zero in B, so
    every point of B on E has p_1 != 0 (the zero set of N is that of (x_2, x_3, p_1), x = r Y, p = P at s = 1).
    Failing batches are halved (a single failing box gets ok False)."""
    import eight_field_lohner as L
    old_order = L.ORDER
    L.ORDER = BOX_ORDER                     # one fixed order for this call; the previous value is restored below
    try:
        return _enclose_k(B)
    finally:
        L.ORDER = old_order


def _enclose_k(B):
    import eight_field as EF
    import eight_field_lohner as L
    n = len(B)
    ok = np.zeros(n, bool)
    excl = np.zeros(n, bool)
    Y2, Y3, P1 = (np.zeros((n, 2)) for _ in range(3))
    Wlo = np.full(n, -np.inf)
    cen = 0.5 * (B[:, 0:3] + B[:, 3:6])

    def params(lo3, hi3):
        Ell = I(lo3[:, 0], hi3[:, 0])
        eps = EF.iexp(Ell * (-1.5))
        return Ell, eps, I(np.stack([eps.lo, lo3[:, 1], lo3[:, 2]], 1), np.stack([eps.hi, hi3[:, 1], hi3[:, 2]], 1))

    def rec(ix):
        if len(ix) == 0:
            return
        m = len(ix)
        lo3 = np.concatenate([B[ix, 0:3], cen[ix]])
        hi3 = np.concatenate([B[ix, 3:6], cen[ix]])
        Ell, eps, P = params(lo3, hi3)
        try:
            Z, D, mq, ns = L.flow(P)
            if not (np.all(np.isfinite(Z.lo)) and np.all(np.isfinite(Z.hi)) and np.all(np.isfinite(D.lo))
                    and np.all(np.isfinite(D.hi)) and np.all(mq > 0)):
                raise FloatingPointError("non-finite enclosure or collision")
        except Exception:
            if m == 1:
                return
            h = m // 2
            rec(ix[:h])
            rec(ix[h:])
            return
        sl = slice(0, m)
        ok[ix] = True
        Y2[ix] = np.stack([Z.lo[sl, 1], Z.hi[sl, 1]], 1)
        Y3[ix] = np.stack([Z.lo[sl, 2], Z.hi[sl, 2]], 1)
        P1[ix] = np.stack([Z.lo[sl, 3], Z.hi[sl, 3]], 1)
        e_b = eps[sl]
        W = (I(Z.lo[sl, 6], Z.hi[sl, 6]) + e_b * I(Z.lo[sl, 7], Z.hi[sl, 7])) * EF.iexp(Ell[sl])
        Wlo[ix] = W.lo
        # Krawczyk exclusion
        rows = (1, 2, 3)
        J = [[None] * 3 for _ in range(3)]
        for a_, r_ in enumerate(rows):
            J[a_][0] = I(D.lo[sl, r_, 0], D.hi[sl, r_, 0]) * (e_b * (-1.5))
            J[a_][1] = I(D.lo[sl, r_, 1], D.hi[sl, r_, 1])
            J[a_][2] = I(D.lo[sl, r_, 2], D.hi[sl, r_, 2])
        Nc = [I(Z.lo[m:, r_], Z.hi[m:, r_]) for r_ in rows]
        Jm = np.stack([np.stack([J[a_][j].mid for j in range(3)], -1) for a_ in range(3)], -2)    # (m, 3, 3)
        try:
            Cm = np.linalg.inv(Jm)
        except np.linalg.LinAlgError:
            return
        Bx = [I(B[ix, k], B[ix, 3 + k]) for k in range(3)]
        for i in range(3):
            acc = I(cen[ix, i])
            for j in range(3):
                acc = acc - Nc[j] * Cm[:, i, j]
            for j in range(3):
                Mij = I(np.full(m, 1.0 if i == j else 0.0))
                for l_ in range(3):
                    Mij = Mij - J[l_][j] * Cm[:, i, l_]
                acc = acc + Mij * (Bx[j] - cen[ix, j])
            dis = (acc.hi < Bx[i].lo) | (acc.lo > Bx[i].hi)
            excl[ix] |= np.where(np.isfinite(acc.lo) & np.isfinite(acc.hi), dis, False)
    rec(np.arange(n))
    return ok, Y2, Y3, P1, Wlo, excl


def shell_boxes(Hin, Hout):
    """the cube of half-width Hout around a8 minus the open cube of half-width Hin, as 26 boxes (3 x 3 x 3 grid with
    the centre removed); none of them contains a8 in its closure"""
    e = [np.array([A8_LAB[k] - Hout, A8_LAB[k] - Hin, A8_LAB[k] + Hin, A8_LAB[k] + Hout]) for k in range(3)]
    out = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == j == k == 1:
                    continue
                out.append([e[0][i], e[1][j], e[2][k], e[0][i + 1], e[1][j + 1], e[2][k + 1]])
    return np.array(out)


def l2_cover(H, workers=1, maxlev=40, Hin=K1_HALF):
    """adaptive (X) check on the shell between the cubes of half-width Hin (default K1) and H around a8, with the
    direct Lohner of
    verify/eight_iiiL.py (enclose: C1 Lohner of label boxes at s = 1; bisect: widest scaled side, M weighted 4x).
    Returns (counts, number of boxes evaluated, seconds, stuck boxes)."""
    import eight_iiiL as XL
    B = shell_boxes(Hin, H)
    stats = dict(alpha=0, beta=0, gamma=0, delta=0, k1=0, lohner_fail=0)
    nev, t0, stuck, lev = 0, time.time(), [], 0
    while len(B) and lev < maxlev:
        in_k1 = np.all(B[:, 0:3] >= K1_LO, 1) & np.all(B[:, 3:6] <= K1_HI, 1)
        stats["k1"] += int(in_k1.sum())
        B = B[~in_k1]
        if not len(B):
            break
        ok, Y2, Y3, P1, Wlo, ex = enclose_k(B)
        nev += len(B)
        alpha = ok & ((Y2[:, 0] > 0) | (Y2[:, 1] < 0) | (Y3[:, 0] > 0) | (Y3[:, 1] < 0))
        beta = ok & ~alpha & (Wlo > A8_HI)
        gamma = ok & ~alpha & ~beta & ((P1[:, 0] > 0) | (P1[:, 1] < 0))
        delta = ok & ~alpha & ~beta & ~gamma & ex
        acc = alpha | beta | gamma | delta
        for k, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma), ("delta", delta)):
            stats[k] += int(v.sum())
        stats["lohner_fail"] += int((~ok).sum())
        nxt = []
        for b in B[~acc]:
            if np.max(b[3:6] - b[0:3]) < 1e-8:
                stuck.append(b)
            else:
                nxt += XL.bisect(b)
        B = np.array(nxt) if nxt else np.zeros((0, 6))
        lev += 1
        print(f"    H {H:g} level {lev}: evaluated {nev}, next {len(B)}, {stats}, {time.time() - t0:.0f} s", flush=True)
    return stats, nev, time.time() - t0, stuck + list(B)


def l2eta():
    SR.lowprio()
    import eight_field_lohner as L
    L.ORDER = int(os.environ.get("LORDER", "12"))
    out = []
    Hs = [K1_HALF] + [float(x) for x in os.environ.get("X_H", "6e-5,1.2e-4,2.4e-4,4.8e-4").split(",")]
    for Hin, H in zip(Hs[:-1], Hs[1:]):
        st, nev, dt, left = l2_cover(H, Hin=Hin)
        out.append((Hin, H, nev, dt, len(left)))
        print(f"  shell {Hin:g} .. {H:g} around a8: {nev} boxes evaluated, {dt:.0f} s, left {len(left)}, {st}",
              flush=True)
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "l2eta":
    l2eta()


MAXB = int(os.environ.get("X_MAXB", "3000"))


def l2_boxes(B, maxlev=30, tag="", leaves=None):
    """adaptive (X) check of the label boxes B (n, 6) at s = 1 (level-1 survivors): alpha / beta / gamma / delta
    (Krawczyk exclusion) with the batched C1 Lohner; boxes inside K1 skipped; the rest bisected (eight_iiiL.bisect)"""
    import eight_iiiL as XL
    stats = dict(alpha=0, beta=0, gamma=0, delta=0, k1=0, lohner_fail=0)
    nev, t0, lev = 0, time.time(), 0
    while len(B) and lev < maxlev:
        in_k1 = np.all(B[:, 0:3] >= K1_LO, 1) & np.all(B[:, 3:6] <= K1_HI, 1)
        stats["k1"] += int(in_k1.sum())
        B = B[~in_k1]
        if not len(B):
            break
        ok, Y2, Y3, P1, Wlo, ex = enclose_k(B)
        nev += len(B)
        alpha = ok & ((Y2[:, 0] > 0) | (Y2[:, 1] < 0) | (Y3[:, 0] > 0) | (Y3[:, 1] < 0))
        beta = ok & ~alpha & (Wlo > A8_HI)
        gamma = ok & ~alpha & ~beta & ((P1[:, 0] > 0) | (P1[:, 1] < 0))
        delta = ok & ~alpha & ~beta & ~gamma & ex
        acc = alpha | beta | gamma | delta
        for k, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma), ("delta", delta)):
            stats[k] += int(v.sum())
            if leaves is not None and v.any():
                leaves.append((k, B[v].copy()))
        stats["lohner_fail"] += int((~ok).sum())
        nxt = []
        for b in B[~acc]:
            nxt += XL.bisect(b)
        B = np.array(nxt) if nxt else np.zeros((0, 6))
        lev += 1
        print(f"    {tag} level {lev}: evaluated {nev}, next {len(B)}, {stats}, {time.time() - t0:.0f} s", flush=True)
        if len(B) > MAXB:
            print(f"    {tag}: more than {MAXB} open boxes, stopped (reported as left)", flush=True)
            break
    return stats, nev, time.time() - t0, B


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "l2surv":
    # l2surv ell_lo ell_hi [max boxes]: level 2 on the level-1 survivors (iiiL_l1.npz) with centre ell in the window
    SR.lowprio()
    import eight_field_lohner as L
    L.ORDER = int(os.environ.get("LORDER", "12"))
    z = np.load(OUT)
    S = z["surv"]
    lo, hi = S[:, 7:10], S[:, 10:13]
    c = 0.5 * (lo[:, 0] + hi[:, 0])
    sel = (c >= float(sys.argv[2])) & (c < float(sys.argv[3]))
    B = np.concatenate([lo[sel], hi[sel]], 1)
    if len(sys.argv) > 4:
        B = B[:int(sys.argv[4])]
    # merge: survivors grouped by the grid cell (2e-3, 2e-3, 1e-3) of their centre; each group's hull is one box
    # (it contains every survivor of the group)
    lim = np.array([1e-3, 1e-3, 5e-4])
    key = np.floor(0.5 * (B[:, 0:3] + B[:, 3:6]) / (2 * lim)).astype(np.int64)
    groups = {}
    for i, k in enumerate(map(tuple, key)):
        groups.setdefault(k, []).append(i)
    B = np.array([np.concatenate([B[ix, 0:3].min(0), B[ix, 3:6].max(0)]) for ix in groups.values()])
    print(f"l2surv: merged into {len(B)} hull boxes", flush=True)
    # pre-split to the size where the C1 Lohner works (half-widths <= 1e-3 in ell and phi, 5e-4 in M)
    out_ = []
    stack = list(B)
    while stack:
        b = stack.pop()
        w = (b[3:6] - b[0:3]) / 2 / lim
        k = int(np.argmax(w))
        if w[k] <= 1:
            out_.append(b)
            continue
        m = 0.5 * (b[k] + b[3 + k])
        b1, b2 = b.copy(), b.copy()
        b1[3 + k] = m
        b2[k] = m
        stack += [b1, b2]
    B = np.array(out_)
    print(f"l2surv: {len(B)} survivor boxes with centre ell in [{sys.argv[2]}, {sys.argv[3]}); ell length "
          f"{float(np.sum(B[:, 3] - B[:, 0])):.5f} (summed)", flush=True)
    st, nev, dt, left = l2_boxes(B, tag="surv")
    print(f"l2surv: {nev} boxes evaluated, {dt:.0f} s, left {len(left)}, {st}", flush=True)


# ---------------------------------------------------------------- parallel driver: level 1 over files, then level 2
# python verify/eight_iiiL_shadow.py par --workers N dir1[,dir2,...]
# Phase 1: a pool over the saved files that can have s = 1 labels (eps_c >= r_max^(-3/2)); each worker runs level 1
# (with refinement) and writes its own shard tubeb/iiiL_shard_<pid>_<t>.npz (restartable: files in the shards are
# skipped). Phase 2: the shells around a8 from K1 to H_SHELL (SHELLS) plus the merged level-1 survivors outside the
# cube of half-width H_SHELL, in tasks over the pool (l2_boxes, with the Krawczyk exclusion). Output
# tubeb/iiiL_final.npz (task table, every accepted leaf with its test, what is left) and the verdict.
H_SHELL = float(os.environ.get("X_HSHELL", "1.92e-3"))
SHELLS = [3e-5, 6e-5, 1.2e-4, 2.4e-4, 4.8e-4, 9.6e-4, 1.92e-3]
SHARD_GLOB = os.path.join(ROOT, "words/out/eight/tubeb/iiiL_shard_*.npz")
FINAL = os.path.join(ROOT, "words/out/eight/tubeb/iiiL_final.npz")
_X = {}


def _xinit():
    SR.lowprio()
    import eight_field_lohner as L
    L.ORDER = int(os.environ.get("LORDER", "12"))
    T = SR.tube()
    _X.update(T=T, er=SR.ell_range(T), rows=[], surv=[], done=[],
              shard=os.path.join(ROOT, f"words/out/eight/tubeb/iiiL_shard_{os.getpid()}_{int(time.time())}.npz"))


def _xfile(f):
    t0 = time.time()
    try:
        D, res = level1_file(_X["T"], f, _X["er"][0], _X["er"][1])
    except Exception as e:
        return f, None, f"error {e!r}"[:200], time.time() - t0
    cnt = {}
    for b, rows in res.items():
        for r in rows:
            cnt[r[0]] = cnt.get(r[0], 0) + 1
            if r[0] == "surv":
                lo = r[2] if r[2] is not None else np.full(3, np.nan)
                hi = r[3] if r[3] is not None else np.full(3, np.nan)
                _X["surv"].append([float(b), *lo, *hi])
                _X["rows"].append(os.path.basename(f))
    _X["done"].append((os.path.basename(f), cnt.get("alpha", 0), cnt.get("beta", 0), cnt.get("gamma", 0),
                       cnt.get("out", 0), cnt.get("surv", 0), cnt.get("notclosed", 0), cnt.get("notthin", 0),
                       time.time() - t0))
    dn_ = _X["done"]
    np.savez(_X["shard"], fname=np.array([d[0] for d in dn_], dtype="<U64"),
             counts=np.array([d[1:] for d in dn_], float),
             counts_cols=np.array(["alpha", "beta", "gamma", "out", "surv", "notclosed", "notthin", "sec"]),
             surv_fname=np.array(_X["rows"], dtype="<U64"), surv=np.array(_X["surv"], float).reshape(-1, 7))
    return f, cnt, "", time.time() - t0


def _xl2(task):
    # the C1 Lohner order for level 2: refine_many (level 1, same worker) sets eight_field_lohner.ORDER to SR_ORDER
    # (6) for the thin centres; the box enclosures of level 2 need the order they were timed with (LORDER, 12)
    import eight_field_lohner as L
    L.ORDER = int(os.environ.get("LORDER", "12"))
    kind, B = task
    leaves = []
    st, nev, dt, left = l2_boxes(B, tag=kind, leaves=leaves)
    return kind, st, nev, dt, left, leaves


def _shards(totals=None):
    """files done, level-1 survivors and (if totals is a dict) the summed per-file counts over all shards"""
    done, surv = set(), []
    for sh in glob.glob(SHARD_GLOB):
        # referee 5c C5: an unreadable shard must not be skipped (its survivors would be missing from phase 2)
        z = np.load(sh)
        done |= set(str(x) for x in z["fname"])
        if len(z["surv"]):
            surv.append(z["surv"])
        if totals is not None and len(z["fname"]):
            for c, v in zip(z["counts_cols"], z["counts"].sum(0)):
                totals[str(c)] = totals.get(str(c), 0) + float(v)
    return done, (np.concatenate(surv) if surv else np.zeros((0, 7)))


def _presplit(B, lim):
    out_, stack = [], list(B)
    while stack:
        b = stack.pop()
        w = (b[3:6] - b[0:3]) / 2 / lim
        k = int(np.argmax(w))
        if w[k] <= 1:
            out_.append(b)
            continue
        m = 0.5 * (b[k] + b[3 + k])
        b1, b2 = b.copy(), b.copy()
        b1[3 + k] = m
        b2[k] = m
        stack += [b1, b2]
    return np.array(out_).reshape(-1, 6)


def par(src, workers, chunk=48):
    import multiprocessing as mp
    SR.lowprio()
    T = SR.tube()
    ell_lo, ell_hi = SR.ell_range(T)
    emin = math.exp(ell_hi) ** -1.5 * (1 - 1e-9)        # eps_c below this: no s = 1 labels in the ell range
    files = [f for d in src.split(",") for f in sorted(glob.glob(os.path.join(d, "b_*.npz")))]
    cand = []
    for f in files:
        try:
            if float(np.load(f)["boxes"][:, 0].max()) >= emin:
                cand.append(f)
        except Exception:
            print(f"  {os.path.basename(f)}: unreadable, skipped (rerun later)", flush=True)
    done, _ = _shards()
    todo = [f for f in cand if os.path.basename(f) not in done]
    print(f"iiiL par: {len(files)} files, {len(cand)} with eps_c >= {emin:.4f}, {len(todo)} to do; {workers} workers",
          flush=True)
    t0 = time.time()
    pool = mp.Pool(workers, initializer=_xinit)
    tot = {}
    for k, (f, cnt, err, dt) in enumerate(pool.imap_unordered(_xfile, todo)):
        if cnt is None:
            print(f"  {os.path.basename(f)}: {err}", flush=True)
            tot["error"] = tot.get("error", 0) + 1
            continue
        for c, v in cnt.items():
            tot[c] = tot.get(c, 0) + v
        if cnt.get("surv", 0) or k % 50 == 0:
            print(f"  [{k + 1}/{len(todo)}] {os.path.basename(f)}: {cnt} ({dt:.0f} s)", flush=True)
    print(f"phase 1 done (this run): {tot}; {time.time() - t0:.0f} s", flush=True)
    allt = {}
    done, S = _shards(allt)
    allt["error"] = tot.get("error", 0)
    print(f"phase 1 over all shards: {len(done)} files, " + ", ".join(f"{k} {v:.0f}" for k, v in allt.items()
                                                                       if k != "sec"), flush=True)
    lo, hi = S[:, 1:4], S[:, 4:7]
    ok = np.all(np.isfinite(lo), 1) & np.all(np.isfinite(hi), 1)
    bad_surv = int((~ok).sum())                     # survivors of unclosed sub-boxes without an s = 1 box: reported
    lo, hi = lo[ok], hi[ok]
    inside = np.all(lo >= A8_LAB - H_SHELL, 1) & np.all(hi <= A8_LAB + H_SHELL, 1)
    B = np.concatenate([lo[~inside], hi[~inside]], 1)
    lim = np.array([1e-3, 1e-3, 5e-4])
    if len(B):
        key = np.floor(0.5 * (B[:, 0:3] + B[:, 3:6]) / (2 * lim)).astype(np.int64)
        groups = {}
        for i, kk in enumerate(map(tuple, key)):
            groups.setdefault(kk, []).append(i)
        B = np.array([np.concatenate([B[ix, 0:3].min(0), B[ix, 3:6].max(0)]) for ix in groups.values()])
        B = _presplit(B, lim)
    tasks = [(f"shell {SHELLS[i]:g}..{SHELLS[i + 1]:g}", shell_boxes(SHELLS[i], SHELLS[i + 1]))
             for i in range(len(SHELLS) - 1) if SHELLS[i + 1] <= H_SHELL * (1 + 1e-12)]
    tasks += [(f"surv chunk {i // chunk}", B[i:i + chunk]) for i in range(0, len(B), chunk)]
    print(f"phase 2: {len(S)} level-1 survivors ({bad_surv} without an s = 1 box), {int(inside.sum())} inside the "
          f"shell cube {H_SHELL:g}; {len(B)} Lohner boxes for the rest; {len(tasks)} tasks", flush=True)
    res = pool.map(_xl2, tasks, chunksize=1)
    pool.close()
    pool.join()
    table, leaves, left = [], [], []
    for (kind, st, nev, dt, lf, lv) in res:
        table.append((kind, nev, dt, len(lf), st))
        leaves += [(kind, t, b) for (t, bb) in lv for b in bb]
        left += list(lf)
        print(f"  {kind}: {nev} boxes evaluated, left {len(lf)}, {st}, {dt:.0f} s", flush=True)
    np.savez(FINAL, table_kind=np.array([t[0] for t in table]), table_nev=np.array([t[1] for t in table]),
             table_sec=np.array([t[2] for t in table]), table_left=np.array([t[3] for t in table]),
             table_stats=np.array([str(t[4]) for t in table]),
             leaf_task=np.array([x[0] for x in leaves]), leaf_test=np.array([x[1] for x in leaves]),
             leaf_box=np.array([x[2] for x in leaves]).reshape(-1, 6), left=np.array(left).reshape(-1, 6),
             k1_half=K1_HALF, a8=A8_LAB, h_shell=H_SHELL, shells=np.array(SHELLS), phase1=str(allt),
             bad_surv=bad_surv)
    okall = (not left) and bad_surv == 0 and allt.get("notclosed", 0) == 0 and allt.get("notthin", 0) == 0 \
        and allt.get("error", 0) == 0
    print(f"(iii-L) (X) on the given saved boxes: {'VERIFIED' if okall else 'NOT verified'} (left {len(left)}, "
          f"survivors without s = 1 box {bad_surv}, boxes not closed {allt.get('notclosed', 0):.0f}, not thin "
          f"{allt.get('notthin', 0):.0f}, errors {allt.get('error', 0)}); {time.time() - t0:.0f} s; -> {FINAL}", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "par":
    a_ = sys.argv[2:]
    nw_ = 8
    if "--workers" in a_:
        k_ = a_.index("--workers")
        nw_ = int(a_[k_ + 1])
        a_ = a_[:k_] + a_[k_ + 2:]
    par(a_[0], nw_)


# ---------------------------------------------------------------- (ii-1) with refinement on given files
# python verify/eight_iiiL_shadow.py ii1fix [ledger]  (files with ii1_fail > 0 in the ledger, default the merged one)
def ii1fix(ledger):
    global MODE
    MODE = "ii1"
    SR.lowprio()
    T = SR.tube()
    ell_lo, ell_hi = SR.ell_range(T)
    Lz = np.load(ledger)
    m = Lz["ii1_fail"] > 0
    files = sorted(set(str(f) for f in Lz["fname"][m]))
    dirs = [os.path.join(ROOT, "words/out/eight", d) for d in ("saves_step3", "saves_rerun", "saves_slabrec", "saves_gap", "saves_gap2")]
    print(f"ii1fix: {len(files)} files with (ii-1) failures in {os.path.basename(ledger)} ({int(m.sum())} boxes)",
          flush=True)
    rows_out = []
    t0 = time.time()
    for f in files:
        path = next((os.path.join(d, f) for d in dirs if os.path.exists(os.path.join(d, f))), None)
        if path is None:
            print(f"  {f}: not found", flush=True)
            continue
        t1 = time.time()
        D, res = level1_file(T, path, ell_lo, ell_hi)
        for b, rows in res.items():
            cnt = {}
            for r in rows:
                cnt[r[0]] = cnt.get(r[0], 0) + 1
            rows_out.append((f, b, cnt.get("alpha", 0), cnt.get("beta", 0), cnt.get("inside", 0), cnt.get("out", 0),
                             cnt.get("surv", 0), cnt.get("notclosed", 0)))
            if cnt.get("surv", 0):
                print(f"  {f} box {b}: {cnt}", flush=True)
        print(f"  {f}: done ({time.time() - t1:.0f} s)", flush=True)
    R = np.array([r[2:] for r in rows_out], float).reshape(-1, 6)
    np.savez(os.path.join(ROOT, "words/out/eight/tubeb/ii1_fix.npz"), fname=np.array([r[0] for r in rows_out]),
             box=np.array([r[1] for r in rows_out]), counts=R,
             cols=np.array(["alpha", "beta", "inside", "out", "surv", "notclosed"]))
    nsurv = int(R[:, 4].sum()) if len(R) else 0
    print(f"ii1fix: {len(rows_out)} boxes; label boxes accepted alpha {int(R[:, 0].sum())}, beta {int(R[:, 1].sum())}, "
          f"inside {int(R[:, 2].sum())}, out {int(R[:, 3].sum())}; surviving {nsurv}; "
          f"{'(ii-1) VERIFIED on these boxes' if nsurv == 0 else '(ii-1) NOT verified'}; {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ii1fix":
    ii1fix(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "words/out/eight/tubeb/iiL_ledger_merged.npz"))
