"""Stage 21, (ii-L) coverage tools (paper/review/eight_tube_boundary.md section 11): which boundary labels of tube A
need a record, the thin boxes below EPS_MIN, and the global coverage statement against the saved boxes.

needs   Rigorous classification of the boundary of Lambda(s), s in [S0, 1] (S0 = 1.165e-4, small-s lemma below), by
        interval patches (cube-sphere angles, EllTubeFit.label_box): (I) q_D(ell, phi) > 0.86 certainly; (w) W > A8_hi
        certainly by the energy identity W = 2 r M^2/s + s r^(-1/2)(2 Uhat - U(Y0)) >= 2 r M^2/s - s r^(-1/2) max U(Y0)
        (Uhat >= 0); otherwise the patch NEEDS a record. For needing patches: eps = s^2 r^(-3/2), kappa = M/sqrt(eps)
        (the shadow curve M = kappa sqrt(eps) at fixed phi carries the extremal). Output: the needing patches with
        their (phi, kappa, eps) hulls, the minimum eps that needs a record, and the part below EPS_MIN = 1e-3.
slab    Cells (phi, kappa) for thin-in-eps boxes below EPS_MIN: a grid over the (phi, kappa) hull of the needing patches
        with eps < 1e-3, minus cells inside the existing level-0 cover (R_shadow_v2 proved + R_level1_triage failures);
        eps_top with the slicing of eight_R_shadow at RS_EPSMIN = e_lo; kappa split to M half-width <= HM; output
        cells, tops and boxes (eps_top, phi, M range) for a closure-only saved run.
cover   Global statement: every needing extremal lies in the (phi, kappa) cell of some saved box whose records all pass
        (ledger), i.e. the required cells (level-0 cover + slab cells) are covered by the union of the passing saved
        cells. Exact for axis-parallel rectangles (elementary rectangles of the edge arrangement). Reports gaps.
    python verify/eight_iiL_cover.py needs [nslab] [nuv]
    python verify/eight_iiL_cover.py slab [hp] [hm]
    python verify/eight_iiL_cover.py cover"""
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
import eight_iiL_shadowrec as SR  # noqa: E402

S0 = 1.165e-4
EPS_MIN = 1e-3
OUTD = os.path.join(ROOT, "words/out/eight/tubeb")
NEEDS = os.path.join(OUTD, "iiL_needs.npz")
SLAB = os.path.join(OUTD, "iiL_slab_cells.npz")


def isincos_gen(a, b):
    """interval sin and cos on [a, b] (any a <= b): endpoint values and the extrema at the critical points k pi/2
    inside, pushed outward by 3 ulps"""
    import eight_tube_boundary as TB
    sa, sb, ca, cb = np.sin(a), np.sin(b), np.cos(a), np.cos(b)
    slo, shi = np.minimum(sa, sb), np.maximum(sa, sb)
    clo, chi = np.minimum(ca, cb), np.maximum(ca, cb)
    big = (b - a) >= 2 * math.pi
    k0 = np.ceil(a / (math.pi / 2) - 1e-12)
    for d in range(5):
        k = k0 + d
        c = k * (math.pi / 2)
        inn = (c >= a - 1e-12) & (c <= b + 1e-12)
        km = np.mod(k, 4)
        shi = np.where(inn & (km == 1), 1.0, shi)
        slo = np.where(inn & (km == 3), -1.0, slo)
        chi = np.where(inn & (km == 0), 1.0, chi)
        clo = np.where(inn & (km == 2), -1.0, clo)
    slo, shi = np.where(big, -1.0, slo), np.where(big, 1.0, shi)
    clo, chi = np.where(big, -1.0, clo), np.where(big, 1.0, chi)
    S = I(np.maximum(TB.push(slo, 3, -np.inf), -1.0), np.minimum(TB.push(shi, 3, np.inf), 1.0))
    C = I(np.maximum(TB.push(clo, 3, -np.inf), -1.0), np.minimum(TB.push(chi, 3, np.inf), 1.0))
    return S, C


def u_max(phi_lo, phi_hi):
    """upper bound of U(Y0), Y0 = cos phi e_I + sin phi e_Z, over phi intervals (arrays)"""
    import eight_tube_boundary as TB
    sn, cn = isincos_gen(np.asarray(phi_lo, float), np.asarray(phi_hi, float))
    tot = None
    for u in TB.UKI:
        d = I(1.0) - (cn * (u[0] * 0.5 + u[1] * TB.S3H) + sn * u[2])
        v = I(dn(1.0 / up(np.sqrt(up(d.hi)))), up(1.0 / dn(np.sqrt(np.maximum(dn(d.lo), 1e-300)))))
        tot = v if tot is None else tot + v
    return tot.hi


def needs(nslab=60, nuv=48):
    """classify boundary patches of tube A for s in [S0, 1]"""
    SR.lowprio()
    T = SR.tube()
    Tq = SR.section_tube(T, SR.Q_STAR)
    sg = np.unique(np.concatenate([np.geomspace(S0, 0.05, nslab + 1), np.linspace(0.05, 1.0, 77)]))
    sg[0], sg[-1] = S0, 1.0
    nslab = len(sg) - 1
    g = np.linspace(-1, 1, nuv + 1)
    t0 = time.time()
    rows = []
    cnt = {"I": 0, "w": 0, "need": 0}
    for k in range(nslab):
        F, UL, UH, VL, VH = [], [], [], [], []
        for f in range(6):
            for i in range(nuv):
                for j in range(nuv):
                    F.append(f); UL.append(g[i]); UH.append(g[i + 1]); VL.append(g[j]); VH.append(g[j + 1])
        F = np.array(F); n = len(F)
        slo, shi = np.full(n, sg[k]), np.full(n, sg[k + 1])
        alo, ahi = T.label_box(F, slo, shi, np.array(UL), np.array(UH), np.array(VL), np.array(VH))
        z = np.zeros(n)
        isI = Tq.box_certainly_outside(np.stack([alo[:, 0], alo[:, 1], z], 1), np.stack([ahi[:, 0], ahi[:, 1], z], 1), z, z)
        r_lo = dn(np.exp(alo[:, 0]))
        Mabs_lo = np.where((alo[:, 2] <= 0) & (ahi[:, 2] >= 0), 0.0, np.minimum(np.abs(alo[:, 2]), np.abs(ahi[:, 2])))
        Um = u_max(alo[:, 1], ahi[:, 1])
        Wlo = dn(dn(2.0 * r_lo * Mabs_lo ** 2) / up(sg[k + 1])) - up(up(sg[k + 1] * up(r_lo ** -0.5)) * Um)
        isW = dn(Wlo) > SR.A8_HI
        bad = ~np.all(np.isfinite(alo) & np.isfinite(ahi), 1) | np.any(ahi - alo > 1.0, 1)
        isI &= ~bad
        isW &= ~bad
        need = ~isI & ~isW
        cnt["I"] += int(isI.sum()); cnt["w"] += int((isW & ~isI).sum()); cnt["need"] += int(need.sum())
        nbad = int(bad.sum())
        if nbad:
            print(f"  slab {k}: {nbad} patches with an unusable enclosure (counted as need)", flush=True)
        for idx in np.flatnonzero(need & ~bad):
            el, eh = alo[idx, 0], ahi[idx, 0]
            eps_lo = dn(sg[k] ** 2 * math.exp(-1.5 * eh))
            eps_hi = up(sg[k + 1] ** 2 * math.exp(-1.5 * el))
            ka = np.array([alo[idx, 2], ahi[idx, 2]]) / np.sqrt(np.array([eps_lo, eps_hi]))[:, None]
            kap = alo[idx, 2] / math.sqrt(eps_hi) if alo[idx, 2] >= 0 else alo[idx, 2] / math.sqrt(eps_lo)
            kbp = ahi[idx, 2] / math.sqrt(eps_lo) if ahi[idx, 2] >= 0 else ahi[idx, 2] / math.sqrt(eps_hi)
            rows.append((sg[k], sg[k + 1], el, eh, alo[idx, 1], ahi[idx, 1], alo[idx, 2], ahi[idx, 2], eps_lo, eps_hi,
                         dn(kap), up(kbp)))
        if (k + 1) % 20 == 0:
            print(f"  slab {k + 1}/{nslab} (s up to {sg[k + 1]:.4g}): I {cnt['I']}, w {cnt['w']}, need {cnt['need']}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    R = np.array(rows)
    np.savez(NEEDS, rows=R, cols=np.array(["s_lo", "s_hi", "ell_lo", "ell_hi", "phi_lo", "phi_hi", "M_lo", "M_hi",
                                            "eps_lo", "eps_hi", "kappa_lo", "kappa_hi"]), cnt=np.array([cnt["I"], cnt["w"], cnt["need"]]))
    below = R[R[:, 8] < EPS_MIN] if len(R) else R
    print(f"needs: s in [{S0}, 1], {nslab} slabs x {6 * nuv * nuv} patches: I {cnt['I']}, w {cnt['w']}, need "
          f"{cnt['need']}; min eps needing a record {R[:, 8].min():.3e} (at s {R[np.argmin(R[:, 8]), 0]:.4g}); "
          f"needing patches with eps below {EPS_MIN}: {len(below)}"
          + (f", their kappa range [{below[:, 10].min():.2f}, {below[:, 11].max():.2f}], phi range "
             f"[{below[:, 4].min():.3f}, {below[:, 5].max():.3f}], s range [{below[:, 0].min():.4g}, {below[:, 1].max():.4g}]"
             if len(below) else "") + f"  ({time.time() - t0:.0f} s)", flush=True)
    return R


def classify_patches(T, Tq, P):
    """P (n, 7) = face, s_lo, s_hi, u_lo, u_hi, v_lo, v_hi -> (alo, ahi, isI, isW, bad)"""
    F = P[:, 0].astype(int)
    alo, ahi = T.label_box(F, P[:, 1], P[:, 2], P[:, 3], P[:, 4], P[:, 5], P[:, 6], nsub=2)
    n = len(P)
    z = np.zeros(n)
    bad = ~np.all(np.isfinite(alo) & np.isfinite(ahi), 1) | np.any(ahi - alo > 1.0, 1)
    alo_s = np.where(bad[:, None], 0.0, alo)
    ahi_s = np.where(bad[:, None], 0.0, ahi)
    isI = Tq.box_certainly_outside(np.stack([alo_s[:, 0], alo_s[:, 1], z], 1), np.stack([ahi_s[:, 0], ahi_s[:, 1], z], 1), z, z)
    r_lo = dn(np.exp(alo_s[:, 0]))
    Mabs_lo = np.where((alo_s[:, 2] <= 0) & (ahi_s[:, 2] >= 0), 0.0, np.minimum(np.abs(alo_s[:, 2]), np.abs(ahi_s[:, 2])))
    Um = u_max(alo_s[:, 1], ahi_s[:, 1])
    Wlo = dn(dn(2.0 * r_lo * Mabs_lo ** 2) / up(P[:, 2])) - up(up(P[:, 2] * up(r_lo ** -0.5)) * Um)
    isW = dn(Wlo) > SR.A8_HI
    return alo, ahi, isI & ~bad, isW & ~bad & ~isI, bad


OCC_P0, OCC_DP, OCC_K0, OCC_DK = 0.30, 0.01, -8.0, 0.02
SPLIT_SMAX = 0.003
OCC = np.full((115, 800), np.inf)          # min eps of needing patches per (phi, asinh kappa) cell


def needs_below(depth=3, nuv=48):
    """adaptive version for the part that matters for the thin boxes: boundary patches whose eps range reaches below
    EPS_MIN (s up to sqrt(EPS_MIN) e^(0.75 ell_max)); a needing patch is split 2 x 2 in (u, v) and 2 in s up to `depth`
    levels before it is recorded as needing a record"""
    SR.lowprio()
    T = SR.tube()
    Tq = SR.section_tube(T, SR.Q_STAR)
    ell_lo, ell_hi = SR.ell_range(T)
    smax = min(1.0, math.sqrt(EPS_MIN) * math.exp(0.75 * ell_hi) * 1.01)
    sg = np.unique(np.concatenate([np.geomspace(S0, min(0.05, smax), 60), np.linspace(0.05, smax, max(2, int((smax - 0.05) / 0.0125) + 1))]))         if smax > 0.05 else np.geomspace(S0, smax, 60)
    g = np.linspace(-1, 1, nuv + 1)
    t0 = time.time()
    rows, cnt = [], {"I": 0, "w": 0, "need": 0, "above": 0}
    for k in range(len(sg) - 1):
        P = np.array([[f, sg[k], sg[k + 1], g[i], g[i + 1], g[j], g[j + 1]]
                      for f in range(6) for i in range(nuv) for j in range(nuv)])
        dep_k = depth if sg[k] < SPLIT_SMAX else 0       # beyond SPLIT_SMAX nearly every patch needs: no splitting
        for lev in range(dep_k + 1):
            alo, ahi, isI, isW, bad = classify_patches(T, Tq, P)
            eps_lo = dn(P[:, 1] ** 2 * np.exp(-1.5 * np.where(bad, 10.0, ahi[:, 0])))
            above = ~bad & (eps_lo >= EPS_MIN)          # needs nothing from the thin boxes (eps >= EPS_MIN)
            need = ~isI & ~isW & ~above
            cnt["I"] += int(isI.sum()); cnt["w"] += int(isW.sum()); cnt["above"] += int((above & ~isI & ~isW).sum())
            if lev == dep_k or not need.any():
                idx = np.flatnonzero(need & ~bad)
                nbad_ = int((need & bad).sum())
                cnt["need"] += int(need.sum())
                if len(idx):
                    el, eh = alo[idx, 0], ahi[idx, 0]
                    e_lo = dn(P[idx, 1] ** 2 * np.exp(-1.5 * eh))
                    e_hi = up(P[idx, 2] ** 2 * np.exp(-1.5 * el))
                    Ml, Mh = alo[idx, 2], ahi[idx, 2]
                    kl = np.where(Ml >= 0, Ml / np.sqrt(e_hi), Ml / np.sqrt(np.maximum(e_lo, 1e-300)))
                    kh = np.where(Mh >= 0, Mh / np.sqrt(np.maximum(e_lo, 1e-300)), Mh / np.sqrt(e_hi))
                    # occupancy of the (phi, asinh kappa) grid
                    i0 = np.floor((alo[idx, 1] - OCC_P0) / OCC_DP).astype(int)
                    i1 = np.floor((ahi[idx, 1] - OCC_P0) / OCC_DP).astype(int)
                    j0 = np.floor((np.arcsinh(kl) - OCC_K0) / OCC_DK).astype(int)
                    j1 = np.floor((np.arcsinh(kh) - OCC_K0) / OCC_DK).astype(int)
                    # referee 5b D3: a needing patch outside the grid must not be booked into an edge cell
                    assert (i0.min() >= 0 and i1.max() < OCC.shape[0] and j0.min() >= 0 and j1.max() < OCC.shape[1]), \
                        "needing patch outside the occupancy grid: enlarge OCC"
                    for a_, b_, c_, d_, ee in zip(i0, i1, j0, j1, e_lo):
                        OCC[a_:b_ + 1, c_:d_ + 1] = np.minimum(OCC[a_:b_ + 1, c_:d_ + 1], ee)
                    absM = np.where((Ml <= 0) & (Mh >= 0), 0.0, np.minimum(np.abs(Ml), np.abs(Mh)))
                    rows.append((sg[k], sg[k + 1], len(idx), nbad_, float(e_lo.min()), float(e_hi.max()),
                                 float(absM.min()), float(np.maximum(np.abs(Ml), np.abs(Mh)).max()),
                                 float(alo[idx, 1].min()), float(ahi[idx, 1].max()), float(kl.min()), float(kh.max())))
                elif nbad_:
                    rows.append((sg[k], sg[k + 1], 0, nbad_, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                break
            Q = P[need]
            um, vm, sm = 0.5 * (Q[:, 3] + Q[:, 4]), 0.5 * (Q[:, 5] + Q[:, 6]), np.sqrt(Q[:, 1] * Q[:, 2])
            parts = []
            for (sa, sb) in ((Q[:, 1], sm), (sm, Q[:, 2])):
                for (ua, ub) in ((Q[:, 3], um), (um, Q[:, 4])):
                    for (va, vb) in ((Q[:, 5], vm), (vm, Q[:, 6])):
                        parts.append(np.stack([Q[:, 0], sa, sb, ua, ub, va, vb], 1))
            P = np.concatenate(parts)
        if (k + 1) % 10 == 0:
            print(f"  slab {k + 1}/{len(sg) - 1} (s up to {sg[k + 1]:.4g}): I {cnt['I']}, w {cnt['w']}, above {cnt['above']}, "
                  f"need {cnt['need']}  ({time.time() - t0:.0f} s)", flush=True)
    R = np.array(rows) if rows else np.zeros((0, 12))
    np.savez(NEEDS, rows=R, cols=np.array(["s_lo", "s_hi", "n_need", "n_bad", "eps_min", "eps_max", "absM_min", "absM_max",
                                            "phi_lo", "phi_hi", "kappa_lo", "kappa_hi"]),
             cnt=np.array([cnt["I"], cnt["w"], cnt["need"], cnt["above"]]), smax=smax, depth=depth,
             occ=OCC, occ_grid=np.array([OCC_P0, OCC_DP, OCC_K0, OCC_DK]))
    ok = np.isfinite(R[:, 4]) if len(R) else np.zeros(0, bool)
    occ = np.isfinite(OCC)
    print(f"needs below EPS_MIN: s in [{S0}, {smax:.4f}], adaptive depth {depth}: I {cnt['I']}, w {cnt['w']}, eps >= EPS_MIN "
          f"{cnt['above']}, need {cnt['need']} ({int(R[:, 3].sum()) if len(R) else 0} unusable enclosures)"
          + (f"; needing slabs s in [{R[ok, 0].min():.4g}, {R[ok, 1].max():.4g}], min eps {R[ok, 4].min():.3e}, |M| in "
             f"[{R[ok, 6].min():.4f}, {R[ok, 7].max():.4f}], kappa in [{R[ok, 10].min():.1f}, {R[ok, 11].max():.1f}], "
             f"occupied (phi, asinh kappa) cells {occ.sum()}" if ok.any() else "")
          + f"  ({time.time() - t0:.0f} s)", flush=True)
    return R


# ---------------------------------------------------------------- rectangle coverage (exact)
def covered(req, cov):
    """req (n, 4), cov (m, 4) rectangles [x0, x1, y0, y1]: for each req rectangle, the uncovered area fraction (0 =
    fully covered), by the elementary rectangles of the edge arrangement restricted to it"""
    out = np.zeros(len(req))
    for i, (x0, x1, y0, y1) in enumerate(req):
        sel = (cov[:, 0] < x1) & (cov[:, 1] > x0) & (cov[:, 2] < y1) & (cov[:, 3] > y0)
        C = cov[sel]
        if not len(C):
            out[i] = 1.0
            continue
        xs = np.unique(np.clip(np.concatenate([[x0, x1], C[:, 0], C[:, 1]]), x0, x1))
        ys = np.unique(np.clip(np.concatenate([[y0, y1], C[:, 2], C[:, 3]]), y0, y1))
        unc = 0.0
        for a, b in zip(xs[:-1], xs[1:]):
            xm = 0.5 * (a + b)
            colC = C[(C[:, 0] <= a) & (C[:, 1] >= b)]
            for c, d in zip(ys[:-1], ys[1:]):
                if not np.any((colC[:, 2] <= c) & (colC[:, 3] >= d)):
                    unc += (b - a) * (d - c)
        out[i] = unc / max((x1 - x0) * (y1 - y0), 1e-300)
    return out


def level0_cells():
    d = np.load(os.path.join(ROOT, "words/out/eight/R_shadow_v2.npz"))
    t = np.load(os.path.join(ROOT, "words/out/eight/R_level1_triage.npz"))
    return np.concatenate([d["proved_cells"][:, :4], t["cells"]])


def slab_cells(hp=0.02, hm=0.005, kcoarse=5):
    """cells below EPS_MIN from the occupancy grid of needs_below (min eps of needing patches per (phi, asinh kappa)
    cell): candidate cells phi width 2 hp, asinh-kappa width kcoarse fine steps, kept where some occupied fine cell
    has eps < EPS_MIN; minus cells inside the level-0 cover; eps_top by the shadow slicing with RS_EPSMIN = e_lo; kappa
    split until the box's M half-width <= hm"""
    SR.lowprio()
    z = np.load(NEEDS)
    occ = z["occ"]
    P0, DP, K0, DK = z["occ_grid"]
    fine = occ < EPS_MIN
    if not fine.any():
        print("slab cells: no needing curve below EPS_MIN", flush=True)
        return None
    e_lo = float(occ[fine].min()) * 0.99
    os.environ["RS_EPSMIN"] = repr(e_lo)
    import importlib
    import eight_R_shadow as RS
    importlib.reload(RS)
    tb = SR.tube()
    plo, phi_, mlo, mhi, emax = RS.q_bounds(tb)
    pstep = int(round(2 * hp / DP))
    cells = []
    for i in range(0, occ.shape[0], pstep):
        for j in range(0, occ.shape[1], kcoarse):
            if fine[i:i + pstep, j:j + kcoarse].any():
                cells.append([P0 + i * DP, P0 + (i + pstep) * DP, math.sinh(K0 + j * DK), math.sinh(K0 + (j + kcoarse) * DK)])
    cells = np.array(cells)
    t0 = time.time()
    inside_old = covered(cells, level0_cells()) == 0.0
    cells = cells[~inside_old]
    tops = RS.eps_top(tb, cells, emax)
    cells, tops = RS.split_cells(cells, tops, hm)
    tops = RS.eps_top(tb, cells, emax)
    keep = np.isfinite(tops)
    cells, tops = cells[keep], tops[keep]
    boxes = np.array([RS.make_box(c, t) for c, t in zip(cells, tops)]) if len(cells) else np.zeros((0, 6))
    np.savez(SLAB, cells=cells, tops=tops, boxes=boxes, e_lo=e_lo, hp=hp, hm=hm)
    q = np.quantile(tops, [0, 0.25, 0.5, 0.75, 1]) if len(tops) else []
    print(f"slab cells below EPS_MIN = {EPS_MIN}: e_lo = {e_lo:.3e}; occupied fine cells {int(fine.sum())}; "
          f"{int(inside_old.sum())} candidate cells already inside the level-0 cover; {len(cells)} cells with a finite "
          f"eps_top (box M half-width <= {hm}, phi width {2 * hp}); eps_top quantiles {np.round(q, 8)}; "
          f"{time.time() - t0:.0f} s; saved {SLAB}", flush=True)
    return cells, tops, boxes


def cover(ledger=None, ii1fix=None, tag="", topfile=None):
    """global coverage of the required cells by the passing saved boxes (ledger; default the main ledger). With
    ii1fix (tubeb/ii1_fix.npz) a box also needs (ii-1): ii1_fail == 0 in the ledger, or no surviving label box in
    the refined (ii-1) run. A saved box covers its own (phi, kappa) cell: its eps is the eps_top of that cell
    (eight_R_shadow.make_box), so it covers every needing label of the cell."""
    L_ = SR.load_ledger(ledger) if ledger else SR.load_ledger()
    rids = sorted(set(str(x) for x in L_["run_id"]))
    assert len(rids) == 1 and rids[0], f"the cover needs a ledger from ONE run (run ids {rids})"   # referee 5b D4
    print(f"cover{tag}: ledger run id {rids[0]}")
    ok = (L_["n_fail"] == 0) & (L_["n_unres"] == 0)
    if ii1fix is not None:
        z = np.load(ii1fix)
        fixed = {(str(f), int(b)) for f, b, c in zip(z["fname"], z["box"], z["counts"]) if c[4] == 0 and c[5] == 0}
        ii1ok = np.array([(L_["ii1_fail"][i] <= 0) or ((str(L_["fname"][i]), int(L_["box"][i])) in fixed)
                          for i in range(len(ok))])
        print(f"cover{tag}: (ii-1) failing boxes in the ledger {int(np.sum(L_['ii1_fail'] > 0))}, repaired by the "
              f"refined run {int(np.sum((L_['ii1_fail'] > 0) & ii1ok))}")
        ok &= ii1ok
    if topfile is not None:
        zt = np.load(topfile)
        tres = {(str(f), int(b)): int(r) for f, b, r in zip(zt["fname"], zt["box"], zt["res"])}
        assert set(str(x) for x in zt["run_id"]) <= set(str(x) for x in L_["run_id"]), "top check of another run"
        topok = np.array([tres.get((str(L_["fname"][i]), int(L_["box"][i])), 2) != 2 for i in range(len(ok))])
        print(f"cover{tag}: top check (referee 5c C2): passing boxes not certified at the top of Q: "
              f"{int(np.sum(ok & ~topok))}")
        ok &= topok
    eps = L_["eps"]
    sq = np.sqrt(eps)
    k1 = np.minimum(L_["M_lo"] / sq, L_["M_hi"] / sq)
    k2 = np.maximum(L_["M_lo"] / sq, L_["M_hi"] / sq)
    covs = np.stack([L_["phi_lo"], L_["phi_hi"], k1, k2], 1)[ok]
    req = level0_cells()
    names = ["level-0 cover"] * len(req)
    if os.path.exists(SLAB):
        sc = np.load(SLAB)["cells"]
        req = np.concatenate([req, sc])
        names += ["slab"] * len(sc)
    names = np.array(names)
    unc = covered(req, covs) if len(covs) else np.ones(len(req))
    for nm in ("level-0 cover", "slab"):
        sel = names == nm
        if sel.any():
            print(f"cover: {nm}: {sel.sum()} required cells; fully covered {np.sum(unc[sel] == 0)}, partly "
                  f"{np.sum((unc[sel] > 0) & (unc[sel] < 1))}, not at all {np.sum(unc[sel] == 1)}; uncovered area "
                  f"fraction {np.sum(unc[sel] * ((req[sel, 1] - req[sel, 0]) * (req[sel, 3] - req[sel, 2]))) / np.sum((req[sel, 1] - req[sel, 0]) * (req[sel, 3] - req[sel, 2])):.4f}")
    print(f"cover{tag}: saved boxes in the ledger {len(ok)}, passing {ok.sum()}, not passing {np.sum(~ok)} "
          f"(tube not closed {int(np.sum(L_['n_fail'] == -1))})")
    np.savez(os.path.join(OUTD, f"iiL_cover_gaps{tag}.npz"), req=req, names=names, uncovered=unc)
    return unc



# ---------------------------------------------------------------- top check (referee 5c C2, C7)
# For (X) and (ii-1) a passing box must sit at or above the top of Q on its own (phi, kappa) rectangle: every eps
# above eps_B (up to emax of Q) with the rectangle's (phi, kappa) must be certainly outside Q. Tested on slices in
# log eps (ratio 1.25 first, failing slices split in 4, down to ratio 1.002) with eight_R_cover2.outside at the
# production slab count (RC_NSLAB 1000): first for tube A; a slice that tube A does not certify is tested for the
# earlier tube (the one the level-0 / step-1 cells and tops were made with), which is enough because tube A lies inside
# it for all s (tube fact F6, verify/eight_tube_facts.py), hence Q(A) inside Q(old). A box passes if every slice is
# certified; the result per box (A / old / FAIL) goes to tubeb/iiL_topcheck.npz.
OLD_TUBE = os.path.join(ROOT, "words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper.npz")
TOPCHECK = os.path.join(OUTD, "iiL_topcheck.npz")
_TC = {}


def _tc_init():
    SR.lowprio()
    os.environ["RC_NSLAB"] = os.environ.get("TC_NSLAB", "1000")
    import importlib
    import eight_R_cover2 as RC
    importlib.reload(RC)
    import words_eight_tubeb as W
    import eight_R_shadow as RS
    tA, tO = W.EllTubeFit(SR.TUBE_A), W.EllTubeFit(OLD_TUBE)
    emax = max(RS.q_bounds(tA)[4], RS.q_bounds(tO)[4]) * 1.02
    _TC.update(RC=RC, tA=tA, tO=tO, emax=emax)


def _slices_boxes(sl):
    """sl rows (le_a, le_b, phi_a, phi_b, ka, kb) -> parameter boxes (log eps lo, hi, phi lo, hi, M lo, hi)"""
    ea, eb = np.exp(sl[:, 0]), np.exp(sl[:, 1])
    sa, sb = np.sqrt(ea), np.sqrt(eb)
    v = np.stack([sl[:, 4] * sa, sl[:, 4] * sb, sl[:, 5] * sa, sl[:, 5] * sb])
    lo, hi = v.min(0), v.max(0)
    lo = lo - 4 * np.spacing(np.abs(lo)) - 1e-300
    hi = hi + 4 * np.spacing(np.abs(hi)) + 1e-300
    return np.stack([sl[:, 0] - 1e-12, sl[:, 1] + 1e-12, sl[:, 2], sl[:, 3], lo, hi], 1)


def _tc_task(rows):
    """rows (n, 5) = eps_B, phi_lo, phi_hi, kappa_lo, kappa_hi -> per row 0 (A), 1 (old tube via F6), 2 (FAIL)"""
    RC, tA, tO, emax = _TC["RC"], _TC["tA"], _TC["tO"], _TC["emax"]
    n = len(rows)
    res = np.zeros(n, int)
    work = []                                  # (row index, le_a, le_b)
    for i, (e, pa, pb, ka, kb) in enumerate(rows):
        if e * (1 + 1e-12) >= emax:
            continue
        m = max(1, int(math.ceil(math.log(emax / e) / math.log(1.25))))
        ed = np.linspace(math.log(e), math.log(emax), m + 1)
        work += [(i, ed[j], ed[j + 1]) for j in range(m)]
    while work:
        W_ = np.array(work)
        idx = W_[:, 0].astype(int)
        sl = np.stack([W_[:, 1], W_[:, 2], rows[idx, 1], rows[idx, 2], rows[idx, 3], rows[idx, 4]], 1)
        B = _slices_boxes(sl)
        oA = RC.outside(B, tA)
        oO = np.ones(len(B), bool)
        if (~oA).any():
            oO[~oA] = RC.outside(B[~oA], tO)
            res[np.unique(idx[~oA & oO])] = np.maximum(res[np.unique(idx[~oA & oO])], 1)
        nxt = []
        for j in np.flatnonzero(~oA & ~oO):
            a, b = W_[j, 1], W_[j, 2]
            if b - a <= math.log(1.002):
                res[idx[j]] = 2
                continue
            q = np.linspace(a, b, 5)
            nxt += [(idx[j], q[k], q[k + 1]) for k in range(4)]
        work = [w for w in nxt if res[w[0]] != 2]
    return res


def topcheck(ledger, workers=8, chunk=64):
    import multiprocessing as mp
    SR.lowprio()
    L_ = SR.load_ledger(ledger)
    ok = (L_["n_fail"] == 0) & (L_["n_unres"] == 0)
    sq = np.sqrt(L_["eps"])
    k1 = np.minimum(L_["M_lo"] / sq, L_["M_hi"] / sq)
    k2 = np.maximum(L_["M_lo"] / sq, L_["M_hi"] / sq)
    rows = np.stack([L_["eps"], L_["phi_lo"], L_["phi_hi"], k1, k2], 1)[ok]
    t0 = time.time()
    parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with mp.Pool(workers, initializer=_tc_init) as pool:
        res = np.concatenate(pool.map(_tc_task, parts, chunksize=1)) if parts else np.zeros(0, int)
    np.savez(TOPCHECK, fname=L_["fname"][ok], box=L_["box"][ok], res=res, run_id=L_["run_id"][ok],
             nslab=int(os.environ.get("TC_NSLAB", "1000")))
    print(f"topcheck ({os.path.basename(ledger)}, RC_NSLAB {os.environ.get('TC_NSLAB', '1000')}): {len(res)} passing "
          f"boxes; at the top of Q(A): {int(np.sum(res == 0))}, of Q(old) (tube A inside it, F6): {int(np.sum(res == 1))}, "
          f"NOT at the top: {int(np.sum(res == 2))}; {time.time() - t0:.0f} s -> {TOPCHECK}", flush=True)
    for i in np.flatnonzero(res == 2)[:20]:
        print(f"  {L_['fname'][ok][i]} box {int(L_['box'][ok][i])}: eps {rows[i, 0]:.6g} phi [{rows[i, 1]:.4f}, "
              f"{rows[i, 2]:.4f}] kappa [{rows[i, 3]:.4f}, {rows[i, 4]:.4f}]", flush=True)
    return res

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "needs":
        needs(int(sys.argv[2]) if len(sys.argv) > 2 else 60, int(sys.argv[3]) if len(sys.argv) > 3 else 48)
    elif cmd == "needsbelow":
        needs_below(int(sys.argv[2]) if len(sys.argv) > 2 else 3, int(sys.argv[3]) if len(sys.argv) > 3 else 48)
    elif cmd == "slab":
        slab_cells(float(sys.argv[2]) if len(sys.argv) > 2 else 0.02, float(sys.argv[3]) if len(sys.argv) > 3 else 0.005)
    elif cmd == "cover":                    # cover [ledger] [ii1fix npz] [topcheck npz]
        led = sys.argv[2] if len(sys.argv) > 2 else None
        fix = sys.argv[3] if len(sys.argv) > 3 else None
        top = sys.argv[4] if len(sys.argv) > 4 else None
        cover(led, tag="_records")
        if fix:
            cover(led, fix, tag="_with_ii1", topfile=top)
    elif cmd == "topcheck":                 # topcheck <ledger> [workers]
        topcheck(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
