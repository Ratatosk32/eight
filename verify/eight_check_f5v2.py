"""Figure-eight certificate checker: step F5 (cover) of stage F, version 2, from the saved outputs of one or several
stage F runs (the official run's F plus F runs on added save files, e.g. re-topped leaves of a repaired certificate).
Version 1 (verify/eight_check_f5.py, frozen) is unchanged; differences: (a) several stage F output directories, each
with the same log requirements; (b) the universe rule below EPS_MIN accepts a needing fine cell whose part outside the
passing leaf rectangles holds no point of Q with eps >= e_lo (eps_top nan, tube A, eight_R_cover2.outside at
RC_NSLAB 1000, NSL 240 eps slices from e_lo = 0.99 x the smallest eps of a needing patch): the rule of the slab-cell
construction (eight_iiL_cover.slab_cells) and of stage E; (c) the boxes that F4 left not at the top are re-tested by
the same top check (eight_check_records._top_task) at a finer s-slab count (TOP_LADDER = (4000,); any slicing is
sound, the slab count only changes the strength of the outside test).

    stage_f5v2(run_dir, f_dirs, leaves, workers=2, old_tube=None, check_universe=True, mutation=None) -> (passed, reason)

Inputs: for every F dir: stage_f.npz, stage_f.log (F1: failing rows, unresolved, refinement exceptions, unclosed
sub-boxes, violations and tiling failures 0; F2: still failing after refinement 0; F3: survivors without an s = 1 box 0
and left 0; F4 present and consistent with the npz); needs_below.npz from the first F dir that has one (it depends on
the tube only); run_dir/k1_a8.npz; the certificate leaves (n, 6) = phi_a, phi_b, kappa_a, kappa_b, eps_top, type.
Steps: the good boxes of every F dir (closed and thin, records pass, ii1 ok, at the top of Q) are pooled; every leaf
is matched to a good box by eight_check.leaf_boxes (same eps and phi edges bit for bit, M containment); a leaf without
one passes only if the label tests show it needs no record (eight_check_records._gap_task); the needing fine cells
below EPS_MIN must lie in the union of the passing leaf rectangles or have an empty uncovered part (above). One line
"STAGE F5v2 PASS" / "STAGE F5v2 FAIL <reason>"; any exception is a FAIL. Mutation m11: the good box of one leaf that
needs records is deleted (must FAIL)."""
import math
import os
import sys
import time

for _k in list(os.environ):
    if _k.startswith(("SR_", "X_", "RC_", "RS_", "CT_", "TC_")) or _k == "LORDER":
        del os.environ[_k]

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

EMPTY = dict(NSLAB=1000, NSL=240, E_LO_FACTOR=0.99)
TOP_LADDER = (4000,)        # s-slab counts for re-testing the boxes F4 left not at the top (any slicing is sound)


def _top_retest(args):
    """F4's top check (eight_check_records._top_task: tube A, then the declared earlier tube) at a finer s-slab count"""
    import eight_check_records as F
    rows, nslab = args
    RC = F._G["RC"]
    old = RC.NSLAB
    RC.NSLAB = nslab
    try:
        return F._top_task(rows)
    finally:
        RC.NSLAB = old
_E = {}


def _empty_init(tube_a, e_lo):
    import eight_iiL_shadowrec as SR
    SR.lowprio()
    import eight_R_cover2 as RC
    import eight_R_shadow as RS
    import words_eight_tubeb as W
    RC.NSLAB = EMPTY["NSLAB"]
    RS.EPS_MIN = e_lo
    RS.NSL = EMPTY["NSL"]
    tb = W.EllTubeFit(tube_a)
    _E.update(RS=RS, tb=tb, emax=RS.q_bounds(tb)[4])


def _empty_task(rects):
    """eps_top of the rectangles (n, 4) with eps slices from e_lo: True where nan (no point of Q with eps >= e_lo)"""
    t = _E["RS"].eps_top(_E["tb"], np.asarray(rects, float).reshape(-1, 4), _E["emax"])
    return ~np.isfinite(t)


def uncovered_parts(cell, rects):
    """the elementary rectangles of the edge arrangement inside cell that no rectangle of rects covers"""
    x0, x1, y0, y1 = cell
    sel = (rects[:, 0] < x1) & (rects[:, 1] > x0) & (rects[:, 2] < y1) & (rects[:, 3] > y0)
    Cr = rects[sel]
    xs = np.unique(np.clip(np.concatenate([[x0, x1], Cr[:, 0], Cr[:, 1]]), x0, x1))
    ys = np.unique(np.clip(np.concatenate([[y0, y1], Cr[:, 2], Cr[:, 3]]), y0, y1))
    out = []
    for a, b in zip(xs[:-1], xs[1:]):
        col = Cr[(Cr[:, 0] <= a) & (Cr[:, 1] >= b)]
        for c, d in zip(ys[:-1], ys[1:]):
            if not np.any((col[:, 2] <= c) & (col[:, 3] >= d)):
                out.append((a, b, c, d))
    return out


def stage_f5v2(run_dir, f_dirs, leaves, workers=2, old_tube=None, check_universe=True, mutation=None):
    try:
        return _stage_f5v2(run_dir, f_dirs, leaves, workers, old_tube, check_universe, mutation)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"STAGE F5v2 FAIL exception {type(e).__name__}: {e}"[:400], flush=True)
        return False, f"exception {type(e).__name__}: {e}"


def _good_boxes(Fd):
    import eight_check_f5 as F5v1
    lg = F5v1.parse_log(os.path.join(Fd, "stage_f.log"))
    req = dict(fail_rows=0, unres=0, referr=0, subnc=0, viol=0, tiling=0, ii1_still=0, x_nobox=0, x_left=0)
    bad = [k for k, v in req.items() if lg[k] != v]
    if bad:
        raise RuntimeError(f"{Fd}: stage F log does not allow the cover: {bad}")
    z = np.load(os.path.join(Fd, "stage_f.npz"), allow_pickle=False)
    n = len(z["fname"])
    assert n == lg["boxes"], f"{Fd}: npz and log disagree on the box count"
    okb = np.flatnonzero(z["closed"].astype(bool) & z["thin"].astype(bool))
    assert len(okb) == lg["okb"]
    rp = ((z["n_fail"][okb] == 0) & (z["n_unres"][okb] == 0) & (z["n_referr"][okb] == 0) & (z["viol"][okb] == 0)
          & z["vol_ok"][okb].astype(bool))
    passing = okb[rp]
    top = z["top"]
    assert len(top) == len(passing) == lg["top_n"] and int(np.sum(top == 2)) == lg["top_bad"], f"{Fd}: top misaligned"
    good = passing[top != 2]
    nt = passing[top == 2]
    nontop_rows = [(float(z["eps"][i]), float(z["phi_lo"][i]), float(z["phi_hi"][i]), float(z["M_lo"][i]),
                    float(z["M_hi"][i]), f"{os.path.basename(Fd)}/{z['fname'][i]}#{int(z['box'][i])}") for i in nt]
    rows = [(float(z["eps"][i]), float(z["phi_lo"][i]), float(z["phi_hi"][i]), float(z["M_lo"][i]), float(z["M_hi"][i]),
             f"{os.path.basename(Fd)}/{z['fname'][i]}#{int(z['box'][i])}") for i in good]
    nontop = [f"{z['fname'][i]} box {int(z['box'][i])} (eps {z['eps'][i]:.6g}, phi [{z['phi_lo'][i]:.5f}, "
              f"{z['phi_hi'][i]:.5f}])" for i in passing[top == 2]]
    return lg, rows, nontop, n, nontop_rows


def _stage_f5v2(run_dir, f_dirs, leaves, workers, old_tube, check_universe, mutation):
    import multiprocessing as mp
    import eight_check_records as F
    import eight_check as CK
    import eight_iiL_cover as C
    import eight_iiL_shadowrec as SR
    t0 = time.time()
    SR.lowprio()
    print(f"STAGE F5v2: run {run_dir}, F dirs {f_dirs}, workers {workers}, mutation {mutation}; empty rule {EMPTY}",
          flush=True)
    print("  code md5: " + "; ".join(f"{os.path.basename(p)} {SR._md5(p)}" for p in
                                    [os.path.abspath(__file__), os.path.join(HERE, "eight_check_f5.py"), F.__file__,
                                     CK.__file__, C.__file__, SR.__file__]), flush=True)
    good, nontop_all = [], []
    for Fd in f_dirs:
        lg, rows, nontop, n, ntr = _good_boxes(Fd)
        good += rows
        nontop_all += ntr
        print(f"  {Fd}: {n} boxes, good {len(rows)}, not at the top {len(nontop)}" +
              (": " + ", ".join(nontop) if nontop else ""), flush=True)
    leaves = np.asarray(leaves, float).reshape(-1, 6)
    LB = CK.leaf_boxes(leaves[:, :4], leaves[:, 4])
    byk = {}
    for (e, pl, ph, ml, mh, tag) in good:
        byk.setdefault((e, pl, ph), []).append((ml, mh, tag))
    hits = [[t for (ml, mh, t) in byk.get((float(b[0]), float(b[2]), float(b[3])), ()) if ml <= b[4] and mh >= b[5]]
            for b in LB]
    missing = [li for li in range(len(leaves)) if not hits[li]]
    cfg = dict(const=F.CONST, k1_path=os.path.join(run_dir, "k1_a8.npz"),
               a8_hi=float(np.load(os.path.join(run_dir, "k1_a8.npz"))["a8_act_hi"]), tube_a=SR.TUBE_A, goal=SR.GOAL,
               old_tube=old_tube, mutation=None, mut_file=None)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=F._init, initargs=(cfg,)) as pool:
        # the boxes F4 left not at the top: re-tested on the ladder TOP_LADDER (finer s-slabs; the test is the same)
        left = list(nontop_all)
        for ns in TOP_LADDER:
            if not left:
                break
            rws = []
            for (e, pl, ph, ml, mh, tag) in left:
                sq = math.sqrt(e)
                k1, k2 = sorted((ml / sq, mh / sq))
                rws.append([e, pl, ph, k1, k2])
            res = pool.apply(_top_retest, ((np.array(rws), ns),))
            cert = [b for b, r in zip(left, res) if r != 2]
            print(f"  top re-test at RC_NSLAB {ns}: {len(cert)} of {len(left)} certified at the top "
                  f"({', '.join(f'{b[5]}: ' + ('tube A' if r == 0 else 'earlier tube') for b, r in zip(left, res) if r != 2)})",
                  flush=True)
            good += cert
            left = [b for b, r in zip(left, res) if r == 2]
        for (e, pl, ph, ml, mh, tag) in good[len(good) - (len(nontop_all) - len(left)):]:
            byk.setdefault((e, pl, ph), []).append((ml, mh, tag))
        hits = [[t for (ml, mh, t) in byk.get((float(b[0]), float(b[2]), float(b[3])), ()) if ml <= b[4] and mh >= b[5]]
                for b in LB]
        missing = [li for li in range(len(leaves)) if not hits[li]]
        if mutation == "m11":
            for li in [li for li in range(len(leaves)) if hits[li]]:
                if not pool.apply(F._gap_task, (leaves[li],)):
                    print(f"  m11: good box(es) of leaf {li} deleted: {hits[li]}", flush=True)
                    hits[li] = []
                    missing = sorted(missing + [li])
                    break
            else:
                raise RuntimeError("m11: no matched leaf needs records")
        gap_ok = pool.map(F._gap_task, [leaves[i] for i in missing], chunksize=1) if missing else []
    nmiss = sum(1 for v in gap_ok if not v)
    reasons = []
    print(f"F5 cover: {len(leaves)} leaves (R {int(np.sum(leaves[:, 5] == 0))}, record-only "
          f"{int(np.sum(leaves[:, 5] == 1))}); matched to a good box {len(leaves) - len(missing)}; without one "
          f"{len(missing)}, of which needing a record {nmiss} ({time.time() - t0:.0f} s)", flush=True)
    if nmiss:
        reasons.append(f"{nmiss} leaves need records and have no good box")
    if check_universe:
        nb_path = next((os.path.join(Fd, "needs_below.npz") for Fd in f_dirs
                        if os.path.exists(os.path.join(Fd, "needs_below.npz"))), None)
        if nb_path is None:
            raise RuntimeError("no needs_below.npz in the F dirs")
        zz = np.load(nb_path)
        nbadp = int(np.sum(zz["rows"][:, 3])) if len(zz["rows"]) else 0
        occ = zz["occ"]
        P0, DP, K0, DK = zz["occ_grid"]
        ii, jj = np.nonzero(np.isfinite(occ))
        cells = np.stack([P0 + ii * DP, P0 + (ii + 1) * DP, np.sinh(K0 + jj * DK), np.sinh(K0 + (jj + 1) * DK)], 1)
        gapd = dict(zip(missing, gap_ok))
        ok_rect = np.array([lf[:4] for li, lf in enumerate(leaves) if gapd.get(li, True)]).reshape(-1, 4)
        unc = C.covered(cells, ok_rect) if len(cells) else np.zeros(0)
        gaps = np.flatnonzero(unc > 0)
        e_lo = float(occ[np.isfinite(occ)].min()) * EMPTY["E_LO_FACTOR"]
        parts, owner = [], []
        for k in gaps:
            for p in uncovered_parts(cells[k], ok_rect):
                parts.append(p)
                owner.append(k)
        parts = np.array(parts).reshape(-1, 4)
        owner = np.array(owner, int)
        empty = np.zeros(len(parts), bool)
        if len(parts):
            with ctx.Pool(workers, initializer=_empty_init, initargs=(SR.TUBE_A, e_lo)) as pool2:
                chunks = [parts[i:i + 20] for i in range(0, len(parts), 20)]
                empty = np.concatenate(pool2.map(_empty_task, chunks, chunksize=1))
        bad_cells = sorted(set(owner[~empty].tolist()))
        print(f"F5 universe below EPS_MIN: needing fine cells {len(cells)}, not inside the passing leaf rectangles "
              f"{len(gaps)}; their uncovered parts {len(parts)}, empty (no point of Q with eps >= e_lo = {e_lo:.3e}) "
              f"{int(empty.sum())}; cells with a non-empty uncovered part {len(bad_cells)}; patches with an unusable "
              f"enclosure {nbadp} ({time.time() - t0:.0f} s)", flush=True)
        for k in bad_cells[:20]:
            print(f"   cell phi [{cells[k, 0]:.4f}, {cells[k, 1]:.4f}] kappa [{cells[k, 2]:.4f}, {cells[k, 3]:.4f}]",
                  flush=True)
        if bad_cells or nbadp:
            reasons.append(f"universe: {len(bad_cells)} needing cells with a non-empty part outside the leaves, "
                           f"{nbadp} unusable patches")
    passed = not reasons
    out_dir = f_dirs[0]
    np.savez(os.path.join(out_dir, f"stage_f5v2{'_' + mutation if mutation else ''}.npz"), missing=np.array(missing, int),
             gap_ok=np.array(gap_ok, bool), reasons=np.array(reasons))
    print(f"STAGE F5v2 {'PASS' if passed else 'FAIL ' + '; '.join(reasons)} ({time.time() - t0:.0f} s)", flush=True)
    return passed, "; ".join(reasons)


if __name__ == "__main__":
    # python verify/eight_check_f5v2.py <run_dir> <F dir>[,<F dir>...] <leaves.npy> [workers] [old_tube|-] [universe 1|0]
    #        [m11|-]
    a = sys.argv[1:]
    ok_, _ = stage_f5v2(a[0], a[1].split(","), np.load(a[2]), int(a[3]) if len(a) > 3 else 2,
                        (a[4] if len(a) > 4 and a[4] != "-" else None), (a[5] != "0") if len(a) > 5 else True,
                        (a[6] if len(a) > 6 and a[6] != "-" else None))
    sys.exit(0 if ok_ else 1)
