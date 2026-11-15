"""Stage 21, hypothesis (R) by shadows (theorist, paper/review/eight_stage2_secondorder_theorist.md): a 2-D cover.

Scaling: the solution with parameters (tau1^2 eps, phi, tau1 M) on [0, 1] is the initial piece [0, tau1] of the solution
with (eps, phi, M), reparametrised (Y~(tau) = Y(tau1 tau), P~ = tau1 P(tau1 tau)). A box proved by the scalar Sturm test
(no focal point of I on (0, 1]) therefore proves (R) at all its shadows, i.e. along the curves M = kappa sqrt(eps) at
fixed phi from the box back to eps -> 0.
Cover of Q (eps >= EPS_MIN): cells [phi_a, phi_b] x [kappa_a, kappa_b]. For a cell, eps_top = the upper end of the
highest log-eps slice whose region {phi in cell, M = kappa sqrt(eps), kappa in cell} is NOT certainly outside Q
(rigorous outside test of the tube code); everything above eps_top is certainly outside Q. The cell's box is THIN in
eps: {eps_top} x [phi_a, phi_b] x hull(kappa sqrt(eps_top)); proving it proves every point of Q in the cell with
eps >= EPS_MIN (its curve point at eps_top is in the box, and the point is a shadow of that one). kappa cells are split
until the box's M half-width is at most HM; failing boxes split their cell (phi or kappa) and are retried.
The slab eps < EPS_MIN (s -> 0, next to I) is handled separately.
    python verify/eight_R_shadow.py out.npz [workers] [h]"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import eight_R_cover2 as RC  # noqa: E402

EPS_MIN = float(os.environ.get("RS_EPSMIN", "1e-3"))
NSL = int(os.environ.get("RS_SLICES", "120"))
MAXSPLIT = int(os.environ.get("RS_MAXSPLIT", "4"))
SPLIT = int(os.environ.get("RS_SPLIT", "4"))
MW = float(os.environ.get("RS_MWEIGHT", "2.0"))


def q_bounds(tb):
    """rigorous outer bounds of Q = {(eps, phi, M) : label in Lambda(s), s in [0, 1]}: c(s) and G(s) are linear between
    the nodes, the extent of the ellipsoid along axis i is sqrt((G^-1)_ii) and (G^-1)_ii is convex in G (operator
    convexity of the inverse), so per node segment |a_i - c_i| <= max over the two nodes; c_i is linear. A margin
    covers the float inverse (referee eight_slab_referee.md (3))."""
    sn = tb.s_nodes
    Cn = tb.centre(sn)
    half = np.sqrt(np.stack([np.diagonal(np.linalg.inv(g)) for g in tb.G])) * (1 + 1e-9) + 1e-12
    lo_seg = np.minimum(Cn[:-1], Cn[1:]) - np.maximum(half[:-1], half[1:])
    hi_seg = np.maximum(Cn[:-1], Cn[1:]) + np.maximum(half[:-1], half[1:])
    emax = float(np.max(sn[1:] ** 2 * np.exp(-1.5 * lo_seg[:, 0]))) * (1 + 1e-12)
    return lo_seg[:, 1].min(), hi_seg[:, 1].max(), lo_seg[:, 2].min(), hi_seg[:, 2].max(), emax


def grid(a, b, w):
    """cell edges of width w covering [a, b] (explicit count, the last edge >= b)"""
    n = max(1, int(math.ceil((b - a) / w)) + 1)
    return a + w * np.arange(n + 1)


def m_range(ka, kb, ea, eb):
    """hull of kappa sqrt(eps) over the rectangle"""
    sa, sb = math.sqrt(ea), math.sqrt(eb)
    v = [ka * sa, ka * sb, kb * sa, kb * sb]
    lo, hi = min(v), max(v)                          # rounded outward (a few ulps)
    return lo - 4 * np.spacing(abs(lo)) - 1e-300, hi + 4 * np.spacing(abs(hi)) + 1e-300


CHUNK = int(os.environ.get("RS_CHUNK", "40"))


def eps_top(tb, cells, emax):
    """cells (n, 4) = phi_a, phi_b, kappa_a, kappa_b -> eps_top (n,) (nan: no point of Q with eps >= EPS_MIN);
    processed in chunks of CHUNK cells so that the outside test never holds more than CHUNK x NSL boxes"""
    if len(cells) > CHUNK:
        return np.concatenate([eps_top(tb, cells[i:i + CHUNK], emax) for i in range(0, len(cells), CHUNK)])
    edges = np.exp(np.linspace(math.log(EPS_MIN), math.log(emax * 1.02), NSL + 1))
    n = len(cells)
    boxes, owner, sl = [], [], []
    for i, (pa, pb, ka, kb) in enumerate(cells):
        for j in range(NSL):
            ma, mb = m_range(ka, kb, edges[j], edges[j + 1])
            boxes.append([math.log(edges[j]), math.log(edges[j + 1]), pa, pb, ma, mb])
            owner.append(i)
            sl.append(j)
    boxes = np.array(boxes)
    out = RC.outside(boxes, tb)
    top = np.full(n, np.nan)
    owner, sl = np.array(owner), np.array(sl)
    for i in range(n):
        inside = sl[(owner == i) & ~out]
        if len(inside):
            top[i] = edges[inside.max() + 1]
    return top


def make_box(cell, et):
    pa, pb, ka, kb = cell
    ma, mb = m_range(ka, kb, et, et)
    return np.array([et, et, pa, pb, ma, mb])


def split_cells(cells, tops, hm):
    """split kappa until the M half-width at eps_top is <= hm"""
    out_c, out_t = [], []
    for c, t in zip(cells, tops):
        if not np.isfinite(t):
            continue
        stack = [c]
        while stack:
            pa, pb, ka, kb = stack.pop()
            if 0.5 * (kb - ka) * math.sqrt(t) > hm:
                km = 0.5 * (ka + kb)
                stack += [np.array([pa, pb, ka, km]), np.array([pa, pb, km, kb])]
            else:
                out_c.append(np.array([pa, pb, ka, kb]))
                out_t.append(t)
    return np.array(out_c), np.array(out_t)


def work(B):
    """prove a batch; if the batch raises (e.g. one centre trajectory fails), retry box by box so that one bad box
    does not take its neighbours down"""
    RC.keep_awake_low()
    import eight_comptube as CT
    import eight_field_lohner as L
    L.ORDER = 12
    sdir = os.environ.get("RS_SAVE_DIR", "")
    light = os.environ.get("RS_LIGHT", "") == "1"
    try:
        if sdir or light:
            closed, ok, ymin, r1 = CT.check(B, m=4 if light else 8, save=bool(sdir), sturm=not light)
            if light:
                ok = closed                     # records need only a closed tube
            if sdir:
                import hashlib
                key = hashlib.md5(np.ascontiguousarray(B).tobytes()).hexdigest()[:16]
                np.savez_compressed(os.path.join(sdir, f"b_{key}.npz"), boxes=B, light=light, **CT.check.saved)
        else:
            closed, ok, ymin, r1 = CT.check(B)
        return ok, ymin, ""
    except Exception as e:
        err = repr(e)[:60]
        ok, ymin = np.zeros(len(B), bool), np.full(len(B), -np.inf)
        for k in range(len(B)):
            try:
                c1, o1, y1, _ = CT.check(B[k:k + 1])
                ok[k], ymin[k] = o1[0], y1[0]
            except Exception:
                pass
        return ok, ymin, f"{err} (retried box by box: {int(ok.sum())} proved)"


def top_work(cells):
    RC.keep_awake_low()
    tb = RC.tube()
    plo, phi_, mlo, mhi, emax = q_bounds(tb)
    return eps_top(tb, cells, emax)


def tops_parallel(pool, cells, nchunk=400):
    if len(cells) == 0:
        return np.zeros(0)
    parts = [cells[i:i + 40] for i in range(0, len(cells), 40)]
    return np.concatenate(pool.map(top_work, parts))


def main():
    RC.keep_awake_low()
    out = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    limit = int(os.environ.get("RS_LIMIT", "0"))
    tb = RC.tube()
    plo, phi_, mlo, mhi, emax = q_bounds(tb)
    kmax = (max(abs(mlo), abs(mhi)) / math.sqrt(EPS_MIN)) * (1 + 1e-12)
    t0 = time.time()
    # initial cells: phi width 2h, kappa coarse (split adaptively)
    hm = float(os.environ.get("RS_HM", str(h)))
    cache = f"{out}.cells_h{h}_hm{hm}_e{EPS_MIN}.npz"
    if os.environ.get("RS_CELLS"):                  # an explicit list (cells, tops), e.g. the failures of a level
        z = np.load(os.environ["RS_CELLS"])
        cells, tops = z["cells"], z["tops"]
    elif os.environ.get("RS_RESUME"):               # the cells come from the order file below
        cells, tops = np.zeros((0, 4)), np.zeros(0)
    elif os.path.exists(cache):
        z = np.load(cache)
        cells, tops = z["cells"], z["tops"]
    else:
        pg = grid(plo, phi_, 2 * h)
        kg = np.linspace(-kmax, kmax, 65)
        cells = np.array([[a, a + 2 * h, k1, k2] for a in pg[:-1] for k1, k2 in zip(kg[:-1], kg[1:])])
        tops = eps_top(tb, cells, emax)
        cells, tops = split_cells(cells, tops, hm)
        tops = eps_top(tb, cells, emax)            # refined cells may have a lower top
        keep = np.isfinite(tops)
        cells, tops = cells[keep], tops[keep]
        np.savez(cache, cells=cells, tops=tops, tube=os.environ.get("RC_TUBE", ""))
    pre = float(os.environ.get("RS_PRESPLIT_EPS", "0.05"))   # tubes need half-width ~0.005 at large eps: split first
    big = tops > pre
    if big.any() and not os.environ.get("RS_RESUME") and not os.environ.get("RS_CELLS"):
        nx = []
        for pa, pb, ka, kb in cells[big]:
            pm, km = 0.5 * (pa + pb), 0.5 * (ka + kb)
            nx += [[pa, pm, ka, km], [pm, pb, ka, km], [pa, pm, km, kb], [pm, pb, km, kb]]
        nx = np.array(nx)
        nt = eps_top(tb, nx, emax)
        keep = np.isfinite(nt)
        cells = np.concatenate([cells[~big], nx[keep]])
        tops = np.concatenate([tops[~big], nt[keep]])
        print(f"pre-split {big.sum()} cells with eps_top > {pre} into {keep.sum()} nonempty quarters", flush=True)
    order = np.argsort(tops)
    cells, tops = cells[order], tops[order]
    resume = os.environ.get("RS_RESUME", "")
    proved, proved_cells, stuck = [], [], []
    import multiprocessing as mp
    pool = mp.Pool(workers)
    if resume:
        # RS_RESUME = order.npz,log: the cell order of an earlier run and its log; batches proved in full are kept
        # (their boxes sit at an older, higher eps_top: still valid, a box proves all its shadows below it); the other
        # cells get a fresh, tighter eps_top (RC_NSLAB / RS_SLICES) and are proved again
        oz, lg = resume.split(",")
        z = np.load(oz)
        ocells, otops = z["cells"], z["tops"]
        bs = int(os.environ.get("RS_RESUME_BATCH", "4"))
        lines = [l for l in open(lg) if "level 0 batch" in l]
        done = np.zeros(len(ocells), bool)
        for l in lines:
            bi = int(l.split("batch ")[1].split("/")[0]) - 1
            k, m = l.split("proved ")[1].split(";")[0].split()[0].split("/")
            if int(k) == int(m):
                done[bi * bs: bi * bs + int(m)] = True
        for c, t in zip(ocells[done], otops[done]):
            proved.append(make_box(c, t))
            proved_cells.append(np.concatenate([c, [t]]))
        rest = ocells[~done]
        rc = f"{oz}.retop_ns{RC.NSLAB}_nsl{NSL}.npz"
        if os.path.exists(rc) and len(np.load(rc)["tops"]) == len(rest):
            nt = np.load(rc)["tops"]
        else:
            nt = tops_parallel(pool, rest)
            np.savez(rc, tops=nt)
        keep = np.isfinite(nt)
        cells, tops = rest[keep], nt[keep]
        o = np.argsort(tops)
        cells, tops = cells[o], tops[o]
        print(f"resume: {done.sum()} cells proved in full batches kept; {len(rest)} re-topped, {keep.sum()} nonempty; "
              f"tops {np.quantile(tops, [0, .5, 1]).round(4)} ({time.time() - t0:.0f} s)", flush=True)
    if limit:                                  # a spread sample (low to high eps_top) for smoke tests
        ix = np.unique(np.linspace(0, len(cells) - 1, limit).astype(int))
        cells, tops = cells[ix], tops[ix]
    print(f"Q bounds: phi [{plo:.3f}, {phi_:.3f}], M [{mlo:.3f}, {mhi:.3f}], eps <= {emax:.4f}; cells to prove: {len(cells)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    level = 0
    with pool:
        while len(cells):
            B = np.array([make_box(c, t) for c, t in zip(cells, tops)])
            chunks = [np.arange(i, min(i + RC.BATCH, len(B))) for i in range(0, len(B), RC.BATCH)]
            nxt = []
            for ci, (ok, ymin, err) in enumerate(pool.imap(work, [B[ix] for ix in chunks])):
                ix = chunks[ci]
                for j, o in zip(ix, ok):
                    if o:
                        proved.append(B[j])
                        proved_cells.append(np.concatenate([cells[j], [tops[j]]]))
                    elif level < MAXSPLIT:
                        pa, pb, ka, kb = cells[j]
                        pm, km = 0.5 * (pa + pb), 0.5 * (ka + kb)
                        if SPLIT == 2:
                            if (pb - pa) >= MW * (kb - ka) * math.sqrt(tops[j]):    # M weighs ~2x (theorist)
                                nxt += [np.array([pa, pm, ka, kb]), np.array([pm, pb, ka, kb])]
                            else:
                                nxt += [np.array([pa, pb, ka, km]), np.array([pa, pb, km, kb])]
                        else:
                            nxt += [np.array([pa, pm, ka, km]), np.array([pm, pb, ka, km]),
                                    np.array([pa, pm, km, kb]), np.array([pm, pb, km, kb])]
                    else:
                        stuck.append(np.concatenate([cells[j], [tops[j]]]))
                if (ci + 1) % 25 == 0:          # save the progress during a level, not only at its end
                    np.savez(out + ".partial.npz", proved=np.array(proved), proved_cells=np.array(proved_cells),
                             stuck=np.array(stuck), level=level, batch=ci + 1)
                print(f"  level {level} batch {ci + 1}/{len(chunks)}: proved {int(ok.sum())}/{len(ix)}"
                      + (f" ERROR {err}" if err else "") + f"; elapsed {time.time() - t0:.0f} s", flush=True)
                if ci == 0 and level == 0 and ok.sum() == 0:
                    print("  WARNING: the first batch proved nothing", flush=True)
            failed_level = [np.concatenate([cells[j], [tops[j]]]) for j in range(len(cells))
                            if not any(np.array_equal(cells[j], pc[:4]) for pc in proved_cells[-len(cells):])]
            np.savez(out + f".failed_level{level}.npz", failed=np.array(failed_level))
            if nxt:
                cells = np.array(nxt)
                tops = tops_parallel(pool, cells)
                keep = np.isfinite(tops)
                cells, tops = cells[keep], tops[keep]
            else:
                cells = np.zeros((0, 4))
            level += 1
            print(f"level {level - 1} done: proved {len(proved)} so far; next level {len(cells)} cells", flush=True)
            np.savez(out, proved=np.array(proved), proved_cells=np.array(proved_cells), stuck=np.array(stuck),
                     eps_min=EPS_MIN, h=h)
    print(f"done: proved {len(proved)}, stuck {len(stuck)}; {time.time() - t0:.0f} s", flush=True)
    if not limit:
        print("SHADOW COVER COMPLETE (stuck 0)" if len(stuck) == 0 else "SHADOW COVER INCOMPLETE: stuck cells remain",
              flush=True)





# ---------------------------------------------------------------- the small-eps slab (theorist eight_shadow_slab_theorist.md)
def slab_main(out, workers, E, hp, hm):
    """boxes eps in [0, E] x [phi] x [M] proved by comparison tubes in slab mode (eps as a box parameter, eps = 0
    included); grid over Q's (phi, M) bounds, dropped where certainly outside Q for eps in (1e-12, E]; failures split"""
    import multiprocessing as mp
    RC.keep_awake_low()
    tb = RC.tube()
    plo, phi_, mlo, mhi, emax = q_bounds(tb)
    assert E >= EPS_MIN, "the slab must reach the shadow cover's EPS_MIN"
    pg = grid(plo, phi_, 2 * hp)
    mg = grid(mlo, mhi, 2 * hm)
    B = np.array([[0.0, E, a, a + 2 * hp, b, b + 2 * hm] for a in pg[:-1] for b in mg[:-1]])
    t0 = time.time()

    def outside_slab(B):
        L = B.copy()
        L[:, 0], L[:, 1] = math.log(1e-12), math.log(E)
        res = np.zeros(len(B), bool)
        for i in range(0, len(B), 400):
            res[i:i + 400] = RC.outside(L[i:i + 400], tb)
        return res
    o = outside_slab(B)
    B = B[~o]
    print(f"slab eps in [0, {E}]: grid {len(o)}, certainly outside {o.sum()}, to prove {len(B)} ({time.time() - t0:.0f} s)", flush=True)
    proved, stuck = [], []
    level = 0
    with mp.Pool(workers) as pool:
        while len(B):
            chunks = [B[i:i + RC.BATCH] for i in range(0, len(B), RC.BATCH)]
            nxt = []
            for ci, (ok, ymin, err) in enumerate(pool.imap(slab_work, [(c, E) for c in chunks])):
                ch = chunks[ci]
                for b, o_ in zip(ch, ok):
                    if o_:
                        proved.append(b)
                    elif level < MAXSPLIT:
                        pm, mm = 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])
                        nxt += [np.array([0.0, E, b[2], pm, b[4], mm]), np.array([0.0, E, pm, b[3], b[4], mm]),
                                np.array([0.0, E, b[2], pm, mm, b[5]]), np.array([0.0, E, pm, b[3], mm, b[5]])]
                    else:
                        stuck.append(b)
                print(f"  slab level {level} batch {ci + 1}/{len(chunks)}: proved {int(ok.sum())}/{len(ch)}"
                      + (f" ERROR {err}" if err else "") + f"; elapsed {time.time() - t0:.0f} s", flush=True)
                if ci == 0 and level == 0 and ok.sum() == 0:
                    print("  WARNING: the first batch proved nothing", flush=True)
            B = np.array(nxt) if nxt else np.zeros((0, 6))
            if len(B):
                B = B[~outside_slab(B)]
            level += 1
            print(f"slab level {level - 1} done: proved {len(proved)} so far; next level {len(B)}", flush=True)
            np.savez(out, proved=np.array(proved), stuck=np.array(stuck), E=E)
    print(f"slab done: proved {len(proved)}, stuck {len(stuck)}; {time.time() - t0:.0f} s", flush=True)
    print("SLAB COMPLETE (stuck 0)" if len(stuck) == 0 else "SLAB INCOMPLETE: stuck boxes remain", flush=True)


def slab_work(args):
    B, E = args
    RC.keep_awake_low()
    import eight_comptube as CT
    import eight_field_lohner as L
    L.ORDER = 12
    try:
        closed, ok, ymin, r1 = CT.check(B, slab=E)
        return ok, ymin, ""
    except Exception as e:
        return np.zeros(len(B), bool), np.full(len(B), -np.inf), repr(e)[:100]



if __name__ == "__main__":
    if sys.argv[1] == "slab":
        slab_main(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
    else:
        main()
