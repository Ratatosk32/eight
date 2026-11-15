"""Stage 21, hypothesis (R) over Q by comparison tubes (verify/eight_comptube.py): cover, filter, prove, bisect.

Q = {(eps, phi, M) : eps = s^2 exp(-1.5 ell), (ell, phi, M) in Lambda(s), s in [0, 1]} with Lambda(s) the fitted label
ellipsoid (tools/words_eight_tubeb.py EllTubeFit). This driver covers the part eps >= EPS_MIN by boxes with half-widths
(1.5 h, h, 0.25 h) in (log eps, phi, M) (log eps half-width 1.5 h = label ell half-width h), drops boxes that are
certainly outside Lambda(s) for every s (rigorous, Cauchy-Schwarz test of the tube code, slab-wise in s), proves the
others (comparison tube closes and the scalar Sturm test passes), and bisects the failures down to H_MIN.
Output: npz with proved boxes (lo/hi in (eps, phi, M)), outside boxes, and the stuck ones (for the S-chart barrier).
    python verify/eight_R_cover2.py out.npz [workers] [h0]"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

EPS_MIN = float(os.environ.get("RC_EPSMIN", "1e-4"))
H_MIN = float(os.environ.get("RC_HMIN", "0.0025"))
BATCH = int(os.environ.get("RC_BATCH", "12"))
NSLAB = int(os.environ.get("RC_NSLAB", "40"))


def keep_awake_low():
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
        k.SetThreadExecutionState(0x80000000 | 0x00000001)


def tube():
    import words_eight_tubeb as T
    return T.EllTubeFit(os.environ.get("RC_TUBE", T.ELL_FIT_NPZ))     # RC_TUBE: another EllTubeFit npz (e.g. the taper)


def outside(boxes, tb):
    """True where the parameter box (log eps, phi, M) is certainly outside Q: for every s-slab [sa, sb] the label box
    ell in (2/3)(2 log s - log eps), phi, M is certainly outside Lambda(s)"""
    n = len(boxes)
    # the s floor below is sound only for eps_hi >= ~4e-13 (referee eight_slab_referee.md (2)); boxes with smaller
    # eps reach the first s-slab, where the full ell range of the tube is tested (that covers eps -> 0, eps = 0)
    assert np.all(boxes[:, 1] >= math.log(4e-13)), "outside(): eps_hi below 4e-13"
    out = np.ones(n, bool)
    edges = np.linspace(0.0, 1.0, NSLAB + 1)
    # the tube's own ell range (with a margin): label boxes are clipped to it; points outside it are outside Lambda
    if not hasattr(outside, "ell_range") or outside.ell_range[0] is not tb:
        ss = np.linspace(0.0, 1.0, 801)
        C = tb.centre(ss)
        half = np.sqrt(np.stack([np.diagonal(np.linalg.inv(g)) for g in tb.Gs(ss)]))
        outside.ell_range = (tb, float(np.min(C[:, 0] - half[:, 0])) - 0.05, float(np.max(C[:, 0] + half[:, 0])) + 0.05)
    _, ell_min, ell_max = outside.ell_range
    for sa, sb in zip(edges[:-1], edges[1:]):
        sa_ = max(sa, 1e-6)
        ell_lo = (2.0 / 3.0) * (2 * math.log(sa_) - boxes[:, 1])
        ell_hi = (2.0 / 3.0) * (2 * math.log(sb) - boxes[:, 0])
        empty = (ell_hi < ell_min) | (ell_lo > ell_max)            # no ell of the tube in this slab: outside here
        ell_lo_c = np.clip(ell_lo, ell_min, ell_max)
        ell_hi_c = np.clip(ell_hi, ell_min, ell_max)
        a_lo = np.stack([ell_lo_c - 1e-9, boxes[:, 2], boxes[:, 4]], 1)
        a_hi = np.stack([ell_hi_c + 1e-9, boxes[:, 3], boxes[:, 5]], 1)
        o = tb.box_certainly_outside(a_lo, a_hi, np.full(n, sa), np.full(n, sb))
        o |= _axis_outside(tb, a_lo, a_hi, sa, sb)
        out &= (o | empty)
    return out


def _axis_outside(tb, a_lo, a_hi, sa, sb):
    """second sufficient test: Lambda(s) lies in its bounding box c_i(s) +- sqrt((G(s)^-1)_ii); c is linear and
    (G^-1)_ii convex in G (G linear) between the nodes, so over [sa, sb] both are bounded by their values at sa, sb
    and the nodes inside; a margin covers the float inverse. Outside if some coordinate range misses the box."""
    key = (id(tb), sa, sb)
    cache = _axis_outside.cache
    if key not in cache:
        sn = tb.s_nodes
        pts = np.concatenate([[sa, sb], sn[(sn > sa) & (sn < sb)]])
        C = tb.centre(pts)
        half = np.sqrt(np.stack([np.diagonal(np.linalg.inv(g)) for g in tb.Gs(pts)])) * (1 + 1e-9) + 1e-12
        cache[key] = (C.min(0) - half.max(0) - 1e-12, C.max(0) + half.max(0) + 1e-12)
    blo, bhi = cache[key]
    return np.any((a_lo > bhi[None, :]) | (a_hi < blo[None, :]), axis=1)


_axis_outside.cache = {}


def initial_boxes(tb, h):
    """grid over the bounding region of Q in (log eps, phi, M)"""
    s = np.linspace(0.0, 1.0, 401)
    C = tb.centre(s)
    G = tb.Gs(s)
    half = np.sqrt(np.stack([np.diagonal(np.linalg.inv(g)) for g in G]))
    lo, hi = C - half, C + half
    with np.errstate(divide="ignore"):
        le_lo = np.min(2 * np.log(np.maximum(s, 1e-300)) - 1.5 * hi[:, 0])
        le_hi = np.max(2 * np.log(np.maximum(s, 1e-300)) - 1.5 * lo[:, 0])
    le_lo = max(le_lo, math.log(EPS_MIN))
    le_hi = min(le_hi, float(os.environ.get("RC_LOGEPS_MAX", "1e9")))     # restrict to a sub-region (easy part)
    rng = [(le_lo, le_hi), (lo[:, 1].min(), hi[:, 1].max()), (lo[:, 2].min(), hi[:, 2].max())]
    w = 2 * h * np.array([1.5, 1.0, 0.25])
    g = [np.arange(r[0], r[1] + w[j], w[j]) for j, r in enumerate(rng)]
    B = []
    for a in g[0][:-1]:
        for b in g[1][:-1]:
            for c in g[2][:-1]:
                B.append([a, a + w[0], b, b + w[1], c, c + w[2]])
    return np.array(B)


def to_param(B):
    """(log eps, phi, M) boxes -> (eps, phi, M) boxes for eight_comptube.check"""
    P = B.copy()
    P[:, 0], P[:, 1] = np.exp(B[:, 0]), np.exp(B[:, 1])
    return P


def work(B):
    keep_awake_low()
    import eight_comptube as CT
    import eight_field_lohner as L
    L.ORDER = 12
    try:
        closed, ok, ymin, r1 = CT.check(to_param(B))
        return ok, ymin, ""
    except Exception as e:                       # a centre that cannot be validated etc.: bisect
        return np.zeros(len(B), bool), np.full(len(B), -np.inf), repr(e)[:120]


def bisect(b):
    w = (b[1::2] - b[0::2]) / np.array([1.5, 1.0, 0.25])
    j = int(np.argmax(w))
    m = 0.5 * (b[2 * j] + b[2 * j + 1])
    b1, b2 = b.copy(), b.copy()
    b1[2 * j + 1] = m
    b2[2 * j] = m
    return [b1, b2]


def presize(B, hmap_path, h0, factor):
    """split each grid box (h0) in halves until its h <= factor x (smallest float-admissible hmax among the 3 nearest
    map points, in coordinates scaled by the box proportions); floors at H_MIN"""
    d = np.load(hmap_path)
    Pm, Hm = d["P"], d["hmax"]
    scale = np.array([1.5, 1.0, 0.25])
    out = []
    for b in B:
        c = 0.5 * (b[0::2] + b[1::2])
        dist = np.linalg.norm((Pm - c) / scale, axis=1)
        hpred = max(H_MIN, factor * np.min(Hm[np.argsort(dist)[:3]]))
        level = 0
        while h0 / 2 ** level > hpred + 1e-12:
            level += 1
        # halve h: split once along each of the 3 directions per level
        boxes = [b]
        for _ in range(level):
            for j in range(3):
                nb = []
                for x in boxes:
                    m = 0.5 * (x[2 * j] + x[2 * j + 1])
                    x1, x2 = x.copy(), x.copy()
                    x1[2 * j + 1] = m
                    x2[2 * j] = m
                    nb += [x1, x2]
                boxes = nb
        out += boxes
    return np.array(out)


def main():
    keep_awake_low()
    out = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    h0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    limit = int(os.environ.get("RC_LIMIT", "0"))
    tb = tube()
    B = initial_boxes(tb, h0)
    t0 = time.time()
    outs = outside(B, tb)
    todo = B[~outs]
    hmap = os.environ.get("RC_HMAP")
    if hmap:
        todo = presize(todo, hmap, h0, float(os.environ.get("RC_HFACTOR", "0.25")))
        o2 = outside(todo, tb)
        todo = todo[~o2]
        print(f"presized by the float map: {len(todo)} boxes", flush=True)
    todo = todo[np.argsort(todo[:, 0])]          # chunk by eps (similar trajectories in a batch)
    if limit:
        todo = todo[:limit]
    print(f"grid {len(B)} boxes at h = {h0}; certainly outside Q: {outs.sum()}; to prove: {len(todo)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    proved, stuck, outside_all = [], [], list(B[outs])
    import multiprocessing as mp
    h = h0
    with mp.Pool(workers) as pool:
        while len(todo):
            chunks = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
            nxt, nproved = [], 0
            for ci, (ok, ymin, err) in enumerate(pool.imap(work, chunks)):
                ch = chunks[ci]
                nproved += int(ok.sum())
                proved += [b for b, o in zip(ch, ok) if o]
                for b, o in zip(ch, ok):
                    if not o:
                        if h / 2 >= H_MIN - 1e-15:
                            nxt += bisect(b)
                        else:
                            stuck.append(b)
                print(f"  h {h:.4f} batch {ci + 1}/{len(chunks)}: proved {int(ok.sum())}/{len(ch)}"
                      + (f" ERROR {err}" if err else "") + f"; elapsed {time.time() - t0:.0f} s", flush=True)
                if ci == 0 and ok.sum() == 0:
                    print("  WARNING: the first batch proved nothing", flush=True)
            nxt = np.array(nxt)
            if len(nxt):
                o2 = outside(nxt, tb)
                outside_all += list(nxt[o2])
                nxt = nxt[~o2]
            print(f"level h {h:.4f} done: proved {nproved}; next level {len(nxt)} boxes", flush=True)
            todo, h = nxt, h / 2
            np.savez(out, proved=np.array(proved), stuck=np.array(stuck), outside=np.array(outside_all), h0=h0,
                     eps_min=EPS_MIN, h_min=H_MIN)
    print(f"done: proved {len(proved)}, stuck {len(stuck)}, outside {len(outside_all)}; {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
