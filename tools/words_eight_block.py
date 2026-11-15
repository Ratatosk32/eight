"""Stage 21, certificate step: memory-lean coarse fit by overlapping block-coordinate ascent (multiplicative Schwarz).

Same certificate as tools/words_eight_coarse.py (phi continuous piecewise Q1 on hexahedra in (sigma, ua, ub), constant
below sigma_lo, far field above sigma_hi, the SOCP constraints F >= 0 at 27 points per hexahedron with the dip-safe
components), but the SOCP is never assembled whole. The unknowns are split into overlapping blocks of node layers in
sigma (optionally also in angle); one block is solved at a time with all other nodal values fixed. Every hexahedron
that touches a free node keeps all its constraint points in the block problem, so a block solve never breaks a
constraint that held before: started from a feasible phi (a rescaled warm start), every iterate stays feasible at the
sample points and the objective increases monotonically. The objective is the coarse fit's target functional plus a
small positive weight on every node (it pushes towards the maximal subsolution, so blocks without targets still raise
the values the targets depend on downstream). Characteristics run from high sigma to low (towards the E ray / the
plane I as t -> 0), so sweeps go top-down then bottom-up.
Warm start: the nu = 4 fit, prolongated exactly (nested meshes: the fine sigma grid contains the coarse layers, the
fine angular grid is the coarse one subdivided in u), times lam0 so that the fine sample constraints hold.
Then: cutting planes from the float cell check (only the blocks with failing hexahedra are re-solved), the lam
rescale (finalize) and the JSON export in the verifier's format.
    python tools/words_eight_block.py fit kind nu ref tag [sweeps] [cut_rounds]
    python tools/words_eight_block.py final kind tag
    python tools/words_eight_block.py json kind tag
ref: sigma refinement pattern (see sigma_grid)."""
import os
import sys
import time
import json
import numpy as np
import scipy.sparse as sp
import clarabel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import words_eight_coarse as C  # noqa: E402

OUT = "D:/nature-math/math-gaps/words/out/eight/block/"
MAXHEX = int(os.environ.get("BLOCK_MAXHEX", "2600"))
SOLVER = os.environ.get("BLOCK_SOLVER", "qdldl")
MASS = float(os.environ.get("BLOCK_MASS", "0.02"))


def sigma_grid(ref):
    """the coarse grid (nu4, dsig_band 0.1) with intervals halved by ref: 'b' halves [0.6, 5], 'o' halves [5, 11],
    'l' halves [0.3, 0.6], 'B' quarters [0.6, 5]; a letter may repeat"""
    m = C.Mesh(1, 0.1)
    s = list(m.sig)
    for ch in ref.replace("0", "").strip("'\""):
        lo, hi, times = {"b": (0.6, 5.0, 1), "o": (5.0, 11.0, 1), "l": (0.3, 0.6, 1), "B": (0.6, 5.0, 2),
                         "m": (-3.0, 0.3, 1)}[ch]
        for _ in range(times):
            new = [s[0]]
            for a, b in zip(s[:-1], s[1:]):
                if a >= lo - 1e-9 and b <= hi + 1e-9:
                    new.append(0.5 * (a + b))
                new.append(b)
            s = new
    return np.array(s)


def make_mesh(kind, nu, ref):
    m = C.Mesh(nu, 0.1, rot=C.ROT[kind])
    m.sig = sigma_grid(ref)
    m.slo, m.shi = m.sig[0], m.sig[-1]
    m.nl = len(m.sig)
    m.nvar = 1 + (m.nl - 1) * m.na
    m.nhex = m.nq * (m.nl - 1)
    m.quad_arr = np.array([qq[4] for qq in m.quads])
    return m


def sample_rows(mesh, kind, hexes, extra):
    """constraint rows (as words_eight_coarse.sample_blocks, dip-safe) for the 27 points of each hexahedron in hexes
    plus the extra (hex, local point) cutting-plane points of those hexahedra"""
    pts = np.array([[a, b, c] for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)])
    H = np.repeat(hexes, len(pts))
    P = np.tile(pts, (len(hexes), 1))
    if extra is not None and len(extra[0]):
        sel = np.isin(extra[0], hexes)
        H = np.concatenate([H, extra[0][sel]]); P = np.vstack([P, extra[1][sel]])
    R = C.rows_at(mesh, kind, H, P[:, 0], P[:, 1], P[:, 2])
    X = []
    hh = 1e-4
    for ax in range(3):
        Pp = P.copy(); Pm = P.copy()
        Pp[:, ax] += hh; Pm[:, ax] -= hh
        Rp = C.rows_at(mesh, kind, H, Pp[:, 0], Pp[:, 1], Pp[:, 2])
        Rm = C.rows_at(mesh, kind, H, Pm[:, 0], Pm[:, 1], Pm[:, 2])
        for key in ("Ws", "D1", "D2"):
            X.append((Rp[key] - Rm[key]) / (2 * hh) / 4)
    R["X"] = X
    return R


def far_sample(mesh, kind, extra_far):
    fp = np.array([[a, b] for a in (0.0, 0.25, 0.5, 0.75, 1.0) for b in (0.0, 0.25, 0.5, 0.75, 1.0)])
    qf = np.repeat(np.arange(mesh.nq), len(fp))
    la_f, lb_f = np.tile(fp[:, 0], mesh.nq), np.tile(fp[:, 1], mesh.nq)
    if extra_far is not None and len(extra_far[0]):
        qf = np.concatenate([qf, extra_far[0]]); la_f = np.concatenate([la_f, extra_far[1]])
        lb_f = np.concatenate([lb_f, extra_far[2]])
    return C.far_rows(mesh, kind, qf, la_f, lb_f)


def socp_block(blocks, free, coef, ot):
    """maximise ot[free] . x over the free unknowns, others fixed at coef; cones as words_eight_coarse.socp"""
    nv = len(coef)
    pos = np.full(nv, -1)
    pos[free] = np.arange(len(free))
    fixed = coef.copy()
    fixed[free] = 0.0
    rows, cols, vals, bs, cones = [], [], [], [], []
    off = 0
    for B in blocks:
        m = len(B["U0"])
        if m == 0:
            continue
        far = "sig" not in B
        q = B["q"] if far else 2 * np.exp(-B["sig"])
        al2 = 1 / (1 + B["U0"])
        sq = np.sqrt(q * al2)
        comps = [B["Ws"], B["D1"], B["D2"]] + ([B["W"]] if far else []) + B.get("X", [])
        dim = 2 + len(comps)
        Cc = B["cols"]
        P = pos[Cc]
        isf = P >= 0
        cf = fixed[Cc]
        coefs = [al2[:, None] * B["Lin"]] + [2 * sq[:, None] * X for X in comps] + [al2[:, None] * B["Lin"]]
        bb = np.zeros(m * dim)
        for j, X in enumerate(coefs):
            r = off + np.arange(m) * dim + j
            rr = np.repeat(r, Cc.shape[1]).reshape(m, -1)
            rows.append(rr[isf]); cols.append(P[isf]); vals.append(-X[isf])
            bb[j::dim] += np.sum(X * cf, axis=1)
        bb[0::dim] += al2 * B["U0"] + 1
        bb[dim - 1::dim] += al2 * B["U0"] - 1
        bs.append(bb)
        cones += [clarabel.SecondOrderConeT(dim)] * m
        off += m * dim
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(off, len(free)))
    b = np.concatenate(bs)
    P0 = sp.csc_matrix((len(free), len(free)))
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.max_iter = 400
    st.direct_solve_method = SOLVER
    st.max_threads = 2
    solver = clarabel.DefaultSolver(P0, -ot[free], A, b, cones, st)
    sol = solver.solve()
    return np.array(sol.x), str(sol.status), off


def ang_groups(mesh, kind, G):
    """G groups of angular quads (k-means on the quad centres), ordered downstream first (E: by n1 descending; I: by
    |n . n_I| ascending); each group: (node ids of its quads)"""
    if G <= 1:
        return [None]
    ctr = mesh.angN[mesh.quad_arr].mean(1)
    ctr /= np.linalg.norm(ctr, axis=1)[:, None]
    rng = np.random.default_rng(0)
    cen = [ctr[rng.integers(len(ctr))]]
    for _ in range(G - 1):
        d = np.min([1 - ctr @ c for c in cen], axis=0)
        cen.append(ctr[np.argmax(d)])
    cen = np.array(cen)
    for _ in range(50):
        lab = np.argmax(ctr @ cen.T, axis=1)
        cen = np.array([ctr[lab == g].mean(0) if np.any(lab == g) else cen[g] for g in range(G)])
        cen /= np.linalg.norm(cen, axis=1)[:, None]
    key = -cen[:, 0] if kind == "E" else np.abs(cen @ C.NI)
    out = []
    for g in np.argsort(key):
        out.append(np.unique(mesh.quad_arr[lab == g].ravel()))
    return out


def hexes_of_block(mesh, k0, k1, nodes):
    """hexahedra touching the free nodes: cell layers k0 - 1 .. k1, quads with a free angular node (all quads in cell
    layer 0 if node layer 0, the single constant, is free)"""
    a, b = max(k0 - 1, 0), min(k1, mesh.nl - 2)
    qs = np.arange(mesh.nq) if nodes is None else np.flatnonzero(np.isin(mesh.quad_arr, nodes).any(1))
    out = [k * mesh.nq + qs for k in range(a, b + 1)]
    if k0 == 0 and nodes is not None:
        out.append(np.arange(mesh.nq))
    return np.unique(np.concatenate(out))


def free_of_block(mesh, k0, k1, nodes):
    out = []
    for k in range(k0, k1 + 1):
        if k == 0:
            out.append(np.array([0]))
        else:
            out.append(1 + (k - 1) * mesh.na + (np.arange(mesh.na) if nodes is None else nodes))
    return np.unique(np.concatenate(out))


def make_blocks(mesh, kind, maxhex=MAXHEX, overlap=1):
    """blocks (k0, k1, angular node set or None): node-layer ranges, consecutive ranges sharing `overlap` layers, and
    for fine angular meshes angular groups (overlapping by the quads around the group's nodes)"""
    L = maxhex // mesh.nq - 1
    G = 1
    if os.environ.get("BLOCK_L"):
        L = int(os.environ["BLOCK_L"])
        G = max(1, int(np.ceil((L + 1) * mesh.nq * 1.15 / maxhex)))
    elif L < overlap + 1:
        L = overlap + 1
        G = int(np.ceil((L + 1) * mesh.nq * 1.3 / maxhex))
    groups = ang_groups(mesh, kind, G)
    out = []
    k0 = 0
    while True:
        k1 = min(k0 + L - 1, mesh.nl - 1)
        for g in groups:
            out.append((k0, k1, g))
        if k1 == mesh.nl - 1:
            break
        k0 = k1 + 1 - overlap
    return out


def prolong(mesh, kind, src):
    """values of the nu4 ds0.1 coarse fit (file src) at the nodes of mesh (exact on nested meshes)"""
    d = np.load(src)
    nu0 = int(d["nu"]); ds0 = float(d["dsb"]) if "dsb" in d else 0.1
    rot0 = float(d["rot"]) if "rot" in d else float(os.environ.get("BLOCK_SRC_ROT", C.ROT[kind]))
    m0 = C.Mesh(nu0, ds0, rot=rot0)
    if "sig" in d:                     # a block fit as the source
        m0.sig = d["sig"]; m0.slo, m0.shi = m0.sig[0], m0.sig[-1]; m0.nl = len(m0.sig)
    coef0 = d["coef"]
    out = np.zeros(mesh.nvar)
    out[0] = coef0[0]
    for k in range(1, mesh.nl):
        xi = np.exp(mesh.sig[k]) * mesh.angN
        if mesh.sig[k] <= m0.slo + 1e-12:
            out[1 + (k - 1) * mesh.na: 1 + k * mesh.na] = coef0[0]
            continue
        out[1 + (k - 1) * mesh.na: 1 + k * mesh.na] = C.eval_phi(m0, coef0, xi)
    return out


def objective(mesh, kind):
    from words_eight_shape import eight_discrete
    c = eight_discrete(48)[0][24]
    xi, wt = C.targets(kind, c)
    ot = C.build_objective(mesh, xi, wt)
    # positive weight on every node: (sigma extent x angular area share), normalised to MASS x the target weight
    ds = np.diff(mesh.sig)
    wl = np.zeros(mesh.nl)
    wl[:-1] += ds / 2
    wl[1:] += ds / 2
    wl *= np.exp(-np.maximum(mesh.sig - 3.0, 0) / 2)      # far layers matter less
    mass = np.zeros(mesh.nvar)
    mass[0] = wl[0]
    for k in range(1, mesh.nl):
        mass[1 + (k - 1) * mesh.na: 1 + k * mesh.na] = wl[k] / mesh.na
    mass *= MASS * wt.sum() / mass.sum()
    return ot + mass, ot, c


def min_slack(mesh, kind, coef, hexes, extra):
    """min over the sample points of (U0 + Lin phi - q |comps|^2), relative to U0 (chunked)"""
    out = np.inf
    for i0 in range(0, len(hexes), 3000):
        R = sample_rows(mesh, kind, hexes[i0:i0 + 3000], extra)
        cc = coef[R["cols"]]
        q = 2 * np.exp(-R["sig"])
        comps = [R["Ws"], R["D1"], R["D2"]] + R["X"]
        sl = R["U0"] + np.sum(R["Lin"] * cc, 1) - q * sum(np.sum(X * cc, 1) ** 2 for X in comps)
        out = min(out, np.min(sl / R["U0"]))
    return out


def mem_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return -1


def peak_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2 ** 20
    except Exception:
        return -1


def free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return 99.0


def wait_ram():
    while free_ram_gb() < 3.0:
        print(f"   free RAM {free_ram_gb():.2f} GB < 3 GB: waiting", flush=True)
        time.sleep(60)


def solve_block(mesh, kind, coef, ot, blk, extra, extra_far):
    wait_ram()
    t0 = time.time()
    k0, k1, nodes = blk
    hexes = hexes_of_block(mesh, k0, k1, nodes)
    free = free_of_block(mesh, k0, k1, nodes)
    blocks = [sample_rows(mesh, kind, hexes, extra)]
    if k1 == mesh.nl - 1:
        blocks.append(far_sample(mesh, kind, extra_far))
    x, st, nrows = socp_block(blocks, free, coef, ot)
    del blocks
    old = ot[free] @ coef[free]
    ok = st in ("Solved", "AlmostSolved") and np.all(np.isfinite(x))
    if ok and ot[free] @ x >= old - 1e-9 * abs(old):
        coef = coef.copy()
        coef[free] = x
        acc = True
    else:
        acc = False
    print(f"   block layers {k0}-{k1}{'' if nodes is None else f' ({len(nodes)} ang. nodes)'} (sigma {mesh.sig[k0]:.2f}-{mesh.sig[k1]:.2f}): {len(hexes)} hexahedra, {len(free)} "
          f"unknowns, {nrows} rows: {st}, objective {old:.5f} -> {ot[free] @ x:.5f} ({'accepted' if acc else 'REJECTED'}); "
          f"{time.time() - t0:.0f} s, peak {peak_mb():.0f} MB", flush=True)
    return coef


def report(mesh, kind, coef, ot_t, c, tag, extra):
    tau = C.TS if kind == "E" else 1 - C.TS
    Vc = 1.22789891 if kind == "E" else 1.29195718
    t0 = time.time()
    worst, wrel, fails, ncheck = C.check_cells(mesh, kind, coef)
    fworst = C.far_check(mesh, kind, coef)
    Sc = C.S_value(mesh, coef, kind, c[None], tau)[0]
    kk = np.arange(mesh.nhex) // mesh.nq
    print(f"   check ({time.time() - t0:.0f} s, {ncheck} boxes): worst cell bound {worst.min():+.3e} (rel {wrel.min():+.3e}); "
          f"failing hexahedra {np.sum(worst < 0)}; far worst {fworst.min():+.3e}; target objective {ot_t @ coef:.5f}; "
          f"S(c) = {Sc:.5f} (V = {Vc}, deficit {100 * (Vc - Sc) / Vc:.2f} %)", flush=True)
    print(f"   failing by sigma layer: {np.bincount(kk[worst < 0], minlength=mesh.nl - 1).tolist()}", flush=True)
    return worst, fails, fworst, Sc


def fit(kind, nu, ref, tag, sweeps=2, cut_rounds=3, src=None):
    mesh = make_mesh(kind, nu, ref)
    print(f"{kind}: nu = {nu} ({mesh.nq} quads, {mesh.na} angular nodes), sigma grid '{ref}': {mesh.nl} layers, "
          f"{mesh.nhex} hexahedra, {mesh.nvar} unknowns; MAXHEX {MAXHEX}, solver {SOLVER}, mass {MASS}", flush=True)
    ot, ot_t, c = objective(mesh, kind)
    extra = (np.zeros(0, int), np.zeros((0, 3)))
    extra_far = (np.zeros(0, int), np.zeros(0), np.zeros(0))
    path = OUT + f"block_{kind}_{tag}.npz"
    if src is None:
        src = C.OUT + f"coarse_{kind}_nu4_ds0.1.npz"
    if os.path.exists(path) and os.environ.get("BLOCK_RESUME", "1") == "1":
        d = np.load(path)
        coef = d["coef"]
        if "ex_h" in d:
            extra = (d["ex_h"], d["ex_p"]); extra_far = (d["ef_q"], d["ef_a"], d["ef_b"])
        print(f"   resumed from {path}", flush=True)
    else:
        coef0 = prolong(mesh, kind, src)
        # rescale so that the fine sample constraints hold: F[lam phi] >= lam F[phi] + (1 - lam) U0
        lam = 1.0
        for _ in range(30):
            ms = min_slack(mesh, kind, lam * coef0, np.arange(mesh.nhex), None)
            if ms >= 1e-6:
                break
            lam *= 0.99 if ms > -0.02 else 0.95
        coef = lam * coef0
        print(f"   warm start from {os.path.basename(src)}: lam0 = {lam:.4f}, min slack {ms:+.2e}; S(c) "
              f"{C.S_value(mesh, coef, kind, c[None], C.TS if kind == 'E' else 1 - C.TS)[0]:.5f}", flush=True)
    blocks = make_blocks(mesh, kind)
    print(f"   {len(blocks)} blocks: layer ranges {sorted(set((b[0], b[1]) for b in blocks))}", flush=True)
    for sw in range(sweeps):
        order = blocks[::-1] if sw % 2 == 0 else blocks
        t0 = time.time()
        for blk in order:
            coef = solve_block(mesh, kind, coef, ot, blk, extra, extra_far)
        np.savez(path, coef=coef, ex_h=extra[0], ex_p=extra[1], ef_q=extra_far[0], ef_a=extra_far[1], ef_b=extra_far[2],
                 nu=nu, ref=ref, sig=mesh.sig, rot=C.ROT[kind])
        print(f" sweep {sw}: {time.time() - t0:.0f} s", flush=True)
        report(mesh, kind, coef, ot_t, c, tag, extra)
    for rd in range(cut_rounds):
        worst, fails, fworst, Sc = report(mesh, kind, coef, ot_t, c, tag, extra) if rd else (None, None, None, None)
        if rd == 0:
            worst, wrel, fails, _ = C.check_cells(mesh, kind, coef)
            fworst = C.far_check(mesh, kind, coef)
        if np.sum(worst < 0) == 0 and fworst.min() >= 0:
            break
        g = np.array([[a, b, cc] for a in (0, 0.5, 1) for b in (0, 0.5, 1) for cc in (0, 0.5, 1)])
        eh, ep = [], []
        for h, ctr, lo, hi, _ in sorted(fails, key=lambda f: f[4])[:8000]:
            eh.append(np.full(len(g), h)); ep.append(lo + g * (hi - lo))
        if eh:
            extra = (np.concatenate([extra[0]] + eh), np.vstack([extra[1]] + ep))
        fq = np.flatnonzero(fworst < 0)
        if len(fq):
            g2 = np.linspace(0, 1, 5)
            A_, B_ = np.meshgrid(g2, g2, indexing="ij")
            extra_far = (np.concatenate([extra_far[0], np.repeat(fq, 25)]), np.concatenate([extra_far[1], np.tile(A_.ravel(), len(fq))]),
                         np.concatenate([extra_far[2], np.tile(B_.ravel(), len(fq))]))
        # re-solve the blocks whose free layers contain both node layers of a failing cell layer
        badh = np.flatnonzero(worst < 0)
        if len(fq):
            badh = np.union1d(badh, (mesh.nl - 2) * mesh.nq + fq)
        todo = []
        for blk in blocks:
            hb = hexes_of_block(mesh, *blk)
            k0, k1 = blk[0], blk[1]
            hk = badh // mesh.nq
            sel = badh[(hk >= k0) & (hk + 1 <= k1)]
            if blk[2] is not None:
                sel = sel[np.isin(mesh.quad_arr[sel % mesh.nq], blk[2]).all(1)]
            if len(sel):
                todo.append(blk)
        print(f" cut round {rd + 1}: {len(eh)} failing boxes, {len(fq)} far quads; re-solving {len(todo)} blocks", flush=True)
        t0 = time.time()
        for blk in todo[::-1]:
            coef = solve_block(mesh, kind, coef, ot, blk, extra, extra_far)
        np.savez(path, coef=coef, ex_h=extra[0], ex_p=extra[1], ef_q=extra_far[0], ef_a=extra_far[1], ef_b=extra_far[2],
                 nu=nu, ref=ref, sig=mesh.sig, rot=C.ROT[kind])
        print(f"   {time.time() - t0:.0f} s", flush=True)
    report(mesh, kind, coef, ot_t, c, tag, extra)


def finalize(kind, tag, depth=3):
    d = np.load(OUT + f"block_{kind}_{tag}.npz")
    nu, ref = int(d["nu"]), str(d["ref"])
    mesh = make_mesh(kind, nu, ref)
    coef = d["coef"]
    t0 = time.time()
    worst, wrel, fails, ncheck = C.check_cells(mesh, kind, coef, depth)
    fw = C.far_check(mesh, kind, coef)
    Ulo = np.zeros(mesh.nhex)
    for i0 in range(0, mesh.nhex, 20000):
        h = np.arange(i0, min(i0 + 20000, mesh.nhex))
        F, U0, Q, SIG = C.F_hex_grid(mesh, kind, coef, h, 4, np.zeros((len(h), 3)), np.ones((len(h), 3)))
        Ulo[h] = C.split_bound(np.zeros_like(F) + U0, U0, Q, SIG)
    lam = 1.0
    neg = worst < 0
    if neg.any():
        lam = min(lam, np.min(Ulo[neg] / (Ulo[neg] - worst[neg])))
    if (fw < 0).any():
        lam = min(lam, np.min(3.0 / (3.0 - fw[fw < 0])))
    lam *= 0.9999
    coef2 = coef * lam
    worst2, wrel2, _, ncheck2 = C.check_cells(mesh, kind, coef2, depth)
    fw2 = C.far_check(mesh, kind, coef2)
    tau = C.TS if kind == "E" else 1 - C.TS
    from words_eight_shape import eight_discrete
    c = eight_discrete(48)[0][24]
    Sc0 = C.S_value(mesh, coef, kind, c[None], tau)[0]
    Sc = C.S_value(mesh, coef2, kind, c[None], tau)[0]
    Vc = 1.22789891 if kind == "E" else 1.29195718
    print(f"{kind} {tag}: before: worst cell bound {worst.min():+.3e}, failing {np.sum(worst < 0)}, far {fw.min():+.3e}, "
          f"S(c) {Sc0:.5f} ({100 * (Vc - Sc0) / Vc:.2f} %); lam = {lam:.6f}; after: worst cell bound {worst2.min():+.3e} "
          f"(failing {np.sum(worst2 < 0)}), far worst {fw2.min():+.3e}; boxes {ncheck2}; S(c) = {Sc:.5f}, deficit "
          f"{Vc - Sc:.4f} ({100 * (Vc - Sc) / Vc:.2f} %); {time.time() - t0:.0f} s", flush=True)
    np.savez(OUT + f"block_{kind}_{tag}_final.npz", coef=coef2, lam=lam, worst=worst2, fworst=fw2, nu=nu, ref=ref)


def export_json(kind, tag):
    d = np.load(OUT + f"block_{kind}_{tag}_final.npz")
    nu, ref = int(d["nu"]), str(d["ref"])
    mesh = make_mesh(kind, nu, ref)
    coef = d["coef"]
    sgn = np.array([-1.0, 1.0])
    # monomial coefficients of the trilinear interpolant on [-1, 1]^3: C[i,j,l] = sum_v v s^i s^j s^l / 8
    T = np.zeros((8, 8))
    for i in range(2):
        for j in range(2):
            for l in range(2):
                for a in range(2):
                    for cidx in range(4):
                        b, cc = cidx & 1, cidx >> 1
                        T[4 * i + 2 * j + l, 4 * a + cidx] = sgn[a] ** i * sgn[b] ** j * sgn[cc] ** l / 8
    cells = []
    for k in range(mesh.nl - 1):
        cols = np.concatenate([C.Mesh.vid(mesh, np.full((mesh.nq, 1), k), mesh.quad_arr),
                               C.Mesh.vid(mesh, np.full((mesh.nq, 1), k + 1), mesh.quad_arr)], axis=1)
        M = coef[cols] @ T.T
        for qi, (fc, ua0, ub0, du, ids) in enumerate(mesh.quads):
            cells.append({"face": int(fc), "sigma": [float(mesh.sig[k]), float(mesh.sig[k + 1])],
                          "ua": [float(ua0), float(ua0 + du)], "ub": [float(ub0), float(ub0 + du)],
                          "deg": [1, 1, 1], "coef": M[qi].reshape(2, 2, 2).tolist()})
    out = {"side": kind,
           "faces": [{"centre": c.tolist(), "ea": ea.tolist(), "eb": eb.tolist(), "ub_range": list(r)} for c, ea, eb, r in mesh.faces],
           "cells": cells, "far": {"sigma_hi": float(mesh.shi)},
           "low": {"sigma_lo": float(mesh.slo), "phi_lo": float(coef[0])},
           "source": f"block_{kind}_{tag}_final.npz"}
    path = OUT + f"cert_pieces_{kind}_block.json"
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {path}: {len(cells)} cells")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5],
            int(sys.argv[6]) if len(sys.argv) > 6 else 2, int(sys.argv[7]) if len(sys.argv) > 7 else 3,
            sys.argv[8] if len(sys.argv) > 8 else None)
    elif cmd == "final":
        finalize(sys.argv[2], sys.argv[3])
    elif cmd == "json":
        export_json(sys.argv[2], sys.argv[3])
