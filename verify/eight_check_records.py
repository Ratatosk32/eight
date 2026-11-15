"""Figure-eight certificate checker, stage F: (ii-L) records, (ii-1), (iii-L) and their cover
(paper/review/eight_certificate_spec.md C5, C6; referee notes eight_certificate_spec_referee.md G1, G2, G6).

    stage_f(run_dir, saves_dir, out_dir, leaves, workers=8, old_tube=None, check_universe=True, mutation=None)

Inputs (explicit, nothing else is read): run_dir/k1_a8.npz (stage B: Krawczyk K1 and the enclosure of A8), the saved
per-batch npz files of stage E in saves_dir (b_*.npz, full_*.npz, light_*.npz; an empty directory is a FAIL: boxes, closed, light, piece grid, centre and tube boxes, k2,
y1, y2, ...), the certificate's final leaves (n, 6): phi_a, phi_b, kappa_a, kappa_b, eps_top, type (0 = R leaf,
1 = record-only leaf below EPS_MIN), the tube files (tube A, and optionally the earlier tube for the top check, which
is then a declared input: tube A inside it is the tube fact F6 of stage B). Everything is written to out_dir.
No environment variable is read: the constants below are set explicitly in every process and printed; the Taylor
order is set per call (refine_many: THIN_ORDER, enclose_k: BOX_ORDER) and restored after it. Any exception fails the
stage (no per-file catch); a refinement exception inside the record search (counted by boxes_result) is a FAIL.

Steps (one pool of `workers` processes, spawn):
 F1 records: for every closed thin saved box, eight_iiL_shadowrec.boxes_result (classify, records, refinement) and
    ii1_box; then an independent VERIFICATION of the returned rows of every unit (verify_units):
      out: certainly outside Lambda(s) on every piece (not on truncated data); I: q_D > q* certainly;
      s1: certainly inside on every piece up to tau = min(1, 1/stop_lo) for every r of the scale interval (D2);
      T: the node interval inside its piece, every earlier piece and the node's part certainly inside and below s = 1
         (G1), lb > 0; fail / unres: FAIL;
      tiling (G6): the row label boxes of a box lie in the box and their volumes add up to the volume of
         [ell grid] x phi range x M range (relative 1e-9).
 F2 (ii-1) with refinement (eight_iiiL_shadow mode "ii1") for every box whose ii1_box left a failing piece.
 F3 (iii-L) level 1 (mode "X") on every file with a box of eps_c >= exp(-1.5 ell_hi); level 2: the shells from K1
    (K1_HALF 2^k, k = 0..NSHELL) and the merged level-1 survivors outside the shell cube (alpha, beta, gamma, delta).
 F4 top check (referee 5c C2): every passing box sits at the top of Q on its (phi, kappa) rectangle (tube A, or the
    declared earlier tube for slices tube A does not certify).
 F5 cover: every certificate leaf is matched to a saved box built from it (eps = eps_top, phi range, M range
    containing hull(kappa sqrt(eps_top)), G4) which passed F1 to F4; a leaf without such a box passes only if the
    label tests alone show it needs no record (gap check: every label box out, I or s1). With check_universe:
    the boundary patches below EPS_MIN that need a record (eight_iiL_cover.needs_below, recomputed into out_dir) have
    their (phi, kappa) cells inside the union of passing leaf rectangles (G2), and no patch has an unusable enclosure.
Output: out_dir/stage_f.npz and stage_f.log, and one line "STAGE F PASS" or "STAGE F FAIL <reason>".
Mutations (each must FAIL): mutation = "m6" (the first T row of the first file forced to class "out") or "m7" (the
node of that row moved to the first piece after the inside run)."""
import glob
import math
import os
import sys
import time

for _k in list(os.environ):                         # this stage reads no environment variable of the search code
    if _k.startswith(("SR_", "X_", "RC_", "RS_", "CT_", "TC_")) or _k == "LORDER":
        del os.environ[_k]

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

CONST = dict(DELL=0.0025, LB_MIN=1e-4, NSUB=3, FDEPTH=1, RLEVELS=2, REL_F=0.005, REL_MIN=2e-4, REL_MAX=0.01, EDEPTH=4,
             REF_N=2, THIN_ORDER=6, BOX_ORDER=12, CHUNK=12, MFINE_MAX=256, PFRAC=(0.875, 0.75, 0.5, 0.25), Q_STAR=0.86,
             X_ELEV=3, X_RLEV=5, X_REFN=2, FINE_W=0.001, MAXB=3000, NSHELL=6, TC_NSLAB=1000, EPS_MIN=1e-3,
             NEEDS_DEPTH=2, NEEDS_NUV=48, VOL_RTOL=1e-9)
_G = {}
SAVE_PATTERNS = ("b_*.npz", "full_*.npz", "light_*.npz")    # saved batch files of the search and of stage E


# ---------------------------------------------------------------- configuration (every process)
def _configure(cfg):
    import eight_iiL_shadowrec as SR
    import eight_iiiL_shadow as X
    import eight_iiL_cover as C
    import eight_R_cover2 as RC
    import words_eight_tubeb as W
    c = cfg["const"]
    art = X.load_k1(cfg["k1_path"], a8_hi=cfg["a8_hi"])
    SR.A8_HI = X.A8_HI = cfg["a8_hi"]
    SR.LB_MIN, SR.REL_F, SR.REL_MIN, SR.REL_MAX = c["LB_MIN"], c["REL_F"], c["REL_MIN"], c["REL_MAX"]
    SR.NSUB, SR.FDEPTH, SR.RLEVELS, SR.EDEPTH, SR.REF_N = c["NSUB"], c["FDEPTH"], c["RLEVELS"], c["EDEPTH"], c["REF_N"]
    SR.THIN_ORDER, SR.CHUNK, SR.MFINE_MAX, SR.PFRAC = c["THIN_ORDER"], c["CHUNK"], c["MFINE_MAX"], c["PFRAC"]
    X.BOX_ORDER, X.ELEV, X.X_RLEV, X.X_REFN, X.FINE_W, X.MAXB = (c["BOX_ORDER"], c["X_ELEV"], c["X_RLEV"], c["X_REFN"],
                                                                 c["FINE_W"], c["MAXB"])
    RC.NSLAB = c["TC_NSLAB"]
    assert SR.Q_STAR == c["Q_STAR"] and C.EPS_MIN == c["EPS_MIN"]
    # default arguments bound at definition time must equal the constants (no environment at import)
    import inspect

    def dflt(f, name):
        return inspect.signature(f).parameters[name].default
    assert dflt(SR.classify_box, "fdepth") == c["FDEPTH"] and dflt(SR.classify_box, "edepth") == c["EDEPTH"],         "classify_box defaults"
    assert dflt(SR.records, "nsub") == c["NSUB"], "records default nsub"
    assert dflt(SR.refine_many, "nphi") == c["REF_N"] and dflt(SR.refine_many, "nM") == c["REF_N"], "refine_many"
    assert dflt(SR.classify_box, "dell") == c["DELL"], "classify_box default dell"
    T = SR.tube()
    assert os.path.abspath(SR.TUBE_A) == os.path.abspath(cfg["tube_a"])
    _G.update(cfg=cfg, SR=SR, X=X, C=C, RC=RC, T=T, Tq=SR.section_tube(T, c["Q_STAR"]),
              co=__import__("eight_tube_boundary").CoarseE(cfg["goal"]), er=SR.ell_range(T),
              tA=W.EllTubeFit(cfg["tube_a"]), tO=(W.EllTubeFit(cfg["old_tube"]) if cfg["old_tube"] else None), art=art)
    if cfg.get("mutation"):
        _install_mutation(cfg["mutation"], cfg["mut_file"])


def _init(cfg):
    import eight_iiL_shadowrec as SR
    SR.lowprio()
    _configure(cfg)


# ---------------------------------------------------------------- mutations (tests of the stage; each must FAIL)
def _install_mutation(kind, mut_file):
    SR = _G["SR"]
    orig = SR.classify_box
    state = {"done": False}

    def mutated(T, Tq, co, D, b, *a, **k):
        rows = orig(T, Tq, co, D, b, *a, **k)
        if state["done"] or os.path.basename(str(D.get("_fname", ""))) != mut_file:
            return rows
        for i, r in enumerate(rows):
            if r[0] == "T":
                n_ = int(r[7])
                if kind == "m6":            # a boundary label sub-box forced to class "out"
                    rows[i] = ("out", *r[1:7], -1, np.nan, np.nan, np.nan, np.nan)
                else:                       # m7: the node moved to the first piece after the inside run
                    ta, tb = D["ta"], D["tb"]
                    n2 = min(len(ta) - 1, n_ + 1)
                    while n2 < len(ta) - 1 and _row_inside(D, b, r, ta[n2], tb[n2]):
                        n2 += 1
                    rows[i] = ("T", *r[1:7], n2, r[8], float(ta[n2]), float(tb[n2]), np.nan)
                state["done"] = True
                break
        return rows
    SR.classify_box = mutated


def _row_inside(D, b, r, t_lo, t_hi):
    SR, T = _G["SR"], _G["T"]
    W_ = np.array([r[1:7]], float)
    i_, _, s1_ = SR._slab_tests(T, W_, D["boxes"][b][0], np.array([float(t_lo)]), np.array([float(t_hi)]),
                                need_out=False)
    return bool(i_[0] and not s1_[0])


# ---------------------------------------------------------------- F1: records + verification of the rows
def verify_units(units, ells):
    """independent re-check of the class of every row (see the module text); returns (violations, details)"""
    SR, T, Tq = _G["SR"], _G["T"], _G["Tq"]
    bad = []
    vol_by_box = {}
    for (k, Du, bu, rows, _) in units:
        ta, tb = Du["ta"], Du["tb"]
        npc = len(ta)
        eps = float(Du["boxes"][bu][0])
        trunc = bool(Du.get("trunc", False))
        for r in rows:
            cls = r[0]
            W1 = np.array([r[1:7]], float)
            if not np.all(np.isfinite(W1)):
                bad.append(("row without coordinates", k))
                continue
            vol_by_box[k] = vol_by_box.get(k, 0.0) + float(np.prod(W1[0, 1::2] - W1[0, 0::2]))
            if cls == "out":
                if trunc:
                    bad.append(("out on truncated data", k))
                    continue
                Wn = np.repeat(W1, npc, 0)
                _, o_, _ = SR._slab_tests(T, Wn, eps, ta.astype(float), tb.astype(float), need_in=False)
                if not o_.all():
                    bad.append(("out not certainly outside on every piece", k))
            elif cls == "I":
                z = np.zeros(1)
                if not Tq.box_certainly_outside(np.stack([W1[:, 0], W1[:, 2], z], 1),
                                                np.stack([W1[:, 1], W1[:, 3], z], 1), z, z)[0]:
                    bad.append(("I not certain", k))
            elif cls == "s1":
                stop_lo = math.sqrt(eps) * math.exp(W1[0, 0]) ** 0.75 * (1 - 1e-12)
                tau_end = min(1.0, 1.0 / stop_lo)
                sel = np.flatnonzero(ta < tau_end)
                hi = np.minimum(tb[sel], tau_end).astype(float)
                hi[-1] = tau_end
                Wn = np.repeat(W1, len(sel), 0)
                i_, _, _ = SR._slab_tests(T, Wn, eps, ta[sel].astype(float), hi, need_out=False)
                if not i_.all() or (trunc and tau_end > tb[-1]):
                    bad.append(("s1 not certainly inside up to s = 1", k))
            elif cls == "T":
                n_ = int(r[7])
                t_a, t_b = float(r[9]), float(r[10])
                if not (0 <= n_ < npc and ta[n_] <= t_a <= t_b <= tb[n_] and t_a == ta[n_]):
                    bad.append(("T node interval not an initial part of its piece", k))
                    continue
                lo = np.concatenate([ta[:n_], [ta[n_]]]).astype(float)
                hi = np.concatenate([tb[:n_], [t_b]]).astype(float)
                Wn = np.repeat(W1, len(lo), 0)
                i_, _, s1_ = SR._slab_tests(T, Wn, eps, lo, hi, need_out=False)
                if not (i_.all() and not s1_.any()):
                    bad.append(("T node not preceded by a certainly-inside run (G1)", k))
                if not (r[8] > 0):
                    bad.append(("T with lb <= 0", k))
            else:
                bad.append((f"class {cls}", k))
    return bad, vol_by_box


def _rec_file(path):
    """F1 for one saved file: per box (fname, box, eps, phi_lo, phi_hi, M_lo, M_hi, closed, thin, counts..., min lb,
    n_referr, n_subnc, ii1 counts, violations, vol_ok)"""
    SR, T, Tq, co = _G["SR"], _G["T"], _G["Tq"], _G["co"]
    ell_lo, ell_hi = _G["er"]
    c = _G["cfg"]["const"]
    D = dict(np.load(path, allow_pickle=True))
    if bool(np.asarray(D.get("slab", False))):
        raise RuntimeError(f"{os.path.basename(path)}: slab box in the saves directory")
    D["_fname"] = os.path.basename(path)
    nb = D["boxes"].shape[0]
    thin = np.asarray(D.get("dl", np.zeros(nb)), float) <= 1e-300
    cl = [b for b in range(nb) if bool(D["closed"][b]) and thin[b]]
    units = []
    res = SR.boxes_result(T, Tq, co, D, cl, ell_lo, ell_hi, c["DELL"], units_out=units) if cl else []
    ells = SR.ell_grid(ell_lo, ell_hi, c["DELL"])
    bad, vols = verify_units(units, ells)
    out = []
    br = dict(zip(cl, res))
    for b in range(nb):
        Pb = D["boxes"][b]
        row = dict(fname=os.path.basename(path), box=b, eps=float(Pb[0]), phi_lo=float(Pb[2]), phi_hi=float(Pb[3]),
                   M_lo=float(Pb[4]), M_hi=float(Pb[5]), closed=bool(D["closed"][b]), thin=bool(thin[b]))
        if b in br:
            k = cl.index(b)
            cnt, mlb, nu, rows, tight, ne, nn = br[b]
            c1, w1 = SR.ii1_box(T, D, b, ell_lo, ell_hi, c["DELL"])
            total = (ells[-1] - ells[0]) * (Pb[3] - Pb[2]) * (Pb[5] - Pb[4])
            vol = vols.get(k, 0.0)
            row.update(n_T=cnt["T"], n_out=cnt["out"], n_I=cnt["I"], n_s1=cnt["s1"], n_fail=cnt["fail"],
                       n_unres=cnt["unres"], min_lb=mlb, n_referr=ne, n_subnc=nn, ii1_fail=c1["n_fail"],
                       ii1_bdry=c1["n_bdry"], viol=sum(1 for v in bad if v[1] == k),
                       viol_kinds=sorted(set(v[0] for v in bad if v[1] == k)),
                       vol_ok=bool(abs(vol - total) <= c["VOL_RTOL"] * total))
        out.append(row)
    return out


# ---------------------------------------------------------------- F2, F3 tasks
def _ii1_file(path):
    X = _G["X"]
    X.MODE = "ii1"
    try:
        D, res = X.level1_file(_G["T"], path, *_G["er"], dell=_G["cfg"]["const"]["DELL"])
    finally:
        X.MODE = "X"
    return [(os.path.basename(path), b, sum(1 for r in rows if r[0] == "surv"),
             sum(1 for r in rows if r[0] in ("notclosed", "notthin"))) for b, rows in res.items()]


def _x_file(path):
    X = _G["X"]
    X.MODE = "X"
    D, res = X.level1_file(_G["T"], path, *_G["er"], dell=_G["cfg"]["const"]["DELL"])
    surv, cnt = [], {}
    for b, rows in res.items():
        for r in rows:
            cnt[r[0]] = cnt.get(r[0], 0) + 1
            if r[0] == "surv":
                if r[2] is None:
                    surv.append(np.full(6, np.nan))
                else:
                    surv.append(np.concatenate([r[2], r[3]]))
    return os.path.basename(path), cnt, surv


def _l2_task(task):
    X = _G["X"]
    kind, B = task
    leaves = []
    st, nev, dt, left = X.l2_boxes(B, tag=kind, leaves=leaves)
    return kind, st, nev, len(left), [(t, b) for (t, bb) in leaves for b in bb]


def _top_task(rows):
    """top check of the rows (eps_B, phi_lo, phi_hi, kappa_lo, kappa_hi): 0 tube A, 1 earlier tube (F6), 2 FAIL"""
    C = _G["C"]
    import eight_R_shadow as RS
    tA, tO = _G["tA"], _G["tO"]
    emax = max(RS.q_bounds(tA)[4], RS.q_bounds(tO)[4] if tO is not None else 0.0) * 1.02
    C._TC.update(RC=_G["RC"], tA=tA, tO=(tO if tO is not None else tA), emax=emax)
    return C._tc_task(np.asarray(rows, float))


def _needs_task(out_path):
    C = _G["C"]
    c = _G["cfg"]["const"]
    C.NEEDS = out_path
    C.needs_below(c["NEEDS_DEPTH"], c["NEEDS_NUV"])
    return out_path


def _gap_task(leaf):
    """label tests only on the leaf's box (eps_top, cell): True if no label box needs a record"""
    SR = _G["SR"]
    import eight_R_shadow as RS
    B = RS.make_box(leaf[:4], leaf[4])
    tau = np.linspace(0, 1, 161)
    D = {"boxes": B[None, :], "ta": tau[:-1], "tb": tau[1:]}
    orig = SR.records

    class Needs(Exception):
        pass

    def stub(*a, **k):
        raise Needs()
    SR.records = stub
    try:
        rows = SR.classify_box(_G["T"], _G["Tq"], None, D, 0, *_G["er"], _G["cfg"]["const"]["DELL"])
        return not any(r[0] in ("unres", "fail") for r in rows)
    except Needs:
        return False
    finally:
        SR.records = orig


# ---------------------------------------------------------------- the stage
def stage_f(run_dir, saves_dir, out_dir, leaves, workers=8, old_tube=None, check_universe=True, mutation=None,
            files=None):
    """run stage F; returns (passed, reason). leaves: (n, 6) phi_a, phi_b, kappa_a, kappa_b, eps_top, type.
    Any exception is a FAIL of the stage (printed with its type and message)."""
    try:
        return _stage_f(run_dir, saves_dir, out_dir, leaves, workers, old_tube, check_universe, mutation, files)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"STAGE F FAIL exception {type(e).__name__}: {e}"[:400], flush=True)
        return False, f"exception {type(e).__name__}: {e}"


def _stage_f(run_dir, saves_dir, out_dir, leaves, workers, old_tube, check_universe, mutation, files):
    import multiprocessing as mp
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "stage_f.log"), "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    import eight_iiL_shadowrec as SR
    import eight_iiiL_shadow as X
    art = np.load(os.path.join(run_dir, "k1_a8.npz"))
    a8_hi = float(art["a8_act_hi"])
    cfg = dict(const=CONST, k1_path=os.path.join(run_dir, "k1_a8.npz"), a8_hi=a8_hi, tube_a=SR.TUBE_A, goal=SR.GOAL,
               old_tube=old_tube, mutation=mutation, mut_file=None)
    if files is None:
        files = sorted(set(f for pat in SAVE_PATTERNS for f in glob.glob(os.path.join(saves_dir, pat))))
    files = list(files)
    if not files:                           # an empty saves directory must never reach the cover step
        raise RuntimeError(f"no saved files ({', '.join(SAVE_PATTERNS)}) in {saves_dir}")
    fpath = {os.path.basename(f): f for f in files}
    assert len(fpath) == len(files), "two saved files with the same name"
    if mutation:
        cfg["mut_file"] = os.path.basename(files[0])
    say(f"STAGE F: saves {saves_dir} ({len(files)} files), out {out_dir}, workers {workers}, mutation {mutation}")
    say(f"  constants {CONST}")
    say(f"  A8_hi = {a8_hi!r} (k1_a8.npz a8_act_hi), K1 half-width {float(art['k1_half'])}, Y_1 >= "
        f"{float(art['k1_y1_lo'])}; tube A {SR.TUBE_A} md5 {SR._md5(SR.TUBE_A)}; goal md5 {SR._md5(SR.GOAL)}; "
        f"earlier tube {old_tube} md5 {SR._md5(old_tube) if old_tube else '-'}")
    say("  code md5: " + "; ".join(f"{os.path.basename(p)} {SR._md5(p)}" for p in
                                    [os.path.abspath(__file__), SR.__file__, X.__file__, SR.TB.__file__] +
                                    [q for q in SR.DEP_FILES if os.path.exists(q)]))
    _configure(cfg)
    reasons = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        needs_out = os.path.join(out_dir, "needs_below.npz")
        needs_job = pool.apply_async(_needs_task, (needs_out,)) if check_universe else None
        # F1
        rec = []
        for k, rows in enumerate(pool.imap_unordered(_rec_file, files, chunksize=1)):
            rec += rows
            if (k + 1) % 200 == 0:
                say(f"  F1 records: {k + 1}/{len(files)} files ({time.time() - t0:.0f} s)")
        R = {key: [r.get(key, np.nan) for r in rec] for key in ("fname", "box", "eps", "phi_lo", "phi_hi", "M_lo",
                                                              "M_hi", "closed", "thin", "n_T", "n_fail", "n_unres",
                                                              "min_lb", "n_referr", "n_subnc", "ii1_fail", "viol",
                                                              "vol_ok")}
        okb = [r for r in rec if r["closed"] and r["thin"]]
        nviol = sum(r["viol"] for r in okb)
        kinds = sorted(set(k for r in okb for k in r["viol_kinds"]))
        rec_pass = {(r["fname"], r["box"]): (r["n_fail"] == 0 and r["n_unres"] == 0 and r["n_referr"] == 0
                                             and r["viol"] == 0 and r["vol_ok"]) for r in okb}
        lbs = [r["min_lb"] for r in okb if np.isfinite(r["min_lb"])]
        say(f"F1 records: {len(rec)} boxes, closed and thin {len(okb)}, passing {sum(rec_pass.values())}; failing rows "
            f"{sum(r['n_fail'] for r in okb)}, unresolved {sum(r['n_unres'] for r in okb)}, refinement exceptions "
            f"{sum(r['n_referr'] for r in okb)}, unclosed sub-boxes {sum(r['n_subnc'] for r in okb)}; verification "
            f"violations {nviol} {kinds}; tiling failures {sum(1 for r in okb if not r['vol_ok'])}; min lb "
            f"{min(lbs) if lbs else float('nan'):+.4e} ({time.time() - t0:.0f} s)")
        if nviol:
            reasons.append(f"F1 verification: {nviol} rows ({', '.join(kinds)})")
        # F2
        ii1_files = sorted(set(r["fname"] for r in okb if r["ii1_fail"] > 0))
        ii1_ok = {(r["fname"], r["box"]): r["ii1_fail"] <= 0 for r in okb}
        if ii1_files:
            paths = [fpath[f] for f in ii1_files]
            for rows in pool.imap_unordered(_ii1_file, paths, chunksize=1):
                for (f, b, ns, nc) in rows:
                    if (f, b) in ii1_ok and not ii1_ok[(f, b)]:
                        ii1_ok[(f, b)] = (ns == 0 and nc == 0)
        say(f"F2 (ii-1): boxes with a failing piece {sum(1 for r in okb if r['ii1_fail'] > 0)} in {len(ii1_files)} "
            f"files; still failing after refinement {sum(1 for v in ii1_ok.values() if not v)} ({time.time() - t0:.0f} s)")
        # F3
        ell_hi = _G["er"][1]
        emin = math.exp(ell_hi) ** -1.5 * (1 - 1e-9)
        assert emin > CONST["EPS_MIN"], "an s = 1 label with eps < EPS_MIN"
        xfiles = sorted(set(fpath[r["fname"]] for r in okb if r["eps"] >= emin))
        surv, xcnt = [], {}
        for (f, cnt, sv) in pool.imap_unordered(_x_file, xfiles, chunksize=1):
            surv += sv
            for k_, v in cnt.items():
                xcnt[k_] = xcnt.get(k_, 0) + v
        S = np.array(surv).reshape(-1, 6)
        nbad_surv = int(np.sum(~np.all(np.isfinite(S), 1)))
        S = S[np.all(np.isfinite(S), 1)]
        K1 = float(art["k1_half"])
        shells = [K1 * 2 ** k for k in range(CONST["NSHELL"] + 1)]
        H = shells[-1]
        inside = np.all(S[:, 0:3] >= X.A8_LAB - H, 1) & np.all(S[:, 3:6] <= X.A8_LAB + H, 1)
        B = S[~inside]
        lim = np.array([1e-3, 1e-3, 5e-4])
        if len(B):
            key = np.floor(0.5 * (B[:, 0:3] + B[:, 3:6]) / (2 * lim)).astype(np.int64)
            groups = {}
            for i, kk in enumerate(map(tuple, key)):
                groups.setdefault(kk, []).append(i)
            B = X._presplit(np.array([np.concatenate([B[ix, 0:3].min(0), B[ix, 3:6].max(0)])
                                      for ix in groups.values()]), lim)
        tasks = [(f"shell {shells[i]:g}..{shells[i + 1]:g}", X.shell_boxes(shells[i], shells[i + 1]))
                 for i in range(len(shells) - 1)]
        tasks += [(f"surv {i // 48}", B[i:i + 48]) for i in range(0, len(B), 48)]
        left, l2 = 0, []
        for (kind, st, nev, nleft, lv) in pool.imap_unordered(_l2_task, tasks, chunksize=1):
            left += nleft
            l2.append((kind, nev, nleft, str(st)))
        say(f"F3 (iii-L): level 1 on {len(xfiles)} files {xcnt}; survivors {len(S)} (+{nbad_surv} without an s = 1 "
            f"box), {int(inside.sum())} inside the shell cube {H:g}; level 2: {len(tasks)} tasks, "
            f"{sum(t[1] for t in l2)} boxes evaluated, left {left} ({time.time() - t0:.0f} s)")
        for t in sorted(l2):
            say(f"    {t[0]}: {t[1]} boxes, left {t[2]}, {t[3]}")
        if left or nbad_surv or xcnt.get("notclosed", 0) or xcnt.get("notthin", 0):
            x_fail_files = True
        else:
            x_fail_files = False
        # F4 top check of the passing boxes
        passing = [r for r in okb if rec_pass[(r["fname"], r["box"])] and ii1_ok[(r["fname"], r["box"])]]
        rows = []
        for r in passing:
            sq = math.sqrt(r["eps"])
            k1, k2 = sorted((r["M_lo"] / sq, r["M_hi"] / sq))
            rows.append([r["eps"], r["phi_lo"], r["phi_hi"], k1, k2])
        rows = np.array(rows).reshape(-1, 5)
        parts = [rows[i:i + 64] for i in range(0, len(rows), 64)]
        top = np.concatenate(pool.map(_top_task, parts, chunksize=1)) if parts else np.zeros(0, int)
        say(f"F4 top check: {len(top)} boxes; tube A {int(np.sum(top == 0))}, earlier tube {int(np.sum(top == 1))}, "
            f"NOT at the top {int(np.sum(top == 2))} ({time.time() - t0:.0f} s)")
        good = {(r["fname"], r["box"]): r for r, t in zip(passing, top) if t != 2}
        # F5 cover
        leaves = np.asarray(leaves, float).reshape(-1, 6)
        import eight_R_shadow as RS
        byk = {}
        for g in good.values():
            byk.setdefault((g["eps"], g["phi_lo"], g["phi_hi"]), []).append((g["M_lo"], g["M_hi"]))
        missing = []
        for li, lf in enumerate(leaves):
            box = RS.make_box(lf[:4], lf[4])
            hit = any(ml <= box[4] and mh >= box[5] for (ml, mh) in byk.get((box[0], box[2], box[3]), ()))
            if not hit:
                missing.append(li)
        gap_ok = pool.map(_gap_task, [leaves[i] for i in missing], chunksize=1) if missing else []
        nmiss = sum(1 for v in gap_ok if not v)
        say(f"F5 cover: {len(leaves)} leaves (R {int(np.sum(leaves[:, 5] == 0))}, record-only "
            f"{int(np.sum(leaves[:, 5] == 1))}); without a passing saved box {len(missing)}, of which needing a record "
            f"{nmiss} ({time.time() - t0:.0f} s)")
        if nmiss:
            reasons.append(f"F5 cover: {nmiss} leaves need records and have no passing box")
        if x_fail_files:
            reasons.append(f"F3 (iii-L): left {left}, survivors without s = 1 box {nbad_surv}, not closed / not thin "
                           f"{xcnt.get('notclosed', 0)} / {xcnt.get('notthin', 0)}")
        if check_universe:
            needs_job.get()
            z = np.load(needs_out)
            nbad = int(np.sum(z["rows"][:, 3])) if len(z["rows"]) else 0
            occ = z["occ"]
            P0, DP, K0, DK = z["occ_grid"]
            ii, jj = np.nonzero(np.isfinite(occ))
            cells = np.stack([P0 + ii * DP, P0 + (ii + 1) * DP, np.sinh(K0 + jj * DK), np.sinh(K0 + (jj + 1) * DK)], 1)
            gapd = dict(zip(missing, gap_ok))
            ok_rect = np.array([lf[:4] for li, lf in enumerate(leaves) if gapd.get(li, True)]).reshape(-1, 4)
            unc = __import__("eight_iiL_cover").covered(cells, ok_rect) if len(cells) else np.zeros(0)
            nunc = int(np.sum(unc > 0))
            say(f"F5 universe below EPS_MIN: needing fine cells {len(cells)}, not inside the passing leaf rectangles "
                f"{nunc}; patches with an unusable enclosure {nbad} ({time.time() - t0:.0f} s)")
            if nunc or nbad:
                reasons.append(f"F5 universe: {nunc} needing cells outside the leaves, {nbad} unusable patches")
    np.savez(os.path.join(out_dir, "stage_f.npz"), **{k: np.array(v) for k, v in R.items()},
             top=top, leaves=leaves, missing=np.array(missing, int), constants=str(CONST), reasons=np.array(reasons))
    passed = not reasons
    say(f"STAGE F {'PASS' if passed else 'FAIL ' + '; '.join(reasons)} ({time.time() - t0:.0f} s)")
    log.close()
    return passed, "; ".join(reasons)


if __name__ == "__main__":
    # python verify/eight_check_records.py <run_dir> <saves_dir> <out_dir> <leaves.npy> [workers] [old_tube|-]
    #        [universe 1|0] [mutation m6|m7|-]
    a = sys.argv[1:]
    lv = np.load(a[3])
    ok_, _ = stage_f(a[0], a[1], a[2], lv, int(a[4]) if len(a) > 4 else 8,
                     (a[5] if len(a) > 5 and a[5] != "-" else None), (a[6] != "0") if len(a) > 6 else True,
                     (a[7] if len(a) > 7 and a[7] != "-" else None))
    sys.exit(0 if ok_ else 1)
