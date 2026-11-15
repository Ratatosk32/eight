"""Stage 21: CHECKER of the figure-eight certificate, stages A-E (spec: paper/review/eight_certificate_spec.md with the
theorist's and the referee's reviews). The certificate (verify/eight_make_certificate.py, never imported here) is
untrusted: it supplies only the leaf rectangles with their eps_top, the record-only rectangles, the slab boxes and a
header; every box, tube, bound, universe and remainder is computed here. Any exception or any failed test is a FAIL
of the whole run. Settings are constants in this file (printed); no environment variable is read by this file (the
trusted modules read theirs, which are set below before they are imported).
    python verify/eight_check.py <certificate.npz> <run_dir> [stages, default ABCDE]
Stage F (records, (ii-1), (iii-L), record cover, top check) is verify/eight_check_records.py, run on <run_dir>."""
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TUBE_A = os.path.join(ROOT, "words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper_b070_t02_03.npz")
EPS_MIN = 1e-3
NSLAB = 1000            # s-slabs of the outside test
NSL = 240               # log-eps slices of the eps_top computation
LORDER = 12             # Lohner Taylor order
BATCH = 16              # boxes per comparison-tube call
WORKERS = 10
H = 0.01                # level-0 grid: phi width 2H, 64 kappa cells over [-kmax, kmax] (as the search; any grid works)
os.environ.update(RC_TUBE=TUBE_A, RS_EPSMIN=repr(EPS_MIN), RC_NSLAB=str(NSLAB), RS_SLICES=str(NSL), LORDER=str(LORDER),
                  CT_SPLIT="2", CT_BALLS="1", CT_ADAPT="0")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from prove_newton_1212 import I, dn, up  # noqa: E402


class Fail(Exception):
    pass


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(run, msg):
    print(msg, flush=True)
    with open(os.path.join(run, "check.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def m_hull(cells, tops):
    """(fresh) outward-rounded hull of kappa sqrt(eps_top) over the kappa range: the box's M range"""
    s = I(np.asarray(tops, float), np.asarray(tops, float).copy()).sqrt()
    a = I(cells[:, 2], cells[:, 2].copy()) * s
    b = I(cells[:, 3], cells[:, 3].copy()) * s
    return np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi)


def leaf_boxes(cells, tops):
    lo, hi = m_hull(cells, tops)
    return np.stack([tops, tops, cells[:, 0], cells[:, 1], lo, hi], 1)


# ---------------------------------------------------------------- workers
def _w_init():
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import eight_field_lohner as L
    L.ORDER = LORDER


def _w_tube(args):
    """one batch: full mode (closure + Sturm) or light (closure only) or slab; saved data written for full/light"""
    kind, B, path, E = args
    import eight_comptube as CT
    if kind == "slab":
        closed, ok, ymin, r1 = CT.check(B, slab=E)
        return kind, path, closed, ok, ymin
    closed, ok, ymin, r1 = CT.check(B, m=(4 if kind == "light" else 8), save=True, sturm=(kind == "full"))
    sv = CT.check.saved
    np.savez_compressed(path, boxes=B, light=(kind == "light"), slab=False, **sv)
    return kind, path, closed, ok, ymin


def _w_top(cells):
    import eight_R_shadow as RS
    tb = RS.RC.tube()
    plo, phi_, mlo, mhi, emax = RS.q_bounds(tb)
    return RS.eps_top(tb, cells, emax)


# ---------------------------------------------------------------- stages
def stage_A(run, cert, hdr):
    for k, p in (("goal_md5", hdr["goal_json"]), ("tube_md5", hdr["tube"]), ("old_tube_md5", hdr["old_tube"])):
        got = md5(os.path.join(ROOT, p))
        if got != hdr[k]:
            raise Fail(f"A: md5 of {p} is {got}, header says {hdr[k]}")
    if os.path.abspath(os.path.join(ROOT, hdr["tube"])) != os.path.abspath(TUBE_A):
        raise Fail("A: header tube is not the adopted tube A")
    p = subprocess.run([sys.executable, os.path.join(HERE, "eight_tube_facts.py"), TUBE_A], capture_output=True, text=True)
    if "TUBE FACTS VERIFIED" not in p.stdout:
        raise Fail("A: tube facts F1-F6 not verified:\n" + p.stdout[-800:])
    log(run, "A PASS: input md5 match the header; tube facts F1-F6 verified")


def stage_B(run, hdr):
    import eight_field as EF
    import eight_iiL_records as R
    a8 = np.array(hdr["a8"], float)
    out = {}
    for name, d in (("k1", hdr["k1_half"]), ("small", hdr["small_cube"])):
        inside, X, K, yT, minq = EF.krawczyk(a8, [d] * 3)
        y1lo = float(yT[0][0].lo)
        if not inside or not y1lo > 0:
            raise Fail(f"B: Krawczyk cube {d}: proved {inside}, Y_1 lower {y1lo}")
        out[name] = (d, [float(x.lo) for x in X], [float(x.hi) for x in X], y1lo)
    d, xl, xh, _ = out["small"]
    A = np.array([[xl[0], xl[1], xl[2], xh[0], xh[1], xh[2]]])
    o = R.lohner_boxes((A, np.array([[1.0, 1.0]])))
    if not o["ok"][0]:
        raise Fail("B: Lohner over the small Krawczyk cube failed")
    a8_lo, a8_hi = float(o["W_lo"][0]), float(o["W_hi"][0])
    if not a8_hi <= hdr["A8_hi"]:
        raise Fail(f"B: A8 upper bound {a8_hi} exceeds the header's A8_hi {hdr['A8_hi']}")
    np.savez(os.path.join(run, "k1_a8.npz"), k1_half=out["k1"][0], k1_lo=out["k1"][1], k1_hi=out["k1"][2],
             k1_y1lo=out["k1"][3], small_lo=xl, small_hi=xh, A8_lo=a8_lo, A8_hi_proved=a8_hi, A8_hi=hdr["A8_hi"],
             a8=a8,
             # the keys stage F (verify/eight_check_records.py) reads
             ok=True, k1_proved=True, k1_y1_lo=out["k1"][3], a8_lab=a8, a8_act_hi=a8_hi)
    log(run, f"B PASS: Krawczyk K1 (half-width {out['k1'][0]}) and small cube ({d}) proved, Y_1 > 0; "
             f"A8 in [{a8_lo!r}, {a8_hi!r}] <= A8_hi {hdr['A8_hi']!r}")


def stage_C(run, hdr):
    J = os.path.join(ROOT, hdr["goal_json"])
    env = dict(os.environ, VERIFY_CHUNKED="1", VERIFY_WORKERS="8")
    p = subprocess.run([sys.executable, "-u", os.path.join(HERE, "eight_cell_verify.py"), J, "12"],
                       capture_output=True, text=True, env=env)
    with open(os.path.join(run, "cell_verify.log"), "w", encoding="utf-8") as f:
        f.write(p.stdout + p.stderr)
    if "ALL VERIFIED" not in p.stdout:
        raise Fail("C: cell verifier did not verify the coarse piece:\n" + p.stdout[-800:])
    log(run, "C PASS: " + " | ".join(ln.strip() for ln in p.stdout.splitlines() if ln.strip())[:400])


def stage_D(run, cert, hdr):
    """(R) on every leaf (full mode: closed tube and Sturm), the record-only leaves (closed tube), the slab boxes"""
    import multiprocessing as mp
    t0 = time.time()
    saves = os.path.join(run, "saves")
    os.makedirs(saves, exist_ok=True)
    Rc, Rt = cert["R_cells"], cert["R_tops"]
    Ec, Et = cert["rec_cells"], cert["rec_tops"]
    if not (np.all(np.isfinite(Rt)) and np.all(Rt >= EPS_MIN * (1 - 1e-12)) and np.all(Rc[:, 1] > Rc[:, 0])
            and np.all(Rc[:, 3] > Rc[:, 2])):
        raise Fail("D: malformed (R) leaves (non-finite or eps_top < EPS_MIN or empty rectangle)")
    # reuse: a leaf whose box was already PROVED in this run directory (same box, bit for bit, same kind, same checker
    # file) is not recomputed: a box proved in any batch is proved (the batch only shares the time grid of the centre
    # Lohner); the saved file of that batch stays the record of the proof
    done = {"full": set(), "light": set()}
    for f in sorted(os.listdir(saves)):
        if not f.endswith(".npz") or not (f.startswith("full_") or f.startswith("light_")):
            continue
        z = np.load(os.path.join(saves, f), allow_pickle=True)
        kind = "light" if bool(z["light"]) else "full"
        good = z["closed"] if kind == "light" else (z["closed"] & (z["ymin"] > 0))
        for bx, g in zip(z["boxes"], good):
            if g:
                done[kind].add(np.ascontiguousarray(bx, float).tobytes())
    tasks, nreuse = [], {"full": 0, "light": 0}
    tag = time.strftime("%Y%m%d%H%M%S")
    for kind, C, T_ in (("full", Rc, Rt), ("light", Ec, Et)):
        B = leaf_boxes(C, T_)
        need = np.array([np.ascontiguousarray(b_, float).tobytes() not in done[kind] for b_ in B], bool)
        nreuse[kind] = int(np.sum(~need))
        o = [i for i in np.argsort(T_) if need[i]]
        for i in range(0, len(o), BATCH):
            ix = o[i:i + BATCH]
            tasks.append((kind, B[ix], os.path.join(saves, f"{kind}_{tag}_{i // BATCH:05d}.npz"), None))
    log(run, f"  D: reused proved boxes from this run directory: {nreuse}; to compute: {len(tasks)} batches")
    S = np.asarray(cert["slab_boxes"], float)
    E = float(hdr["slab_E"])
    if not (np.all(S[:, 0] == 0.0) and np.all(S[:, 1] == E)):
        raise Fail("D: slab boxes must have eps range exactly [0, E]")
    for i in range(0, len(S), BATCH):
        tasks.append(("slab", S[i:i + BATCH], None, E))
    nfail = {"full": 0, "light": 0, "slab": 0}
    nbox = {"full": 0, "light": 0, "slab": 0}
    ymin = np.inf
    with mp.Pool(WORKERS, _w_init) as pool:
        for k, (kind, path, closed, ok, ym) in enumerate(pool.imap_unordered(_w_tube, tasks)):
            good = closed if kind == "light" else ok
            nbox[kind] += len(good)
            nfail[kind] += int(np.sum(~good))
            if kind != "light":
                ymin = min(ymin, float(np.min(ym)))
            if (k + 1) % 50 == 0:
                log(run, f"  D: {k + 1}/{len(tasks)} batches; failures {nfail}; {time.time() - t0:.0f} s")
    if any(nfail.values()):
        raise Fail(f"D: boxes not proved: {nfail} of {nbox}")
    nbox["full"] += nreuse["full"]
    nbox["light"] += nreuse["light"]
    log(run, f"D PASS: (R) full mode {nbox['full']} leaves (closed tube + Sturm), record-only {nbox['light']} "
             f"(closed tube), slab {nbox['slab']} boxes; min Sturm y {ymin:.3e}; saved data in {saves} "
             f"({time.time() - t0:.0f} s)")


def leftovers(cell, rects):
    """parts of the rectangle cell not covered by the union of rects (exact coordinate compression)"""
    pa, pb, ka, kb = cell
    r = rects[(rects[:, 0] < pb) & (rects[:, 1] > pa) & (rects[:, 2] < kb) & (rects[:, 3] > ka)]
    xs = np.unique(np.clip(np.concatenate([[pa, pb], r[:, 0], r[:, 1]]), pa, pb))
    ys = np.unique(np.clip(np.concatenate([[ka, kb], r[:, 2], r[:, 3]]), ka, kb))
    out = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            if x1 > x0 and y1 > y0 and not (len(r) and np.any((r[:, 0] <= x0) & (r[:, 1] >= x1) & (r[:, 2] <= y0)
                                                                & (r[:, 3] >= y1))):
                out.append([x0, x1, y0, y1])
    return out


def stage_E(run, cert, hdr):
    """the universe (built here from the rigorous bounds of Q on tube A), covered by the (R) leaves up to parts that
    are certainly empty of Q; the slab universe covered by the slab boxes up to parts certainly outside Q"""
    import multiprocessing as mp
    import eight_R_shadow as RS
    import eight_R_cover2 as RC
    t0 = time.time()
    tb = RC.tube()
    plo, phi_, mlo, mhi, emax = RS.q_bounds(tb)
    kmax = float(up(max(abs(mlo), abs(mhi)) / float(dn(math.sqrt(EPS_MIN)))))
    pg = RS.grid(plo, phi_, 2 * H)
    kg = np.linspace(-kmax, kmax, 65)
    kg[0], kg[-1] = -kmax, kmax
    universe = np.array([[a, b, c, d] for a, b in zip(pg[:-1], pg[1:]) for c, d in zip(kg[:-1], kg[1:])])
    rects = np.asarray(cert["R_cells"], float)
    left = []
    for c in universe:
        left += leftovers(c, rects)
    left = np.array(left) if left else np.zeros((0, 4))
    with mp.Pool(WORKERS, _w_init) as pool:
        parts = [left[i:i + 40] for i in range(0, len(left), 40)]
        tops = np.concatenate(pool.map(_w_top, parts)) if parts else np.zeros(0)
    bad = int(np.sum(np.isfinite(tops)))
    if bad:
        raise Fail(f"E: {bad} of {len(left)} uncovered parts of the universe are not certainly empty of Q")
    log(run, f"E1 PASS: universe phi [{plo:.4f}, {phi_:.4f}] x kappa [-{kmax:.3f}, {kmax:.3f}] ({len(universe)} grid "
             f"cells) covered by {len(rects)} leaves up to {len(left)} parts certainly empty of Q ({time.time() - t0:.0f} s)")
    # slab: the (phi, M) universe at eps in [0, E] covered by the slab boxes up to parts certainly outside Q
    S = np.asarray(cert["slab_boxes"], float)
    E = float(hdr["slab_E"])
    su = np.array([[plo, phi_, mlo, mhi]])
    sleft = []
    for c in su:
        sleft += leftovers(c, S[:, 2:6])
    sleft = np.array(sleft) if sleft else np.zeros((0, 4))
    if len(sleft):
        Lb = np.stack([np.full(len(sleft), math.log(1e-12)), np.full(len(sleft), math.log(E)),
                       sleft[:, 0], sleft[:, 1], sleft[:, 2], sleft[:, 3]], 1)
        o = np.concatenate([RC.outside(Lb[i:i + 400], tb) for i in range(0, len(Lb), 400)])
        if not np.all(o):
            raise Fail(f"E: {int(np.sum(~o))} of {len(sleft)} uncovered slab parts are not certainly outside Q")
    log(run, f"E2 PASS: slab eps in [0, {E}] universe covered by {len(S)} slab boxes up to {len(sleft)} parts certainly "
             f"outside Q ({time.time() - t0:.0f} s)")


def main():
    cert_path, run = sys.argv[1], sys.argv[2]
    stages = sys.argv[3] if len(sys.argv) > 3 else "ABCDE"
    os.makedirs(run, exist_ok=True)
    cert = dict(np.load(cert_path, allow_pickle=False))
    hdr = json.loads(str(cert["header"]))
    if len(sys.argv) > 4:                       # smoke test on a spread subset (stage E then cannot pass); 2 workers
        global WORKERS
        WORKERS = 2
        n = int(sys.argv[4])
        for c_, t_ in (("R_cells", "R_tops"), ("rec_cells", "rec_tops")):
            ix = np.linspace(0, len(cert[t_]) - 1, n).astype(int)
            cert[c_], cert[t_] = cert[c_][ix], cert[t_][ix]
        ix = np.linspace(0, len(cert["slab_boxes"]) - 1, max(4, n // 8)).astype(int)
        cert["slab_boxes"] = cert["slab_boxes"][ix]
    # the leaf list for stage F: (phi_a, phi_b, kappa_a, kappa_b, eps_top, type), type 0 = (R) leaf, 1 = record-only
    lv = np.concatenate([np.column_stack([cert["R_cells"], cert["R_tops"], np.zeros(len(cert["R_tops"]))]),
                         np.column_stack([cert["rec_cells"], cert["rec_tops"], np.ones(len(cert["rec_tops"]))])])
    np.save(os.path.join(run, "leaves.npy"), lv)
    log(run, f"checker {md5(os.path.abspath(__file__))}; certificate {cert_path} md5 {md5(cert_path)}; python "
             f"{sys.version.split()[0]} numpy {np.__version__} {platform.platform()}; settings EPS_MIN {EPS_MIN} NSLAB "
             f"{NSLAB} NSL {NSL} LORDER {LORDER} BATCH {BATCH} H {H}")
    t0 = time.time()
    try:
        if "A" in stages:
            stage_A(run, cert, hdr)
        if "B" in stages:
            stage_B(run, hdr)
        if "D" in stages:
            stage_D(run, cert, hdr)
        if "E" in stages:
            stage_E(run, cert, hdr)
        if "C" in stages:
            stage_C(run, hdr)
    except Fail as e:
        log(run, f"CHECK FAIL: {e}")
        sys.exit(1)
    except Exception as e:                      # any error fails the run
        log(run, f"CHECK FAIL (exception): {e!r}")
        raise
    log(run, f"STAGES {stages} PASS ({time.time() - t0:.0f} s); stage F: verify/eight_check_records.py on {run}")


if __name__ == "__main__":
    main()
