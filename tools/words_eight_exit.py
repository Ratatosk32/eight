"""Stage 21, exit-time test (theorist's plan, step 1; float).

Position tube T(t) around the eight: ball |y - x8(t)| < 2|x8(t)| d (mode ball) or an ellipsoid with semi-axes
(0.30, 0.13, 0.10) along the eigenvectors (soft first) of the slice Hessian at the nearest node (mode ellip).
(a) Focal map: the E-field (t in (0, 1]) and the I-field (t in [0, 1)) are inverted on rays from x8(t) to the tube
    boundary (continuation in the radius); along every ray sigma_min and det of the relative Jacobian
    J(t, a) J(t, a8)^(-1) (scale-free near the ends) are recorded. A focal point inside the tube shows as det -> 0 or a
    sign change, or as a failed inversion.
(b) Exit margins on the tube boundary:
      m_i(t)  = min ( V_E^f + S_I^c ) - A8   (first exit at t <= t_E; t = 0: points of E, V_E^f = 0)
      m_ii(t) = min ( S_E^c + V_I^f ) - A8   (last exit at t >= t_I; t = 1: points of I, V_I^f = 0)
    S^c: the coarse pieces (E: the verified nu = 4 piece, cert_pieces_E.json; I: the unrotated nu = 4 piece).
    python tools/words_eight_exit.py ball d1,d2,...  [ndir]
    python tools/words_eight_exit.py ellip a1,a2,a3  [ndir]
"""
import sys
import time
import numpy as np
import words_eight_patch as wp
from words_eight_field import Field, jac, fib, A8, startE, startI, flow, shootE, shootI, EI, EZ, NI
from words_eight_shape import eight_discrete, f_EI, pack_EI

OUT = "D:/nature-math/math-gaps/words/out/eight/"
TIMES = np.array([0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                  0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99])


def invert(F, y, lab0, t, iters=15):
    lab = lab0.copy()
    done = np.zeros(len(y), bool)
    ok = np.zeros(len(y), bool)
    for _ in range(iters):
        act = np.flatnonzero(~done)
        if not len(act):
            break
        Y, J = jac(F.shoot, lab[act], t)
        r = Y[:, 0:3] - y[act]
        sc = np.linalg.norm(y[act], axis=1)
        conv = np.max(np.abs(r), axis=1) < 1e-11 * np.maximum(sc, 1)
        bad = ~np.isfinite(r).all(1) | ~np.isfinite(J).all((1, 2))
        ok[act[conv]] = True
        done[act[conv | bad]] = True
        go = ~conv & ~bad
        if go.any():
            try:
                st = np.linalg.solve(J[go], r[go][:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                done[act[go]] = True
                continue
            lab[act[go]] -= st
    Y, J = jac(F.shoot, lab, t)
    ok &= np.isfinite(Y).all(1)
    return lab, Y, J, ok


def eight_track(F, t):
    Y, J = jac(F.shoot, F.lab8[None, :], t)
    return Y[0], J[0]


def frames_hessian():
    """eigenvectors of the slice Hessian (Schur complement) at every node of the N = 48 discrete eight"""
    X8 = eight_discrete(48)[0]
    N = 48
    z = pack_EI(X8)
    n = z.size
    H = np.zeros((n, n))
    for i in range(n):
        zp = z.copy(); zp[i] += 1e-6
        zm = z.copy(); zm[i] -= 1e-6
        H[:, i] = (f_EI(zp, N)[1] - f_EI(zm, N)[1]) / 2e-6
    Hi = np.linalg.inv((H + H.T) / 2)
    fr = {}
    for k in range(1, N):
        iy = np.arange(1 + 3 * (k - 1), 1 + 3 * k)
        HG = np.linalg.inv(Hi[np.ix_(iy, iy)])
        w, V = np.linalg.eigh(HG)
        fr[k] = V        # columns: soft first
    return fr


def tube_dirs(t, x8, mode, par, U, fr):
    """boundary offsets (relative: y = x8 + 2|x8| * offset) for unit directions U"""
    if mode == "ball":
        return par * U
    k = int(np.clip(round(t * 48), 1, 47))
    V = fr[k]
    a = np.array(par)
    return (U * a[None, :]) @ V.T


def run(mode, pars, ndir):
    d = np.load(OUT + "field_labels.npz")
    FE, FI = Field("E", d["a8"]), Field("I", d["b8"])
    mcE = wp.load_coarse("E", 4, 0.1)
    mcI = wp.load_coarse("I", 4, 0.1)
    # second E piece: the wedge Q(2,1,1) fit of the block agent (not finalised; nu4 fit outside the wedge)
    try:
        import words_eight_block2 as B2
        dw = np.load(OUT + "block/block2_E_wE_nu4p2.npz")
        meshW = B2.make_mesh("E", int(dw["nu"]), str(dw["ref"]), int(dw["p"]))
        SW = lambda y, t: B2.S_value(meshW, dw["coef"], "E", y, t)
    except Exception as e:
        print("wedge piece not available:", e)
        SW = None
    fr = frames_hessian() if mode == "ellip" else None
    U = fib(ndir)
    radii = np.linspace(0, 1, 9)[1:]          # continuation fractions of the boundary offset
    print(f"mode {mode}, parameters {pars}, {ndir} directions, {len(TIMES)} times; S_E^c = coarse_E_nu4_ds0.1_final "
          f"(= cert_pieces_E.json), S_I^c = coarse_I_nu4_ds0.1_final (unrotated)", flush=True)
    rows = []
    for par in pars:
        print(f"\n=== tube parameter {par} ===", flush=True)
        for t in TIMES:
            t0 = time.time()
            out = {}
            for nm, F in (("E", FE), ("I", FI)):
                Y8, J8 = eight_track(F, t)
                x8 = Y8[0:3]
                sc = 2 * np.linalg.norm(x8)
                J8i = np.linalg.inv(J8)
                off = tube_dirs(t, x8, mode, par, U, fr)
                lab = np.repeat(F.lab8[None, :], len(U), axis=0)
                smin = np.full(len(U), np.inf)
                dmin = np.full(len(U), np.inf)
                okall = np.ones(len(U), bool)
                for f in radii:
                    y = x8 + sc * f * off
                    guess = lab if f > radii[0] else F.lab8 + (y - x8) @ J8i.T
                    lab, Y, J, ok = invert(F, y, guess, t)
                    Jr = J @ J8i
                    sv = np.linalg.svd(Jr, compute_uv=False)[:, -1]
                    dt = np.linalg.det(Jr)
                    okall &= ok
                    smin = np.minimum(smin, np.where(ok, sv, 0))
                    dmin = np.minimum(dmin, np.where(ok, dt, -1))
                V = Y[:, 6]
                out[nm] = (y, V, okall, smin, dmin)
            # margins on the boundary (the boundary points of the two fields coincide: same x8(t), same offsets)
            yE, VE, okE, sE, dE = out["E"]
            yI, VI, okI, sI, dI = out["I"]
            ScE = wp.coarse_S(mcE, "E", yI, t)
            ScI = wp.coarse_S(mcI, "I", yE, 1 - t)
            mi = np.where(okE, VE + ScI - A8, np.nan)
            mii = np.where(okI, VI + ScE - A8, np.nan)
            mw = np.where(okI, VI + SW(yI, t) - A8, np.nan) if SW is not None else np.full(len(yI), np.nan)
            row = (par if np.isscalar(par) else 0, t, np.nanmin(mi) if okE.any() else np.nan,
                   np.nanmin(mii) if okI.any() else np.nan, sE.min(), dE.min(), okE.mean(), sI.min(), dI.min(), okI.mean(),
                   np.nanmin(mw) if okI.any() else np.nan,
                   np.nanmin(np.where(okE & okI, VE + VI - A8, np.nan)))
            rows.append(row)
            print(f"  t = {t:.3f}: m_i (V_E^f + S_I^c - A8) min {row[2]:+.4e} | m_ii (S_E^c + V_I^f - A8) min {row[3]:+.4e} "
                  f"(wedge Q211 E: {row[10]:+.4e}) | tube gap (V_E^f + V_I^f - A8) min {row[11]:+.4e} | "
                  f"E-field: sigma_rel min {sE.min():.3f}, det_rel min {dE.min():+.3f}, inverted {okE.mean():.2f} | "
                  f"I-field: sigma_rel min {sI.min():.3f}, det_rel min {dI.min():+.3f}, inverted {okI.mean():.2f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
        # end points: t = 0 (points of E on the tube boundary, V_E^f = 0) and t = 1 (points of I)
        Y8, J8 = eight_track(FE, 1e-9)
        rho8 = FE.lab8[0]
        for dr in (-1, 1):
            p = par if np.isscalar(par) else max(par)
            y0 = np.array([[rho8 * (1 + 2 * p * dr), 0, 0]])
            v = wp.coarse_S(mcI, "I", y0, 1.0)[0]
            print(f"  t = 0, E-point rho = rho8 (1 {'+' if dr > 0 else '-'} 2 d): S_I^c - A8 = {v - A8:+.4e}", flush=True)
        xI = FI.lab8[0] * EI + FI.lab8[1] * EZ
        sc1 = 2 * np.linalg.norm(xI)
        th = np.linspace(0, 2 * np.pi, 73)[:-1]
        p = par if np.isscalar(par) else max(par)
        ring = xI[None, :] + sc1 * p * (np.cos(th)[:, None] * EI[None, :] + np.sin(th)[:, None] * EZ[None, :])
        v = wp.coarse_S(mcE, "E", ring, 1.0)
        vw = SW(ring, 1.0) if SW is not None else np.full(len(ring), np.nan)
        print(f"  t = 1, I-ring at d: S_E^c - A8 min = {np.min(v) - A8:+.4e} (wedge Q211 E: {np.min(vw) - A8:+.4e})", flush=True)
    np.savetxt(OUT + f"exit_{mode}.tsv", np.array(rows), delimiter="\t", fmt="%.6e",
               header="par\tt\tm_i\tm_ii\tE_sigma_rel\tE_det_rel\tE_ok\tI_sigma_rel\tI_det_rel\tI_ok\tm_ii_wedge\tgap")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ball":
        pars = [float(v) for v in sys.argv[2].split(",")]
    else:
        pars = [tuple(float(v) for v in sys.argv[2].split(","))]
    ndir = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    run(mode, pars, ndir)
