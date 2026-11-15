"""Stage 21, week-1 gate, step 3, task 2 (float): the slice action gap eta(t*, d) of the 1/12 problem in shape space.

G(y) = V_E(t*, y) + V_I(t*, y) = least action of E -> I paths (time tau = 1) through y at time t*.
eta(t*, d) = min { G(y) - A8 : dist(y, c) >= d and dist(y, m) >= d }, c = eight(t*), m = its mirror (w3 -> -w3),
dist(y, c) = |y - c| / (2|c|) (the metric distance frozen at c, relative to the size r = sqrt|c|).
Computed on the discrete problem (N nodes), A8 its discrete minimum, so the discretisation error cancels to O(h^2 eta).
The minimum over {dist >= d} is taken on the sphere dist = d (no other local minimum of G exists: multistart).
Small-d curvature: the Schur complement of the discrete Hessian at the eight on the slice node, eta ~ C d^2.
    python tools/words_eight_gap.py N k1,k2,...  [nstart]
"""
import sys
import numpy as np
from words_eight_shape import (action_nodes, eight_discrete, refine, pack_EI, unpack_EI, f_EI, solve, EI, EZ,
                               latlon, min_rij, potential_only)

OUT = "D:/nature-math/math-gaps/words/out/eight/"
DS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def make_f(N, k, c, m, d, P=1e6):
    sc = 2 * np.linalg.norm(c)

    def unpack(z):
        X = np.empty((N + 1, 3))
        X[0] = (z[0], 0, 0)
        inner = z[1:1 + 3 * (N - 2)].reshape(N - 2, 3)
        X[1:k] = inner[:k - 1]
        X[k + 1:N] = inner[k - 1:]
        X[N] = z[-5] * EI + z[-4] * EZ
        u = z[-3:]
        nu = np.linalg.norm(u)
        X[k] = c + sc * d * u / nu
        return X, u, nu

    def f(z):
        X, u, nu = unpack(z)
        A, G = action_nodes(X)
        gk = G[k]
        # dX_k/du = sc d (I - u u^T / nu^2) / nu
        gu = sc * d * (gk - u * (u @ gk) / nu ** 2) / nu
        # second constraint: dist(X_k, m) >= d
        dm = np.linalg.norm(X[k] - m) / sc
        if dm < d:
            A += P * (d - dm) ** 2
            gx = -2 * P * (d - dm) * (X[k] - m) / (sc * max(dm, 1e-300) * sc)
            gu += sc * d * (gx - u * (u @ gx) / nu ** 2) / nu
        inner = np.concatenate([G[1:k], G[k + 1:N]]).ravel()
        return A, np.concatenate([[G[0, 0]], inner, [G[N] @ EI, G[N] @ EZ], gu])

    def pack(X, u):
        inner = np.concatenate([X[1:k], X[k + 1:N]]).ravel()
        return np.concatenate([[X[0, 0]], inner, [X[N] @ EI, X[N] @ EZ], u])

    return f, unpack, pack


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def schur(X8, N, k):
    """Hessian of G at c on the slice node, by central differences of the analytic gradient"""
    z = pack_EI(X8)
    n = z.size
    H = np.zeros((n, n))
    e = 1e-6
    for i in range(n):
        zp = z.copy()
        zp[i] += e
        zm = z.copy()
        zm[i] -= e
        H[:, i] = (f_EI(zp, N)[1] - f_EI(zm, N)[1]) / (2 * e)
    H = (H + H.T) / 2
    ev = np.linalg.eigvalsh(H)
    iy = np.arange(1 + 3 * (k - 1), 1 + 3 * k)
    ir = np.setdiff1d(np.arange(n), iy)
    HG = H[np.ix_(iy, iy)] - H[np.ix_(iy, ir)] @ np.linalg.solve(H[np.ix_(ir, ir)], H[np.ix_(ir, iy)])
    return HG, ev


def frame(c):
    """unit vectors at c: radial, east (longitude), north (latitude)"""
    er = c / np.linalg.norm(c)
    ee = np.cross(EZ, er)
    ee /= np.linalg.norm(ee)
    en = np.cross(er, ee)
    return er, ee, en


def main():
    N = int(sys.argv[1])
    ks = [int(v) for v in sys.argv[2].split(",")]
    nst = int(sys.argv[3]) if len(sys.argv) > 3 else 26
    X8 = None
    for M in (48, 96, 192):
        if M > N:
            break
        X8, A8, o = eight_discrete(M, 1.0, None if X8 is None else pack_EI(refine(X8, M)))
    print(f"N = {N}: discrete A8 = {A8:.12f}", flush=True)
    rows = []
    for k in ks:
        ts = k / N
        c = X8[k].copy()
        m = c * np.array([1, 1, -1])
        sc = 2 * np.linalg.norm(c)
        la, lo, nn = latlon(c)
        HG, ev = schur(X8, N, k)
        # in frozen-metric units y = c + sc d u, |u| = 1: G - A8 ~ (1/2) sc^2 u^T HG u d^2
        w, V = np.linalg.eigh(HG * sc ** 2 / 2)
        er, ee, en = frame(c)
        print(f"\n== slice t* = {ts:.4f} (k = {k}): c = {np.round(c, 5)}, lat {la:.3f} lon {lo:.3f} |c| {nn:.5f}; "
              f"dist(c, mirror) = {np.linalg.norm(c - m) / sc:.4f}", flush=True)
        print(f"   Hessian of the discrete problem: smallest eigenvalues {np.round(ev[:3], 6)} (x N: {np.round(ev[:3] * N, 4)})")
        print(f"   eta ~ C d^2, C over directions: {np.round(w, 5)} (C/A8 {np.round(w / A8, 5)}); softest direction "
              f"(radial, east, north) = {np.round([V[:, 0] @ er, V[:, 0] @ ee, V[:, 0] @ en], 4)}", flush=True)
        dirs = fib_sphere(nst)
        dirs = np.vstack([dirs, V.T, -V.T])
        t = np.linspace(0, 1, N + 1)
        bump = np.where(t <= ts, t / ts, (1 - t) / (1 - ts))
        bump = np.sin(np.pi / 2 * bump) ** 2
        prev = None
        for d in DS:
            f, unpack, pack = make_f(N, k, c, m, d)
            best = None
            starts = [u for u in dirs]
            if prev is not None:
                starts = [prev[1]] + starts
            for j, u in enumerate(starts):
                u = u / np.linalg.norm(u)
                if j == 0 and prev is not None:
                    X0 = prev[0].copy()
                else:
                    y = c + sc * d * u
                    X0 = X8 + bump[:, None] * (y - c)[None, :]
                z0 = pack(X0, u)
                o = solve(f, z0, gtol=1e-12, maxiter=20000)
                if best is None or o.fun < best[0]:
                    best = (o.fun, o.x)
            Xb, ub, nub = unpack(best[1])
            ub = ub / nub
            Ab = best[0]
            dm = np.linalg.norm(Xb[k] - m) / sc
            yla, ylo, yn = latlon(Xb[k])
            ela, elo, en_ = latlon(Xb[N])
            eta = Ab - A8
            quad = (ub @ (HG * sc ** 2 / 2) @ ub) * d ** 2
            print(f"  d = {d:7.4f}: eta = {eta:.6e}  eta/A8 = {eta / A8:.4e}  eta/d^2 = {eta / d ** 2:.5f} (quadratic model "
                  f"{quad / d ** 2:.5f})  dir (r, e, n) = {np.round([ub @ er, ub @ ee, ub @ en], 3)}  y lat {yla:+.2f} "
                  f"lon {ylo:+.2f} |y|/|c| {yn / nn:.3f}  dist to mirror {dm:.3f}  min r_ij {min_rij(Xb):.4f}  "
                  f"start |x0| {Xb[0, 0]:.3f}  end lat {ela:+.2f}", flush=True)
            rows.append((ts, d, eta, eta / A8, ub @ er, ub @ ee, ub @ en, yla, ylo, yn / nn, min_rij(Xb), dm))
            prev = (Xb, ub)
    np.savetxt(OUT + f"gap_N{N}_{'_'.join(map(str, ks))}.tsv", np.array(rows), delimiter="\t",
               header="tstar\td\teta\teta_over_A8\tdir_r\tdir_e\tdir_n\ty_lat\ty_lon\ty_size_ratio\tmin_rij\tdist_mirror",
               fmt="%.8e")


if __name__ == "__main__":
    main()
