"""Stage 21, step 3 (float test): the Riccati barrier for regularity (R) of the I-field, theorist's recipe
(paper/review/eight_field_c1_obstacle_theorist.md, section 3).

Fixed-eps scaled flow, H = 2|Y||P|^2 - eps U(Y): Y' = -H_P = -4|Y|P, P' = H_Y = 2|P|^2 Y/|Y| - eps grad U.
Jacobi fields (X, Xi) = d(Y, P)/da for the labels a = (ell, phi, M) at fixed s:
    X' = -(H_PY X + H_PP Xi),  Xi' = H_YY X + H_YP Xi,  (X, Xi)(0) = ((Y0, Y0_phi, 0), (0, 0, n_I)),
R = Xi X^(-1):  R' = H_YY + H_YP R + R H_PY + R H_PP R   (H_YP[i, j] = d^2 H / dY_i dP_j).
Barrier: B(tau0) = R_c(tau0) + d0 I, B' = RHS_c(B) + d1 I (centre coefficients). It is an upper barrier for the box if
B(tau0) >= R(tau0) and B' - RHS_z(B) >= 0 for every trajectory z of the box (Loewner order); then X(1) is invertible.
This float test samples the trajectories at the 8 corners and 6 face centres of each box and reports the smallest
(d0, d1) that work and the margin.
    python tools/words_eight_riccati.py nboxes d [seed]"""
import math
import sys

import numpy as np

S3 = math.sqrt(3.0)
UK = np.array([[-1.0, 0, 0], [0.5, S3 / 2, 0], [0.5, -S3 / 2, 0]])
EI, EZ, NI = np.array([0.5, S3 / 2, 0]), np.array([0, 0, 1.0]), np.array([S3 / 2, -0.5, 0])
A8L = np.array([1.5424624176685366, 0.7066425303908384, -0.30889005565526345])     # (ell, phi, mu) of the eight
NT = 400
TAU0 = 0.05


def derivs(Y, P, eps):
    R = np.linalg.norm(Y)
    Yh = Y / R
    q = R - UK @ Y
    g = Yh[None, :] - UK                                   # grad q_k
    gU = (-0.5 * q ** -1.5) @ g
    I3 = np.eye(3)
    d2q = (I3 - np.outer(Yh, Yh)) / R
    HU = sum(0.75 * q[k] ** -2.5 * np.outer(g[k], g[k]) - 0.5 * q[k] ** -1.5 * d2q for k in range(3))
    PP = P @ P
    HP = 4 * R * P
    HY = 2 * PP * Yh - eps * gU
    HPP = 4 * R * I3
    HYP = 4 * np.outer(Yh, P)
    HYY = 2 * PP * d2q - eps * HU
    return HP, HY, HPP, HYP, HYY


def rhs(state, eps):
    Y, P = state[0:3], state[3:6]
    X, Xi = state[6:15].reshape(3, 3), state[15:24].reshape(3, 3)
    HP, HY, HPP, HYP, HYY = derivs(Y, P, eps)
    dX = -(HYP.T @ X + HPP @ Xi)
    dXi = HYY @ X + HYP @ Xi
    return np.concatenate([-HP, HY, dX.ravel(), dXi.ravel()])


def trajectory(eps, phi, M):
    Y0 = math.cos(phi) * EI + math.sin(phi) * EZ
    Yphi = -math.sin(phi) * EI + math.cos(phi) * EZ
    X0 = np.stack([Y0, Yphi, np.zeros(3)], 1)
    Xi0 = np.stack([np.zeros(3), np.zeros(3), NI], 1)
    st = np.concatenate([Y0, M * NI, X0.ravel(), Xi0.ravel()])
    h = 1.0 / NT
    out = [st]
    for _ in range(NT):
        k1 = rhs(st, eps)
        k2 = rhs(st + h / 2 * k1, eps)
        k3 = rhs(st + h / 2 * k2, eps)
        k4 = rhs(st + h * k3, eps)
        st = st + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(st)
    return np.array(out)


def riccati_rhs(B, Y, P, eps):
    _, _, HPP, HYP, HYY = derivs(Y, P, eps)
    return HYY + HYP @ B + B @ HYP.T + B @ HPP @ B


def barrier(trc, epsc, d0, d1):
    """B on the grid tau_k = k / NT for k >= k0 (RK4 with the centre coefficients interpolated linearly)"""
    k0 = int(round(TAU0 * NT))
    Xc, Xic = trc[k0, 6:15].reshape(3, 3), trc[k0, 15:24].reshape(3, 3)
    B = Xic @ np.linalg.inv(Xc)
    B = 0.5 * (B + B.T) + d0 * np.eye(3)
    h = 1.0 / NT
    Bs = [B]
    for k in range(k0, NT):
        Ya, Pa = trc[k, 0:3], trc[k, 3:6]
        Yb, Pb = trc[k + 1, 0:3], trc[k + 1, 3:6]
        Ym, Pm = 0.5 * (Ya + Yb), 0.5 * (Pa + Pb)
        f = lambda B_, Y_, P_: riccati_rhs(B_, Y_, P_, epsc) + d1 * np.eye(3)
        k1 = f(B, Ya, Pa)
        k2 = f(B + h / 2 * k1, Ym, Pm)
        k3 = f(B + h / 2 * k2, Ym, Pm)
        k4 = f(B + h * k3, Yb, Pb)
        B = B + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        B = 0.5 * (B + B.T)
        if not np.all(np.isfinite(B)) or np.max(np.abs(B)) > 1e6:
            return None
        Bs.append(B)
    return np.array(Bs)


def check_box(pc, w, d0s=(0.003, 0.01, 0.03, 0.1), d1s=(0.003, 0.01, 0.03, 0.1, 0.3)):
    """pc = (eps, phi, M), w = half-widths; returns (best (d0, d1, margin) or None, info)"""
    trc = trajectory(*pc)
    samples = []
    for sgn in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T:
        samples.append(pc + sgn * w)
    for j in range(3):
        for sg in (-1, 1):
            e = np.zeros(3)
            e[j] = sg
            samples.append(pc + e * w)
    trs = [trajectory(*p) for p in samples]
    k0 = int(round(TAU0 * NT))
    Rs0 = []
    for tr in trs:
        X, Xi = tr[k0, 6:15].reshape(3, 3), tr[k0, 15:24].reshape(3, 3)
        R = Xi @ np.linalg.inv(X)
        Rs0.append(0.5 * (R + R.T))
    Rc0 = trc[k0, 15:24].reshape(3, 3) @ np.linalg.inv(trc[k0, 6:15].reshape(3, 3))
    Rc0 = 0.5 * (Rc0 + Rc0.T)
    var0 = max(np.max(np.linalg.eigvalsh(R - Rc0)) for R in Rs0)
    best = None
    for d1 in d1s:
        for d0 in d0s:
            if d0 < var0:
                continue
            Bs = barrier(trc, pc[0], d0, d1)
            if Bs is None:
                continue
            m0 = min(np.min(np.linalg.eigvalsh(Bs[0] - R)) for R in Rs0)
            m1 = np.inf
            for p, tr in zip(samples, trs):
                for i, k in enumerate(range(k0, NT + 1)):
                    D = riccati_rhs(Bs[i], trc[k, 0:3], trc[k, 3:6], pc[0]) + d1 * np.eye(3) \
                        - riccati_rhs(Bs[i], tr[k, 0:3], tr[k, 3:6], p[0])
                    m1 = min(m1, np.min(np.linalg.eigvalsh(0.5 * (D + D.T))))
            if m0 >= 0 and m1 >= 0:
                cand = (d0, d1, min(m0, m1), float(np.max(np.linalg.eigvalsh(Bs[-1]))))
                if best is None or d0 + d1 < best[0] + best[1]:
                    best = cand
                break
        if best is not None:
            break
    X1 = trc[-1, 6:15].reshape(3, 3)
    return best, {"var_R0": var0, "sv_X1": np.linalg.svd(X1, compute_uv=False)}


def main():
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    dd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    ell8, phi8, mu8 = A8L
    ok, rows = 0, []
    for b in range(nb):
        s = rng.uniform(0.05, 1.0)
        ell = ell8 + rng.uniform(-0.3, 0.3)
        phi = phi8 + rng.uniform(-0.3, 0.3)
        M = s * mu8 + rng.uniform(-0.075, 0.075)
        eps = s * s * math.exp(-1.5 * ell)
        pc = np.array([eps, phi, M])
        w = dd * np.array([0.1, 1.0, 0.25])
        best, info = check_box(pc, w)
        rows.append((s, ell - ell8, phi - phi8, M - s * mu8, best, info))
        ok += best is not None
        print(f"box {b:2d}: s {s:.3f} dl {ell - ell8:+.2f} dphi {phi - phi8:+.2f} dM {M - s * mu8:+.3f} eps {eps:.4f}: "
              f"var R0 {info['var_R0']:.3f}, sv X(1) {np.round(info['sv_X1'], 3)}; barrier "
              + ("FAILED" if best is None else f"d0 {best[0]} d1 {best[1]} margin {best[2]:.2e} max eig B(1) {best[3]:.2f}"),
              flush=True)
    print(f"d = {dd}: barrier closes on {ok} of {nb} boxes")


if __name__ == "__main__":
    main()
