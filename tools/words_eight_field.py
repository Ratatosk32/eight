"""Stage 21, certificate step: the fields of extremals around the eight in shape space (float).

Hamiltonian H = 2|x||p|^2 - U(x) (x in shape space, L = 0): x' = 4|x| p, p' = -2|p|^2 x/|x| + grad U, action
A' = 2|x||p|^2 + U.
E-field: extremals leaving the Euler ray orthogonally: x(0) = (rho, 0, 0), p(0) = (0, p2, p3); labels a = (rho, p2, p3).
I-field: extremals arriving on the plane I orthogonally, integrated backward from t = 1: x(1) = u EI + v EZ,
p(1) = mu n_I; labels b = (u, v, mu).
V_E(t, x) = action from 0 to t along the E-extremal through (t, x); V_I(t, x) = action from t to 1 along the I-extremal.
Penalty: psi_E = kE t |aT|^2, psi_I = kI (1 - t) |bT|^2, aT = LamE (a - a8), LamE = J_E(t*, a8) / (2|c|), so |aT| ~ d at
the slice (same for I). S = V - psi is a subsolution on the field domain iff D psi/Dt >= 2|x||grad psi|^2, implied by
    kE <= kmax_E = min over the domain of sigma_min(dx/daT)^2 / (8 |x| t^2)     (and the same with 1 - t for I).
This module: RK4 integrator (vectorised), the eight's labels, the slice maps, kmax over a label ball, field values on
the slice compared with the minimisation values.
    python tools/words_eight_field.py
"""
import sys
import numpy as np
from scipy.optimize import fsolve
from words_eight_shape import potential, EI, EZ, NI, eight_discrete

A8 = 2.5198560926
TS = 0.5


def rhs(Y):
    x, p = Y[:, 0:3], Y[:, 3:6]
    n = np.linalg.norm(x, axis=1)
    U, GU = potential(x)
    pp = np.sum(p * p, axis=1)
    dx = 4 * n[:, None] * p
    dp = -2 * (pp / n)[:, None] * x + GU
    dA = 2 * n * pp + U
    return np.concatenate([dx, dp, dA[:, None]], axis=1)


def flow(Y0, t0, t1, nsteps, keep=None):
    """RK4 from t0 to t1 (either direction); keep: list of step indices to store"""
    h = (t1 - t0) / nsteps
    Y = Y0.copy()
    out = {}
    if keep is not None and 0 in keep:
        out[0] = Y.copy()
    for k in range(1, nsteps + 1):
        k1 = rhs(Y)
        k2 = rhs(Y + h / 2 * k1)
        k3 = rhs(Y + h / 2 * k2)
        k4 = rhs(Y + h * k3)
        Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if keep is not None and k in keep:
            out[k] = Y.copy()
    return Y, out


def startE(a):
    a = np.atleast_2d(a)
    Y = np.zeros((a.shape[0], 7))
    Y[:, 0] = a[:, 0]
    Y[:, 4] = a[:, 1]
    Y[:, 5] = a[:, 2]
    return Y


def startI(b):
    b = np.atleast_2d(b)
    Y = np.zeros((b.shape[0], 7))
    Y[:, 0:3] = b[:, 0:1] * EI + b[:, 1:2] * EZ
    Y[:, 3:6] = b[:, 2:3] * NI
    return Y


NSTEP = 2000      # RK4 steps on [0, 1]


def shootE(a, t1, nst=None):
    nst = nst or max(int(NSTEP * t1), 50)
    return flow(startE(a), 0.0, t1, nst)[0]


def shootI(b, t0, nst=None):
    nst = nst or max(int(NSTEP * (1 - t0)), 50)
    Y = flow(startI(b), 1.0, t0, nst)[0]
    Y[:, 6] *= -1          # action from t0 to 1
    return Y


def eight_labels():
    X8 = eight_discrete(96)[0]
    h = 1 / 96
    v0 = (X8[1] - X8[0]) / h
    p0 = v0 / (4 * np.linalg.norm(X8[0]))
    a0 = np.array([X8[0, 0], p0[1], p0[2]])

    def res(a):
        Y = shootE(a, 1.0)[0]
        x, p = Y[0:3], Y[3:6]
        return [x @ NI, p @ EI, p @ EZ]
    a8 = fsolve(res, a0, xtol=1e-13)
    Y = shootE(a8, 1.0)[0]
    x1, p1 = Y[0:3], Y[3:6]
    b8 = np.array([x1 @ EI, x1 @ EZ, p1 @ NI])
    return a8, b8, Y[6]


def jac(shoot, lab, t, eps=1e-6):
    """x(t) and its Jacobian w.r.t. the labels, many labels at once (central differences)"""
    lab = np.atleast_2d(lab)
    m = lab.shape[0]
    L = np.vstack([lab] + [lab + eps * e for e in np.eye(3)] + [lab - eps * e for e in np.eye(3)])
    Y = shoot(L, t)
    x = Y[:m, 0:3]
    J = np.stack([(Y[(1 + k) * m:(2 + k) * m, 0:3] - Y[(4 + k) * m:(5 + k) * m, 0:3]) / (2 * eps) for k in range(3)], axis=2)
    return Y[:m], J


def invert(shoot, y, lab0, t, iters=30):
    """labels with x(t; lab) = y (Newton), vectorised"""
    lab = np.array(lab0, float).copy()
    for _ in range(iters):
        Y, J = jac(shoot, lab, t)
        r = Y[:, 0:3] - y
        if np.max(np.abs(r)) < 1e-11:
            break
        lab = lab - np.linalg.solve(J, r[:, :, None])[:, :, 0]
    return lab, Y


class Field:
    def __init__(self, kind, lab8, ts=TS):
        self.kind = kind
        self.ts = ts
        self.shoot = shootE if kind == "E" else shootI
        self.lab8 = lab8
        Y, J = jac(self.shoot, lab8, ts)
        self.c = Y[0, 0:3]
        self.V8 = Y[0, 6]
        self.Lam = J[0] / (2 * np.linalg.norm(self.c))       # aT = Lam (a - a8)
        self.Laminv = np.linalg.inv(self.Lam)

    def labels(self, aT):
        return self.lab8 + np.atleast_2d(aT) @ self.Laminv.T

    def tilde(self, lab):
        return (np.atleast_2d(lab) - self.lab8) @ self.Lam.T

    def kmax(self, aT, times):
        """min over the given labels and times of sigma_min(dx/daT)^2 / (8|x| tau^2), tau = t (E) or 1 - t (I)"""
        lab = self.labels(aT)
        best = np.inf
        rows = []
        for t in times:
            Y, J = jac(self.shoot, lab, t)
            Jt = J @ self.Laminv
            sv = np.linalg.svd(Jt, compute_uv=False)[:, -1]
            tau = t if self.kind == "E" else 1 - t
            k = sv ** 2 / (8 * np.linalg.norm(Y[:, 0:3], axis=1) * tau ** 2)
            rows.append((t, k.min(), sv.min()))
            best = min(best, k.min())
        return best, rows


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def main():
    a8, b8, A = eight_labels()
    print(f"eight's E-labels (rho, p2, p3) = {a8}, action E -> I along it = {A:.10f} (A8 = {A8})")
    print(f"eight's I-labels (u, v, mu) = {b8}")
    FE = Field("E", a8)
    FI = Field("I", b8)
    print(f"slice t* = {TS}: x_E(t*) = {FE.c}, x_I(t*) = {FI.c}, V_E = {FE.V8:.8f}, V_I = {FI.V8:.8f}, "
          f"sum = {FE.V8 + FI.V8:.10f}")
    # field values against the minimisation values on spheres around c (both fields, both directions)
    c = FE.c
    sc = 2 * np.linalg.norm(c)
    U = fib(40)
    for d in (0.05, 0.1, 0.2, 0.3):
        y = c + sc * d * U
        lE, YE = invert(FE.shoot, y, FE.labels(d * U), TS)
        lI, YI = invert(FI.shoot, y, FI.labels(d * U), TS)
        G = YE[:, 6] + YI[:, 6]
        print(f"  d = {d}: field G - A8 min {np.min(G - A8):.4e} max {np.max(G - A8):.4e}; |aT| in "
              f"[{np.min(np.linalg.norm(FE.tilde(lE), axis=1)):.3f}, {np.max(np.linalg.norm(FE.tilde(lE), axis=1)):.3f}], "
              f"|bT| in [{np.min(np.linalg.norm(FI.tilde(lI), axis=1)):.3f}, {np.max(np.linalg.norm(FI.tilde(lI), axis=1)):.3f}]",
              flush=True)
    # kmax over label balls
    times = np.concatenate([np.geomspace(1e-3, 0.05, 8), np.linspace(0.075, TS, 8)])
    for r in (0.1, 0.2, 0.3, 0.4, 0.5):
        aT = r * fib(60)
        kE, rowsE = FE.kmax(aT, times)
        kI, rowsI = FI.kmax(aT, 1 - times)
        print(f"  label radius {r}: kmax_E = {kE:.4f}, kmax_I = {kI:.4f}", flush=True)
        if r == 0.3:
            print("     E: t, kmin, smin:", [(round(a, 4), round(b, 4), round(s, 3)) for a, b, s in rowsE])
            print("     I: t, kmin, smin:", [(round(a, 4), round(b, 4), round(s, 3)) for a, b, s in rowsI])
    np.savez("D:/nature-math/math-gaps/words/out/eight/field_labels.npz", a8=a8, b8=b8, LamE=FE.Lam, LamI=FI.Lam)


if __name__ == "__main__":
    main()
