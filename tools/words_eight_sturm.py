"""Stage 21, (R) by a scalar Sturm comparison (theorist, paper/review/eight_riccati_obstacle_theorist.md section 3), floats.

Scaled system: L = |Y'|^2/(8|Y|) + eps U = (1/2)|Y'|_g^2 + eps U, g = e^(2f) |dY|^2, f = -(1/2) log(4|Y|): a cone over
the round sphere of radius 1/2 (curvature 4), so for tangential parts (orthogonal to Y)
    Rm(J, v) v = (3/|Y|) (<v_T, v_T>_g J_T - <J_T, v_T>_g v_T),
and Hess_g U(h, k) = h.U''k - (grad f.h)(k.grad U) - (grad f.k)(h.grad U) + (h.k)(grad f.grad U) (Euclidean dots).
Jacobi equation: nabla_t nabla_t J = -Rm(J, v) v + eps Hess_g U(J)^sharp, nabla_t J = J' + Gamma(v, J),
Gamma(a, b) = a (grad f.b) + b (grad f.a) - (a.b) grad f.
Unit test: the label Jacobi fields X (columns) of tools/words_eight_riccati.py satisfy this equation (finite
differences in tau). Then the Sturm test: w(tau) = largest eigenvalue of Rop = Rm(., v) v - eps Hess_g U relative to g,
y'' + w y = 0, y(0) = 1, y'(0) = 0; y > 0 on [0, 1] implies no focal point of I, i.e. X(1) invertible.
    python tools/words_eight_sturm.py [npoints]"""
import math
import sys

import numpy as np

import words_eight_riccati as W


def gradU_hessU(Y):
    R = np.linalg.norm(Y)
    Yh = Y / R
    q = R - W.UK @ Y
    g = Yh[None, :] - W.UK
    gU = (-0.5 * q ** -1.5) @ g
    d2q = (np.eye(3) - np.outer(Yh, Yh)) / R
    HU = sum(0.75 * q[k] ** -2.5 * np.outer(g[k], g[k]) - 0.5 * q[k] ** -1.5 * d2q for k in range(3))
    return gU, HU


def gradf(Y):
    return -Y / (2 * (Y @ Y))


def Gamma(Y, a, b):
    gf = gradf(Y)
    return a * (gf @ b) + b * (gf @ a) - (a @ b) * gf


def hessg_matrix(Y, eps):
    """matrix of eps Hess_g U as a bilinear form in Euclidean coordinates"""
    gU, HU = gradU_hessU(Y)
    gf = gradf(Y)
    return eps * (HU - np.outer(gf, gU) - np.outer(gU, gf) + (gf @ gU) * np.eye(3))


def rop_matrix(Y, v, eps):
    """the bilinear form <Rop h, h>_g = <Rm(h, v) v, h>_g - eps Hess_g U(h, h), Euclidean coordinates"""
    R = np.linalg.norm(Y)
    c = 1.0 / (4 * R)                       # g = c |.|^2
    Yh = Y / R
    PT = np.eye(3) - np.outer(Yh, Yh)
    vT = PT @ v
    curv = (3 / R) * c * c * ((vT @ vT) * PT - np.outer(vT, vT))
    return curv - hessg_matrix(Y, eps)


def w_of(Y, P, eps):
    """largest eigenvalue of Rop relative to g: max_h q(h) / (c |h|^2)"""
    v = -4 * np.linalg.norm(Y) * P
    Q = rop_matrix(Y, v, eps)
    c = 1.0 / (4 * np.linalg.norm(Y))
    return float(np.max(np.linalg.eigvalsh(0.5 * (Q + Q.T)))) / c


def jacobi_residual(eps, phi, M):
    tr = W.trajectory(eps, phi, M)
    h = 1.0 / W.NT
    taus = np.arange(W.NT + 1) * h
    Ys, Ps = tr[:, 0:3], tr[:, 3:6]
    vs = np.array([-4 * np.linalg.norm(Y) * P for Y, P in zip(Ys, Ps)])
    worst, scale = 0.0, 0.0
    for col in range(3):
        J = tr[:, 6:15].reshape(-1, 3, 3)[:, :, col]
        dJ = np.gradient(J, h, axis=0, edge_order=2)
        Wv = np.array([dJ[k] + Gamma(Ys[k], vs[k], J[k]) for k in range(len(taus))])
        dW = np.gradient(Wv, h, axis=0, edge_order=2)
        lhs = np.array([dW[k] + Gamma(Ys[k], vs[k], Wv[k]) for k in range(len(taus))])
        rhs = []
        for k in range(len(taus)):
            Y, v = Ys[k], vs[k]
            R = np.linalg.norm(Y)
            c = 1.0 / (4 * R)
            Yh = Y / R
            PT = np.eye(3) - np.outer(Yh, Yh)
            vT, JT = PT @ v, PT @ J[k]
            Rm = (3 / R) * (c * (vT @ vT) * JT - c * (JT @ vT) * vT)
            Hs = hessg_matrix(Y, eps) @ J[k] / c          # raise the index: g^(-1) = 1/c
            rhs.append(-Rm + Hs)
        rhs = np.array(rhs)
        sl = slice(5, -5)
        worst = max(worst, np.max(np.abs(lhs[sl] - rhs[sl])))
        scale = max(scale, np.max(np.abs(rhs[sl])))
    return worst, scale


def sturm(eps, phi, M):
    tr = W.trajectory(eps, phi, M)
    h = 1.0 / W.NT
    ws = np.array([w_of(tr[k, 0:3], tr[k, 3:6], eps) for k in range(W.NT + 1)])
    y, yp, ymin = 1.0, 0.0, 1.0
    for k in range(W.NT):                    # RK4 with w linear on the step
        wa, wb = ws[k], ws[k + 1]
        wm = 0.5 * (wa + wb)
        f = lambda Y_, w_: np.array([Y_[1], -w_ * Y_[0]])
        Y = np.array([y, yp])
        k1 = f(Y, wa); k2 = f(Y + h / 2 * k1, wm); k3 = f(Y + h / 2 * k2, wm); k4 = f(Y + h * k3, wb)
        Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y, yp = Y
        ymin = min(ymin, y)
    return ymin, y, ws.min(), ws.max()


def main():
    npts = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ell8, phi8, mu8 = W.A8L
    e8 = math.exp(-1.5 * ell8)
    res, sc = jacobi_residual(e8, phi8, mu8)
    print(f"unit test (Jacobi equation for the label fields along the eight): max residual {res:.2e} "
          f"(scale of the terms {sc:.2e})")
    ymin, y1, wl, wh = sturm(e8, phi8, mu8)
    print(f"Sturm along the eight: w in [{wl:.3f}, {wh:.3f}], y(1) = {y1:.3f}, min y {ymin:.3f}")
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(npts):
        s = rng.uniform(0.02, 1.0)
        ell = ell8 + rng.uniform(-0.3, 0.3)
        phi = phi8 + rng.uniform(-0.3, 0.3)
        M = s * mu8 + rng.uniform(-0.075, 0.075)
        eps = s * s * math.exp(-1.5 * ell)
        ymin, y1, wl, wh = sturm(eps, phi, M)
        rows.append((ymin, s, ell - ell8, phi - phi8, M - s * mu8, wh))
    rows.sort()
    npass = sum(r[0] > 0 for r in rows)
    print(f"{npts} points of the +-(0.3, 0.3, 0.075) label box: Sturm passes at {npass}; worst:")
    for r in rows[:6]:
        print(f"   min y {r[0]:+.3f} at s {r[1]:.2f} dl {r[2]:+.2f} dphi {r[3]:+.2f} dM {r[4]:+.3f} (max w {r[5]:.2f})")


if __name__ == "__main__":
    main()
