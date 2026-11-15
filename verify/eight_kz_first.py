"""Along the eight's twelfth, the isosceles end is the FIRST time at which q1 is orthogonal to q1' (needed to identify
x8 with the zero of the map Phi of Kapela-Zgliczynski, whose Poincare section is the first such time).

Shape formula (checked numerically against positions): |q1|^2 = |x|/3 + x1/6 + x2/(2 sqrt3). Along the field extremal
of the eight, scaled time tau runs from the isosceles end (tau = 0) to the Euler end (tau = 1), physical time
t = 1 - tau (twelfth of duration 1, s = 1), x = r Y, dx/dt = -r dY/dtau = 4 r |Y| P. Hence
    q1 . q1' = (1/2) d|q1|^2/dt = 2 r H,   H(tau) = |Y| g(Y) . P,   g(Y) = Y/(3|Y|) + (1/6, 1/(2 sqrt3), 0).
H(0) = 0 exactly (Y(0) in I, P(0) = M n_I and g(Y0) . n_I = sqrt3/12 - sqrt3/12 = 0). Claim: H > 0 on (0, 1].
Check: the validated Taylor flow of verify/eight_field.py (the same integrator as the Krawczyk proof), over the
Krawczyk cube of half-width 1e-12 around the eight's label; on every step the a priori enclosure B of the state over
the whole step gives an enclosure of H; steps with H > 0 certainly are accepted; the steps before the first such
step must form an initial segment [0, tau0] on which H' > 0 certainly, with
    H' = (Y . Y'/|Y|) g . P + |Y| (Dg Y') . P + |Y| g . P',   Y' = -4|Y| P,   P' = 2|P|^2 Y/|Y| - eps grad U(Y),
    Dg = (Id - Y Y^T/|Y|^2)/(3|Y|),   grad U = sum_k -1/2 (|Y| - u_k . Y)^(-3/2) (Y/|Y| - u_k).
Then H(tau) = int_0^tau H' > 0 on (0, tau0] and H > 0 on [tau0, 1].
    python verify/eight_kz_first.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import eight_field as EF  # noqa: E402
from prove_newton_1212 import I, dn, up, peval, tpow  # noqa: E402

S3 = I(3.0).sqrt()
G0 = (I(1.0) / I(6.0), I(1.0) / (I(2.0) * S3), I(0.0))


def val(c):
    """value part (row 0) of a jet: the enclosure over the label box"""
    return I(c.lo[0], c.hi[0])


def H_and_dH(B, eps):
    Y = [val(B[i]) for i in range(3)]
    P = [val(B[3 + i]) for i in range(3)]
    R = (Y[0].sq() + Y[1].sq() + Y[2].sq()).sqrt()
    g = [Y[i] / (I(3.0) * R) + G0[i] for i in range(3)]
    gP = g[0] * P[0] + g[1] * P[1] + g[2] * P[2]
    H = R * gP
    Yd = [P[i] * R * (-4.0) for i in range(3)]
    YYd = Y[0] * Yd[0] + Y[1] * Yd[1] + Y[2] * Yd[2]
    # (Dg Y') . P = (Y' . P - (Y . Y')(Y . P)/|Y|^2) / (3|Y|)
    YdP = Yd[0] * P[0] + Yd[1] * P[1] + Yd[2] * P[2]
    YP = Y[0] * P[0] + Y[1] * P[1] + Y[2] * P[2]
    DgYdP = (YdP - YYd * YP / R.sq()) / (I(3.0) * R)
    P2 = P[0].sq() + P[1].sq() + P[2].sq()
    gradU = [I(0.0)] * 3
    for u in EF.UK:
        q = R - (Y[0] * u[0] + Y[1] * u[1] + Y[2] * u[2])
        f = q.sqrt() * q                                   # q^(3/2)
        for i in range(3):
            gradU[i] = gradU[i] - (Y[i] / R - u[i]) / (f * I(2.0))
    Pd = [P2 * Y[i] / R * I(2.0) - eps * gradU[i] for i in range(3)]
    gPd = g[0] * Pd[0] + g[1] * Pd[1] + g[2] * Pd[2]
    dH = YYd / R * gP + R * DgYdP + R * gPd
    return H, dH


def main():
    a8 = EF.floats()
    d = 1e-12
    X = [I(dn(a8[i] - d), up(a8[i] + d)) for i in range(3)]
    y, E = EF.start_jets(*X)
    eps = val(E)
    t_lo, t_hi, hmax = 0.0, 0.0, 0.02
    steps, phase, n_dH, tau0 = 0, "start", 0, None
    worst_H, worst_dH = np.inf, np.inf
    while True:
        rem_hi = float(up(1.0 - t_lo))
        last = rem_hi <= hmax
        hstep = rem_hi if last else hmax
        res = EF.step(y, E, hstep)
        if res is None:
            hmax /= 2
            if hmax < 1e-7:
                raise RuntimeError(f"step collapsed at tau {t_lo}")
            continue
        cs, Rm, B = res
        steps += 1
        H, dH = H_and_dH(B, eps)
        if float(H.lo) > 0:
            if phase == "start":
                phase, tau0 = "positive", t_lo
            worst_H = min(worst_H, float(H.lo))
        elif phase == "start" and float(dH.lo) > 0:
            n_dH += 1
            worst_dH = min(worst_dH, float(dH.lo))
        else:
            if hstep > 1e-6:                    # refine: redo this step with half the size
                steps -= 1
                hmax = hstep / 2
                continue
            print(f"FAIL at step {steps}, tau in [{t_lo:.6f}, {t_lo + hstep:.6f}]: H in [{float(H.lo):.3e}, "
                  f"{float(H.hi):.3e}], H' in [{float(dH.lo):.3e}, {float(dH.hi):.3e}], phase {phase}")
            print("KZ FIRST-TIME CHECK FAIL")
            return
        hmax = min(0.02, hstep * 1.5)
        if last:
            break
        tP = tpow(I(hstep), EF.PORDER + 1)
        y = [peval(cs[i], I(hstep)) + Rm[i] * tP for i in range(EF.NC)]
        t_lo, t_hi = float(dn(t_lo + hstep)), float(up(t_hi + hstep))
    print(f"{steps} steps over tau in [0, 1]; H' > 0 certainly on the first {n_dH} steps (tau in [0, {tau0:.4f}], "
          f"min lower bound of H' {worst_dH:.4e}); H > 0 certainly on all later steps (min lower bound {worst_H:.4e})")
    print("KZ FIRST-TIME CHECK PASS: the isosceles end is the first time with q1 orthogonal to q1'")


if __name__ == "__main__":
    main()
