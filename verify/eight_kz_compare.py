"""The eight x8 of this paper and the orbit of Kapela-Zgliczynski (Nonlinearity 16 (2003), Lemma 3.5, Table 1).

KZ (masses 1, G = 1, as here) start the twelfth at the Euler configuration q1 = (1, 0), q2 = (-1, 0), q3 = (0, 0) with
velocities q1' = q2' = (v, u), q3' = -2 (v, u), and prove that their map Phi has a unique zero in
X = x0 +- 1e-6, x0 = (0.347116768716, 0.532724944657), enclosed in
N(x0, X) = [0.347116886243943, 0.347116889993313] x [0.532724941587373, 0.532724949187495].

Our eight at its Euler end (time to go s = 1): shape point x = rho e1 (rho = |x| = r Y_1(1)), shape momentum
p = (0, p2, p3) = P(1). With z1 = (q2 - q1)/sqrt2, z2 = sqrt(2/3) q3 (centre of mass 0, q3 = 0 at the Euler
configuration), x = (|z1|^2 - |z2|^2, 2 z1 conj z2), L = 0 and p1 = 0 give z1' = 0 and
z2' = (x2' - i x3')/(2 z1), x' = 4 rho p; so q1' = q2' = sqrt(2/3) sqrt(rho) (p2 - i p3) and |q1| = d = sqrt(rho/2).
Rescaling to |q1| = 1 (q -> q/d, t -> t d^(-3/2)) multiplies velocities by d^(1/2):
    (v, u) = sqrt(2/3) rho^(1/2) d^(1/2) (p2, -p3)     (the sign of u is the mirror, a symmetry of both problems).
This script encloses (v, |u|) in interval arithmetic over the Krawczyk cube of half-width 1e-12 (stage B) and checks
the inclusion in KZ's N(x0, X).
    python verify/eight_kz_compare.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import eight_field as EF  # noqa: E402
from prove_newton_1212 import I  # noqa: E402

KZ_N = ((0.347116886243943, 0.347116889993313), (0.532724941587373, 0.532724949187495))
KZ_X = ((0.347116768716 - 1e-6, 0.347116768716 + 1e-6), (0.532724944657 - 1e-6, 0.532724944657 + 1e-6))


def main():
    a8 = EF.floats()
    inside, X, K, yT, minq = EF.krawczyk(a8, [1e-12] * 3)
    assert inside, "Krawczyk cube not proved"
    r = EF.iexp(X[0])
    rho = r * yT[0][0]
    p2, p3 = yT[4][0], yT[5][0]
    d = (rho * 0.5).sqrt()
    c = (I(2.0) / I(3.0)).sqrt() * rho.sqrt() * d.sqrt()
    v, u = c * p2, c * p3                      # |u| = c |p3| (p3 > 0 here)
    print(f"Krawczyk cube 1e-12 proved: {inside}; rho in [{float(rho.lo)!r}, {float(rho.hi)!r}]")
    print(f"(v, |u|) in [{float(v.lo)!r}, {float(v.hi)!r}] x [{float(u.lo)!r}, {float(u.hi)!r}]")
    inN = KZ_N[0][0] <= float(v.lo) and float(v.hi) <= KZ_N[0][1] and KZ_N[1][0] <= float(u.lo) and float(u.hi) <= KZ_N[1][1]
    inX = KZ_X[0][0] <= float(v.lo) and float(v.hi) <= KZ_X[0][1] and KZ_X[1][0] <= float(u.lo) and float(u.hi) <= KZ_X[1][1]
    print(f"inside KZ's box X = x0 +- 1e-6: {inX}; inside their enclosure N(x0, X): {inN}")
    print("KZ COMPARISON PASS" if (inN and inX) else "KZ COMPARISON FAIL")


if __name__ == "__main__":
    main()
