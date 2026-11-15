"""Stage 21, referee round 5c C3/C4: the proof artifacts behind K1_HALF and A8_HI, stored so that the (iii-L) code reads
them instead of hard-coding them.
 K1: Krawczyk (verify/eight_field.krawczyk via eight_iiiL.k1_box) on the cube of half-width K1_HALF around a8:
     proved (K inside X: a8 is the only zero of N = (Y_2, Y_3, P_1)(1; .) in X), and Y_1(1) > 0 on X (the zero is on
     the ray E, not on -E) is part of the verdict.
 A8: the action W = r (k + eps m)(1) enclosed by the C1 Lohner (eight_iiL_records.a8_upper) over the Krawczyk cube of
     half-width A8_HALF (default 1e-12; proved here too; it contains the zero): A8 <= act_hi; the records' A8_HI must be >= act_hi.
Writes words/out/eight/tubeb/k1_a8_artifact.npz (values, the md5 of the code used).
    python verify/eight_k1_artifact.py [K1 half-width]"""
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eight_iiiL as XL  # noqa: E402

OUT = os.path.join(ROOT, "words/out/eight/tubeb/k1_a8_artifact.npz")


def md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    k1 = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-5
    t0 = time.time()
    p1, lo1, hi1, act1, y1 = XL.k1_box(k1)
    print(f"K1 half-width {k1:g}: Krawczyk proved {p1}; Y_1(1) >= {y1:.6f}; action over K1 <= {act1!r} ({time.time() - t0:.0f} s)",
          flush=True)
    ha = float(os.environ.get("A8_HALF", "1e-12"))
    p2, lo2, hi2, actk, y2 = XL.k1_box(ha)
    # the action over the Krawczyk box of half-width ha by the C1 Lohner (as eight_iiL_records.a8_upper, which uses
    # 1e-6; the Taylor-model action enclosure of the Krawczyk run, actk, is much wider and is only reported)
    import eight_iiL_records as RR
    import eight_field_lohner as L
    L.ORDER = 12
    A = np.array([[lo2[0], lo2[1], lo2[2], hi2[0], hi2[1], hi2[2]]])
    o = RR.lohner_boxes((A, np.array([[1.0, 1.0]])))
    assert o["ok"][0]
    act2, act2_lo = float(o["W_hi"][0]), float(o["W_lo"][0])
    print(f"cube {ha:g}: Krawczyk proved {p2}; Y_1(1) >= {y2:.6f}; A8 in [{act2_lo!r}, {act2!r}] (C1 Lohner over "
          f"the Krawczyk box); Krawczyk-run action bound {actk!r} ({time.time() - t0:.0f} s)", flush=True)
    ok = bool(p1) and bool(p2) and y1 > 0 and y2 > 0
    codes = {os.path.basename(f): md5(f) for f in [os.path.join(HERE, x) for x in
                                                     ("eight_iiiL.py", "eight_iiL_records.py", "eight_field.py",
                                                      "eight_field_lohner.py", "prove_newton_1212.py")]}
    np.savez(OUT, k1_half=k1, k1_proved=bool(p1), k1_lo=lo1, k1_hi=hi1, k1_y1_lo=y1, k1_act_hi=act1,
             a8_half=ha, a8_proved=bool(p2), a8_act_hi=act2, a8_act_lo=act2_lo, a8_y1_lo=y2, a8_lab=XL.A8_LAB, ok=ok,
             codes=np.array([f"{k} {v}" for k, v in codes.items()]))
    print(f"artifact {'OK' if ok else 'NOT OK'} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
