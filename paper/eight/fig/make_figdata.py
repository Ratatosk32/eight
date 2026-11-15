"""Data for the figures of paper/eight/main.tex (pgfplots / TikZ read the files written here).
    python paper/eight/fig/make_figdata.py
fig_shape_*.dat : the eight's twelfth and its 11 images on the shape sphere (longitude, latitude in degrees)
fig_cover.tex   : TikZ fills of the (R) leaves of the certificate in (phi, kappa), coloured by log10 eps_top
fig_cellmin.dat : per cell of the subsolution: sigma of the cell centre and the float minimum of F (7^3 points)"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "verify"))


def shape():
    from scipy.integrate import solve_ivp
    import eight_field as EF
    a8, _ = EF.newton_f(np.array([math.log(math.hypot(3.55638, 3.03611)), math.atan2(3.03611, 3.55638), -0.30889]))
    ell, phi, M = a8
    eps = math.exp(ell) ** -1.5
    y0 = np.concatenate([math.cos(phi) * EF.EI_F + math.sin(phi) * EF.EZ_F, M * EF.NI_F, [0.0, 0.0]])
    sol = solve_ivp(EF.rhs_f(eps), (0, 1), y0, method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True)
    Y = sol.sol(np.linspace(0, 1, 241))[0:3].T          # tau = 0 on I ... tau = 1 on E
    n = Y / np.linalg.norm(Y, axis=1)[:, None]
    # the group: rotations by 120 deg about e3 (relabelling), reflection in the plane I (through e3 and e_I),
    # rotation by pi about the Euler axis e1; their products give the 12 pieces of the loop
    c, s = -0.5, math.sqrt(3) / 2
    R3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    nI = np.array([math.sqrt(3) / 2, -0.5, 0])
    MI = np.eye(3) - 2 * np.outer(nI, nI)
    RE = np.diag([1.0, -1.0, -1.0])
    G = []
    for k in range(3):
        Rk = np.linalg.matrix_power(R3, k)
        for A in (np.eye(3), MI):
            for B in (np.eye(3), RE):
                G.append(Rk @ A @ B)
    pieces = []
    for g in G:
        m = n @ g.T
        if not any(np.allclose(m, p) or np.allclose(m[::-1], p) for p in pieces):
            pieces.append(m)
    print("distinct pieces", len(pieces))

    def write(name, m):
        lon = np.degrees(np.arctan2(m[:, 1], m[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(m[:, 2], -1, 1)))
        with open(os.path.join(HERE, name), "w") as f:
            f.write("lon lat\n")
            prev = None
            for a, b in zip(lon, lat):
                if prev is not None and abs(a - prev) > 180:      # wrap: break the line
                    f.write("\n")
                f.write(f"{a:.4f} {b:.4f}\n")
                prev = a
    write("fig_shape_twelfth.dat", pieces[0])
    for i, m in enumerate(pieces[1:]):
        write(f"fig_shape_copy{i}.dat", m)
    return len(pieces) - 1


def cover():
    z = np.load(os.path.join(ROOT, "words/out/eight/certificate/eight_cert_v4.npz"))
    C, T = z["R_cells"], z["R_tops"]
    lt = np.log10(T)
    lo, hi = -3.0, math.log10(0.2)
    ramp = ["cde2fb", "9ec5f4", "6da7ec", "3987e5", "256abf", "184f95", "0d366b"]   # light -> dark
    with open(os.path.join(HERE, "fig_cover.tex"), "w") as f:
        for i, h in enumerate(ramp):
            f.write(f"\\definecolor{{cv{i}}}{{HTML}}{{{h.upper()}}}\n")
        for (pa, pb, ka, kb), v in zip(C, lt):
            k = int(np.clip((v - lo) / (hi - lo) * len(ramp), 0, len(ramp) - 1))
            f.write(f"\\fill[cv{k}] ({pa:.4f},{ka:.4f}) rectangle ({pb:.4f},{kb:.4f});\n")
    print("cover rects", len(C), "log10 eps_top range", lt.min(), lt.max())


def cellmin():
    import json
    import eight_cell_mpiv as M
    d = json.load(open(os.path.join(ROOT, "words/out/eight/block/cert_pieces_E_goalEfit.json"), encoding="utf-8"))
    mins = M.scan(d)
    with open(os.path.join(HERE, "fig_cellmin.dat"), "w") as f:
        f.write("sigma fmin\n")
        for c, m in zip(d["cells"], mins):
            f.write(f"{0.5 * (c['sigma'][0] + c['sigma'][1]):.4f} {m:.6e}\n")
    print("cells", len(mins), "min", mins.min())


if __name__ == "__main__":
    shape()
    cover()
    cellmin()
