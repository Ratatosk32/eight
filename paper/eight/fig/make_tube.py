"""Data for the tube figure: for several times t = 1 - s, the image of the boundary of the label ellipsoid Lambda(s)
(tube A, tapered) under the field map f_s, as an outline on the shape sphere (longitude, latitude of x/|x|; convex hull
of the image points), and the eight's position at that time.  python paper/eight/fig/make_tube.py
Writes fig_tube_t<k>.dat (outline, closed) and fig_tube_eight.dat (the eight's points at those times)."""
import math
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "verify"))
import eight_field as EF  # noqa: E402

z = np.load(os.path.join(ROOT, "words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper_b070_t02_03.npz"))
SN, C, G = z["s_nodes"], z["centre"], z["G"]


def tube_at(s):
    k = min(int(np.searchsorted(SN, s, side="right")) - 1, len(SN) - 2)
    w = (s - SN[k]) / (SN[k + 1] - SN[k])
    return (1 - w) * C[k] + w * C[k + 1], (1 - w) * G[k] + w * G[k + 1]


def lonlat(x):
    n = x / np.linalg.norm(x)
    return math.degrees(math.atan2(n[1], n[0])), math.degrees(math.asin(max(-1.0, min(1.0, n[2]))))


rng = np.random.default_rng(1)
u = rng.normal(size=(500, 3))
u /= np.linalg.norm(u, axis=1)[:, None]
times = [0.05, 0.25, 0.5, 0.75, 0.95]
pts8 = []
for k, t in enumerate(times):
    s = 1.0 - t
    c, Gm = tube_at(s)
    w, V = np.linalg.eigh(Gm)
    A = V @ np.diag(w ** -0.5) @ V.T                    # boundary: a = c + A u, |u| = 1
    P = []
    for uu in u:
        x, _, _ = EF.physical(c + A @ uu, s)
        P.append(lonlat(x))
    P = np.array(P)
    h = ConvexHull(P)
    with open(os.path.join(HERE, f"fig_tube_t{k}.dat"), "w") as f:
        f.write("lon lat\n")
        for i in list(h.vertices) + [h.vertices[0]]:
            f.write(f"{P[i, 0]:.4f} {P[i, 1]:.4f}\n")
    a8 = np.array([z["lab8"][0], z["lab8"][1], s * z["lab8"][2] / 1.0])
    x8, _, _ = EF.physical(a8, s)
    pts8.append(lonlat(x8))
    print(f"t = {t}: outline {len(h.vertices)} points, lon {P[:, 0].min():.1f}..{P[:, 0].max():.1f}, "
          f"lat {P[:, 1].min():.1f}..{P[:, 1].max():.1f}; eight at {pts8[-1]}", flush=True)
with open(os.path.join(HERE, "fig_tube_eight.dat"), "w") as f:
    f.write("lon lat\n")
    for a, b in pts8:
        f.write(f"{a:.4f} {b:.4f}\n")
