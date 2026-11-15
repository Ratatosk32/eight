"""Rasterise the (R) leaves of the certificate in (phi, kappa) into fig/fig_cover.png (pure python: zlib PNG),
coloured by log10 eps_top on the sequential blue ramp; white where no leaf.  python fig/make_cover_png.py"""
import math, os, struct, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
X0, X1, Y0, Y1, NX, NY = 0.37, 1.32, -3.3, 1.55, 950, 970
ramp = ["cde2fb", "9ec5f4", "6da7ec", "3987e5", "256abf", "184f95", "0d366b"]
rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in ramp], np.uint8)
z = np.load(os.path.join(ROOT, "words/out/eight/certificate/eight_cert_v4.npz"))
C, T = z["R_cells"], np.log10(z["R_tops"])
lo, hi = -3.0, math.log10(0.2)
img = np.full((NY, NX, 3), 252, np.uint8)
for (pa, pb, ka, kb), v in zip(C, T):
    k = int(np.clip((v - lo) / (hi - lo) * len(ramp), 0, len(ramp) - 1))
    i0, i1 = int((pa - X0) / (X1 - X0) * NX), int(math.ceil((pb - X0) / (X1 - X0) * NX))
    j0, j1 = int((Y1 - kb) / (Y1 - Y0) * NY), int(math.ceil((Y1 - ka) / (Y1 - Y0) * NY))
    img[max(j0, 0):j1, max(i0, 0):i1] = rgb[k]
raw = b"".join(b"\x00" + img[j].tobytes() for j in range(NY))
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", NX, NY, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
open(os.path.join(HERE, "fig_cover.png"), "wb").write(png)
print("written", NX, NY)
