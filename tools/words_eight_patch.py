"""Stage 21, certificate step: the max-type patch and the slice check (float).

S_E = max(V_E^field - psi_E, V_E^mirror-field - psi_E^mirror, S_E^coarse), the same for I, with
    psi_E(t, x) = kE t (|aT| - r0E)_+^2  on the E-field domain |aT| <= rE  (aT the normalised label, ~ d at t*),
    psi_I(t, x) = kI (1 - t) (|bT| - r0I)_+^2  on the I-field domain |bT| <= rI.
Conditions (all checked here on samples):
 (P1) V - psi is a subsolution on the field domain: kE <= sigma_min(dx/daT)^2 / (8 |x| t^2) wherever |aT| > r0E
      (and the field map is a diffeomorphism there: sigma_min > 0, no focal point), the same for I with 1 - t;
 (P2) the patch closes: V^field - psi < S^coarse on the lateral boundary |aT| = rE for every t in (0, t*]
      (and for I on |bT| = rI, t in [t*, 1)); then the max equals the coarse piece near the boundary;
 (P3) the slice: S_E + S_I >= A8 + eta_target at every slice point outside the tube (d >= dT around c and its mirror).
    python tools/words_eight_patch.py nu dsb r0E rE kE r0I rI kI eta_target
"""
import sys
import numpy as np
import words_eight_coarse as wc
from words_eight_field import Field, flow, startE, startI, invert, jac, fib, A8, TS
from words_eight_shape import eight_discrete

OUT = "D:/nature-math/math-gaps/words/out/eight/"


def load_coarse(kind, nu, dsb):
    import os
    f = OUT + f"coarse_{kind}_nu{nu}_ds{dsb}_final.npz"
    d = np.load(f if os.path.exists(f) else OUT + f"coarse_{kind}_nu{nu}_ds{dsb}.npz")
    return wc.Mesh(nu, dsb, rot=float(d["rot"]) if "rot" in d else 0.0), d["coef"]


def coarse_S(mc, kind, y, tau):
    mesh, coef = mc
    return wc.S_value(mesh, coef, kind, y, tau)


def p1_kappa(F, r0, rb, ts_list, ndir=120, a0=np.zeros(3)):
    aT = np.vstack([a0 + r * fib(ndir) for r in np.linspace(max(r0, 1e-3), rb, 5)])
    k, rows = F.kmax(aT, ts_list)
    return k


MIRROR_LAB = {"E": np.array([1.0, 1.0, -1.0]), "I": np.array([1.0, -1.0, 1.0])}   # label of the mirrored extremal


def p2_boundary(F, kind, mc, rb, r0, kap, ndir=200, ntime=40, a0=np.zeros(3)):
    """closure on the boundary |aT - a0| = rb of the branch domain, all times:
    inner (boundary point inside the mirror domain): P_b >= P_m (the min ignores the ending branch);
    outer (otherwise): P_b < S_coarse (the max ignores it)"""
    lab = F.labels(a0 + rb * fib(ndir))
    tau_end = TS if kind == "E" else 1 - TS
    nst = 4000
    taus = np.geomspace(1e-4, tau_end, ntime)
    keep = sorted(set(int(round(tt / tau_end * nst)) for tt in taus) - {0})
    if kind == "E":
        _, out = flow(startE(lab), 0.0, tau_end, nst, keep)
    else:
        _, out = flow(startI(lab), 1.0, 1 - tau_end, nst, keep)
    worst_out, worst_in, where = np.inf, np.inf, None
    n_in = 0
    for kidx in keep:
        tau = kidx / nst * tau_end
        t = tau if kind == "E" else 1 - tau
        Y = out[kidx]
        x = Y[:, 0:3]
        V = Y[:, 6] if kind == "E" else -Y[:, 6]
        Pb = V - kap * tau * (rb - r0) ** 2
        # the mirror branch at x: the b-branch at the mirrored point, label guess = mirrored label
        xm = x * np.array([1, 1, -1])
        try:
            labm, Ym = invert(F.shoot, xm, lab * MIRROR_LAB[kind], t)
            ok = np.linalg.norm(Ym[:, 0:3] - xm, axis=1) < 1e-8
        except np.linalg.LinAlgError:
            ok = np.zeros(len(x), bool); labm = lab; Ym = Y
        rm = np.linalg.norm(F.tilde(labm) - a0, axis=1)
        inside = ok & (rm <= rb)
        Vm = Ym[:, 6] if kind == "E" else Ym[:, 6]
        Pm = Vm - kap * tau * np.maximum(rm - r0, 0) ** 2
        if inside.any():
            n_in += inside.sum()
            worst_in = min(worst_in, np.min((Pb - Pm)[inside]))
        if (~inside).any():
            Sc = coarse_S(mc, kind, x[~inside], tau)
            mg = Sc - Pb[~inside]
            j = np.argmin(mg)
            if mg[j] < worst_out:
                worst_out, where = mg[j], (tau, j)
    return worst_out, worst_in, where, n_in


def field_part(F, y, r0, rb, kap, tau, a0=np.zeros(3)):
    """V - psi where y is in the field domain (else -inf); also returns |aT|"""
    c = F.c
    sc = 2 * np.linalg.norm(c)
    guess = F.labels((y - c) / sc)
    ok = np.linalg.norm((y - c) / sc - a0, axis=1) < 1.6 * rb
    out = np.full(len(y), -np.inf)
    rr = np.full(len(y), np.inf)
    if ok.any():
        lab, Y = invert(F.shoot, y[ok], guess[ok], TS)
        r = np.linalg.norm(F.tilde(lab) - a0, axis=1)
        good = (r <= rb) & (np.linalg.norm(Y[:, 0:3] - y[ok], axis=1) < 1e-8)
        val = Y[:, 6] - kap * tau * np.maximum(r - r0, 0) ** 2
        tmp = np.full(ok.sum(), -np.inf)
        tmp[good] = val[good]
        out[ok] = tmp
        rr[ok] = np.where(good, r, np.inf)
    return out, rr


def main():
    nu, dsb = int(sys.argv[1]), float(sys.argv[2])
    r0E, rE, kE, r0I, rI, kI, eta_t = map(float, sys.argv[3:10])
    offE = np.array([float(v) for v in sys.argv[10].split(",")]) if len(sys.argv) > 10 else np.zeros(3)
    offI = np.array([float(v) for v in sys.argv[11].split(",")]) if len(sys.argv) > 11 else np.zeros(3)
    dT = 0.1
    d = np.load(OUT + "field_labels.npz")
    FE, FI = Field("E", d["a8"]), Field("I", d["b8"])
    mcE, mcI = load_coarse("E", nu, dsb), load_coarse("I", nu, dsb)
    c = FE.c
    m = c * np.array([1, 1, -1])
    sc = 2 * np.linalg.norm(c)
    from words_eight_gap import frame
    fr = np.array(frame(c))                       # rows: radial, east, north unit vectors at c
    a0 = offE @ fr
    b0 = offI @ fr
    assert np.linalg.norm(offE) <= r0E and np.linalg.norm(offI) <= r0I, "c must lie in the zero set of psi"
    print(f"field-domain centres (r, e, n): E {offE}, I {offI}")
    print(f"patch: E-field |aT| <= {rE} (psi from {r0E}, kE = {kE}); I-field |bT| <= {rI} (psi from {r0I}, kI = {kI}); "
          f"coarse nu = {nu}; tube dT = {dT}; eta_target = {eta_t} ({eta_t / A8:.2e} A8)", flush=True)
    # (P1)
    tsE = np.concatenate([np.geomspace(1e-3, 0.05, 6), np.linspace(0.08, TS, 8)])
    kmE = p1_kappa(FE, r0E, rE, tsE, a0=a0)
    kmI = p1_kappa(FI, r0I, rI, 1 - tsE, a0=b0)
    print(f"(P1) kmax_E = {kmE:.4f} >= kE = {kE}: {kmE >= kE};  kmax_I = {kmI:.4f} >= kI = {kI}: {kmI >= kI}", flush=True)
    # (P2)
    for nm, Fx, mc, rb, r0, kp, cen in (("E", FE, mcE, rE, r0E, kE, a0), ("I", FI, mcI, rI, r0I, kI, b0)):
        wo, wi, wh, nin = p2_boundary(Fx, nm, mc, rb, r0, kp, a0=cen)
        print(f"(P2) {nm} boundary: outer closure min (S_coarse - P_b) = {wo:+.4e} (worst at tau = "
              f"{wh[0] if wh else float('nan'):.4g}); inner closure min (P_b - P_mirror) = {wi:+.4e} over {nin} points "
              f"inside the mirror domain", flush=True)
    # (P3) slice points
    Ys = []
    for d_ in np.arange(0.1, 1.0001, 0.025):
        Ys.append(c + sc * d_ * fib(300))
    nn = fib(3000)
    for rr in np.geomspace(0.05, 3.0, 30):
        Ys.append(rr * np.linalg.norm(c) * nn)
    Y = np.vstack(Ys)
    dist = np.minimum(np.linalg.norm(Y - c, axis=1), np.linalg.norm(Y - m, axis=1)) / sc
    nY = Y / np.linalg.norm(Y, axis=1)[:, None]
    Y = Y[(dist >= dT) & (np.min([1 - nY @ u for u in wc.UK], axis=0) > 1e-8)]
    dist = np.minimum(np.linalg.norm(Y - c, axis=1), np.linalg.norm(Y - m, axis=1)) / sc
    SE = coarse_S(mcE, "E", Y, TS)
    SI = coarse_S(mcI, "I", Y, 1 - TS)
    print(f"(P3) {len(Y)} slice points outside the tube; evaluating the field parts ...", flush=True)
    Ym = Y * np.array([1, 1, -1])
    near = np.minimum(np.linalg.norm(Y - c, axis=1), np.linalg.norm(Ym - c, axis=1)) / sc < 0.7
    fE = np.full(len(Y), -np.inf); fEm = fE.copy(); fI = fE.copy(); fIm = fE.copy()
    fE[near], _ = field_part(FE, Y[near], r0E, rE, kE, TS, a0)
    fEm[near], _ = field_part(FE, Ym[near], r0E, rE, kE, TS, a0)
    fI[near], _ = field_part(FI, Y[near], r0I, rI, kI, 1 - TS, b0)
    fIm[near], _ = field_part(FI, Ym[near], r0I, rI, kI, 1 - TS, b0)
    def branch_min(a, b):
        both = np.isfinite(a) & np.isfinite(b)
        out = np.where(np.isfinite(a), a, b)
        out[both] = np.minimum(a[both], b[both])
        return out
    SEt = np.maximum(SE, branch_min(fE, fEm))
    SIt = np.maximum(SI, branch_min(fI, fIm))
    mg = SEt + SIt - A8 - eta_t
    j = np.argmin(mg)
    from words_eight_shape import latlon
    la, lo, nr = latlon(Y[j])
    print(f"(P3) slice margin min (S_E + S_I - A8 - eta_target) = {mg[j]:+.4e} at d = {dist[j]:.3f}, lat {la:.1f}, "
          f"lon {lo:.1f}, |y|/|c| {nr / np.linalg.norm(c):.3f}", flush=True)
    for lo_, hi_ in ((0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 99)):
        s = (dist >= lo_) & (dist < hi_)
        if s.any():
            k = np.argmin(np.where(s, mg, np.inf))
            src = ("fieldE" if SEt[k] > SE[k] else "coarseE") + "+" + ("fieldI" if SIt[k] > SI[k] else "coarseI")
            print(f"     d in [{lo_}, {hi_}): {s.sum()} points, min margin {mg[k]:+.4e} ({src}); coarse-only sum - A8 min "
                  f"{np.min((SE + SI - A8)[s]):+.4e}", flush=True)
    np.savez(OUT + "patch_slice.npz", Y=Y, mg=mg, SE=SE, SI=SI, fE=fE, fI=fI)


if __name__ == "__main__" and sys.argv[1] != "targets":
    main()


def make_targets(rE, rI, ndir=120, ntime=30, w=0.5):
    """objective points for the coarse fits along the patch boundary extremals, all times (xi, weight tau^(1/3) w)"""
    d = np.load(OUT + "field_labels.npz")
    res = {}
    for kind, lab8, rb in (("E", d["a8"], rE), ("I", d["b8"], rI)):
        F = Field(kind, lab8)
        lab = F.labels(rb * fib(ndir))
        tau_end = TS if kind == "E" else 1 - TS
        nst = 4000
        taus = np.geomspace(1e-4, tau_end, ntime)
        keep = sorted(set(int(round(tt / tau_end * nst)) for tt in taus) - {0})
        if kind == "E":
            _, out = flow(startE(lab), 0.0, tau_end, nst, keep)
        else:
            _, out = flow(startI(lab), 1.0, 1 - tau_end, nst, keep)
        xs, ws = [], []
        for kidx in keep:
            tau = kidx / nst * tau_end
            xs.append(out[kidx][:, 0:3] / tau ** (4 / 3)); ws.append(np.full(len(lab), w * tau ** (1 / 3)))
        res[f"xi_{kind}"] = np.vstack(xs); res[f"w_{kind}"] = np.concatenate(ws)
    np.savez(OUT + f"patch_targets_{rE}_{rI}.npz", **res)
    print("targets saved", {k: v.shape for k, v in res.items()})


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "targets":
    make_targets(float(sys.argv[2]), float(sys.argv[3]))
