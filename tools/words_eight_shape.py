"""Stage 21, week-1 gate, step 3 (float): the 1/12 Chenciner-Montgomery problem in the 3-D shape space.

Equal masses 1, G = 1, planar, centre of mass 0. Jacobi r1 = q2 - q1, r2 = q3 - (q1 + q2)/2 (complex numbers),
mass-weighted z1 = sqrt(mu1) r1, z2 = sqrt(mu2) r2, mu1 = 1/2, mu2 = 2/3, so K = (|z1'|^2 + |z2'|^2)/2.
Hopf map: x = (w1, w2, w3), w1 = |z1|^2 - |z2|^2, w2 + i w3 = 2 z1 conj(z2); |x| = |z1|^2 + |z2|^2 = I (moment of
inertia). At zero angular momentum the kinetic energy is K = |x'|^2 / (8 |x|) (quotient metric g = |dx|^2/(4|x|),
the cone dr^2 + r^2 dS^2/4 over the unit sphere, r = sqrt|x|).
Mutual distances: r_ij^2 = |x| - u_k . x (k the third body), u_3 = (-1, 0, 0) (collision 12), u_1 = (1/2, sqrt3/2, 0)
(collision 23), u_2 = (1/2, -sqrt3/2, 0) (collision 13). U = sum_k (|x| - u_k . x)^(-1/2), homogeneous of degree
-1/2 in x.
Euler set E = {r2 = 0} (body 3 at the midpoint of 1 and 2): the ray R+ (1, 0, 0).
Isosceles set I = {r13 = r12}: the plane w2 = sqrt3 w1 (normal n_I = (sqrt3, -1, 0)/2), through the collision ray of
23 and the Euler ray of body 1, containing the w3 axis (Lagrange points).
Self-similar scaling: t -> s t, x -> s^(4/3) x, action -> s^(1/3) action.
Discretisation: nodes x_0..x_N at t_k = k tau/N; kinetic sum_k D(x_k, x_k+1)/(2h) with D the squared geodesic distance
of the cone, D(a, b) = |a| + |b| - sqrt(2(|a||b| + a.b)) (= min over the S^1 phase of |z_a - e^(i phi) z_b|^2),
potential by the trapezoid rule. Second order in h.
    python tools/words_eight_shape.py        (verification: A8 against the Jacobi probe, the reduction checks)
"""
import numpy as np
from scipy.optimize import minimize

S3 = np.sqrt(3.0)
UK = np.array([[-1.0, 0.0, 0.0], [0.5, S3 / 2, 0.0], [0.5, -S3 / 2, 0.0]])   # collision rays 12, 23, 13
EI = np.array([0.5, S3 / 2, 0.0])     # in-plane unit vector of I along the equator (the C23 direction)
EZ = np.array([0.0, 0.0, 1.0])
NI = np.array([S3 / 2, -0.5, 0.0])    # unit normal of I


def jacobi_to_shape(r1, r2):
    """r1, r2 complex arrays -> x (..., 3)"""
    z1 = np.sqrt(0.5) * r1
    z2 = np.sqrt(2.0 / 3.0) * r2
    w = 2 * z1 * np.conj(z2)
    return np.stack([np.abs(z1) ** 2 - np.abs(z2) ** 2, w.real, w.imag], axis=-1)


def shape_to_jacobi(x):
    """one lift (z1 real >= 0)"""
    n = np.linalg.norm(x, axis=-1)
    a = np.sqrt(np.maximum((n + x[..., 0]) / 2, 0))
    b2 = np.maximum((n - x[..., 0]) / 2, 0)
    w = x[..., 1] + 1j * x[..., 2]
    z2 = np.where(a > 1e-300, np.conj(w) / (2 * np.where(a > 1e-300, a, 1)), np.sqrt(b2))
    return a / np.sqrt(0.5), z2 / np.sqrt(2.0 / 3.0)


def potential(X):
    """U and grad U for X (..., 3)"""
    n = np.linalg.norm(X, axis=-1)
    nx = X / n[..., None]
    U = np.zeros(X.shape[:-1])
    G = np.zeros(X.shape)
    for u in UK:
        q = n - X @ u
        qi = q ** -0.5
        U += qi
        G += -0.5 * (qi / q)[..., None] * (nx - u)
    return U, G


def potential_only(X):
    n = np.linalg.norm(X, axis=-1)
    return sum((n - X @ u) ** -0.5 for u in UK)


def dist2(a, b):
    """squared geodesic distance in the cone metric, and its gradients"""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    Q = np.sqrt(np.maximum(2 * (na * nb + np.sum(a * b, axis=-1)), 1e-300))
    D = na + nb - Q
    ua = a / na[..., None]
    ub = b / nb[..., None]
    ga = ua - (nb[..., None] * ua + b) / Q[..., None]
    gb = ub - (na[..., None] * ub + a) / Q[..., None]
    return D, ga, gb


def metric_dist(a, b):
    return np.sqrt(np.maximum(dist2(a, b)[0], 0))


def action_nodes(X, tau=1.0):
    """discrete action of the node path X (N+1, 3) over time tau, and its gradient (N+1, 3)"""
    N = X.shape[0] - 1
    h = tau / N
    D, ga, gb = dist2(X[:-1], X[1:])
    U, GU = potential(X)
    w = np.full(N + 1, h)
    w[0] = w[-1] = h / 2
    A = np.sum(D) / (2 * h) + np.sum(w * U)
    G = w[:, None] * GU
    G[:-1] += ga / (2 * h)
    G[1:] += gb / (2 * h)
    return A, G


def action_nodes_only(X, tau=1.0):
    N = X.shape[0] - 1
    h = tau / N
    na = np.linalg.norm(X, axis=-1)
    Q = np.sqrt(np.maximum(2 * (na[:-1] * na[1:] + np.sum(X[:-1] * X[1:], axis=-1)), 0))
    D = na[:-1] + na[1:] - Q
    U = potential_only(X)
    w = np.full(N + 1, h)
    w[0] = w[-1] = h / 2
    return np.sum(D) / (2 * h) + np.sum(w * U)


# ---------------------------------------------------------------- parametrisations
# full problem E -> I: z = [s, x_1..x_{N-1} (3 each), a, b]; x_0 = (s, 0, 0), x_N = a EI + b EZ
def unpack_EI(z, N):
    X = np.empty((N + 1, 3))
    X[0] = (z[0], 0, 0)
    X[1:N] = z[1:1 + 3 * (N - 1)].reshape(N - 1, 3)
    X[N] = z[-2] * EI + z[-1] * EZ
    return X


def pack_EI(X):
    N = X.shape[0] - 1
    return np.concatenate([[X[0, 0]], X[1:N].ravel(), [X[N] @ EI, X[N] @ EZ]])


def f_EI(z, N, tau=1.0):
    X = unpack_EI(z, N)
    A, G = action_nodes(X, tau)
    return A, np.concatenate([[G[0, 0]], G[1:N].ravel(), [G[N] @ EI, G[N] @ EZ]])


def solve(fun, z0, args=(), gtol=1e-11, maxiter=40000):
    return minimize(fun, z0, args=args, jac=True, method="L-BFGS-B",
                    options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-16, "maxcor": 30})


def eight_guess(N, tau=1.0, lat_end=0.7, s=None):
    """a rough initial path from E (lon 0, equator) to I (lon 60 deg, latitude lat_end), size about 1.5 tau^(4/3)"""
    t = np.linspace(0, 1, N + 1)
    lon = np.deg2rad(60) * t
    lat = lat_end * np.sin(np.pi * t / 2)
    R = (1.5 if s is None else s) * tau ** (4 / 3)
    return R * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def eight_discrete(N, tau=1.0, z0=None):
    """discrete minimiser of the 1/12 problem (northern one), nodes and action"""
    if z0 is None:
        z0 = pack_EI(eight_guess(N, tau))
    o = solve(f_EI, z0, (N, tau))
    X = unpack_EI(o.x, N)
    if X[N, 2] < 0:
        X[:, 2] *= -1
    return X, o.fun, o


def refine(X, M):
    """interpolate a node path to M intervals (linear in x; good enough as a start)"""
    N = X.shape[0] - 1
    t = np.linspace(0, 1, N + 1)
    tt = np.linspace(0, 1, M + 1)
    return np.stack([np.interp(tt, t, X[:, j]) for j in range(3)], axis=1)


def min_rij(X):
    n = np.linalg.norm(X, axis=-1)
    return float(np.sqrt(np.min([n - X @ u for u in UK])))


def latlon(x):
    n = np.linalg.norm(x, axis=-1)
    return np.rad2deg(np.arcsin(x[..., 2] / n)), np.rad2deg(np.arctan2(x[..., 1], x[..., 0])), n


def main():
    # 1. reduction checks on random configurations
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    q -= q.mean(axis=1, keepdims=True)
    r1 = q[:, 1] - q[:, 0]
    r2 = q[:, 2] - (q[:, 0] + q[:, 1]) / 2
    x = jacobi_to_shape(r1, r2)
    U_direct = 1 / abs(q[:, 1] - q[:, 0]) + 1 / abs(q[:, 2] - q[:, 0]) + 1 / abs(q[:, 2] - q[:, 1])
    I_direct = np.sum(abs(q) ** 2, axis=1)
    print("check |x| = moment of inertia:", np.max(abs(np.linalg.norm(x, axis=1) - I_direct)))
    print("check U(x) = sum 1/r_ij:", np.max(abs(potential_only(x) - U_direct)))
    r12, r13 = abs(q[:, 1] - q[:, 0]), abs(q[:, 2] - q[:, 0])
    print("check r13^2 - r12^2 = -(3/2)(w1 - w2/sqrt3) [I: w2 = sqrt3 w1]:",
          np.max(abs((r13 ** 2 - r12 ** 2) - (-(3 / 2) * x[:, 0] + (S3 / 2) * x[:, 1]))))
    # kinetic at zero angular momentum: a horizontal velocity (Im <z, z'> = 0)
    z = np.stack([np.sqrt(0.5) * r1, np.sqrt(2 / 3) * r2], axis=1)
    v = rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape)
    v -= (np.imag(np.sum(np.conj(z) * v, axis=1)) / np.sum(abs(z) ** 2, axis=1))[:, None] * 1j * z
    eps = 1e-6
    xp = jacobi_to_shape((z[:, 0] + eps * v[:, 0]) / np.sqrt(0.5), (z[:, 1] + eps * v[:, 1]) / np.sqrt(2 / 3))
    xm = jacobi_to_shape((z[:, 0] - eps * v[:, 0]) / np.sqrt(0.5), (z[:, 1] - eps * v[:, 1]) / np.sqrt(2 / 3))
    xd = (xp - xm) / (2 * eps)
    K_shape = np.sum(xd ** 2, axis=1) / (8 * np.linalg.norm(x, axis=1))
    K_dir = 0.5 * np.sum(abs(v) ** 2, axis=1)
    print("check K = |x'|^2/(8|x|) at L = 0:", np.max(abs(K_shape - K_dir) / K_dir))
    # 2. A8 at tau = 1, N = 48, 96, 192 and Richardson
    X = None
    res = {}
    for N in (48, 96, 192, 384):
        z0 = None if X is None else pack_EI(refine(X, N))
        X, A, o = eight_discrete(N, 1.0, z0)
        res[N] = (A, X)
        la, lo, n = latlon(X)
        print(f"N = {N}: A = {A:.10f}  (|grad| {np.max(abs(o.jac)):.1e})  start |x| = {X[0, 0]:.6f}, end lat {la[-1]:.4f} "
              f"lon {lo[-1]:.4f} |x| {n[-1]:.6f}; min r_ij {min_rij(X):.4f}")
    Ns = sorted(res)
    for a, b in zip(Ns[:-1], Ns[1:]):
        print(f"Richardson ({a}, {b}): {res[b][0] + (res[b][0] - res[a][0]) / 3:.10f}")
    np.save("D:/nature-math/math-gaps/words/out/eight/eight_shape_N384.npy", res[384][1])
    # 3. multistart E -> I over the full plane I in shape space (other local minima?)
    N = 48
    found = []
    for k in range(60):
        R = rng.uniform(0.4, 3.0)
        lon_end = rng.choice([60.0, 240.0]) + rng.normal(0, 20)
        lat_end = rng.uniform(-80, 80)
        t = np.linspace(0, 1, N + 1)
        lon = np.deg2rad(lon_end * t + rng.normal(0, 30) * np.sin(np.pi * t))
        lat = np.deg2rad(lat_end * t + rng.normal(0, 30) * np.sin(np.pi * t))
        rr = R * (1 + 0.3 * rng.normal() * np.sin(np.pi * t))
        X0 = abs(rr)[:, None] * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
        z0 = pack_EI(X0)
        o = solve(f_EI, z0, (N, 1.0))
        Xk = unpack_EI(o.x, N)
        la, lo, n = latlon(Xk)
        found.append((o.fun, la[-1], lo[-1], np.max(abs(o.jac))))
    found.sort()
    vals = np.array([f[0] for f in found])
    print(f"multistart (60, N = 48, full plane I): distinct action values (rounded 1e-6):",
          sorted(set(np.round(vals, 6)))[:10])
    for f in found[:3] + found[-3:]:
        print("   A = %.8f end lat %+.3f lon %+.3f |grad| %.1e" % f)


if __name__ == "__main__":
    main()
