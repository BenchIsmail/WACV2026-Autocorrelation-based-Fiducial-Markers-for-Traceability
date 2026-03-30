import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import Rbf



def torus_mod(x, n):#[0, n)
    return x - n * np.floor(x / n)

def torus_sub(p, q, n):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return np.array([torus_mod(p[0] - q[0], n), torus_mod(p[1] - q[1], n)], float)

def torus_float_distance(p, q, n):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    dr = abs(p[0] - q[0]); dr = min(dr, n - dr)
    dc = abs(p[1] - q[1]); dc = min(dc, n - dc)
    return float(np.hypot(dr, dc))

def to_centered_offsets(rc, n, center=None):
    if center is None:
        center = (n / 2.0, n / 2.0)
    r, c = float(rc[0]), float(rc[1])
    dr = ((r - center[0] + n / 2) % n) - n / 2
    dc = ((c - center[1] + n / 2) % n) - n / 2
    return np.array([dr, dc], float)



def sample_bilinear_wrap(I, pos):
    I = np.asarray(I, float)
    n = I.shape[0]
    r, c = float(pos[0]), float(pos[1])
    r0 = int(np.floor(r)) % n
    c0 = int(np.floor(c)) % n
    r1 = (r0 + 1) % n
    c1 = (c0 + 1) % n
    fr = r - np.floor(r)
    fc = c - np.floor(c)
    v00 = I[r0, c0]; v01 = I[r0, c1]
    v10 = I[r1, c0]; v11 = I[r1, c1]
    t0 = (1 - fr) * ((1 - fc) * v00 + fc * v01)
    t1 = fr * ((1 - fc) * v10 + fc * v11)
    return float(t0 + t1)



def _get_patch_wrap(R, rc_int, halfwin):
    n, m = R.shape
    r0, c0 = int(rc_int[0]) % n, int(rc_int[1]) % m
    rad = int(np.floor(halfwin))
    rr = [(r0 + dr) % n for dr in range(-rad, rad + 1)]
    cc = [(c0 + dc) % m for dc in range(-rad, rad + 1)]
    patch = R[np.ix_(rr, cc)].astype(float)
    g = np.arange(-rad, rad + 1, dtype=float)
    Y, X = np.meshgrid(g, g, indexing='ij')
    return patch, X, Y, r0, c0


def refine_peak_bilinear(R, rc_int, halfwin=1.5):
    n = R.shape[0]
    r0, c0 = int(rc_int[0]) % n, int(rc_int[1]) % n

    def neg_val(x):
        dy, dx = float(x[0]), float(x[1])
        r = torus_mod(r0 + dy, n)
        c = torus_mod(c0 + dx, n)
        return -sample_bilinear_wrap(R, (r, c))

    inits = [np.zeros(2, float)]
    bounds = [(-halfwin, halfwin), (-halfwin, halfwin)]
    best = None
    best_f = np.inf

    for x0 in inits:
        res = minimize(neg_val, x0, method='L-BFGS-B', bounds=bounds)
        if res.success and res.fun < best_f:
            best = res.x
            best_f = res.fun

    if best is None:
        return float(r0), float(c0), float(R[r0, c0])

    dy, dx = float(best[0]), float(best[1])
    r_star = torus_mod(r0 + dy, n)
    c_star = torus_mod(c0 + dx, n)
    val = sample_bilinear_wrap(R, (r_star, c_star))
    return float(r_star), float(c_star), float(val)


def _tps_fit(patch, X, Y):
    return Rbf(X.ravel(), Y.ravel(), patch.ravel(), function='thin_plate', smooth=0.0)

def refine_peak_tps(R, rc_int, halfwin=2.0, coarse_step=0.25):
    patch, X, Y, r0, c0 = _get_patch_wrap(R, rc_int, halfwin)
    rbf = _tps_fit(patch, X, Y)

    g = np.arange(-int(np.floor(halfwin)), int(np.floor(halfwin)) + coarse_step, coarse_step)
    best_val = -np.inf
    dy_best, dx_best = 0.0, 0.0

    for dy in g:
        for dx in g:
            val = float(rbf(dx, dy))
            if val > best_val:
                best_val = val
                dy_best, dx_best = dy, dx

    n = R.shape[0]
    r_star = torus_mod(r0 + dy_best, n)
    c_star = torus_mod(c0 + dx_best, n)
    val = float(rbf(dx_best, dy_best))
    return float(r_star), float(c_star), val



class LocalModelCache:
    def __init__(self):
        self.tps = {}

def eval_local_model(R, pos, model='bilinear', cache=None, halfwin=2.0):
    if model == 'bilinear':
        return sample_bilinear_wrap(R, pos)

    if model == 'tps':
        if cache is None:
            cache = LocalModelCache()
        n = R.shape[0]
        r_star, c_star = float(pos[0]), float(pos[1])
        r0 = int(np.round(r_star)) % n
        c0 = int(np.round(c_star)) % n
        key = (r0, c0)
        rbf = cache.tps.get(key)
        if rbf is None:
            patch, X, Y, *_ = _get_patch_wrap(R, (r0, c0), halfwin)
            rbf = _tps_fit(patch, X, Y)
            cache.tps[key] = rbf
        dy = r_star - r0
        dx = c_star - c0
        return float(rbf(dx, dy))

    raise ValueError(f"Unknown model '{model}'")
