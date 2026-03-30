import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.interpolate import Rbf

from detect_shifts_outils import (
    torus_float_distance,
    to_centered_offsets,
    eval_local_model,
    refine_peak_bilinear,
    refine_peak_tps,
)


def _refine_one_peak(R, rc_int, *, model='bilinear', halfwin=1.5, tps_coarse_step=0.25):
    m = str(model).lower()
    if m == 'bilinear':
        return refine_peak_bilinear(R, rc_int, halfwin=halfwin)
    if m == 'tps':
        return refine_peak_tps(R, rc_int, halfwin=halfwin, coarse_step=tps_coarse_step)
    raise ValueError(f"Unknown refine model '{model}'")

from scipy.optimize import minimize

def joint_refine_uv_lbfgsb(
    R, u0, v0, *,
    h=1.5,
    integrated=True,
    energy_halfwin=2.0,
    center=None,
    d=2, R_int=1.5, lam=1.0,
    n_samples=200,
    seed=0,
):
    n = R.shape[0]
    if center is None:
        center = (n/2.0, n/2.0)
    center = np.asarray(center, float)

    u0 = np.asarray(u0, float)
    v0 = np.asarray(v0, float)

    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2*np.pi, n_samples)
    radii = np.sqrt(rng.uniform(0, R_int**2, n_samples))
    dr = radii * np.sin(thetas)
    dc = radii * np.cos(thetas)
    mc_offsets = np.stack([dr, dc], axis=1)  

    def energy(u, v):
        if integrated:
            E, _ = energy_uv_integrated(R, u, v, d=d, R_int=R_int, lam=lam, n_samples=n_samples)
            return E
        else:
            E, _ = energy_uv_simple(R, u, v, halfwin=energy_halfwin, center=center)
            return E

    def obj(x):
        r1 = x[0:2]
        r2 = x[2:4]
        u = (u0 + r1) % n
        v = (v0 + r2) % n
        return -float(energy(u, v))

    x0 = np.zeros(4, float)
    bounds = [(-h, h), (-h, h), (-h, h), (-h, h)]

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)

    r1 = res.x[0:2]
    r2 = res.x[2:4]
    u_hat = (u0 + r1) % n
    v_hat = (v0 + r2) % n
    E_hat = -float(res.fun)

    w_hat = (u_hat - v_hat + center) % n

    return u_hat, v_hat, w_hat, E_hat, res


def detect_candidates_subpixel(
    R, *,
    k=100,
    nms_size=51,
    exclude_center_radius=20.0,
    min_separation=3.0,
    refine_model='bilinear',
    refine_halfwin=1.5,
    tps_coarse_step=0.25,
    center=None
):
    n = R.shape[0]
    if center is None:
        center = (n / 2.0, n / 2.0)
    cy, cx = float(center[0]), float(center[1])

    local_max = (maximum_filter(R, size=nms_size, mode='wrap') == R)
    local_min = minimum_filter(R, size=nms_size, mode='wrap')
    prom = np.where(local_max, R - local_min, 0.0)

    yy, xx = np.indices(R.shape, float)
    dist_center = np.hypot(yy - cy, xx - cx)
    prom[dist_center < float(exclude_center_radius)] = 0.0

    flat = prom.ravel()
    if not np.any(flat):
        return np.zeros((0, 2), float)

    kk = min(int(k), np.count_nonzero(flat))
    idx = np.argpartition(flat, -kk)[-kk:]
    idx = idx[np.argsort(flat[idx])][::-1]
    r, c = np.unravel_index(idx, R.shape)
    cand = np.stack([r, c], axis=1).astype(int)

    keep = []
    for p in cand:
        if all(torus_float_distance(p, q, n) >= float(min_separation) for q in keep):
            keep.append((int(p[0]), int(p[1])))
        if len(keep) >= kk:
            break
    cand = np.array(keep, int)

    refined = []
    for rc in cand:
        r_star, c_star, _ = _refine_one_peak(
            R, rc, model=refine_model, halfwin=refine_halfwin, tps_coarse_step=tps_coarse_step
        )
        refined.append([r_star, c_star])
    return np.array(refined, float)




def energy_uv_simple(R, u, v, halfwin=2.0, center=None):
    n = R.shape[0]
    if center is None:
        center = (n / 2.0, n / 2.0)

    u, v = np.asarray(u, float), np.asarray(v, float)
    w = (u - v) + np.asarray(center, float)
    w = np.mod(w, n)

    Ru = eval_local_model(R, u, halfwin=halfwin)
    Rv = eval_local_model(R, v, halfwin=halfwin)
    Rw = eval_local_model(R, w, halfwin=halfwin)
    return 2.0 * (Ru + Rv + Rw), w


def energy_uv_integrated(R, u, v, *, d=2, R_int=2.0, lam=1.0, n_samples=100):# E_int(u,v) = 2 * (∫Ω R̄(τ-u) + ∫Ω R̄(τ-v) + λ ∫Ω R̄(τ-(u-v+center)))
    n = R.shape[0]
    center = np.array([n / 2, n / 2], float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    w = (u - v + center) % n

    def _fit_tps_local(center):
        r0, c0 = np.floor(center).astype(int) % n
        rr = np.arange(r0 - d, r0 + d + 1) % n
        cc = np.arange(c0 - d, c0 + d + 1) % n
        Y, X = np.meshgrid(rr, cc, indexing='ij')
        coords = np.stack([Y.ravel(), X.ravel()], axis=1).astype(float)
        vals = np.array([R[int(y) % n, int(x) % n] for y, x in coords])
        return Rbf(coords[:, 0], coords[:, 1], vals, function='thin_plate', smooth=0.0)

    def _integrate_model(f, center):
        thetas = np.random.uniform(0, 2 * np.pi, n_samples)
        radii = np.sqrt(np.random.uniform(0, R_int ** 2, n_samples))
        dr = radii * np.sin(thetas)
        dc = radii * np.cos(thetas)
        pts = np.stack([center[0] + dr, center[1] + dc], axis=1)
        vals = f(pts[:, 0] % n, pts[:, 1] % n)
        return float(np.mean(vals) * np.pi * R_int ** 2)

    f_u = _fit_tps_local(u)
    f_v = _fit_tps_local(v)
    f_w = _fit_tps_local(w)

    Iu = _integrate_model(f_u, u)
    Iv = _integrate_model(f_v, v)
    Iw = _integrate_model(f_w, w)

    E_int = 2.0 * (Iu + Iv + lam * Iw)
    return float(E_int), w


def best_pair_max_energy(
    R, positions, *,
    energy_halfwin=3.0,
    min_dist=3.0,
    antipodal_tol=2.0,
    angle_min_deg=5.0,
    w_exclude_center_radius=0.0,
    center=None,
    integrated=False,
    d=2, R_int=2.0, lam=1.0
):
    if positions is None or len(positions) < 2:
        return None

    n = R.shape[0]
    if center is None:
        center = (n / 2.0, n / 2.0)
    cy, cx = float(center[0]), float(center[1])
    sin_min = np.sin(np.deg2rad(angle_min_deg))

    def _dist_center(p):
        return float(np.hypot(p[0] - cy, p[1] - cx))

    best_E = -np.inf
    best = None
    pos = np.asarray(positions, float)

    for i in range(len(pos) - 1):
        ui = pos[i]
        for j in range(i + 1, len(pos)):
            vj = pos[j]
            if torus_float_distance(ui, vj, n) < float(min_dist):
                continue

            u_off = to_centered_offsets(ui, n, (cy, cx))
            v_off = to_centered_offsets(vj, n, (cy, cx))
            if np.hypot(*(u_off + v_off)) < float(antipodal_tol):
                continue

            nu, nv = np.hypot(*u_off), np.hypot(*v_off)
            if nu < 1e-9 or nv < 1e-9:
                continue
            sinang = abs(u_off[0] * v_off[1] - u_off[1] * v_off[0]) / (nu * nv)
            if sinang < sin_min:
                continue

            if integrated:
                E, w = energy_uv_integrated(R, ui, vj, d=int(d), R_int=float(R_int), lam=float(lam))
            else:
                E, w = energy_uv_simple(R, ui, vj, halfwin=energy_halfwin, center=center)

            if w_exclude_center_radius > 0.0 and _dist_center(w) < float(w_exclude_center_radius):
                continue

            if E > best_E:
                best_E = E
                best = (ui, vj, w, E)

    return best




def find_hexagon(
    R, *,
    k=100,
    nms_size=51,
    exclude_center_radius=10.0,
    min_separation=3.0,
    refine_model='tps',
    refine_halfwin=1.5,
    tps_coarse_step=0.25,
    energy_halfwin=2.0,
    min_dist=3.0,
    antipodal_tol=2.0,
    angle_min_deg=12.0,
    w_exclude_center_radius=None,
    U_ref=None,
    V_ref=None,
    integrated=True,
    d=2,
    R_int=1.5,
    lam=1.0,
    do_joint_refine=True,
    joint_h=None,              
    joint_n_samples=200,       
    joint_seed=0,
):

    import numpy as np

    n = R.shape[0]
    assert R.ndim == 2 and n == R.shape[1]
    center = (n / 2.0, n / 2.0)
    def energy_uv_integrated_seeded(R, u, v, *, d, R_int, lam, n_samples, seed):
        state = np.random.get_state()
        try:
            np.random.seed(int(seed))
            return energy_uv_integrated(
                R, u, v,
                d=int(d), R_int=float(R_int), lam=float(lam),
                n_samples=int(n_samples),
            )
        finally:
            np.random.set_state(state)

    refined = detect_candidates_subpixel(
        R,
        k=k,
        nms_size=nms_size,
        exclude_center_radius=exclude_center_radius,
        min_separation=min_separation,
        refine_model=refine_model,
        refine_halfwin=refine_halfwin,
        tps_coarse_step=tps_coarse_step,
        center=center,
    )
    if refined.size < 2:
        return None

    if w_exclude_center_radius is None:
        w_exclude_center_radius = float(exclude_center_radius)

    best = best_pair_max_energy(
        R,
        refined,
        energy_halfwin=float(energy_halfwin),
        min_dist=float(min_dist),
        antipodal_tol=float(antipodal_tol),
        angle_min_deg=float(angle_min_deg),
        w_exclude_center_radius=float(w_exclude_center_radius),
        center=center,
        integrated=bool(integrated),
        d=int(d),
        R_int=float(R_int),
        lam=float(lam),
    )
    if best is None:
        return None

    u_star, v_star, w_star, E_sel = best  

    optres = None
    if do_joint_refine:
        h_box = float(refine_halfwin if joint_h is None else joint_h)

        u_hat, v_hat, w_hat, E_hat, optres = joint_refine_uv_lbfgsb(
            R, u_star, v_star,
            h=h_box,
            integrated=False,                 
            energy_halfwin=float(energy_halfwin),
            center=center,
            d=int(d), R_int=float(R_int), lam=float(lam),
            n_samples=int(joint_n_samples),
            seed=int(joint_seed),
        )

        u_star, v_star, w_star = u_hat, v_hat, w_hat
        E_joint = float(E_hat)
    else:
        E_joint = float(E_sel)

    u_c = to_centered_offsets(u_star, n, center)
    v_c = to_centered_offsets(v_star, n, center)
    w_c = to_centered_offsets(w_star, n, center)

    def _cos(a, b, eps=1e-9):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        na, nb = np.hypot(*a), np.hypot(*b)
        return 0.0 if na < eps or nb < eps else float(np.dot(a, b) / (na * nb))

    if U_ref is not None and V_ref is not None:
        U_ref = np.asarray(U_ref, float)
        V_ref = np.asarray(V_ref, float)

        cands = [(u_c, v_c), (-u_c, v_c), (u_c, -v_c), (-u_c, -v_c)]
        scores = [_cos(pu, U_ref) + _cos(pv, V_ref) for pu, pv in cands]
        u_fin, v_fin = cands[int(np.argmax(scores))]

        W_ref = U_ref - V_ref
        w_fin = w_c if _cos(w_c, W_ref) >= 0 else -w_c
    else:
        u_fin, v_fin, w_fin = u_c, v_c, w_c


    if integrated:
        E_int_final, _ = energy_uv_integrated_seeded(
            R, u_star, v_star,
            d=d, R_int=R_int, lam=lam,
            n_samples=joint_n_samples,
            seed=joint_seed,
        )
        energy_final = float(E_int_final)
    else:
        energy_final = float(E_joint)

    hex6_centered = np.stack([+u_fin, -u_fin, +v_fin, -v_fin, +w_fin, -w_fin], axis=0)

    return dict(
        u_fin=u_fin,
        v_fin=v_fin,
        w_fin=w_fin,
        hex6_centered=hex6_centered,

        u_star=u_star,
        v_star=v_star,
        w_star=w_star,

        energy_selected=float(E_sel),     
        energy_joint=float(E_joint),      
        energy_final=float(energy_final), 

        refined_peaks=refined,
        integrated=bool(integrated),
        optres=optres,
    )





