import numpy as np
import logging
from typing import List, Tuple, Optional

from autocorrelation import autocorrelation_display,correlation_nomalisee
from subpixel_energy_maximiz import find_hexagon
import operations as oprt

from find_min_stable_patch_size import find_min_stable_patch_size_centered as _find_min_ps

from joblib import Parallel, delayed 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_HEXAGON_KW = dict(
    k=20,
    nms_size=10,
    exclude_center_radius=10.0,
    min_separation=3.0,
    refine_model='tps',     # 'bilinear' | 'quadratic' | 'tps' | 'gaussian'
    refine_halfwin=1.5,
    tps_coarse_step=0.25,
    energy_halfwin=3.0,
    min_dist=3.0,
    antipodal_tol=0.5,
    angle_min_deg=2.0,
    w_exclude_center_radius=10.0
)


def _roll_image_to_center(img: np.ndarray, center_rc: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape
    dy = h // 2 - int(center_rc[0])
    dx = w // 2 - int(center_rc[1])
    out = np.roll(img, dy, axis=0)
    out = np.roll(out, dx, axis=1)
    return out


def _ensure_vec_dcdr(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2,)
    return v


def _A_from_refs(U_ref: np.ndarray, V_ref: np.ndarray) -> np.ndarray:
    U_ref = _ensure_vec_dcdr(U_ref)
    V_ref = _ensure_vec_dcdr(V_ref)
    x1, y1 = float(U_ref[0]), float(U_ref[1])
    x3, y3 = float(V_ref[0]), float(V_ref[1])

    A_ = np.array([
        [x1, y1, 0,  0],
        [0,  0,  x1, y1],
        [x3, y3, 0,  0],
        [0,  0,  x3, y3],
    ], dtype=np.float64)
    return A_


def _B_from_pair(a_dcdr: np.ndarray, b_dcdr: np.ndarray) -> np.ndarray:
    ax, ay = float(a_dcdr[0]), float(a_dcdr[1])
    bx, by = float(b_dcdr[0]), float(b_dcdr[1])
    return np.array([[ax, ay, bx, by]], dtype=np.float64).reshape(4, 1)


def _ordered6_from_u_v_w_dcdr(U_dcdr: np.ndarray, V_dcdr: np.ndarray, W_dcdr: np.ndarray) -> np.ndarray:
    U = np.asarray(U_dcdr, float).reshape(2,)
    V = np.asarray(V_dcdr, float).reshape(2,)
    W = np.asarray(W_dcdr, float).reshape(2,)
    return np.array([
        [ U[0],  U[1]],   # +U
        [ V[0],  V[1]],   # +V
        [-W[0], -W[1]],   # -W
        [-U[0], -U[1]],   # -U
        [-V[0], -V[1]],   # -V
        [ W[0],  W[1]],   # +W
    ], dtype=float)


def compute_ordered_min_patch_uvws(
    image_deformee: np.ndarray,
    *,
    U_ref: Tuple[float, float],
    V_ref: Tuple[float, float],
    start_ps: int = 100,
    min_ps: int = 20,
    max_ps: int = 300,
    step: int = 2,
    tol_abs=1,
    tol_rel_pct=5.0,
    ref_smooth_window=0,
    stable_seq_len=5,
    hex_kwargs: Optional[dict] = None,
    show_tracks: bool = False,
):
    if _find_min_ps is None:
        logging.error("Aucun trouveur de patch minimal disponible.")
        return None, None, None, None, None

    kw = dict(DEFAULT_HEXAGON_KW)
    if hex_kwargs:
        kw.update(hex_kwargs)

    res = _find_min_ps(
        image=image_deformee,
        autocorr_fn=autocorrelation_display,
        find_fn=find_hexagon,
        start_ps=int(start_ps),
        step=int(step),
        min_ps=int(min_ps),
        max_ps=int(max_ps),
        tol_abs=tol_abs,
        tol_rel_pct=tol_rel_pct,
        integrated=kw.get('integrated', True),
        d=kw.get('d', 2),
        R_int=kw.get('R_int', 1.5),
        lam=kw.get('lam', 1.0),
        search_kwargs=kw,
        show_tracks=show_tracks,
        debug=False,
        U_ref=U_ref,
        V_ref=V_ref,
        ref_smooth_window=ref_smooth_window,
        stable_seq_len=stable_seq_len
    )

    if res is None:
        return None, None, None, None, None

    if ("final_u" not in res) or ("final_v" not in res) or ("final_w" not in res):
        return None, None, None, None, None

    u_drdc = np.asarray(res["final_u"], dtype=float)
    v_drdc = np.asarray(res["final_v"], dtype=float)
    w_drdc = np.asarray(res["final_w"], dtype=float)

    U_dcdr = np.array([u_drdc[1], u_drdc[0]], dtype=float)
    V_dcdr = np.array([v_drdc[1], v_drdc[0]], dtype=float)
    W_dcdr = np.array([w_drdc[1], w_drdc[0]], dtype=float)

    ordered6_dcdr = _ordered6_from_u_v_w_dcdr(U_dcdr, V_dcdr, W_dcdr)

    min_stable_patch = res.get("min_patch", None)

    return U_dcdr, V_dcdr, W_dcdr, ordered6_dcdr, min_stable_patch


def _crop_square_centered(img: np.ndarray, center_rc: Tuple[int, int], size: int) -> np.ndarray:
    h, w = img.shape
    r0, c0 = int(center_rc[0]), int(center_rc[1])
    half = int(size) // 2

    y0 = r0 - half
    y1 = r0 + half
    x0 = c0 - half
    x1 = c0 + half

    return img[y0:y1, x0:x1]




def local_affinities_via_min_patch_random(
    image_deformee: np.ndarray,
    *,
    U_ref: Tuple[float, float],
    V_ref: Tuple[float, float],
    num_centers: int = 200,
    rng_seed: Optional[int] = None,
    border_margin: int = 0,

    min_ps: int = 20,
    max_ps: int = 300,
    step: int = 1,
    stable_seq_len: int = 5,
    stable_tol: float = 1.0,

    hex_kwargs: Optional[dict] = None,
    patch_support_size: Optional[int] = None,
    verbose: bool = False,
    n_jobs: int = -1,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    if image_deformee.ndim != 2:
        raise ValueError("image_deformee doit être une image 2D (grayscale)")

    U_ref = _ensure_vec_dcdr(U_ref)
    V_ref = _ensure_vec_dcdr(V_ref)

    H, W = image_deformee.shape
    rng = np.random.default_rng(rng_seed)

    support = int(max_ps) if patch_support_size is None else int(patch_support_size)
    hard_margin = support // 2
    margin = max(int(border_margin), hard_margin)

    if H < 2 * margin or W < 2 * margin:
        logging.info("[local_affinities] image trop petite pour support=%d (marge=%d).", support, margin)
        return [], []

    rows = rng.integers(margin, H - margin, size=int(num_centers))
    cols = rng.integers(margin, W - margin, size=int(num_centers))
    centers = list(zip(rows.tolist(), cols.tolist()))  # (row, col)

    A_ = _A_from_refs(U_ref, V_ref)

    search_kwargs = {} if hex_kwargs is None else dict(hex_kwargs)

    def _process_center(r0: int, c0: int):
        local_affs: List[np.ndarray] = []
        local_ctrs: List[Tuple[int, int]] = []

        try:
            patch = _crop_square_centered(image_deformee, (r0, c0), support)
            if patch.size == 0 or patch.shape[0] != support or patch.shape[1] != support:
                return local_affs, local_ctrs
        except Exception as e:
            if verbose:
                logging.debug("[local_affinities] crop échoué au centre (%d,%d): %s", r0, c0, e)
            return local_affs, local_ctrs

        try:
            res = _find_min_ps(
                patch,
                min_ps=min_ps,
                max_ps=max_ps,
                step=step,
                stable_seq_len=stable_seq_len,
                stable_tol=stable_tol,
                U_ref=tuple(U_ref),
                V_ref=tuple(V_ref),
                search_kwargs=search_kwargs,
                show_tracks=False,
                return_patch=False,   
            )
        except Exception as e:
            if verbose:
                logging.debug("[local_affinities] _find_min_ps failed at (%d,%d): %s", r0, c0, e)
            return local_affs, local_ctrs

        if (res is None) or (res.get("min_stable_ps") is None):
            return local_affs, local_ctrs

        ordered6 = res.get("ordered6_min", None)
        if ordered6 is None or len(ordered6) != 6:
            return local_affs, local_ctrs
        
        pairs = [(0, 1)]

        for i, j in pairs:
            a = ordered6[i]
            b = ordered6[j]
            B = _B_from_pair(a, b)
            try:
                affine = oprt.transf_clac(patch, A_, B)
            except Exception as e:
                if verbose:
                    logging.debug(
                        "[local_affinities] échec transf_clac (%d,%d) pair=(%d,%d): %s",
                        r0, c0, i, j, e
                    )
                continue

            local_affs.append(affine)
            local_ctrs.append((int(r0), int(c0)))

        return local_affs, local_ctrs

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_center)(r0, c0) for (r0, c0) in centers
    )

    local_affinities: List[np.ndarray] = []
    patch_centers: List[Tuple[int, int]] = []
    for affs, ctrs in results:
        local_affinities.extend(affs)
        patch_centers.extend(ctrs)

    return local_affinities, patch_centers




def local_affinity_via_min_patch_center(
    image_deformee: np.ndarray,
    *,
    U_ref: Tuple[float, float],
    V_ref: Tuple[float, float],
    start_ps: int = 100,
    min_ps: int = 20,
    max_ps: int = 300,
    step: int = 1,
    tol_abs: float = 1.0,
    tol_rel_pct: float = 5.0,
    ref_smooth_window: int = 0,
    stable_seq_len: int = 5,
    hex_kwargs: Optional[dict] = None,
    patch_support_size: Optional[int] = None,
    which_pair: Optional[str] = None,
    verbose: bool = False,
    show_tracks: bool = False,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    if image_deformee.ndim != 2:
        raise ValueError("image_deformee doit être une image 2D (grayscale)")

    H, W = image_deformee.shape
    r0, c0 = H // 2, W // 2

    pair_map = {
        "uv":    (0, 1),  # +U, +V
        "v-w":   (1, 2),  # +V, -W
        "w-u":   (2, 3),  # -W, -U
        "-u,-v": (3, 4),  # -U, -V
        "-v,w":  (4, 5),  # -V, +W
        "+w,u":  (5, 0),  # +W, +U
        "all":   None,    # géré à part
    }
    if which_pair is None:
        which_pair = pair_map["uv"]
    elif isinstance(which_pair, str):
        if which_pair not in pair_map:
            raise ValueError(f"which_pair='{which_pair}' inconnu. Choisir parmi {list(pair_map.keys())} ou un tuple (i,j).")
        if which_pair != "all":
            which_pair = pair_map[which_pair]

    U_ref = np.asarray(U_ref, float).reshape(2,)
    V_ref = np.asarray(V_ref, float).reshape(2,)

    support = int(max_ps) if patch_support_size is None else int(patch_support_size)
    half = support // 2
    if r0 - half < 0 or c0 - half < 0 or r0 + half > H or c0 + half > W:
        if verbose:
            logging.info("[center] support=%d ne rentre pas entièrement au centre.", support)
        return [], []

    y0, y1 = r0 - half, r0 + half
    x0, x1 = c0 - half, c0 + half
    patch = image_deformee[y0:y1, x0:x1]

    try:
        U_loc, V_loc, W_loc, ordered6, _ = compute_ordered_min_patch_uvws(
            patch,
            U_ref=tuple(U_ref), V_ref=tuple(V_ref),
            start_ps=start_ps, min_ps=min_ps, max_ps=max_ps, step=step,
            tol_abs=tol_abs, tol_rel_pct=tol_rel_pct,
            ref_smooth_window=ref_smooth_window, stable_seq_len=stable_seq_len,
            hex_kwargs=hex_kwargs,
            show_tracks=show_tracks
        )
    except Exception as e:
        if verbose:
            logging.debug("[center] compute_ordered_min_patch_uvws a échoué: %s", e)
        return [], []

    if ordered6 is None or len(ordered6) != 6:
        return [], []

    A_ = _A_from_refs(U_ref, V_ref)

    affs: List[np.ndarray] = []
    ctrs: List[Tuple[int, int]] = []

    if which_pair == "all":
        idx_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]
        for (i, j) in idx_pairs:
            a = ordered6[i]
            b = ordered6[j]
            B = _B_from_pair(a, b)
            try:
                aff = oprt.transf_clac(patch, A_, B)
            except Exception as e:
                if verbose:
                    logging.debug("[center] transf_clac a échoué pour paire (%d,%d): %s", i, j, e)
                continue
            affs.append(aff)
            ctrs.append((int(r0), int(c0)))
        return affs, ctrs

    a = ordered6[which_pair[0]]
    b = ordered6[which_pair[1]]
    B = _B_from_pair(a, b)
    try:
        affine = oprt.transf_clac(patch, A_, B)
    except Exception as e:
        if verbose:
            logging.debug("[center] transf_clac a échoué: %s", e)
        return [], []

    return [affine], [(int(r0), int(c0))]




def local_affinity_via_min_patch_at(
    image_deformee: np.ndarray,
    *,
    center_rc: Tuple[int, int],
    U_ref: Tuple[float, float],
    V_ref: Tuple[float, float],
    min_ps: int = 20,
    max_ps: int = 300,
    step: int = 1,
    stable_seq_len: int = 5,
    stable_tol: float = 1.0,
    hex_kwargs: Optional[dict] = None,
    patch_support_size: Optional[int] = None,
    which_pair: Optional[str] = None,
    verbose: bool = False,
    show_tracks: bool = False,
    strict_inside: bool = True,
    pad_mode: Optional[str] = "reflect",
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Optional[np.ndarray]]:
    import logging

    if image_deformee.ndim != 2:
        raise ValueError("image_deformee doit être une image 2D (grayscale)")

    H, W = image_deformee.shape
    r0, c0 = map(int, center_rc)

    pair_map = {
        "uv":    (0, 1),
        "v-w":   (1, 2),
        "w-u":   (2, 3),
        "-u,-v": (3, 4),
        "-v,w":  (4, 5),
        "+w,u":  (5, 0),
        "all":   None,
    }
    if which_pair is None:
        which_pair = pair_map["uv"]
    elif isinstance(which_pair, str):
        if which_pair not in pair_map:
            raise ValueError(
                f"which_pair='{which_pair}' inconnu. Choisir parmi {list(pair_map.keys())} "
                "ou un tuple (i,j)."
            )
        if which_pair != "all":
            which_pair = pair_map[which_pair]

    U_ref = np.asarray(U_ref, float).reshape(2,)
    V_ref = np.asarray(V_ref, float).reshape(2,)

    support = int(max_ps) if patch_support_size is None else int(patch_support_size)
    half = support // 2

    y0, y1 = r0 - half, r0 + half
    x0, x1 = c0 - half, c0 + half

    if strict_inside:
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            if verbose:
                logging.info("[at] support=%d dépasse l'image en (r,c)=(%d,%d).", support, r0, c0)
            return [], [], None
        patch_support = image_deformee[y0:y1, x0:x1]
    else:
        pad_top    = max(0, -y0)
        pad_left   = max(0, -x0)
        pad_bottom = max(0,  y1 - H)
        pad_right  = max(0,  x1 - W)
        if (pad_top or pad_left or pad_bottom or pad_right):
            if pad_mode is None:
                if verbose:
                    logging.info("[at] débordement sans pad_mode -> abandon")
                return [], [], None
            padw = ((pad_top, pad_bottom), (pad_left, pad_right))
            img_pad = np.pad(image_deformee, padw, mode=pad_mode)
            y0_p, y1_p = y0 + pad_top, y1 + pad_top
            x0_p, x1_p = x0 + pad_left, x1 + pad_left
            patch_support = img_pad[y0_p:y1_p, x0_p:x1_p]
        else:
            patch_support = image_deformee[y0:y1, x0:x1]

    search_kwargs = {} if hex_kwargs is None else dict(hex_kwargs)

    try:
        res = _find_min_ps(
            patch_support,
            min_ps=min_ps,
            max_ps=max_ps,
            step=step,
            stable_seq_len=stable_seq_len,
            stable_tol=stable_tol,
            U_ref=tuple(U_ref),
            V_ref=tuple(V_ref),
            search_kwargs=search_kwargs,
            show_tracks=show_tracks,
            return_patch=True,
            patch_pad_mode=pad_mode if pad_mode is not None else "reflect",
        )
    except Exception as e:
        if verbose:
            logging.debug("[at] _find_min_ps a échoué: %s", e)
        return [], [], None

    if (res is None) or (res.get("min_stable_ps") is None):
        return [], [], None

    stable_patch = res.get("min_patch", None)

    ordered6 = res.get("ordered6_min", None)
    if ordered6 is None or len(ordered6) != 6:
        return [], [], None

    A_ = _A_from_refs(U_ref, V_ref)

    if which_pair == "all":
        idx_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        affs: List[np.ndarray] = []
        ctrs: List[Tuple[int, int]] = []
        for (i, j) in idx_pairs:
            a = ordered6[i]
            b = ordered6[j]
            B = _B_from_pair(a, b)
            try:
                aff = oprt.transf_clac(patch_support, A_, B)
            except Exception as e:
                if verbose:
                    logging.debug("[at] transf_clac a échoué pour paire (%d,%d): %s", i, j, e)
                continue
            affs.append(aff)
            ctrs.append((int(r0), int(c0)))
        return affs, ctrs, stable_patch

    a = ordered6[which_pair[0]]
    b = ordered6[which_pair[1]]
    B = _B_from_pair(a, b)

    try:
        affine = oprt.transf_clac(patch_support, A_, B)
    except Exception as e:
        if verbose:
            logging.debug("[at] transf_clac a échoué: %s", e)
        return [], [], None

    return [affine], [(int(r0), int(c0))], stable_patch








def choose_best_affinity_patch(
    min_stable_patch: np.ndarray,
    affs: List[np.ndarray],
    image_ref: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not affs or min_stable_patch is None:
        return None, None

    (Hc, Wc) = image_ref.shape
    best_score = -np.inf
    redr_best = None
    H_used_best = None

    for H in affs:
        redr_candidate, H_used_candidate = oprt.centralizer(min_stable_patch, H, (Hc, Wc), 0)

        ref_resized, crop_resized = oprt.padding_zone_corr(image_ref, redr_candidate)
        corr = correlation_nomalisee(ref_resized, crop_resized)

        score = float(corr.max())

        if score > best_score:
            best_score = score
            redr_best = redr_candidate
            H_used_best = H_used_candidate

    return redr_best, H_used_best


