# patchwize_rectification.py
import numpy as np
import sys
sys.path.append('/Users/i.bencheikh/Desktop/ENS/Doctorat/vpv-master/misc/')
sys.path.append('/Users/i.bencheikh/Desktop/ENS/Doctorat/GhostSeal/')
import vpv
from vpv import *
import cv2 as cv
from scipy.ndimage import binary_dilation
import deformation_generator as dg
import autocorrelation as autoc
import detect_shifts as shm
from typing import List, Tuple, Optional
from rectification_energy import local_affinity_via_min_patch_at
import operations as oprt
from joblib import Parallel, delayed  
from rectification import rectification


def image_centralizer(patch, H_affine_3x3, canvas_shape_hw):
    Hc, Wc = map(int, canvas_shape_hw)
    hp, wp = patch.shape

    # centres (en coordonnées image, pixels)
    cx_p, cy_p = (wp - 1) / 2.0, (hp - 1) / 2.0
    cx_c, cy_c = (Wc - 1) / 2.0, (Hc - 1) / 2.0

    # translations de centrage
    T_patch = np.array([[1, 0, -cx_p],
                        [0, 1, -cy_p],
                        [0, 0,   1  ]], dtype=np.float64)
    T_canvas = np.array([[1, 0, cx_c],
                         [0, 1, cy_c],
                         [0, 0,  1  ]], dtype=np.float64)

    # homographie finale : centre->H->centre
    H_final = T_canvas @ H_affine_3x3 @ T_patch
    H_final = H_final.astype(np.float64)

    # warp
    border_val = int(np.round(np.mean(patch))) if patch.ndim == 2 else 0
    out = cv.warpPerspective(patch, H_final, (Wc, Hc),
                             flags=cv.INTER_LANCZOS4,
                             borderMode=cv.BORDER_CONSTANT,
                             borderValue=border_val)
    return out, H_final


def generate_positions_in_roi(
    image_deformee: np.ndarray,
    y1: int, y2: int, x1: int, x2: int,
    *,
    mode: str = "grid",         # "grid" ou "random"
    step: int = 64,             # espacement pour la grille
    n_random: int = 100,        # nombre de points aléatoires
    seed: Optional[int] = None  # graine aléatoire (pour reproductibilité)
) -> List[Tuple[int, int]]:
    H, W = image_deformee.shape[:2]

    if not (0 <= y1 < y2 <= H and 0 <= x1 < x2 <= W):
        raise ValueError("ROI invalide : vérifier 0 <= y1 < y2 <= H et 0 <= x1 < x2 <= W")

    rng = np.random.default_rng(seed)

    if mode == "grid":
        ys = np.arange(y1, y2, step)
        xs = np.arange(x1, x2, step)
        centers = [(int(r), int(c)) for r in ys for c in xs]

    elif mode == "random":
        ys = rng.integers(y1, y2, size=n_random)
        xs = rng.integers(x1, x2, size=n_random)
        centers = list(zip(ys.tolist(), xs.tolist()))

    else:
        raise ValueError("mode doit être 'grid' ou 'random'")

    return centers


#========Ghostseal_encoded========
def convolutional_encode(bits, K=7):
    if K == 3:
        G1 = [1, 1, 1]
        G2 = [1, 0, 1]
    elif K == 5:
        G1 = [1, 1, 1, 0, 1]
        G2 = [1, 0, 0, 1, 1]
    elif K == 7:
        G1 = [1, 1, 1, 1, 0, 0, 1]
        G2 = [1, 0, 1, 1, 0, 1, 1]
    elif K == 9:
        G1 = [1, 1, 1, 1, 0, 1, 0]
        G2 = [1, 0, 1, 1, 1, 0, 0, 0, 1]
    else:
        raise ValueError("K must be 3, 5, 7 or 9")

    memory = [0] * (K - 1)
    encoded = []

    for bit in bits + [0] * (K - 1):  # flush
        shift_register = [bit] + memory
        out1 = sum(a & b for a, b in zip(shift_register, G1)) % 2
        out2 = sum(a & b for a, b in zip(shift_register, G2)) % 2
        encoded.extend([out1, out2])
        memory = shift_register[:-1]

    return encoded


def generate_active_positions(h, w, density, point_size, sk, seed=None):
    if seed is not None:
        np.random.seed(seed)

    step = point_size * sk
    grid_positions = [(i, j) for i in range(0, h, step) for j in range(0, w, step)]

    total_positions = len(grid_positions)
    num_active = int(round(total_positions * density))

    active_indices = np.random.choice(total_positions, size=num_active, replace=False)
    active_positions = [grid_positions[i] for i in active_indices]

    return active_positions


def gen_test_sk4_with_token_convolution(
    active_positions, h, w, point_size,
    angle_shift, norm_shift, angle_base,
    token, K=7
):
    # Étape 1 : Conversion du token en bits
    bits = ''.join(f'{ord(c):08b}' for c in token)
    bit_array = np.array([int(b) for b in bits], dtype=np.uint8)
    # Étape 2 : Codage convolutionnel
    encoded_bits = convolutional_encode(bit_array.tolist(), K=K)
    encoded_len = len(encoded_bits)

    # Étape 3 : Placement dans les positions actives
    sorted_positions = sorted(active_positions, key=lambda pos: (pos[0], pos[1]))
    base = np.zeros((h, w), dtype=np.uint8)

    for idx, (i, j) in enumerate(sorted_positions):
        bit = encoded_bits[idx % encoded_len]
        if bit == 1:
            base[i:i+point_size, j:j+point_size] = 1

    # Étape 4 : Application des décalages
    k = np.deg2rad(angle_shift)
    module = norm_shift * point_size
    rad = np.deg2rad(angle_base)

    shift1_x = int(round(module * np.cos(rad)))
    shift1_y = int(round(module * np.sin(rad)))
    shift2_x = int(round(shift1_x * np.cos(k) + shift1_y * np.sin(k)))
    shift2_y = int(round(-shift1_x * np.sin(k) + shift1_y * np.cos(k)))

    shifts = [(0, 0), (shift1_y, shift1_x), (shift2_y, shift2_x)]

    # Étape 5 : Créer les 3 copies (shiftées)
    copies = []
    for dy, dx in shifts:
        shifted = np.zeros_like(base)
        yy, xx = np.nonzero(base)
        for y, x in zip(yy, xx):
            i, j = (y + dy) % h, (x + dx) % w
            shifted[i, j] = 1
        copies.append(shifted)

    # Étape 6 : Dilatation et inversion pour image finale
    combined = np.clip(np.sum(copies, axis=0), 0, 1)
    structuring_element = np.ones((point_size, point_size), dtype=bool)
    final = np.ones_like(combined, dtype=np.uint8) * 255
    final[binary_dilation(combined == 1, structure=structuring_element)] = 0

    return base, final


def generate_gs3d_sk_deformation(h, w, density, point_size, sk, angle_base, norm_shift, angle_shift,
                                 token, active_positions,
                                 scale, rot_z, tilt, tilt_orient, rot_x, rot_y, seed=None, kernel_gauss=(11,11)):

    active_positions = generate_active_positions(h, w, density, point_size, sk, seed=None)
    base_gs3d , ghostseal_gs3d_clean = gen_test_sk4_with_token_convolution(
        active_positions, h, w, point_size,
        angle_shift, norm_shift, angle_base,
        token, K=7
    )
    ghostseal_gs3d = cv.GaussianBlur(ghostseal_gs3d_clean, kernel_gauss, 0)

    final_deformation, final_homography = dg.general_deformer(ghostseal_gs3d, scale, rot_z, tilt, tilt_orient, rot_x, rot_y)

    return ghostseal_gs3d_clean, base_gs3d, final_deformation, final_homography


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
        # Rectification + recentrage sur un canvas de la taille de l'image de référence
        redr_candidate, H_used_candidate = image_centralizer(min_stable_patch, H, (Hc, Wc))

        # Calcul de la corrélation normalisée
        ref_resized, crop_resized = oprt.padding_zone_corr(image_ref, redr_candidate)
        corr = autoc.correlation_nomalisee(ref_resized, crop_resized)

        score = float(corr.max())

        if score > best_score:
            best_score = score
            redr_best = redr_candidate
            H_used_best = H_used_candidate

    return redr_best, H_used_best


def frobenius_norm(A, H):
    A_lin_persp = A[:, :2]
    H_lin_persp = H[:, :2]
    return np.linalg.norm(A_lin_persp - H_lin_persp, "fro")


def _process_center(
    center: Tuple[int, int],
    image_deformee: np.ndarray,
    image_ref: np.ndarray,
    U_ref: Tuple[float, float],
    V_ref: Tuple[float, float],
    start_ps: int,
    min_ps: int,
    max_ps: int,
    step: int,
    tol_abs: float,
    tol_rel_pct: float,
    ref_smooth_window: int,
    stable_seq_len: int,
    hex_kwargs: dict,
):
    # Calcul local (séquentiel) de l’affinité et du patch minimal
    affs, centers, min_stable_patch = local_affinity_via_min_patch_at(
        image_deformee=image_deformee,
        center_rc=center,
        U_ref=U_ref, V_ref=V_ref,
        start_ps=start_ps, min_ps=min_ps, max_ps=max_ps, step=step,
        tol_abs=tol_abs, tol_rel_pct=tol_rel_pct,
        ref_smooth_window=ref_smooth_window, stable_seq_len=stable_seq_len,
        hex_kwargs=hex_kwargs,
        patch_support_size=None,
        which_pair="all",
        verbose=False,
        show_tracks=False
    )

    if not affs or min_stable_patch is None:
        return None, None

    redr_best, H_used_best = choose_best_affinity_patch(
        min_stable_patch=min_stable_patch,
        affs=affs,
        image_ref=image_ref
    )

    if redr_best is None or H_used_best is None:
        return None, None

    return H_used_best, redr_best


# --------- traitement principal + parallélisation sur les positions ---------

from PIL import Image

h, w, density, point_size, sk, angle_base = 2048, 2048, 0.1, 1, 4, 90
norm_shift = 40
angle_shift = 90
token = 'ZAEG4HBR'
active_positions = generate_active_positions(h, w, density, point_size, sk, seed=None)
ghostseal_sk, _, gs3d1, matrix = generate_gs3d_sk_deformation(h, w, density, point_size, sk, 
                        angle_base,norm_shift,angle_shift,  token,active_positions 
                    , 1, 0, 1, 0, 0, 0, seed=None,kernel_gauss = (5,5))
# start_ps = 120
# min_ps = 20
# max_ps = 200
# step = 2
# tol_abs = 1
# tol_rel_pct = 1
# ref_smooth_window = 4
# stable_seq_len = 4
# hex_kwargs = dict(
#     k=40,
#     nms_size=20,
#     exclude_center_radius=10.0,
#     min_separation=8.0,
#     refine_model='tps',
#     refine_halfwin=1.5,
#     tps_coarse_step=0.25,
#     energy_halfwin=1.5,
#     min_dist=1.0,
#     antipodal_tol=2.0,
#     angle_min_deg=10.0,
#     w_exclude_center_radius=10.0,
# )
"""
#large_noise, _, gs3d1, matrix = generate_gs3d_sk_deformation(h, w, density, point_size, sk, 
#                        angle_base,norm_shift,angle_shift,  token,active_positions 
#                    , 1, 0, 1, 0, 20, -20, seed=None,kernel_gauss = (5,5))

from ghostseal_generator import generate_gs3d_noise_deformation
large_noise,gs3d1, final_homography = generate_gs3d_noise_deformation(h, w, (60,0), (0,60), 1, 0, 1, 0, 20, -20, seed=None)

vpv(large_noise,gs3d1)
from time import time
start = time()

final_image, total_h_estim = rectification(
    large_noise, gs3d1,
    iteration=1,
    U_ref=(60, 0), V_ref=(0, 60),
    num_centers=100,
    border_margin = 200,
    rng_seed = 3146412745,
    start_ps=120, min_ps=20, max_ps=200, step=4,
    tol_abs=1, tol_rel_pct=1.0,
    ref_smooth_window=4, stable_seq_len=4,
    use_cond_hard_filter=True,
    cond_max=5.0,
    use_cond_robust_filter=False,
    k_mad_cond=5.0,
    use_A_robust_filter=True,
    k_mad_A=3.0,
)

end = time()
print(end-start)
print(frobenius_norm(total_h_estim, final_homography))
vpv(large_noise,gs3d1,final_image) """

#large_noise = np.array(Image.open("/Users/i.bencheikh/Desktop/ref (1).png").convert("L"))
#gs3d1 = np.array(Image.open("/Users/i.bencheikh/Downloads/gs3d34.PNG").convert("L"))
""" list_positions = generate_positions_in_roi(gs3d1, 100, 800, 200, 600, mode="grid", step=150)


def _process_center(
    center: Tuple[int, int],
    image_deformee: np.ndarray,
    image_ref: np.ndarray,
    U_ref: Tuple[float, float],
    V_ref: Tuple[float, float],
    start_ps: int,
    min_ps: int,
    max_ps: int,
    step: int,
    tol_abs: float,
    tol_rel_pct: float,
    ref_smooth_window: int,
    stable_seq_len: int,
    hex_kwargs: dict,
):
    # Calcul local (séquentiel) de l’affinité et du patch minimal
    affs, centers, min_stable_patch = local_affinity_via_min_patch_at(
        image_deformee=image_deformee,
        center_rc=center,
        U_ref=U_ref, V_ref=V_ref,
        start_ps=start_ps, min_ps=min_ps, max_ps=max_ps, step=step,
        tol_abs=tol_abs, tol_rel_pct=tol_rel_pct,
        ref_smooth_window=ref_smooth_window, stable_seq_len=stable_seq_len,
        hex_kwargs=hex_kwargs,
        patch_support_size=None,
        which_pair="all",
        verbose=False,
        show_tracks=False
    )

    if not affs or min_stable_patch is None:
        return None, None

    redr_best, H_used_best = choose_best_affinity_patch(
        min_stable_patch=min_stable_patch,
        affs=affs,
        image_ref=image_ref
    )

    if redr_best is None or H_used_best is None:
        return None, None

    return H_used_best, redr_best


# PARALLÉLISATION sur les centres :
results = Parallel(n_jobs=-1)(
    delayed(_process_center)(
        center,
        gs3d1,
        large_noise,
        (60, 0), (0, 60),
        start_ps,
        min_ps,
        max_ps,
        step,
        tol_abs,
        tol_rel_pct,
        ref_smooth_window,
        stable_seq_len,
        hex_kwargs
    )
    for center in list_positions
)

affinities: List[np.ndarray] = []
stable_patchs: List[np.ndarray] = []
for H_used_best, redr_best in results:
    if H_used_best is None or redr_best is None:
        continue
    affinities.append(H_used_best)
    stable_patchs.append(redr_best)


def crop_and_align(image_ref, copies):
    ref_resized, crop_resized = oprt.padding_zone_corr(image_ref, copies)
    corr = autoc.correlation_nomalisee(ref_resized, crop_resized)

    h_, w_ = corr.shape
    h, w = copies.shape
    x, y = shm.find_peak_position(corr, method='centroid')
    dx = w / 2 - x - (w - w_) / 2
    dy = h / 2 - y - (h - h_) / 2
    H_affine = np.eye(3)
    H_affine[0, 2] -= dx
    H_affine[1, 2] -= dy

    transformed_crop = oprt.apply_homography(crop_resized, H_affine, (h_, w_))
    return transformed_crop, corr, ref_resized, crop_resized


imm_transl_list = []
for patch in stable_patchs:
    imm_transl, correlation, ref_resized, crop_resized = crop_and_align(large_noise, patch)
    imm_transl_list.append(imm_transl)

image_f = imm_transl_list[0]
for i in range(1, len(imm_transl_list)):
    image_f += imm_transl_list[i]
image_f = image_f / len(imm_transl_list)

vpv(cv.GaussianBlur(large_noise, (11,11), 0), image_f, imm_transl_list[0], imm_transl_list[1], imm_transl_list[2]) """




def _process_center(
    center: Tuple[int, int],
    image_deformee: np.ndarray,
    image_ref: np.ndarray,
    U_ref: Tuple[float, float],
    V_ref: Tuple[float, float],
    start_ps: int,
    min_ps: int,
    max_ps: int,
    step: int,
    tol_abs: float,
    tol_rel_pct: float,
    ref_smooth_window: int,
    stable_seq_len: int,
    hex_kwargs: dict,
):
    
    # Calcul local (séquentiel) de l’affinité et du patch minimal
    affs, centers, min_stable_patch = local_affinity_via_min_patch_at(
        image_deformee=image_deformee,
        center_rc=center,
        U_ref=U_ref, V_ref=V_ref,
        start_ps=start_ps, min_ps=min_ps, max_ps=max_ps, step=step,
        tol_abs=tol_abs, tol_rel_pct=tol_rel_pct,
        ref_smooth_window=ref_smooth_window, stable_seq_len=stable_seq_len,
        hex_kwargs=hex_kwargs,
        patch_support_size=None,
        which_pair="all",
        verbose=False,
        show_tracks=False
    )

    if not affs or min_stable_patch is None:
        return None, None

    redr_best, H_used_best = choose_best_affinity_patch(
        min_stable_patch=min_stable_patch,
        affs=affs,
        image_ref=image_ref
    )

    if redr_best is None or H_used_best is None:
        return None, None

    return H_used_best, redr_best


import numpy as np

def cylindrical_jacobian_at_deformed_point(
    center_uvp: Tuple[float, float],
    src_shape,
    f: float,
    R: float,
    num_tiles: int = 1,
    cx_src: float = None,
    cy_src: float = None,
):
    H_src, W_src = src_shape[:2]

    if cx_src is None:
        cx_src = W_src / 2.0
    if cy_src is None:
        cy_src = H_src / 2.0

    # Taille de l'image projetée
    W_out = int(W_src * num_tiles)
    H_out = H_src
    cx_p = W_out / 2.0
    cy_p = H_out / 2.0

    u_p, v_p = center_uvp

    # --- 1) (u_p, v_p) -> coordonnées "caméra" (x, y) ---
    theta = (u_p - cx_p) / R
    h = (v_p - cy_p) / R

    x = np.tan(theta)
    r = np.sqrt(1.0 + x**2)
    y = h * r

    # --- 2) point source (u, v) ---
    u = f * x + cx_src
    v = f * y + cy_src

    # --- 3) Jacobien analytique d(u_p, v_p) / d(u, v) ---
    # x = (u - cx)/f
    # y = (v - cy)/f
    # theta = arctan(x)
    # r = sqrt(1+x^2)
    # h = y / r
    # u_p = R*theta + cx_p
    # v_p = R*h + cy_p

    # On exprime J en fonction de x,y
    denom1 = 1.0 + x**2
    sqrt1 = np.sqrt(denom1)

    du_p_du = R / (f * denom1)           # ∂u_p/∂u
    du_p_dv = 0.0                        # ∂u_p/∂v

    dv_p_du = - R * x * y / (f * denom1**1.5)   # ∂v_p/∂u
    dv_p_dv = R / (f * sqrt1)                   # ∂v_p/∂v

    J = np.array([
        [du_p_du, du_p_dv],
        [dv_p_du, dv_p_dv]
    ], dtype=float)

    return J, (u, v)


import math

def compute_local_errors_vs_cylindrical_jacobian(
    centers_list: np.ndarray,        # shape (N, 2) -> (u_p, v_p) dans proj_gs3d
    src_shape,                       # shape de large_noise
    proj_img: np.ndarray,            # proj_gs3d
    image_ref: np.ndarray,           # large_noise (référence)
    f: float,
    R: float,
    U_ref: Tuple[float, float],
    V_ref: Tuple[float, float],
    start_ps: int,
    min_ps: int,
    max_ps: int,
    step: int,
    tol_abs: float,
    tol_rel_pct: float,
    ref_smooth_window: int,
    stable_seq_len: int,
    hex_kwargs: dict,
):
    H_src, W_src = src_shape[:2]
    Hp, Wp = proj_img.shape[:2]

    centers_rc = []
    H_list = []
    log10_errs = []

    # 1) conversion de centres (x,y) -> (row,col)
    #    centers_list[i] = (u_p, v_p) = (x, y) dans proj_gs3d
    centers_rc_for_process = []
    for (cx, cy) in centers_list:
        r = int(round(cy))
        c = int(round(cx))
        centers_rc_for_process.append((r, c))
        centers_rc.append((r, c))

    # 2) calcul des affinités en parallèle
    results = Parallel(n_jobs=-1)(
        delayed(_process_center)(
            center_rc,
            proj_img,
            image_ref,
            U_ref, V_ref,
            start_ps,
            min_ps,
            max_ps,
            step,
            tol_abs,
            tol_rel_pct,
            ref_smooth_window,
            stable_seq_len,
            hex_kwargs,
        )
        for center_rc in centers_rc_for_process
    )

    # 3) pour chaque centre, comparer H au Jacobien cylindrique
    for idx, ((cx, cy), center_rc, res) in enumerate(zip(centers_list, centers_rc, results)):
        H_used_best, redr_best = res

        if H_used_best is None:
            H_list.append(None)
            log10_errs.append(np.nan)
            continue

        # Jacobien analytique au centre (en coords projetées)
        J_cyl, (u_src, v_src) = cylindrical_jacobian_at_deformed_point(
            center_uvp=(cx, cy),
            src_shape=src_shape,
            f=f,
            R=R,
            num_tiles=1,
            cx_src=W_src/2.0,
            cy_src=H_src/2.0,
        )

        # Embedding 3x3 pour comparer avec H (3x3)
        A = np.eye(3, dtype=float)
        A[:2, :2] = J_cyl

        # Erreur de Frobenius
        err = frobenius_norm(A, np.linalg.inv(H_used_best)) + 1e-12   # éviter log10(0)
        log10_err = math.log10(err)

        H_list.append(np.linalg.inv(H_used_best))
        log10_errs.append(log10_err)

    return centers_rc, log10_errs, H_list





import cv2

def draw_error_patches(
    proj_img: np.ndarray,
    centers_rc: List[Tuple[int, int]],
    log10_errs: List[float],
    max_ps: int,
    alpha: float = 0.3,
    upscale: int = 2,
):
    # 1) base BGR
    if proj_img.ndim == 2:
        base = cv2.cvtColor(proj_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        base = proj_img.astype(np.uint8).copy()

    H, W = base.shape[:2]
    half = max_ps // 2

    # 2) upscaling
    H_up = H * upscale
    W_up = W * upscale

    base_up = cv2.resize(base, (W_up, H_up), interpolation=cv2.INTER_CUBIC)
    overlay_up = base_up.copy()

    GREEN  = (0, 255, 0)
    ORANGE = (0, 165, 255)
    RED    = (0, 0, 255)

    # 3) remplissage des patches
    for (r, c), loge in zip(centers_rc, log10_errs):
        if np.isnan(loge):
            color = RED
        elif loge <= -1.5:
            color = GREEN
        elif loge> -1.5 and loge <= -1:
            color = ORANGE
        else:
            color = RED

        r0 = max(0, r - half)
        r1 = min(H, r + half)
        c0 = max(0, c - half)
        c1 = min(W, c + half)

        # coords upscalées
        r0u = int(r0 * upscale)
        r1u = int(r1 * upscale)
        c0u = int(c0 * upscale)
        c1u = int(c1 * upscale)

        cv2.rectangle(overlay_up, (c0u, r0u), (c1u, r1u), color, thickness=-1)

    # 4) fusion
    out_up = cv2.addWeighted(overlay_up, alpha, base_up, 1 - alpha, 0)

    # 5) contours
    for (r, c), loge in zip(centers_rc, log10_errs):
        if np.isnan(loge):
            color = RED
        elif loge <= -2.0:
            color = GREEN
        elif loge <= -1.5:
            color = ORANGE
        else:
            color = RED

        r0 = max(0, r - half)
        r1 = min(H, r + half)
        c0 = max(0, c - half)
        c1 = min(W, c + half)

        r0u = int(r0 * upscale)
        r1u = int(r1 * upscale)
        c0u = int(c0 * upscale)
        c1u = int(c1 * upscale)

        cv2.rectangle(out_up, (c0u, r0u), (c1u, r1u), color,
                      thickness=2 * upscale, lineType=cv2.LINE_AA)

    # 6) retour à la taille originale
    out = cv2.resize(out_up, (W, H), interpolation=cv2.INTER_AREA)
    return out


from qr_code import qr_chessboard, draw_qr_overlay_on_projection_from_cells,compute_projected_qr_quads_for_chessboard,cylindrical_projection_realistic,detect_all_qr_corners,build_cell_info_from_detections,get_deformed_qr_centers


if __name__ == "__main__":
    rows = 8
    cols = 8
    cell_size = 256
    margin = 40
    f = 600.0
    R = 600.0
    num_tiles = 1

    # 1) chessboard (pour centres)
    board = qr_chessboard(
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        margin=margin,
        per_cell_payload=True,
        payload_prefix="cell",
    )

    # 2) projection cylindrique de l'image de référence de déformation (ghostseal)
    from ghostseal_generator import generate_gs3d_noise_deformation

    large_noise, gs3d1, matrix = generate_gs3d_noise_deformation(
        2048, 2048, (40, 0), (0, 40),
        1, 0, 1, 0, 0, 0, seed=21353456
    )
    large_noise = cv.GaussianBlur(large_noise,(5,5),0)
    proj_gs3d = cylindrical_projection_realistic(
        large_noise,
        f=f,
        R=R,
        num_tiles=num_tiles,
        border_value=255.0,
    ).astype(np.float32)
    # 3) centres des QR déformés (dans le chessboard projeté)
    centers_array, centers_dict = get_deformed_qr_centers(
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        margin=margin,
        f=f,
        R=R,
        src_shape=board.shape,
        num_tiles=1,
    )
    centers_list = np.array(list(centers_dict.values()), dtype=float)

    # 4) paramètres pour l'affinité locale (à adapter à tes valeurs)
    U_ref = (40, 0)
    V_ref = (0, 40)
    # ces variables doivent déjà exister chez toi :
    # start_ps, min_ps, max_ps, step, tol_abs, tol_rel_pct,
    # ref_smooth_window, stable_seq_len, hex_kwargs


    start_ps = 120
    min_ps = 20
    max_ps = 200
    step = 2
    tol_abs = 1
    tol_rel_pct = 1
    ref_smooth_window = 4
    stable_seq_len = 4
    hex_kwargs = dict(
        k=40,
        nms_size=20,
        exclude_center_radius=10.0,
        min_separation=8.0,
        refine_model='tps',
        refine_halfwin=1.5,
        tps_coarse_step=0.25,
        energy_halfwin=1.5,
        min_dist=1.0,
        antipodal_tol=2.0,
        angle_min_deg=10.0,
        w_exclude_center_radius=10.0,
    )
    centers_rc, log10_errs, H_list = compute_local_errors_vs_cylindrical_jacobian(
        centers_list=centers_list,
        src_shape=large_noise.shape,
        proj_img=proj_gs3d,
        image_ref=large_noise,
        f=f,
        R=R,
        U_ref=U_ref,
        V_ref=V_ref,
        start_ps=start_ps,
        min_ps=min_ps,
        max_ps=max_ps,
        step=step,
        tol_abs=tol_abs,
        tol_rel_pct=tol_rel_pct,
        ref_smooth_window=ref_smooth_window,
        stable_seq_len=stable_seq_len,
        hex_kwargs=hex_kwargs,
    )
    proj_gs3d = cylindrical_projection_realistic(
        ghostseal_sk,
        f=f,
        R=R,
        num_tiles=num_tiles,
        border_value=255.0,
    ).astype(np.float32)
    
    # 5) draw des patches
    vis = draw_error_patches(
        proj_img=proj_gs3d,
        centers_rc=centers_rc,
        log10_errs=log10_errs,
        max_ps=max_ps,
        alpha=0.3,
        upscale=2,
    )

    # 6) sauvegarde
    cv2.imwrite("local_affinity_cylindrical_error_map.png", vis)
    print("Image sauvegardée sous local_affinity_cylindrical_error_map.png")
    vpv(vis)

# from qr_code import qr_chessboard, draw_qr_overlay_on_projection_from_cells,compute_projected_qr_quads_for_chessboard,cylindrical_projection_realistic,detect_all_qr_corners,build_cell_info_from_detections,get_deformed_qr_centers

# # ======================
# #  Exemple d'utilisation
# # ======================
# if __name__ == "__main__":
#     rows = 8
#     cols = 8
#     cell_size = 256
#     margin = 40
#     f = 600.0
#     R = 600.0
#     num_tiles = 1

#     # 1) chessboard de QR (ta fonction)
#     board = qr_chessboard(
#         rows=rows,
#         cols=cols,
#         cell_size=cell_size,
#         margin=margin,
#         per_cell_payload=True,
#         payload_prefix="cell",
#     )

#     # 2) projection cylindrique (ta fonction)
#     proj = cylindrical_projection_realistic(
#         board,
#         f=f,
#         R=R,
#         num_tiles=num_tiles,
#         border_value=255.0,
#     ).astype(np.float32)

#     centers_array, centers_dict = get_deformed_qr_centers(
#         rows=rows,
#         cols=cols,
#         cell_size=cell_size,
#         margin=margin,
#         f=f,
#         R=R,
#         src_shape=board.shape,
#         num_tiles=1,
#     )
#     centers_list = np.array(list(centers_dict.values()), dtype=float)

#     print(centers_list)

#     from ghostseal_generator import generate_gs3d_noise_deformation

#     large_noise, gs3d1, matrix = generate_gs3d_noise_deformation(
#     2048, 2048, (60, 0), (0, 60),
#     1, 0, 1, 0, 0, 0, seed=21353456
# )
#     proj_gs3d = cylindrical_projection_realistic(
#         large_noise,
#         f=f,
#         R=R,
#         num_tiles=num_tiles,
#         border_value=255.0,
#     ).astype(np.float32)

#     results = Parallel(n_jobs=-1)(
#     delayed(_process_center)(
#         center,
#         proj_gs3d,
#         large_noise,
#         (60, 0), (0, 60),
#         start_ps,
#         min_ps,
#         max_ps,
#         step,
#         tol_abs,
#         tol_rel_pct,
#         ref_smooth_window,
#         stable_seq_len,
#         hex_kwargs
#     )
#     for center in centers_list
#     )



    



    # # 3) quadrilatères théoriques des QR après déformation
    # cell_quads = compute_projected_qr_quads_for_chessboard(
    #     rows=rows,
    #     cols=cols,
    #     cell_size=cell_size,
    #     margin=margin,
    #     f=f,
    #     R=R,
    #     src_shape=board.shape,
    #     num_tiles=num_tiles,
    # )

    # # 4) détection réelle dans l'image projetée (TA fonction)
    # detection_qr = detect_all_qr_corners(
    #     proj,
    #     rows=rows,
    #     cols=cols,
    #     cell_size=cell_size,
    #     margin=margin,
    # )

    # # 5) construit l'état par cellule (detected / decoded / non détecté)
    # cell_info = build_cell_info_from_detections(
    #     rows=rows,
    #     cols=cols,
    #     cell_quads=cell_quads,
    #     detections=detection_qr,
    # )

    # # debug console
    # print("État par cellule :")
    # for r in range(rows):
    #     for c in range(cols):
    #         info = cell_info[(r, c)]
    #         print(
    #             f"cell({r},{c}) -> detected={info['detected']}, "
    #             f"decoded={info['decoded']}, msg={info['message']}"
    #         )

    # # 6) dessin final dans l'image projetée
    # vis = draw_qr_overlay_on_projection_from_cells(
    #     proj,
    #     cell_info,
    #     alpha=0.3,
    #     upscale=4
    # )

    # cv.imwrite("qr_chessboard_projection_overlay_cells.png", vis)

