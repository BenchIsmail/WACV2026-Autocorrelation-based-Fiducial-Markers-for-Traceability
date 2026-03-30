
import numpy as np
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from ghostseal_generator import generate_white_noise_and_shifts
from deformation_generator import general_deformer
from rectification_energy import local_affinity_via_min_patch_center



def homography_jacobian(H, x, y):
    H = H / H[2, 2]
    h11, h12, h13 = H[0]
    h21, h22, h23 = H[1]
    h31, h32, h33 = H[2]

    denom = (h31 * x + h32 * y + h33)
    denom2 = denom ** 2

    dx_dx = (h11 * denom - h31 * (h11 * x + h12 * y + h13)) / denom2
    dx_dy = (h12 * denom - h32 * (h11 * x + h12 * y + h13)) / denom2
    dy_dx = (h21 * denom - h31 * (h21 * x + h22 * y + h23)) / denom2
    dy_dy = (h22 * denom - h32 * (h21 * x + h22 * y + h23)) / denom2

    return np.array([[dx_dx, dx_dy],
                     [dy_dx, dy_dy]], dtype=float)





def center_error_log10_and_mats(image_def, H_forw, local_aff_list):
    h, w = image_def.shape
    center_x, center_y = w / 2.0, h / 2.0

    H_inv = np.linalg.inv(H_forw)
    J = homography_jacobian(H_inv, center_x, center_y)  

    if local_aff_list is None or len(local_aff_list) == 0:
        return np.nan, None, None, J

    J_lin = J[:2, :2]

    errors = []
    A_lin_list = []
    A_full_list = []

    for A in local_aff_list:
        A = np.asarray(A, float)
        if A.shape[0] == 3:
            A_lin = A[:2, :2]
        elif A.shape[0] == 2:
            A_lin = A[:, :2]
        else:
            raise ValueError(f"Matrice d'affinité invalide : {A.shape}")

        try:
            A_inv = np.linalg.inv(A_lin)
        except np.linalg.LinAlgError:
            A_inv = None

        errs = [np.linalg.norm(A_lin - J_lin, ord='fro')]
        if A_inv is not None:
            errs.append(np.linalg.norm(A_inv - J_lin, ord='fro'))

        e_min = min(errs)
        errors.append(e_min)
        A_lin_list.append(A_lin)
        A_full_list.append(A)

    e = min(errors)
    val = np.log10(e) if np.isfinite(e) and e > 0 else -5.0

    idx_min = np.argmin(errors)
    best_A_lin = A_lin_list[idx_min]
    best_A_full = A_full_list[idx_min]

    return val, best_A_lin, best_A_full, J




import cv2
def frob_error_for_angles(image,
    angle_x, angle_y,
    U_ref=(50, 0), V_ref=(0, 50),
    image_size=800,
    hex_kwargs=None,
    start_ps=100, min_ps=20, max_ps=200, step=2,
    tol_abs=1.0, tol_rel_pct=5.0,
    ref_smooth_window=0, stable_seq_len=5,
    patch_support_size=None,
    which_pair="uv",
    show_tracks = True
):


    img_def,H_forw = general_deformer(image,1, 0, 1, 0, angle_x, angle_y)

    affs, centers = local_affinity_via_min_patch_center(
        image_deformee=img_def,
        U_ref=U_ref, V_ref=V_ref,
        start_ps=start_ps, min_ps=min_ps, max_ps=max_ps, step=step,
        tol_abs=tol_abs, tol_rel_pct=tol_rel_pct,
        ref_smooth_window=ref_smooth_window, stable_seq_len=stable_seq_len,
        hex_kwargs=hex_kwargs,
        patch_support_size=patch_support_size,
        which_pair=which_pair,
        verbose=False,
        show_tracks = show_tracks
        
    )

    local_aff = affs if affs else None
    logerr, A_lin, A_full, J = center_error_log10_and_mats(img_def, H_forw, local_aff)
    return logerr, A_lin, A_full, J, H_forw




def _compute_one_point(args):
    (ix, iy, ax, ay, image, kwargs) = args

    logerr, A_lin, A_full, J, H_forw = frob_error_for_angles(
        image,
        angle_x=ax, angle_y=ay,
        U_ref=kwargs["U_ref"], V_ref=kwargs["V_ref"],
        image_size=kwargs["image_size"],
        hex_kwargs=kwargs["hex_kwargs"],
        start_ps=kwargs["start_ps"], min_ps=kwargs["min_ps"], max_ps=kwargs["max_ps"], step=kwargs["step"],
        tol_abs=kwargs["tol_abs"], tol_rel_pct=kwargs["tol_rel_pct"],
        ref_smooth_window=kwargs["ref_smooth_window"], stable_seq_len=kwargs["stable_seq_len"],
        patch_support_size=kwargs["patch_support_size"],
        which_pair=kwargs["which_pair"],
        show_tracks=kwargs["show_tracks"],  
    )

    logerr_num = None if (logerr is None or not np.isfinite(logerr)) else float(logerr)

    rec = {
        "angle_x_deg": float(ax),
        "angle_y_deg": float(ay),
        "log10_fro_error": (None if logerr is None else float(logerr)),
        "A_est_lin_2x2": (A_lin.tolist() if A_lin is not None else None),
        "A_est_full": (A_full.tolist() if A_full is not None else None),
        "J_true_2x2": J.tolist(),
        "H_forward_3x3": H_forw.tolist(),
        "params": kwargs
    }
    return ix, iy, logerr_num, rec


def heatmap_affinity_error_parallel(
    U_ref=(50, 0), V_ref=(0, 50),
    image_size=800,
    ax_list=np.linspace(0, 15, 31),
    ay_list=np.linspace(0, 15, 31),
    hex_kwargs=None,
    start_ps=100, min_ps=30, max_ps=150, step=2,
    tol_abs=1.0, tol_rel_pct=5.0,
    ref_smooth_window=0, stable_seq_len=5,
    patch_support_size=None,
    which_pair="uv",
    show=True,
    json_out_path="affinity_error_grid.json",
    show_tracks=False,   
    max_workers=None,    
    chunksize=1          
):
    E = np.full((len(ay_list), len(ax_list)), np.nan, dtype=float)
    records = []

    image = generate_white_noise_and_shifts(image_size, image_size, U_ref, V_ref, seed=None)

    params = {
        "U_ref": list(U_ref),
        "V_ref": list(V_ref),
        "image_size": int(image_size),
        "start_ps": int(start_ps),
        "min_ps": int(min_ps),
        "max_ps": int(max_ps),
        "step": int(step),
        "tol_abs": float(tol_abs),
        "tol_rel_pct": float(tol_rel_pct),
        "ref_smooth_window": int(ref_smooth_window),
        "stable_seq_len": int(stable_seq_len),
        "patch_support_size": (None if patch_support_size is None else int(patch_support_size)),
        "which_pair": str(which_pair),
        "hex_kwargs": hex_kwargs,
        "show_tracks": bool(show_tracks),
    }

    tasks = []
    for iy, ay in enumerate(ay_list):
        for ix, ax in enumerate(ax_list):
            tasks.append((ix, iy, float(ax), float(ay), image, params))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_compute_one_point, t) for t in tasks]
        for fut in as_completed(futures):
            ix, iy, logerr_num, rec = fut.result()
            E[iy, ix] = (np.nan if logerr_num is None else logerr_num)
            records.append(rec)
            print(f"X={rec['angle_x_deg']:.1f}°, Y={rec['angle_y_deg']:.1f}° -> log10(err)={logerr_num}")

    out = {
        "grid": {
            "ax_list_deg": list(map(float, ax_list)),
            "ay_list_deg": list(map(float, ay_list)),
            "log10_fro_error_matrix": E.tolist()
        },
        "results": records
    }
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[OK] Résultats sauvegardés dans: {json_out_path}")

    if show:
        plt.figure(figsize=(10, 10))
        im = plt.imshow(
            E, origin='lower', aspect='auto',
            extent=[ax_list.min(), ax_list.max(), ay_list.min(), ay_list.max()],
            cmap='viridis', vmin=-3, vmax=-1
        )
        plt.colorbar(im, label='Log10(Frobenius Error)')
        plt.title('Affinity error (Frobenius) vs. X/Y inclination (center-only)')
        plt.xlabel('Inclination X (°)')
        plt.ylabel('Inclination Y (°)')
        plt.tight_layout()
        plt.savefig('affinity_error_grid.png')

    return E, ax_list, ay_list



