# code_parallel_precis.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import json

from concurrent.futures import ProcessPoolExecutor

from ghostseal_generator import generate_white_noise_and_shifts
from deformation_generator import general_deformer
from find_min_stable_patch_size import find_min_stable_patch_size_centered



def deform_vector_by_homography_dr_dc(H, vec_dr_dc, origin_xy):
    vec_dr_dc = np.asarray(vec_dr_dc, float)

    # (dr,dc) -> (dx,dy) = (col,row)
    v_xy = np.array([vec_dr_dc[1], vec_dr_dc[0]], dtype=float)

    x0, y0 = origin_xy
    p0 = np.array([x0,         y0,         1.0], dtype=float)
    p1 = np.array([x0+v_xy[0], y0+v_xy[1], 1.0], dtype=float)

    Hp0 = H @ p0
    Hp1 = H @ p1
    Hp0 /= Hp0[2]
    Hp1 /= Hp1[2]

    v_def_xy = Hp1[:2] - Hp0[:2]                  # (dx, dy)
    v_def_dr_dc = np.array([v_def_xy[1], v_def_xy[0]], dtype=float)  # -> (dr, dc)
    return v_def_dr_dc



def best_zero_symmetry_error(fu, fv, fw, U_th, V_th, W_th):
    extr = np.vstack([fu, fv, fw])        # (3,2)
    theo = np.vstack([U_th, V_th, W_th])  # (3,2)

    best_err = np.inf
    best_detail = None

    for perm in itertools.permutations(range(3)):           # permutations U/V/W
        theo_perm = theo[list(perm)]                        # réordonné
        for signs in itertools.product([1, -1], repeat=3):  # signes ±
            signs_arr = np.array(signs, dtype=float)[:, None]  # (3,1)
            theo_signed = theo_perm * signs_arr             # (3,2)
            errs = np.linalg.norm(extr - theo_signed, axis=1)
            mean_err = float(np.mean(errs))
            if mean_err < best_err:
                best_err = mean_err
                best_detail = (perm, signs, errs)

    return best_err, best_detail



def _worker_one_cell(args):
    (ix, iy, tilt, ori,
     image, center_xy, U_ref, V_ref,
     find_kwargs, search_kwargs_local) = args

    # Déformation
    img_def, H_forw = general_deformer(
        image,
        1, 0,          
        tilt,
        ori,
        0, 0
    )

    # Extraction
    res = find_min_stable_patch_size_centered(
        img_def,
        U_ref=U_ref,
        V_ref=V_ref,
        show_tracks=False,
        search_kwargs=search_kwargs_local,
        **find_kwargs
    )

    fu = res.get("final_u", None)
    fv = res.get("final_v", None)
    fw = res.get("final_w", None)

    # Échec extraction
    if (fu is None) or (fv is None) or (fw is None):
        return {
            "ix": ix, "iy": iy,
            "tilt": float(tilt), "ori": float(ori),
            "ok": False,
            "fu": fu, "fv": fv, "fw": fw
        }

    fu_arr = np.asarray(fu, float)
    fv_arr = np.asarray(fv, float)
    fw_arr = np.asarray(fw, float)

    U_th = deform_vector_by_homography_dr_dc(H_forw, U_ref, center_xy)
    V_th = deform_vector_by_homography_dr_dc(H_forw, V_ref, center_xy)
    W_ref = (U_ref[0] - V_ref[0], U_ref[1] - V_ref[1])
    W_th = deform_vector_by_homography_dr_dc(H_forw, W_ref, center_xy)
    

    # Meilleure correspondance
    best_err, detail = best_zero_symmetry_error(fu_arr, fv_arr, fw_arr, U_th, V_th, W_th)
    perm, signs, indiv_errs = detail

    return {
        "ix": ix, "iy": iy,
        "tilt": float(tilt), "ori": float(ori),
        "ok": True,
        "fu": (float(fu_arr[0]), float(fu_arr[1])),
        "fv": (float(fv_arr[0]), float(fv_arr[1])),
        "fw": (float(fw_arr[0]), float(fw_arr[1])),
        "U_th": U_th.astype(float).tolist(),
        "V_th": V_th.astype(float).tolist(),
        "W_th": W_th.astype(float).tolist(),
        "best_err": float(best_err),
        "perm": perm,
        "signs": signs,
        "indiv_errs": indiv_errs.astype(float).tolist(),
    }



def compute_heatmap_new_method_parallel(
    U_ref=(50, 0),
    V_ref=(0, 50),
    image_size=600,
    tilts=np.linspace(0.9, 1.0, 2),
    orientations=np.linspace(10, 180, 2),
    search_kwargs=None,
    max_workers=None,
    chunksize=1,
    **find_kwargs
):
    # Image de référence
    image = generate_white_noise_and_shifts(image_size, image_size, U_ref, V_ref, seed=None)

    tilts = np.array(list(tilts), float)
    orientations = np.array(list(orientations), float)

    Hmap = np.full((len(orientations), len(tilts)), np.nan, dtype=float)
    h, w = image.shape[:2]
    center_xy = (w / 2.0, h / 2.0)

    base_search_kwargs = dict(search_kwargs) if search_kwargs is not None else {}

    tasks = []
    for iy, ori in enumerate(orientations):
        for ix, tilt in enumerate(tilts):
            sk_local = dict(base_search_kwargs)  
            tasks.append((
                ix, iy, float(tilt), float(ori),
                image, center_xy, tuple(U_ref), tuple(V_ref),
                dict(find_kwargs), sk_local
            ))

    results_grid = [[None for _ in range(len(tilts))] for __ in range(len(orientations))]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for out in ex.map(_worker_one_cell, tasks, chunksize=chunksize):
            results_grid[out["iy"]][out["ix"]] = out
            print(f"tilt={out["tilt"]:.1f}, ori={out["ori"]:.1f}° -> log10(err)={out["best_err"]}")

    for iy, ori in enumerate(orientations):
        for ix, tilt in enumerate(tilts):
            out = results_grid[iy][ix]

            print("fu", out["fu"])
            print("fv", out["fv"])
            print("fw", out["fw"])

            if not out["ok"]:
                Hmap[iy, ix] = np.nan
                continue

            print("U_th", np.array(out["U_th"]))
            print("V_th", np.array(out["V_th"]))
            print("W_th", np.array(out["W_th"]))

            print(
                f"  meilleure correspondance perm={out['perm']}, signes={out['signs']}, "
                f"err_individuelles={np.array(out['indiv_errs'])}"
            )

            Hmap[iy, ix] = out["best_err"]
            print(f"tilt={tilt:.3f}, ori={ori:.1f}° → error={Hmap[iy,ix]:.4f}")

    return Hmap, tilts, orientations



if __name__ == "__main__":
    integrated = True
    d = 2
    R_int = 1.5
    lam = 1.0

    find_kwargs = dict(
        start_ps=300,
        min_ps=20,
        max_ps=400,
        step=4,
        tol_abs=0.5,
        tol_rel_pct=1.0,
        ref_smooth_window=4,
        stable_seq_len=4
    )

    search_kwargs = dict(
        k=20,
        nms_size=20,
        exclude_center_radius=9.0,
        min_separation=9.0,
        refine_model='tps',
        refine_halfwin=3,
        tps_coarse_step=0.25,
        energy_halfwin=1.5,
        min_dist=20.0,
        antipodal_tol=1.0,
        angle_min_deg=1.0,
        w_exclude_center_radius=9.0,
        integrated=integrated, d=d, R_int=R_int, lam=lam
    )

    Hmap, tilts, orientations = compute_heatmap_new_method_parallel(
        U_ref=(150, 0),
        V_ref=(0, 150),
        image_size=2000,
        tilts=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        orientations=range(0, 181, 10),
        search_kwargs=search_kwargs,
        max_workers=None,   
        chunksize=1,
        **find_kwargs
    )


    results = {
        "tilts": [float(t) for t in tilts],
        "orientations": [float(o) for o in orientations],
        "Hmap": Hmap.tolist(),
        "U_ref": list(map(float, (150, 0))),
        "V_ref": list(map(float, (0, 150))),
        "find_params": find_kwargs,
    }

    json_out = "extraction_error_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Résultats sauvegardés dans {json_out}")


    plt.figure(figsize=(10, 8))
    plt.title("Extraction Error Heatmap – Nouvelle méthode (avec symétries)")
    plt.xlabel("Tilt")
    plt.ylabel("Orientation (°)")

    plt.imshow(
        Hmap,
        origin="lower",
        aspect="auto",
        extent=[tilts.min(), tilts.max(), orientations.min(), orientations.max()],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0
    )

    cbar = plt.colorbar()
    cbar.set_label("Mean Extraction Error (px)")

    plt.tight_layout()
    png_out = "extraction_error_heatmap.png"
    plt.savefig(png_out, dpi=300)
    print(f"[OK] Heatmap sauvegardée dans {png_out}")

    plt.show()
