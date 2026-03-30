import numpy as np
import operations as oprt
import optimization_better as optmiz

from rectification_energy import local_affinities_via_min_patch_random

from typing import Optional

def rectification(
    reference_image,
    deformed_image,
    iteration,
    U_ref,
    V_ref,
    num_centers,
    rng_seed,
    border_margin,

    min_ps,
    max_ps,
    step,
    stable_seq_len,
    stable_tol,

    hex_kwargs: Optional[dict] = None,

    use_cond_hard_filter: bool = True,
    cond_max: float = 100.0,
    use_cond_robust_filter: bool = True,
    k_mad_cond: float = 3.0,
    use_A_robust_filter: bool = True,
    k_mad_A: float = 3.0,
):
    current_image = deformed_image.copy()
    total_h_estim = np.eye(3, dtype=float)

    for it in range(int(iteration)):
        print(f"--- Iteration {it+1} ---")

        local_affinities, patch_centers = local_affinities_via_min_patch_random(
            image_deformee=current_image,
            U_ref=U_ref,                 # (dc,dr)
            V_ref=V_ref,                 # (dc,dr)
            num_centers=num_centers,
            rng_seed=rng_seed,
            border_margin=border_margin,

            min_ps=min_ps,
            max_ps=max_ps,
            step=step,
            stable_seq_len=stable_seq_len,
            stable_tol=stable_tol,

            hex_kwargs=hex_kwargs,
            patch_support_size=None,
            verbose=False,
            n_jobs=-1,
        )

        n_aff = len(local_affinities)
        n_ctr = len(patch_centers)

        if n_aff < 3 or n_ctr < 3:
            print("[rectification] Pas assez d'affinités/centres, arrêt.")
            break

        A_candidates = []
        b_candidates = []
        cond_candidates = []

        n_centers_avail = min(n_aff, n_ctr)
        for k in range(n_centers_avail):
            aff = local_affinities[k]
            r, c = patch_centers[k]  # (row, col)

            M = np.asarray(aff, float)
            if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] < 2:
                continue

            A = M[:2, :2]
            if not np.isfinite(A).all():
                continue

            try:
                condA = np.linalg.cond(A)
            except Exception:
                continue

            if use_cond_hard_filter and condA > cond_max:
                continue

            A_candidates.append(A)
            b_candidates.append(np.array([float(c), float(r)], dtype=float))  # (x,y)
            cond_candidates.append(float(condA))

        if len(A_candidates) < 3:
            print("[rectification] Moins de 3 affinités valides après cond, arrêt.")
            break

        cond_candidates = np.array(cond_candidates, dtype=float)

        if use_cond_robust_filter and len(cond_candidates) >= 3:
            logc = np.log10(cond_candidates + 1e-15)
            med_c = np.median(logc)
            mad_c = np.median(np.abs(logc - med_c)) + 1e-9
            mask_cond = np.abs(logc - med_c) <= k_mad_cond * mad_c

            A_tmp = [A for A, m in zip(A_candidates, mask_cond) if m]
            b_tmp = [b for b, m in zip(b_candidates, mask_cond) if m]

            print(f"[rectification] après cond-robust-filter: {len(A_tmp)} / {len(A_candidates)}")

            if len(A_tmp) < 3:
                print("[rectification] trop peu après cond-robust-filter -> garde tout.")
                A_tmp = A_candidates
                b_tmp = b_candidates
        else:
            A_tmp = A_candidates
            b_tmp = b_candidates

        if use_A_robust_filter and len(A_tmp) >= 3:
            A_stack = np.array([A.flatten() for A in A_tmp], dtype=float)  # (N,4)
            med_A = np.median(A_stack, axis=0)
            mad_A = np.median(np.abs(A_stack - med_A), axis=0) + 1e-9
            diff = np.abs(A_stack - med_A)
            mask_A = (diff <= k_mad_A * mad_A).all(axis=1)

            A_filtered = [A for A, m in zip(A_tmp, mask_A) if m]
            b_filtered = [b for b, m in zip(b_tmp, mask_A) if m]

            print(f"[rectification] après A-robust-filter: {len(A_filtered)} / {len(A_tmp)}")

            if len(A_filtered) < 3:
                print("[rectification] trop peu après A-robust-filter -> garde A_tmp.")
                A_filtered = A_tmp
                b_filtered = b_tmp
        else:
            A_filtered = A_tmp
            b_filtered = b_tmp

        if len(A_filtered) < 3:
            print("[rectification] Toujours moins de 3 affinités, arrêt.")
            break

        H_raw, cost = optmiz.optimize(A_filtered, b_filtered)
        H_estim = np.vstack((H_raw.T, np.array([[0, 0, 1]]))).T

        if abs(H_estim[2, 2]) > 1e-12:
            H_estim /= H_estim[2, 2]

        current_image = oprt.apply_homography(
            current_image,
            oprt.image_centralizer(reference_image, H_estim),
            reference_image.shape,
            borderValue=int(np.mean(current_image)),
        )
        total_h_estim = H_estim @ total_h_estim

    if abs(total_h_estim[2, 2]) > 1e-12:
        total_h_estim /= total_h_estim[2, 2]

    final_image = oprt.apply_homography(
        deformed_image,
        oprt.image_centralizer(reference_image, total_h_estim),
        reference_image.shape,
        borderValue=int(np.mean(deformed_image)),
    )
    return final_image, total_h_estim


