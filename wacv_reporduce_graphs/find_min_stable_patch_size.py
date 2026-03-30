import numpy as np
import matplotlib.pyplot as plt

from autocorrelation import autocorrelation_display
from subpixel_energy_maximiz import find_hexagon


def _euclid(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _closest_with_sign(cands, ref):
    ref = np.asarray(ref, float).reshape(2,)
    best = (None, +1, -1, float("inf"))
    for i, v in enumerate(cands):
        v = np.asarray(v, float).reshape(2,)
        d_pos = np.hypot(*(v - ref))
        d_neg = np.hypot(*((-v) - ref))
        if d_pos <= d_neg:
            d, s = d_pos, +1
        else:
            d, s = d_neg, -1
        if d < best[3]:
            best = (s * v, s, i, d)
    return best


def _order_u_v_w_by_refs(u_fin, v_fin, w_fin, U_ref_dcdr, V_ref_dcdr):
    if U_ref_dcdr is None or V_ref_dcdr is None:
        return u_fin, v_fin, w_fin, dict(perm=("u", "v", "w"), signs=(+1, +1, +1))

    U_ref_drdc = np.array([U_ref_dcdr[1], U_ref_dcdr[0]], float)
    V_ref_drdc = np.array([V_ref_dcdr[1], V_ref_dcdr[0]], float)

    cands = [np.asarray(u_fin, float), np.asarray(v_fin, float), np.asarray(w_fin, float)]

    u_sel, sU, idxU, _ = _closest_with_sign(cands, U_ref_drdc)

    rem_idx = [i for i in (0, 1, 2) if i != idxU]
    cands_V = [cands[i] for i in rem_idx]
    v_sel, sV, local_idxV, _ = _closest_with_sign(cands_V, V_ref_drdc)
    idxV = rem_idx[local_idxV]

    idxW = [i for i in (0, 1, 2) if i not in (idxU, idxV)][0]
    w_raw = cands[idxW]
    target = u_sel - v_sel
    d_pos = np.hypot(*((w_raw) - target))
    d_neg = np.hypot(*((-w_raw) - target))
    if d_pos <= d_neg:
        w_sel, sW = w_raw, +1
    else:
        w_sel, sW = -w_raw, -1

    names = ["u", "v", "w"]
    perm = (names[idxU], names[idxV], names[idxW])
    signs = (sU, sV, sW)

    return u_sel, v_sel, w_sel, dict(perm=perm, signs=signs)


def _extract_uv_fin(image, ps, autocorr_fn, find_fn, search_kwargs=None):
    if search_kwargs is None:
        search_kwargs = {}

    H, W = image.shape
    cy, cx = H // 2, W // 2
    half = ps // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None, None, None

    patch = image[y0:y1, x0:x1]
    R = autocorr_fn(patch)
    res = find_fn(R, **search_kwargs)

    if res is None or "u_fin" not in res or "v_fin" not in res or "w_fin" not in res:
        return None, None, None

    u_fin = np.asarray(res["u_fin"], float)
    v_fin = np.asarray(res["v_fin"], float)
    w_fin = np.asarray(res["w_fin"], float)
    return u_fin, v_fin, w_fin


def _extract_center_patch(img, ps, pad_mode="reflect"):
    h, w = img.shape[:2]
    rc, cc = h // 2, w // 2
    r0, r1 = rc - ps // 2, rc - ps // 2 + ps
    c0, c1 = cc - ps // 2, cc - ps // 2 + ps

    pad_top = max(0, -r0)
    pad_left = max(0, -c0)
    pad_bottom = max(0, r1 - h)
    pad_right = max(0, c1 - w)

    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
        if img.ndim == 2:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        else:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        img = np.pad(img, pad_width, mode=pad_mode)
        r0 += pad_top
        r1 += pad_top
        c0 += pad_left
        c1 += pad_left

    patch = img[r0:r1, c0:c1]
    if patch.shape[0] != ps or patch.shape[1] != ps:
        patch = patch[:ps, :ps] if img.ndim == 2 else patch[:ps, :ps, ...]
    return patch




def find_min_stable_patch_size_centered(
    image,
    autocorr_fn=autocorrelation_display,
    find_fn=find_hexagon,
    min_ps=40,
    max_ps=500,
    step=2,
    stable_seq_len=3,
    stable_tol=0.75,          
    integrated=True,
    d=2,
    R_int=1.5,
    lam=1.0,
    search_kwargs=None,
    U_ref=None,
    V_ref=None,
    return_patch=True,
    patch_pad_mode="reflect",
    show_tracks=False,
    store_tracks=None,
    marker_size=25,
    colors=None,
    debug=False,
):
    from collections import deque

    if colors is None:
        colors = dict(
            U_pos="#1f77b4",
            U_neg="#aec7e8",
            V_pos="#ff7f0e",
            V_neg="#ffbb78",
            W_pos="#2ca02c",
            W_neg="#98df8a",
        )

    if store_tracks is None:
        store_tracks = bool(show_tracks)

    min_ps = int(min_ps)
    max_ps = int(max_ps)
    step = int(step)
    stable_seq_len = int(stable_seq_len)

    if step <= 0:
        raise ValueError("step doit être > 0")
    if max_ps < min_ps:
        raise ValueError("max_ps doit être >= min_ps")
    if stable_seq_len < 2:
        stable_seq_len = 2

    stable_tol = float(stable_tol)
    if stable_tol <= 0:
        stable_tol = 0.75

    sk = {} if search_kwargs is None else dict(search_kwargs)
    sk.update(dict(integrated=integrated, d=d, R_int=R_int, lam=lam))

    def _euclid2(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return float(np.linalg.norm(a - b))

    def _hex_dist_max(pts6_a: np.ndarray, pts6_b: np.ndarray) -> float: #d(h1,h2)=max_i ||Δpic_i||_2
        dists = np.linalg.norm(pts6_a - pts6_b, axis=1)  # (6,)
        return float(np.max(dists))

    def _build_pts6(u, v, w):
        if (U_ref is not None) and (V_ref is not None):
            uo, vo, wo, _ = _order_u_v_w_by_refs(u, v, w, U_ref, V_ref)
        else:
            uo, vo, wo = u, v, w

        return np.array(
            [
                [uo[1], uo[0]],    # +U
                [vo[1], vo[0]],    # +V
                [-wo[1], -wo[0]],  # -W
                [-uo[1], -uo[0]],  # -U
                [-vo[1], -vo[0]],  # -V
                [wo[1], wo[0]],    # +W
            ],
            dtype=float,
        )

    measures = []
    ordered6_per_ps = {} if store_tracks else {}
    ordered6_min = None

    last_pts6 = None
    run_len = 0
    min_stable_ps = None

    prev_uo = prev_vo = prev_wo = None  

    # fenêtre stable stocke la moyenne (u_avg,v_avg,w_avg)
    stable_window = deque(maxlen=stable_seq_len)
    median_u = median_v = median_w = None
    mid_ps = None

    for ps in range(min_ps, max_ps + 1, step):
        u, v, w = _extract_uv_fin(image, ps, autocorr_fn, find_fn, sk)

        if u is None or v is None or w is None:
            measures.append(dict(ps=ps, ok=False))
            last_pts6 = None
            run_len = 0
            prev_uo = prev_vo = prev_wo = None
            stable_window.clear()
            continue

        # ordonner de façon cohérente (u,v,w en dr,dc)
        if (U_ref is not None) and (V_ref is not None):
            uo, vo, wo, _ = _order_u_v_w_by_refs(u, v, w, U_ref, V_ref)
        else:
            uo, vo, wo = u, v, w

        if prev_uo is None:
            measures.append(
                dict(ps=ps, ok=True, du=0.0, dv=0.0, dw=0.0,
                     stable=False, pair_mean=False, dhex=None, tol=stable_tol)
            )
            prev_uo, prev_vo, prev_wo = uo, vo, wo
            last_pts6 = None
            run_len = 0
            stable_window.clear()
            continue

        # moyenne entre (ps-step) et ps
        u_avg = 0.5 * (prev_uo + uo)
        v_avg = 0.5 * (prev_vo + vo)
        w_avg = 0.5 * (prev_wo + wo)

        pts6_avg = _build_pts6(u_avg, v_avg, w_avg)

        if store_tracks:
            ordered6_per_ps[ps] = pts6_avg  

        # stabilité = comparaison directe des 6 pics de l'hexagone moyen
        if last_pts6 is None:
            stable = False
            run_len = 1
            dhex = None
            stable_window.clear()
            stable_window.append(dict(ps=ps, u=u_avg, v=v_avg, w=w_avg))
        else:
            dhex = _hex_dist_max(pts6_avg, last_pts6)
            stable = (dhex <= stable_tol)
            run_len = (run_len + 1) if stable else 1

            if stable:
                stable_window.append(dict(ps=ps, u=u_avg, v=v_avg, w=w_avg))
            else:
                stable_window.clear()
                stable_window.append(dict(ps=ps, u=u_avg, v=v_avg, w=w_avg))

        # mesures de déplacement 
        du = _euclid2(uo, prev_uo)
        dv = _euclid2(vo, prev_vo)
        dw = _euclid2(wo, prev_wo)

        measures.append(dict(
            ps=ps, ok=True, du=du, dv=dv, dw=dw,
            stable=stable, pair_mean=True, dhex=dhex, tol=stable_tol
        ))

        if debug:
            print(f"ps={ps:4d} pair_mean stable={stable} run={run_len}/{stable_seq_len} dhex={dhex}")

        # update précédent = courant
        prev_uo, prev_vo, prev_wo = uo, vo, wo
        last_pts6 = pts6_avg

        if run_len >= stable_seq_len:
            # à ps, on a stable_seq_len hexagones moyens stables,
            # la première moyenne stable correspond à ps - (stable_seq_len-1)*step.
            # la plus petite taille de patch impliquée est donc 1 pas avant : -step.
            # min_stable_ps = ps - stable_seq_len*step
            min_stable_ps = ps - stable_seq_len * step

            U_arr = np.array([e["u"] for e in stable_window], dtype=float)
            V_arr = np.array([e["v"] for e in stable_window], dtype=float)
            W_arr = np.array([e["w"] for e in stable_window], dtype=float)

            median_u = np.median(U_arr, axis=0)
            median_v = np.median(V_arr, axis=0)
            median_w = np.median(W_arr, axis=0)

            mid_ps = int(min_stable_ps + (stable_seq_len // 2) * step)

            ordered6_min = _build_pts6(median_u, median_v, median_w)
            break

    # si rien trouvé: on choisit le ps avec plus petit dhex 
    if min_stable_ps is None:
        best_val = float("inf")
        best_ps = None
        for m in measures:
            if m.get("ok", False) and m.get("pair_mean", False) and (m.get("dhex", None) is not None):
                if m["dhex"] < best_val:
                    best_val = m["dhex"]
                    best_ps = m["ps"]

        # best_ps ici = ps où la moyenne (ps-step, ps) est "la plus stable"
        # on remonte à la plus petite taille impliquée ~ best_ps - step
        if best_ps is not None:
            min_stable_ps = int(best_ps - step)
            mid_ps = int(min_stable_ps)

            # recalcul cohérent: on reconstruit la moyenne sur (mid_ps, mid_ps+step) 
            uA, vA, wA = _extract_uv_fin(image, int(min_stable_ps), autocorr_fn, find_fn, sk)
            uB, vB, wB = _extract_uv_fin(image, int(min_stable_ps + step), autocorr_fn, find_fn, sk)

            if (uA is not None) and (uB is not None) and (vA is not None) and (vB is not None) and (wA is not None) and (wB is not None):
                if (U_ref is not None) and (V_ref is not None):
                    uAo, vAo, wAo, _ = _order_u_v_w_by_refs(uA, vA, wA, U_ref, V_ref)
                    uBo, vBo, wBo, _ = _order_u_v_w_by_refs(uB, vB, wB, U_ref, V_ref)
                else:
                    uAo, vAo, wAo = uA, vA, wA
                    uBo, vBo, wBo = uB, vB, wB

                median_u = 0.5 * (uAo + uBo)
                median_v = 0.5 * (vAo + vBo)
                median_w = 0.5 * (wAo + wBo)
                ordered6_min = _build_pts6(median_u, median_v, median_w)

    u_ord = v_ord = w_ord = None
    if median_u is not None:
        u_ord, v_ord, w_ord = median_u, median_v, median_w

    min_patch = None
    if return_patch and (mid_ps is not None):
        min_patch = _extract_center_patch(image, int(mid_ps), pad_mode=patch_pad_mode)

    if show_tracks:
        if not store_tracks:
            raise ValueError("show_tracks=True nécessite store_tracks=True (ou store_tracks=None).")
        if len(ordered6_per_ps) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title("Trajectoires multiscales")
            ax.set_xlabel("dc (col offset)")
            ax.set_ylabel("dr (row offset)")
            ax.set_zlabel("Patch size (px)")
            ax.invert_yaxis()

            cols_seq = [
                colors.get("U_pos", "#1f77b4"),
                colors.get("V_pos", "#ff7f0e"),
                colors.get("W_neg", "#98df8a"),
                colors.get("U_neg", "#aec7e8"),
                colors.get("V_neg", "#ffbb78"),
                colors.get("W_pos", "#2ca02c"),
            ]

            for ps_k, pts6_k in ordered6_per_ps.items():
                z = np.full(6, ps_k, dtype=float)
                for i in range(6):
                    ax.scatter(pts6_k[i, 0], pts6_k[i, 1], z[i], s=marker_size, color=cols_seq[i], alpha=0.95)

            if ordered6_min is not None and (min_stable_ps is not None):
                pts6_min = ordered6_min
                closed = np.vstack([pts6_min, pts6_min[0]])
                z = np.full(closed.shape[0], float(min_stable_ps))
                ax.plot(closed[:, 0], closed[:, 1], z, color="black", linewidth=2, label=f"min patch = {min_stable_ps}")

            labels = ["+U", "+V", "-W", "-U", "-V", "+W"]
            for col, lab in zip(cols_seq, labels):
                ax.scatter([], [], [], color=col, label=lab)
            if min_stable_ps is not None:
                ax.plot([], [], [], color="black", linewidth=2, label="Hexagone min patch")
            ax.legend(loc="best")
            plt.tight_layout()
            plt.show()

    return dict(
        min_stable_ps=min_stable_ps,
        mid_ps=mid_ps,
        final_u=None if u_ord is None else tuple(u_ord),  
        final_v=None if v_ord is None else tuple(v_ord),
        final_w=None if w_ord is None else tuple(w_ord),
        integrated=integrated,
        R_int=R_int,
        d=d,
        lam=lam,
        history=measures,
        ordered6_per_ps=(ordered6_per_ps if store_tracks else {}),
        ordered6_min=ordered6_min,
        params=dict(
            stable_seq_len=stable_seq_len,
            step=step,
            min_ps=min_ps,
            max_ps=max_ps,
            stable_tol=stable_tol,
            integrated=integrated,
            d=d,
            R_int=R_int,
            lam=lam,
            store_tracks=bool(store_tracks)
        ),
        min_patch=min_patch,
    )
