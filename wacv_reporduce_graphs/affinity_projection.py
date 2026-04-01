import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift
from find_min_stable_patch_size import find_min_stable_patch_size_centered



def _extract_found_stable_peak_pixels(
    ac,
    stable_kwargs=None,
    U_ref_dcdr=None,
    V_ref_dcdr=None,
):
    if stable_kwargs is None:
        stable_kwargs = {}

    res = find_min_stable_patch_size_centered(
        image=ac,
        U_ref=U_ref_dcdr,
        V_ref=V_ref_dcdr,
        return_patch=True,
        **stable_kwargs
    )

    if res is None:
        return None

    u = res.get("final_u", None)
    v = res.get("final_v", None)
    w = res.get("final_w", None)

    if u is None or v is None or w is None:
        return None

    u = np.asarray(u, dtype=float).reshape(2,)
    v = np.asarray(v, dtype=float).reshape(2,)
    w = np.asarray(w, dtype=float).reshape(2,)

    h_ac, w_ac = ac.shape[:2]
    cx_ac = (w_ac - 1) / 2.0
    cy_ac = (h_ac - 1) / 2.0

    peaks_found = {
        "F+U": np.array([cx_ac + u[1], cy_ac + u[0]], dtype=float),
        "F-U": np.array([cx_ac - u[1], cy_ac - u[0]], dtype=float),
        "F+V": np.array([cx_ac + v[1], cy_ac + v[0]], dtype=float),
        "F-V": np.array([cx_ac - v[1], cy_ac - v[0]], dtype=float),
        "F+W": np.array([cx_ac + w[1], cy_ac + w[0]], dtype=float),
        "F-W": np.array([cx_ac - w[1], cy_ac - w[0]], dtype=float),
    }

    return {
        "raw": res,
        "u_fin": u,
        "v_fin": v,
        "w_fin": w,
        "peak_pixels": peaks_found,
    }


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero vector cannot be normalized")
    return v / n


def matrix_affinity_generator(lambdaa, theta, tilde, phi, translation):
    tx, ty = translation
    affinity = (
        np.array([[lambdaa, 0.0], [0.0, lambdaa]], dtype=float)
        @ np.array(
            [
                [np.cos(-theta * np.pi / 180.0), -np.sin(-theta * np.pi / 180.0)],
                [np.sin(-theta * np.pi / 180.0), np.cos(-theta * np.pi / 180.0)],
            ],
            dtype=float,
        )
        @ np.array([[tilde, 0.0], [0.0, 1.0]], dtype=float)
        @ np.array(
            [
                [np.cos(theta * np.pi / 180.0), -np.sin(theta * np.pi / 180.0)],
                [np.sin(theta * np.pi / 180.0), np.cos(theta * np.pi / 180.0)],
            ],
            dtype=float,
        )
        @ np.array(
            [
                [np.cos(phi * np.pi / 180.0), -np.sin(phi * np.pi / 180.0)],
                [np.sin(phi * np.pi / 180.0), np.cos(phi * np.pi / 180.0)],
            ],
            dtype=float,
        )
    )

    homography = np.eye(3, dtype=float)
    homography[:2, :2] = affinity
    homography[0, 2] = tx
    homography[1, 2] = ty
    return homography


def border_remover(M):
    M = np.asarray(M).copy()
    h, w = M.shape[:2]
    if h < 4 or w < 4:
        return M
    M[:1, :] = np.median(M[:4, :])
    M[:, :1] = np.median(M[:, :4])
    M[h - 1 :, :] = np.median(M[h - 4 :, :])
    M[:, w - 1 :] = np.median(M[:, w - 4 :])
    return M




def _to_uint8_image(img_arr):
    arr = np.asarray(img_arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Format image non supporté, shape={arr.shape}")

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size > 0 and arr.max() <= 1.0:
                arr = np.clip(255.0 * arr, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _apply_H_to_points(H, pts_xy):
    pts_xy = np.asarray(pts_xy, dtype=float)
    pts_h = np.c_[pts_xy, np.ones((len(pts_xy), 1), dtype=float)]
    q = (H @ pts_h.T).T
    q = q[:, :2] / q[:, 2:3]
    return q


def _invert_homography(H):
    return np.linalg.inv(np.asarray(H, dtype=float))


def _centering_translation(image_shape):
    h, w = image_shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    T_to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=float)
    T_back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=float)
    return T_to_origin, T_back


def image_centralizer(image, transf):
    T_to_origin, T_back = _centering_translation(np.asarray(image).shape)
    return T_back @ np.asarray(transf, dtype=float) @ T_to_origin


def apply_homography(image, H, output_shape, border_value=None, flags=cv.INTER_CUBIC):
    arr = _to_uint8_image(image)
    h_out, w_out = output_shape

    if border_value is None:
        if arr.ndim == 2:
            border_value = int(np.mean(arr))
        else:
            m = int(np.mean(arr))
            border_value = (m, m, m)

    return cv.warpPerspective(
        arr,
        np.asarray(H, dtype=float),
        (w_out, h_out),
        borderValue=border_value,
        flags=flags,
    )



def image_deformation_generator(image, transf):
    image_arr = _to_uint8_image(image)
    h, w = image_arr.shape[:2]
    deformation = image_centralizer(image_arr, transf)

    if image_arr.ndim == 2:
        border_val = int(np.mean(image_arr))
    else:
        m = int(np.mean(image_arr))
        border_val = (m, m, m)

    image_deformed = apply_homography(image_arr, deformation, (h, w), border_value=border_val)
    return image_deformed


def affine_deformer(image, lambdaa, theta, tilt, phi, translation):
    image_before = _to_uint8_image(image)
    affinity = matrix_affinity_generator(lambdaa, theta, tilt, phi, translation)
    image_deformed = image_deformation_generator(image_before, affinity)
    return image_deformed, affinity




def deform_image_affinity(
    input_image,
    lambdaa,
    theta,
    tilt,
    phi,
    translation,
    apply_blur=False,
    blur_ksize=(5, 5),
    blur_sigma=0,
    return_array=True,
):
    if isinstance(input_image, str):
        img_arr = np.asarray(Image.open(input_image))
    elif isinstance(input_image, Image.Image):
        img_arr = np.asarray(input_image)
    elif isinstance(input_image, np.ndarray):
        img_arr = np.asarray(input_image)
    else:
        raise TypeError("input_image doit être un chemin, une PIL.Image ou un np.ndarray")

    img_arr = _to_uint8_image(img_arr)

    if apply_blur:
        img_arr = cv.GaussianBlur(img_arr, blur_ksize, blur_sigma)

    result, _ = affine_deformer(img_arr, lambdaa, theta, tilt, phi, translation)

    if return_array:
        return result

    if result.ndim == 2:
        return Image.fromarray(result, mode="L")
    return Image.fromarray(result, mode="RGB")


deform_image = deform_image_affinity



def _centered_affine_homography_for_shape(source_shape, lambdaa, theta, tilt, phi, translation):
    dummy = np.zeros(source_shape[:2], dtype=np.uint8)
    H_aff = matrix_affinity_generator(lambdaa, theta, tilt, phi, translation)
    H_centered = image_centralizer(dummy, H_aff)
    return H_centered


def image_point_to_label_point(point_xy, H):
    point_xy = np.asarray(point_xy, dtype=float).reshape(1, 2)
    Hinv = _invert_homography(H)
    uv = _apply_H_to_points(Hinv, point_xy)[0]
    return uv


def label_point_to_image_point(point_uv, H):
    point_uv = np.asarray(point_uv, dtype=float).reshape(1, 2)
    xy = _apply_H_to_points(H, point_uv)[0]
    return xy


def jacobian_affinity_at_point(H, point_uv=None):
    H = np.asarray(H, dtype=float)
    return H[:2, :2].copy()



def differential_at_image_point(point_xy, lambdaa, theta, tilt, phi, translation, source_shape=None):
    if source_shape is None:
        raise ValueError("source_shape doit être fourni, par exemple input_image.shape")

    H = _centered_affine_homography_for_shape(source_shape, lambdaa, theta, tilt, phi, translation)
    point_label = image_point_to_label_point(point_xy, H)
    J = jacobian_affinity_at_point(H, point_label)

    return {
        "point_image": np.asarray(point_xy, dtype=float),
        "point_label": np.asarray(point_label, dtype=float),
        "J": J,
        "H": H,
    }



def autocorrelation_display(arg):
    u = np.asarray(arg)

    if u.ndim == 3:
        if u.shape[2] >= 3:
            u = 0.299 * u[..., 0] + 0.587 * u[..., 1] + 0.114 * u[..., 2]
        else:
            u = u[..., 0]

    if u.ndim != 2:
        raise ValueError(f"Entrée non supportée, shape={u.shape}")

    u = u.astype(np.float64)

    fimg1 = fft2(u)
    fimg1[0, 0] = 0

    fimgnorm = fimg1 * np.conjugate(fimg1)
    fout1 = ifft2(fimgnorm)
    autocorr = fftshift(fout1).real

    energy = np.sum(np.abs(u) ** 2)
    autocorr_normalized = autocorr / energy if energy != 0 else autocorr
    return autocorr_normalized


def extract_centered_patch(image, center_xy, patch_size, pad_value=0):
    img = np.asarray(image)

    cx, cy = center_xy
    cx = int(round(cx))
    cy = int(round(cy))

    half = patch_size // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    H, W = img.shape[:2]

    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(W, x1)
    iy1 = min(H, y1)

    patch = img[iy0:iy1, ix0:ix1]

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    if img.ndim == 2:
        patch_out = np.full((patch_size, patch_size), pad_value, dtype=img.dtype)
        patch_out[pad_top : patch_size - pad_bottom, pad_left : patch_size - pad_right] = patch
    else:
        patch_out = np.full((patch_size, patch_size, img.shape[2]), pad_value, dtype=img.dtype)
        patch_out[
            pad_top : patch_size - pad_bottom,
            pad_left : patch_size - pad_right,
            :,
        ] = patch

    return patch_out


def autocorr_to_rgb_image(ac):
    ac = np.asarray(ac, dtype=float)

    amin = np.min(ac)
    amax = np.max(ac)

    if amax > amin:
        ac_norm = (ac - amin) / (amax - amin)
    else:
        ac_norm = np.zeros_like(ac)

    ac_u8 = np.clip(np.rint(255.0 * ac_norm), 0, 255).astype(np.uint8)
    return np.stack([ac_u8, ac_u8, ac_u8], axis=-1)



def _prepare_draw_image(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size > 0 and arr.max() <= 1.0:
                arr = np.clip(255.0 * arr, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def draw_cross_on_array(img, x, y, color=(255, 0, 0), size=2, thickness=1,
                        line_type=cv.LINE_AA, shift_bits=8):
    out = _prepare_draw_image(img).copy()

    if out.ndim == 2:
        draw_color = int(color[0] if isinstance(color, (tuple, list, np.ndarray)) else color)
    else:
        draw_color = tuple(int(c) for c in color)

    s = 1 << shift_bits
    xi = int(round(float(x) * s))
    yi = int(round(float(y) * s))
    ds = int(round(float(size) * s))

    cv.line(out, (xi - ds, yi), (xi + ds, yi), draw_color,
            thickness=thickness, lineType=line_type, shift=shift_bits)
    cv.line(out, (xi, yi - ds), (xi, yi + ds), draw_color,
            thickness=thickness, lineType=line_type, shift=shift_bits)
    return out


def draw_circle_on_array(img, x, y, color=(0, 255, 255), radius=4, thickness=-1,
                         line_type=cv.LINE_AA, shift_bits=8):
    out = _prepare_draw_image(img).copy()

    if out.ndim == 2:
        draw_color = int(color[0] if isinstance(color, (tuple, list, np.ndarray)) else color)
    else:
        draw_color = tuple(int(c) for c in color)

    s = 1 << shift_bits
    center = (int(round(float(x) * s)), int(round(float(y) * s)))
    radius_i = max(1, int(round(float(radius) * s)))

    cv.circle(out, center, radius_i, draw_color, thickness=thickness,
              lineType=line_type, shift=shift_bits)
    return out



def autocorr_with_theoretical_peaks(
    deformed_image,
    point_xy,
    patch_size,
    U_init,
    V_init,
    lambdaa,
    theta,
    tilt,
    phi,
    translation,
    source_image_shape=None,
    shift_units="source_px",
    show=True,
    stable_kwargs=None
):
    if source_image_shape is None:
        raise ValueError("source_image_shape doit être fourni, par exemple input_image.shape")

    if isinstance(deformed_image, Image.Image):
        deformed_image = np.asarray(deformed_image)

    diff_info = differential_at_image_point(
        point_xy=point_xy,
        lambdaa=lambdaa,
        theta=theta,
        tilt=tilt,
        phi=phi,
        translation=translation,
        source_shape=source_image_shape,
    )

    J = diff_info["J"]

    patch = extract_centered_patch(deformed_image, point_xy, patch_size)
    ac = autocorrelation_display(patch)

    U_init = np.asarray(U_init, dtype=float).reshape(2)
    V_init = np.asarray(V_init, dtype=float).reshape(2)

    if shift_units == "source_px":
        JU = J @ U_init
        JV = J @ V_init
        JW = J @ (U_init - V_init)
        U_used = U_init
        V_used = V_init
    elif shift_units == "deformed_px":
        JU = U_init
        JV = V_init
        JW = U_init - V_init
        U_used = None
        V_used = None
    else:
        raise ValueError("shift_units doit être 'source_px' ou 'deformed_px'")

    peak_vectors = {
        "+JU": JU,
        "-JU": -JU,
        "+JV": JV,
        "-JV": -JV,
        "+J(U-V)": JW,
        "-J(U-V)": -JW,
    }

    h_ac, w_ac = ac.shape[:2]
    cx_ac = (w_ac - 1) / 2.0
    cy_ac = (h_ac - 1) / 2.0

    peak_pixels = {}
    for name, vec in peak_vectors.items():
        dx, dy = vec
        peak_pixels[name] = np.array([cx_ac + dx, cy_ac + dy], dtype=float)

    U_ref_dcdr = np.array([JU[1], JU[0]], dtype=float)
    V_ref_dcdr = np.array([JV[1], JV[0]], dtype=float)

    found_info = _extract_found_stable_peak_pixels(
        ac,
        stable_kwargs=stable_kwargs,
        U_ref_dcdr=U_ref_dcdr,
        V_ref_dcdr=V_ref_dcdr,
    )

    found_peak_pixels = None if found_info is None else found_info["peak_pixels"]

    ac_rgb = autocorr_to_rgb_image(ac)

    color_map_theoretical = {
        "+JU": (255, 0, 0),
        "-JU": (255, 0, 0),
        "+JV": (0, 255, 0),
        "-JV": (0, 255, 0),
        "+J(U-V)": (255, 255, 0),
        "-J(U-V)": (255, 255, 0),
    }

    color_map_found = {
        "F+U": (255, 128, 128),
        "F-U": (255, 128, 128),
        "F+V": (128, 255, 128),
        "F-V": (128, 255, 128),
        "F+W": (255, 255, 128),
        "F-W": (255, 255, 128),
    }

    ac_rgb = draw_circle_on_array(ac_rgb, cx_ac, cy_ac, color=(0, 255, 255), radius=4)

    for name, p in peak_pixels.items():
        px, py = p
        if 0 <= px < w_ac and 0 <= py < h_ac:
            ac_rgb = draw_cross_on_array(
                ac_rgb,
                px, py,
                color=color_map_theoretical.get(name, (255, 0, 255)),
                size=1,
                thickness=1
            )

    if found_peak_pixels is not None:
        for name, p in found_peak_pixels.items():
            px, py = p
            if 0 <= px < w_ac and 0 <= py < h_ac:
                ac_rgb = draw_circle_on_array(
                    ac_rgb,
                    px, py,
                    color=color_map_found.get(name, (255, 255, 255)),
                    radius=2,
                    thickness=1
                )

    ac_with_peaks_pil = Image.fromarray(ac_rgb)

    fig = None
    ax = None

    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ac_rgb)
        ax.set_title("Autocorrélation du patch avec pics théoriques et pics trouvés")

        if 0 <= cx_ac < w_ac and 0 <= cy_ac < h_ac:
            ax.text(cx_ac + 4, cy_ac + 4, "0", color="cyan", fontsize=9, weight="bold")

        color_map_plot_theoretical = {
            "+JU": "red",
            "-JU": "red",
            "+JV": "lime",
            "-JV": "lime",
            "+J(U-V)": "yellow",
            "-J(U-V)": "yellow",
        }

        for name, p in peak_pixels.items():
            px, py = p
            if 0 <= px < w_ac and 0 <= py < h_ac:
                ax.text(
                    px + 8, py + 8,
                    f"T{name}",
                    color=color_map_plot_theoretical.get(name, "magenta"),
                    fontsize=8,
                    weight="bold"
                )

        if found_peak_pixels is not None:
            color_map_plot_found = {
                "F+U": "#ff9999",
                "F-U": "#ff9999",
                "F+V": "#99ff99",
                "F-V": "#99ff99",
                "F+W": "#ffff99",
                "F-W": "#ffff99",
            }

            for name, p in found_peak_pixels.items():
                px, py = p
                if 0 <= px < w_ac and 0 <= py < h_ac:
                    ax.text(
                        px + 8, py - 8,
                        name,
                        color=color_map_plot_found.get(name, "white"),
                        fontsize=8,
                        weight="bold"
                    )

        ax.set_xlim(0, w_ac - 1)
        ax.set_ylim(h_ac - 1, 0)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

    return {
        "patch": patch,
        "autocorr": ac,
        "point_image": diff_info["point_image"],
        "point_label": diff_info["point_label"],
        "J": J,
        "H": diff_info["H"],
        "U_source_used": U_used,
        "V_source_used": V_used,
        "peak_vectors": peak_vectors,
        "peak_pixels": peak_pixels,
        "autocorr_with_peaks_array": ac_rgb,
        "autocorr_with_peaks_pil": ac_with_peaks_pil,
        "fig": fig,
        "ax": ax,
        "found_hexagon": found_info,
        "found_peak_pixels": found_peak_pixels,
    }
