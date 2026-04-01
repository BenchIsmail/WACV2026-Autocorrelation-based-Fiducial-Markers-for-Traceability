import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift
import cv2 as cv
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


def build_camera_basis(view_dir, world_up=np.array([0.0, 0.0, 1.0]), roll=0.0):
    fwd = normalize(view_dir)
    wu = normalize(world_up)

    if abs(np.dot(fwd, wu)) > 0.98:
        wu = np.array([0.0, 1.0, 0.0], dtype=float)

    right = normalize(np.cross(fwd, wu))
    up = normalize(np.cross(right, fwd))

    if abs(roll) > 1e-12:
        cr, sr = math.cos(roll), math.sin(roll)
        right2 = cr * right + sr * up
        up2 = -sr * right + cr * up
        right, up = right2, up2

    return right, up, fwd




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
    p = u

    fimg1 = fft2(p)
    fimg1[0, 0] = 0

    fimgnorm = fimg1 * np.conjugate(fimg1)
    fout1 = ifft2(fimgnorm)
    autocorr = fftshift(fout1).real

    energy = np.sum(np.abs(p) ** 2)
    autocorr_normalized = autocorr / energy if energy != 0 else autocorr
    return autocorr_normalized



def label_point_to_3d(u_cm, v_cm, radius_cm):
    theta = u_cm / radius_cm
    X = radius_cm * np.cos(theta)
    Y = radius_cm * np.sin(theta)
    Z = v_cm
    return np.array([X, Y, Z], dtype=float)


def project_3d_point_to_image(P, camera_pos, view_dir, focal_px, output_size_px, roll=0.0):
    P = np.asarray(P, dtype=float)
    C = np.asarray(camera_pos, dtype=float)

    right, up, fwd = build_camera_basis(view_dir, roll=roll)

    Q = P - C
    x_cam = np.dot(Q, right)
    y_cam = np.dot(Q, up)
    z_cam = np.dot(Q, fwd)

    if z_cam <= 1e-12:
        raise ValueError("Le point est derrière la caméra ou trop proche.")

    cx = cy = (output_size_px - 1) / 2.0
    x_img = focal_px * (x_cam / z_cam) + cx
    y_img = cy - focal_px * (y_cam / z_cam)

    return np.array([x_img, y_img], dtype=float), float(z_cam)


def label_to_image(u_cm, v_cm, radius_cm, camera_pos, view_dir, focal_px, output_size_px, roll=0.0):
    P = label_point_to_3d(u_cm, v_cm, radius_cm)
    p_img, _ = project_3d_point_to_image(
        P=P,
        camera_pos=camera_pos,
        view_dir=view_dir,
        focal_px=focal_px,
        output_size_px=output_size_px,
        roll=roll
    )
    return p_img


def image_point_to_cylinder_label(x_img, y_img, radius_cm, label_size_factor,
                                  camera_pos, view_dir, focal_px, output_size_px, roll=0.0):
    W_cm = H_cm = label_size_factor * radius_cm
    theta_half = W_cm / (2.0 * radius_cm)

    right, up, fwd = build_camera_basis(view_dir, roll=roll)
    C = np.asarray(camera_pos, dtype=float)
    Cx, Cy, Cz = C

    cx = cy = (output_size_px - 1) / 2.0
    x_cam = x_img - cx
    y_cam = -(y_img - cy)

    D = x_cam * right + y_cam * up + focal_px * fwd
    D = normalize(D)

    Dx, Dy, Dz = D

    a = Dx * Dx + Dy * Dy
    b = 2.0 * (Cx * Dx + Cy * Dy)
    c = Cx * Cx + Cy * Cy - radius_cm * radius_cm

    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [t for t in [t1, t2] if t > 1e-6]
    if len(candidates) == 0:
        return None

    t = min(candidates)
    P = C + t * D
    Px, Py, Pz = P

    theta = math.atan2(Py, Px)

    if not (-theta_half <= theta <= theta_half):
        return None
    if not (-H_cm / 2.0 <= Pz <= H_cm / 2.0):
        return None

    Nx = math.cos(theta)
    Ny = math.sin(theta)
    facing = ((Cx - Px) * Nx + (Cy - Py) * Ny) > 0.0
    if not facing:
        return None

    u_cm = radius_cm * theta
    v_cm = Pz
    return np.array([u_cm, v_cm], dtype=float)




def deform_image_cylindrical(input_image, radius_cm, label_size_factor,
                             camera_pos, view_dir, focal_px, output_size_px,
                             roll=0.0, background=255,
                             apply_blur=False, blur_ksize=(5, 5), blur_sigma=0,
                             return_array=True):

    if isinstance(input_image, str):
        img_arr = np.asarray(Image.open(input_image))
    elif isinstance(input_image, Image.Image):
        img_arr = np.asarray(input_image)
    elif isinstance(input_image, np.ndarray):
        img_arr = np.asarray(input_image)
    else:
        raise TypeError("input_image doit être un chemin, une PIL.Image ou un np.ndarray")

    if img_arr.ndim not in (2, 3):
        raise ValueError(f"Format image non supporté, shape={img_arr.shape}")

    if img_arr.ndim == 3 and img_arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"Format couleur non supporté, shape={img_arr.shape}")


    is_gray = (img_arr.ndim == 2) or (img_arr.ndim == 3 and img_arr.shape[2] == 1)

    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
        img_arr = img_arr[..., :3]

    if img_arr.ndim == 3 and img_arr.shape[2] == 1:
        img_arr = img_arr[..., 0]
        is_gray = True

    if img_arr.dtype != np.uint8:
        if np.issubdtype(img_arr.dtype, np.floating):
            if img_arr.max() <= 1.0:
                img_arr = np.clip(255.0 * img_arr, 0, 255).astype(np.uint8)
            else:
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        else:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

    tex = img_arr.copy()

    if apply_blur:
        tex = cv.GaussianBlur(tex, blur_ksize, blur_sigma)


    if tex.ndim == 2:
        Htex, Wtex = tex.shape
    else:
        Htex, Wtex = tex.shape[:2]

    W_cm = H_cm = label_size_factor * radius_cm
    theta_half = W_cm / (2.0 * radius_cm)


    right, up, fwd = build_camera_basis(view_dir, roll=roll)
    C = np.asarray(camera_pos, dtype=float)


    if is_gray:
        if isinstance(background, (tuple, list, np.ndarray)):
            bg_val = int(background[0])
        else:
            bg_val = int(background)

        out = np.full((output_size_px, output_size_px), bg_val, dtype=np.uint8)
    else:
        if isinstance(background, (int, float)):
            bg_val = np.array([background, background, background], dtype=np.uint8)
        else:
            bg_val = np.array(background, dtype=np.uint8).reshape(3)

        out = np.empty((output_size_px, output_size_px, 3), dtype=np.uint8)
        out[:] = bg_val


    cx = cy = (output_size_px - 1) / 2.0

    xs = np.arange(output_size_px, dtype=float)
    ys = np.arange(output_size_px, dtype=float)
    Xp, Yp = np.meshgrid(xs, ys)

    x_cam = Xp - cx
    y_cam = -(Yp - cy)

    D = x_cam[..., None] * right + y_cam[..., None] * up + focal_px * fwd
    D /= np.linalg.norm(D, axis=-1, keepdims=True)

    Dx, Dy = D[..., 0], D[..., 1]
    Cx, Cy, Cz = C


    a = Dx * Dx + Dy * Dy
    b = 2.0 * (Cx * Dx + Cy * Dy)
    c = Cx * Cx + Cy * Cy - radius_cm * radius_cm

    disc = b * b - 4.0 * a * c
    valid = disc > 0.0

    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[valid] = np.sqrt(disc[valid])

    t1 = np.where(valid, (-b - sqrt_disc) / (2.0 * a), np.inf)
    t2 = np.where(valid, (-b + sqrt_disc) / (2.0 * a), np.inf)

    both_pos = (t1 > 1e-6) & (t2 > 1e-6)
    only_t1 = (t1 > 1e-6) & ~(t2 > 1e-6)
    only_t2 = (t2 > 1e-6) & ~(t1 > 1e-6)

    t = np.full_like(t1, np.inf)
    t[both_pos] = np.minimum(t1[both_pos], t2[both_pos])
    t[only_t1] = t1[only_t1]
    t[only_t2] = t2[only_t2]

    valid &= np.isfinite(t)

    Px = Cx + t * D[..., 0]
    Py = Cy + t * D[..., 1]
    Pz = Cz + t * D[..., 2]

    theta = np.arctan2(Py, Px)

    in_patch = (
        valid
        & (theta >= -theta_half) & (theta <= theta_half)
        & (Pz >= -H_cm / 2.0) & (Pz <= H_cm / 2.0)
    )

    Nx = np.cos(theta)
    Ny = np.sin(theta)
    facing = ((Cx - Px) * Nx + (Cy - Py) * Ny) > 0.0
    in_patch &= facing


    if np.any(in_patch):
        u_norm = (theta[in_patch] + theta_half) / (2.0 * theta_half)
        v_norm = (H_cm / 2.0 - Pz[in_patch]) / H_cm

        u_tex = np.clip(u_norm * (Wtex - 1), 0, Wtex - 1)
        v_tex = np.clip(v_norm * (Htex - 1), 0, Htex - 1)

        ui = np.rint(u_tex).astype(int)
        vi = np.rint(v_tex).astype(int)

        if is_gray:
            out[in_patch] = tex[vi, ui]
        else:
            out[in_patch] = tex[vi, ui]

    if return_array:
        return out
    else:
        if is_gray:
            return Image.fromarray(out, mode="L")
        return Image.fromarray(out, mode="RGB")



def deformation_differential_at_label_point(
    u_cm, v_cm,
    radius_cm, camera_pos, view_dir, focal_px, output_size_px,
    roll=0.0, eps_u=1e-6, eps_v=1e-6
):
    f_pp_u = label_to_image(
        u_cm + eps_u, v_cm,
        radius_cm, camera_pos, view_dir, focal_px, output_size_px, roll
    )
    f_pm_u = label_to_image(
        u_cm - eps_u, v_cm,
        radius_cm, camera_pos, view_dir, focal_px, output_size_px, roll
    )
    dF_du = (f_pp_u - f_pm_u) / (2.0 * eps_u)

    f_pp_v = label_to_image(
        u_cm, v_cm + eps_v,
        radius_cm, camera_pos, view_dir, focal_px, output_size_px, roll
    )
    f_pm_v = label_to_image(
        u_cm, v_cm - eps_v,
        radius_cm, camera_pos, view_dir, focal_px, output_size_px, roll
    )
    dF_dv = (f_pp_v - f_pm_v) / (2.0 * eps_v)

    return np.column_stack([dF_du, dF_dv])


def differential_at_image_point(point_xy, radius_cm, label_size_factor,
                                camera_pos, view_dir, focal_px, output_size_px,
                                roll=0.0, eps_u=1e-6, eps_v=1e-6):
    x_img, y_img = point_xy

    uv = image_point_to_cylinder_label(
        x_img=x_img,
        y_img=y_img,
        radius_cm=radius_cm,
        label_size_factor=label_size_factor,
        camera_pos=camera_pos,
        view_dir=view_dir,
        focal_px=focal_px,
        output_size_px=output_size_px,
        roll=roll
    )

    if uv is None:
        raise ValueError("Ce point image n'appartient pas à l'étiquette cylindrique visible.")

    u_cm, v_cm = uv

    J = deformation_differential_at_label_point(
        u_cm=u_cm,
        v_cm=v_cm,
        radius_cm=radius_cm,
        camera_pos=camera_pos,
        view_dir=view_dir,
        focal_px=focal_px,
        output_size_px=output_size_px,
        roll=roll,
        eps_u=eps_u,
        eps_v=eps_v
    )

    return {
        "point_image": np.array([x_img, y_img], dtype=float),
        "point_label": np.array([u_cm, v_cm], dtype=float),
        "J": J
    }




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
        patch_out[pad_top:patch_size-pad_bottom, pad_left:patch_size-pad_right] = patch
    else:
        patch_out = np.full((patch_size, patch_size, img.shape[2]), pad_value, dtype=img.dtype)
        patch_out[pad_top:patch_size-pad_bottom, pad_left:patch_size-pad_right, :] = patch

    return patch_out


def source_pixel_shift_to_label_shift(shift_px, source_image_shape, radius_cm, label_size_factor):

    shift_px = np.asarray(shift_px, dtype=float).reshape(2)

    if len(source_image_shape) == 2:
        Hsrc, Wsrc = source_image_shape
    else:
        Hsrc, Wsrc = source_image_shape[:2]

    W_cm = label_size_factor * radius_cm
    H_cm = label_size_factor * radius_cm

    sx = W_cm / Wsrc
    sy = H_cm / Hsrc

    return np.array([shift_px[0] * sx, shift_px[1] * sy], dtype=float)


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
            if arr.max() <= 1.0:
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

    cv.line(
        out,
        (xi - ds, yi),
        (xi + ds, yi),
        draw_color,
        thickness=thickness,
        lineType=line_type,
        shift=shift_bits
    )
    cv.line(
        out,
        (xi, yi - ds),
        (xi, yi + ds),
        draw_color,
        thickness=thickness,
        lineType=line_type,
        shift=shift_bits
    )
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

    cv.circle(
        out,
        center,
        radius_i,
        draw_color,
        thickness=thickness,
        lineType=line_type,
        shift=shift_bits
    )
    return out




def autocorr_with_theoretical_peaks(
    deformed_image,
    point_xy,
    patch_size,
    U_init,
    V_init,
    radius_cm,
    label_size_factor,
    camera_pos,
    view_dir,
    focal_px,
    output_size_px,
    roll=0.0,
    eps_u=1e-6,
    eps_v=1e-6,
    shift_units="source_px",   # "source_px", "label_cm", "deformed_px"
    source_image_shape=None,
    show=True,
    stable_kwargs=None
):

    if isinstance(deformed_image, Image.Image):
        deformed_image = np.asarray(deformed_image)

    diff_info = differential_at_image_point(
        point_xy=point_xy,
        radius_cm=radius_cm,
        label_size_factor=label_size_factor,
        camera_pos=camera_pos,
        view_dir=view_dir,
        focal_px=focal_px,
        output_size_px=output_size_px,
        roll=roll,
        eps_u=eps_u,
        eps_v=eps_v
    )

    J = diff_info["J"]

    patch = extract_centered_patch(deformed_image, point_xy, patch_size)
    ac = autocorrelation_display(patch)

    U_init = np.asarray(U_init, dtype=float).reshape(2)
    V_init = np.asarray(V_init, dtype=float).reshape(2)

    if shift_units == "source_px":
        if source_image_shape is None:
            raise ValueError("source_image_shape doit être fourni si shift_units='source_px'")
        U_label = source_pixel_shift_to_label_shift(
            U_init, source_image_shape, radius_cm, label_size_factor
        )
        V_label = source_pixel_shift_to_label_shift(
            V_init, source_image_shape, radius_cm, label_size_factor
        )
        JU = J @ U_label
        JV = J @ V_label
        JW = J @ (U_label - V_label)

    elif shift_units == "label_cm":
        U_label = U_init
        V_label = V_init
        JU = J @ U_label
        JV = J @ V_label
        JW = J @ (U_label - V_label)

    elif shift_units == "deformed_px":
        JU = U_init
        JV = V_init
        JW = U_init - V_init
        U_label = None
        V_label = None

    else:
        raise ValueError("shift_units doit être 'source_px', 'label_cm' ou 'deformed_px'")

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
        "U_label_used": U_label,
        "V_label_used": V_label,
        "peak_vectors": peak_vectors,
        "peak_pixels": peak_pixels,
        "autocorr_with_peaks_array": ac_rgb,
        "autocorr_with_peaks_pil": ac_with_peaks_pil,
        "fig": fig,
        "ax": ax,
        "found_hexagon": found_info,
        "found_peak_pixels": found_peak_pixels,
    }