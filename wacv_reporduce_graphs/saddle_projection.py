import argparse
import json
import math
import random
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
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


DEFAULT_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero vector cannot be normalized")
    return v / n


def build_camera_basis(view_dir, world_up=DEFAULT_WORLD_UP, roll=0.0):
    fwd = normalize(view_dir)
    world_up = normalize(world_up)

    if abs(np.dot(fwd, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)

    right = normalize(np.cross(fwd, world_up))
    up = normalize(np.cross(right, fwd))

    if abs(roll) > 1e-12:
        cr, sr = math.cos(roll), math.sin(roll)
        right2 = cr * right + sr * up
        up2 = -sr * right + cr * up
        right, up = right2, up2

    return right, up, fwd


def compute_intrinsics(
    image_width_px,
    image_height_px,
    focal_px=None,
    focal_mm=None,
    sensor_width_mm=None,
    sensor_height_mm=None,
    principal_point_x=None,
    principal_point_y=None,
):
    if focal_mm is not None:
        if sensor_width_mm is None or sensor_height_mm is None:
            raise ValueError(
                "When --focal-mm is used, both --sensor-width-mm and --sensor-height-mm must be provided."
            )
        fx = focal_mm * image_width_px / sensor_width_mm
        fy = focal_mm * image_height_px / sensor_height_mm
        focal_mode = "mm"
    else:
        if focal_px is None:
            raise ValueError("Provide either --focal-px or --focal-mm.")
        fx = float(focal_px)
        fy = float(focal_px)
        focal_mode = "px"

    cx = principal_point_x if principal_point_x is not None else (image_width_px - 1) / 2.0
    cy = principal_point_y if principal_point_y is not None else (image_height_px - 1) / 2.0

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "mode": focal_mode,
    }


def generate_grid_image(size_px=768, line_step_px=32, line_width=1):
    img = Image.new("RGB", (size_px, size_px), "white")
    draw = ImageDraw.Draw(img)
    for x in range(0, size_px, line_step_px):
        lw = 2 if x % (line_step_px * 4) == 0 else line_width
        draw.line([(x, 0), (x, size_px - 1)], fill=(0, 0, 0), width=lw)
    for y in range(0, size_px, line_step_px):
        lw = 2 if y % (line_step_px * 4) == 0 else line_width
        draw.line([(0, y), (size_px - 1, y)], fill=(0, 0, 0), width=lw)
    draw.line([(size_px // 2, 0), (size_px // 2, size_px - 1)], fill=(220, 0, 0), width=2)
    draw.line([(0, size_px // 2), (size_px - 1, size_px // 2)], fill=(0, 0, 220), width=2)
    return img


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


def _project_points_world_to_image(P, camera_pos, view_dir, intrinsics, roll=0.0):
    C = np.asarray(camera_pos, dtype=float)
    right, up, fwd = build_camera_basis(view_dir, roll=roll)
    Q = P - C
    xc = Q @ right
    yc = Q @ up
    zc = Q @ fwd

    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    x_img = cx + fx * (xc / zc)
    y_img = cy - fy * (yc / zc)
    return x_img, y_img, zc, (right, up, fwd)


def label_point_to_3d(u_cm, v_cm, radius_cm):
    x = float(u_cm)
    y = float(v_cm)
    z = (x * x - y * y) / (2.0 * radius_cm)
    return np.array([x, y, z], dtype=float)


def project_3d_point_to_image(P, camera_pos, view_dir, focal_px, output_size_px, roll=0.0):
    P = np.asarray(P, dtype=float)
    C = np.asarray(camera_pos, dtype=float)

    intrinsics = compute_intrinsics(
        image_width_px=output_size_px,
        image_height_px=output_size_px,
        focal_px=focal_px,
    )
    right, up, fwd = build_camera_basis(view_dir, roll=roll)

    Q = P - C
    x_cam = np.dot(Q, right)
    y_cam = np.dot(Q, up)
    z_cam = np.dot(Q, fwd)

    if z_cam <= 1e-12:
        raise ValueError("Le point est derrière la caméra ou trop proche.")

    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
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
        roll=roll,
    )
    return p_img


def image_point_to_saddle_label(
    x_img,
    y_img,
    radius_cm,
    label_size_factor,
    camera_pos,
    view_dir,
    focal_px,
    output_size_px,
    roll=0.0,
):
    W_cm = H_cm = label_size_factor * radius_cm

    right, up, fwd = build_camera_basis(view_dir, roll=roll)
    C = np.asarray(camera_pos, dtype=float)
    Cx, Cy, Cz = C

    cx = cy = (output_size_px - 1) / 2.0
    x_cam = x_img - cx
    y_cam = -(y_img - cy)

    D = x_cam * right + y_cam * up + focal_px * fwd
    D = normalize(D)
    Dx, Dy, Dz = D

    a = Dx * Dx - Dy * Dy
    b = 2.0 * (Cx * Dx - Cy * Dy - radius_cm * Dz)
    c = Cx * Cx - Cy * Cy - 2.0 * radius_cm * Cz

    candidates = []
    if abs(a) > 1e-12:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(max(0.0, disc))
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        candidates.extend([t for t in (t1, t2) if t > 1e-6])
    elif abs(b) > 1e-12:
        t_lin = -c / b
        if t_lin > 1e-6:
            candidates.append(t_lin)

    if not candidates:
        return None

    t = min(candidates)
    P = C + t * D
    Px, Py, _ = P

    if not (-W_cm / 2.0 <= Px <= W_cm / 2.0):
        return None
    if not (-H_cm / 2.0 <= Py <= H_cm / 2.0):
        return None

    return np.array([Px, Py], dtype=float)


def visible_fraction(
    camera_pos,
    view_dir,
    intrinsics,
    image_width_px,
    image_height_px,
    radius_cm,
    label_size_factor,
    roll=0.0,
    samples_x=120,
    samples_y=120,
):
    W_cm = H_cm = label_size_factor * radius_cm

    xs = np.linspace(-W_cm / 2.0, W_cm / 2.0, samples_x)
    ys = np.linspace(-H_cm / 2.0, H_cm / 2.0, samples_y)
    X, Y = np.meshgrid(xs, ys)
    Z = (X ** 2 - Y ** 2) / (2.0 * radius_cm)
    P = np.stack([X, Y, Z], axis=-1)

    x_img, y_img, zc, _ = _project_points_world_to_image(P, camera_pos, view_dir, intrinsics, roll=roll)

    front = zc > 1e-6
    inside = (x_img >= 0) & (x_img < image_width_px) & (y_img >= 0) & (y_img < image_height_px)
    return float((front & inside).mean())


def random_camera_pose(
    radius_cm,
    label_size_factor,
    intrinsics,
    image_width_px,
    image_height_px,
    rng=None,
    min_visible_fraction=0.25,
    max_tries=2000,
):
    if rng is None:
        rng = random.Random()

    W_cm = H_cm = label_size_factor * radius_cm

    for _ in range(max_tries):
        phi_c = rng.uniform(math.radians(20), math.radians(80))
        theta_c = rng.uniform(-math.pi, math.pi)
        dist = rng.uniform(2.2 * radius_cm, 4.5 * radius_cm)

        C = np.array([
            dist * math.cos(phi_c) * math.cos(theta_c),
            dist * math.cos(phi_c) * math.sin(theta_c),
            dist * math.sin(phi_c),
        ], dtype=float)

        x_t = rng.uniform(-0.45 * W_cm / 2.0, 0.45 * W_cm / 2.0)
        y_t = rng.uniform(-0.35 * H_cm / 2.0, 0.35 * H_cm / 2.0)
        z_t = (x_t ** 2 - y_t ** 2) / (2.0 * radius_cm)
        T = np.array([x_t, y_t, z_t], dtype=float)

        base_dir = normalize(T - C)
        tmp_right, tmp_up, tmp_fwd = build_camera_basis(base_dir)

        yaw = rng.uniform(-math.radians(12), math.radians(12))
        pitch = rng.uniform(-math.radians(10), math.radians(10))
        roll = rng.uniform(-math.radians(8), math.radians(8))

        view_dir = normalize(
            math.cos(yaw) * math.cos(pitch) * tmp_fwd
            + math.sin(yaw) * tmp_right
            + math.sin(pitch) * tmp_up
        )

        frac = visible_fraction(
            C,
            view_dir,
            intrinsics,
            image_width_px,
            image_height_px,
            radius_cm,
            label_size_factor,
            roll=roll,
            samples_x=80,
            samples_y=80,
        )
        if frac >= min_visible_fraction:
            return C, view_dir, roll, float(frac)

    raise RuntimeError("Could not draw a valid random pose satisfying the visibility constraint.")


def render_label_on_shape(
    input_image,
    radius_cm,
    label_size_factor,
    camera_pos,
    view_dir,
    intrinsics,
    image_width_px,
    image_height_px,
    roll=0.0,
    background=(255, 255, 255),
):
    img = input_image.convert("RGB")
    tex = np.asarray(img, dtype=np.uint8)
    Htex, Wtex = tex.shape[:2]

    W_cm = H_cm = label_size_factor * radius_cm

    right, up, fwd = build_camera_basis(view_dir, roll=roll)
    C = np.asarray(camera_pos, dtype=float)

    out = np.empty((image_height_px, image_width_px, 3), dtype=np.uint8)
    out[:] = np.array(background, dtype=np.uint8)

    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

    xs = np.arange(image_width_px, dtype=float)
    ys = np.arange(image_height_px, dtype=float)
    Xp, Yp = np.meshgrid(xs, ys)

    x_cam = (Xp - cx) / fx
    y_cam = -(Yp - cy) / fy

    D = x_cam[..., None] * right + y_cam[..., None] * up + fwd
    D /= np.linalg.norm(D, axis=-1, keepdims=True)

    Dx, Dy, Dz = D[..., 0], D[..., 1], D[..., 2]
    Cx, Cy, Cz = C

    a = Dx * Dx - Dy * Dy
    b = 2.0 * (Cx * Dx - Cy * Dy - radius_cm * Dz)
    c = Cx * Cx - Cy * Cy - 2.0 * radius_cm * Cz

    disc = b * b - 4.0 * a * c
    valid = disc > 0.0

    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[valid] = np.sqrt(disc[valid])

    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        t_lin = -c / b

    valid &= (np.abs(a) > 1e-8)
    t = np.where(valid & (t1 > 1e-6), t1, np.inf)
    t = np.where(valid & (t2 > 1e-6) & (t2 < t), t2, t)

    lin_valid = (np.abs(a) <= 1e-8) & (np.abs(b) > 1e-8)
    t = np.where(lin_valid & (t_lin > 1e-6) & (t_lin < t), t_lin, t)

    valid_t = np.isfinite(t)

    Px = Cx + t * D[..., 0]
    Py = Cy + t * D[..., 1]

    in_patch = (
        valid_t
        & (Px >= -W_cm / 2.0)
        & (Px <= W_cm / 2.0)
        & (Py >= -H_cm / 2.0)
        & (Py <= H_cm / 2.0)
    )

    if np.any(in_patch):
        u_norm = (Px[in_patch] + W_cm / 2.0) / W_cm
        v_norm = (H_cm / 2.0 - Py[in_patch]) / H_cm

        u_tex = np.clip(u_norm * (Wtex - 1), 0, Wtex - 1)
        v_tex = np.clip(v_norm * (Htex - 1), 0, Htex - 1)

        ui = np.rint(u_tex).astype(int)
        vi = np.rint(v_tex).astype(int)
        out[in_patch] = tex[vi, ui]

    return Image.fromarray(out)


def deform_image_saddle(
    input_image,
    radius_cm,
    label_size_factor,
    camera_pos,
    view_dir,
    focal_px,
    output_size_px,
    roll=0.0,
    background=255,
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

    if is_gray:
        tex_rgb = np.stack([tex, tex, tex], axis=-1) if tex.ndim == 2 else np.repeat(tex[..., :1], 3, axis=2)
    else:
        tex_rgb = tex

    intrinsics = compute_intrinsics(
        image_width_px=output_size_px,
        image_height_px=output_size_px,
        focal_px=focal_px,
    )

    if isinstance(background, (tuple, list, np.ndarray)):
        bg = tuple(int(v) for v in np.asarray(background).ravel()[:3])
        if len(bg) == 1:
            bg = (bg[0], bg[0], bg[0])
        elif len(bg) == 2:
            bg = (bg[0], bg[1], bg[1])
    else:
        bg = (int(background), int(background), int(background))

    rendered = render_label_on_shape(
        input_image=Image.fromarray(tex_rgb),
        radius_cm=radius_cm,
        label_size_factor=label_size_factor,
        camera_pos=camera_pos,
        view_dir=view_dir,
        intrinsics=intrinsics,
        image_width_px=output_size_px,
        image_height_px=output_size_px,
        roll=roll,
        background=bg,
    )

    rendered_arr = np.asarray(rendered)

    if is_gray:
        rendered_gray = rendered_arr[..., 0]
        if return_array:
            return rendered_gray
        return Image.fromarray(rendered_gray, mode="L")

    if return_array:
        return rendered_arr
    return Image.fromarray(rendered_arr, mode="RGB")


deform_image = deform_image_saddle


def deformation_differential_at_label_point(
    u_cm,
    v_cm,
    radius_cm,
    camera_pos,
    view_dir,
    focal_px,
    output_size_px,
    roll=0.0,
    eps_u=1e-6,
    eps_v=1e-6,
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


def differential_at_image_point(
    point_xy,
    radius_cm,
    label_size_factor,
    camera_pos,
    view_dir,
    focal_px,
    output_size_px,
    roll=0.0,
    eps_u=1e-6,
    eps_v=1e-6,
):
    x_img, y_img = point_xy

    uv = image_point_to_saddle_label(
        x_img=x_img,
        y_img=y_img,
        radius_cm=radius_cm,
        label_size_factor=label_size_factor,
        camera_pos=camera_pos,
        view_dir=view_dir,
        focal_px=focal_px,
        output_size_px=output_size_px,
        roll=roll,
    )

    if uv is None:
        raise ValueError("Ce point image n'appartient pas à l'étiquette saddle visible.")

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
        eps_v=eps_v,
    )

    return {
        "point_image": np.array([x_img, y_img], dtype=float),
        "point_label": np.array([u_cm, v_cm], dtype=float),
        "J": J,
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
        patch_out[pad_top:patch_size - pad_bottom, pad_left:patch_size - pad_right] = patch
    else:
        patch_out = np.full((patch_size, patch_size, img.shape[2]), pad_value, dtype=img.dtype)
        patch_out[pad_top:patch_size - pad_bottom, pad_left:patch_size - pad_right, :] = patch

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
    radius_cm,
    label_size_factor,
    camera_pos,
    view_dir,
    focal_px,
    output_size_px,
    roll=0.0,
    eps_u=1e-6,
    eps_v=1e-6,
    shift_units="source_px",
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
        eps_v=eps_v,
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


def main():
    parser = argparse.ArgumentParser(description="Project a flat image onto a saddle and render it from a camera.")
    parser.add_argument("--input", type=str, help="Input texture file. If omitted with --make-grid, a grid is generated.")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--radius", type=float, default=8.0, help="Saddle scale radius in cm")
    parser.add_argument("--size-factor", type=float, default=3.0, help="Label width and height = size_factor * radius")

    parser.add_argument("--focal-px", type=float, default=700.0, help="Focal length in pixels (legacy/simple mode)")
    parser.add_argument("--focal-mm", type=float, default=None, help="Focal length in mm (real camera mode)")
    parser.add_argument("--sensor-width-mm", type=float, default=None, help="Sensor width in mm")
    parser.add_argument("--sensor-height-mm", type=float, default=None, help="Sensor height in mm")
    parser.add_argument("--principal-point-x", type=float, default=None, help="Principal point x in pixels; defaults to image center")
    parser.add_argument("--principal-point-y", type=float, default=None, help="Principal point y in pixels; defaults to image center")

    parser.add_argument("--output-width", type=int, default=768, help="Output image width in pixels")
    parser.add_argument("--output-height", type=int, default=768, help="Output image height in pixels")
    parser.add_argument("--roll-deg", type=float, default=0.0, help="Optional camera roll in degrees")
    parser.add_argument("--camera-x", type=float, default=None)
    parser.add_argument("--camera-y", type=float, default=None)
    parser.add_argument("--camera-z", type=float, default=None)
    parser.add_argument("--view-x", type=float, default=None)
    parser.add_argument("--view-y", type=float, default=None)
    parser.add_argument("--view-z", type=float, default=None)
    parser.add_argument("--random-pose", action="store_true", help="Draw camera pose at random under visibility constraint")
    parser.add_argument("--min-visible-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--make-grid", action="store_true", help="Generate a regular grid as input texture")
    parser.add_argument("--grid-size", type=int, default=768)
    parser.add_argument("--grid-step", type=int, default=32)
    parser.add_argument("--metadata", type=str, default=None, help="Optional JSON metadata output path")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.make_grid:
        input_image = generate_grid_image(size_px=args.grid_size, line_step_px=args.grid_step)
    else:
        if args.input is None:
            raise ValueError("Provide --input, or use --make-grid")
        input_image = Image.open(args.input).convert("RGB")

    intrinsics = compute_intrinsics(
        image_width_px=args.output_width,
        image_height_px=args.output_height,
        focal_px=args.focal_px if args.focal_mm is None else None,
        focal_mm=args.focal_mm,
        sensor_width_mm=args.sensor_width_mm,
        sensor_height_mm=args.sensor_height_mm,
        principal_point_x=args.principal_point_x,
        principal_point_y=args.principal_point_y,
    )

    if args.random_pose:
        camera_pos, view_dir, roll_rad, frac = random_camera_pose(
            radius_cm=args.radius,
            label_size_factor=args.size_factor,
            intrinsics=intrinsics,
            image_width_px=args.output_width,
            image_height_px=args.output_height,
            rng=rng,
            min_visible_fraction=args.min_visible_fraction,
        )
    else:
        needed = [args.camera_x, args.camera_y, args.camera_z, args.view_x, args.view_y, args.view_z]
        if any(v is None for v in needed):
            raise ValueError(
                "Either use --random-pose, or provide --camera-x --camera-y --camera-z --view-x --view-y --view-z"
            )
        camera_pos = np.array([args.camera_x, args.camera_y, args.camera_z], dtype=float)
        view_dir = np.array([args.view_x, args.view_y, args.view_z], dtype=float)
        roll_rad = math.radians(args.roll_deg)
        frac = visible_fraction(
            camera_pos,
            view_dir,
            intrinsics,
            args.output_width,
            args.output_height,
            args.radius,
            args.size_factor,
            roll=roll_rad,
            samples_x=100,
            samples_y=100,
        )

    rendered = render_label_on_shape(
        input_image=input_image,
        radius_cm=args.radius,
        label_size_factor=args.size_factor,
        camera_pos=camera_pos,
        view_dir=view_dir,
        intrinsics=intrinsics,
        image_width_px=args.output_width,
        image_height_px=args.output_height,
        roll=roll_rad,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(out_path)

    if args.metadata:
        meta = {
            "radius_cm": args.radius,
            "size_factor": args.size_factor,
            "label_size_cm": args.radius * args.size_factor,
            "camera_position_cm": camera_pos.tolist(),
            "view_direction": normalize(view_dir).tolist(),
            "roll_deg": math.degrees(roll_rad),
            "estimated_visible_fraction": float(frac),
            "image_width_px": args.output_width,
            "image_height_px": args.output_height,
            "intrinsics": intrinsics,
            "focal_px_input": args.focal_px if args.focal_mm is None else None,
            "focal_mm_input": args.focal_mm,
            "sensor_width_mm": args.sensor_width_mm,
            "sensor_height_mm": args.sensor_height_mm,
        }
        meta_path = Path(args.metadata)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
