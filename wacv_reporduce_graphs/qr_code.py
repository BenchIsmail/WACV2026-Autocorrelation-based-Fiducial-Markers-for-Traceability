import os
import cv2
import numpy as np
import qrcode



def generate_qrcode(marker_size=512):
    desired_modules = 33
    border = 1
    box_size = marker_size // (desired_modules + 2 * border)

    qr = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size,
        border=border,
    )
    qr.add_data("test")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    img = np.array(img)

    h, w = img.shape
    padded_img = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
    y_offset = (marker_size - h) // 2
    x_offset = (marker_size - w) // 2
    padded_img[y_offset:y_offset + h, x_offset:x_offset + w] = img

    return padded_img


def estimate_qrcode_homography(image):
    detector = cv2.QRCodeDetector()
    retval, points = detector.detect(image)
    if retval and points is not None:
        points = points[0]  # Shape: (4, 2)
        model_pts = np.array([
            [38, 38],
            [961, 38],
            [961, 961],
            [38, 961]
        ], dtype=np.float32)
        H, _ = cv2.findHomography(model_pts, points)
        #estimated_H = np.linalg.inv(H)
        #estimated_H /= estimated_H[2,2]
        return H
    return None








def make_qr(content: str,
            qr_size: int = 200,
            border: int = 1,
            version: int = 4,
            error_correction=None):
    if error_correction is None:
        error_correction = qrcode.constants.ERROR_CORRECT_L

    desired_modules = 4 * version + 17
    box_size = max(1, qr_size // (desired_modules + 2 * border))

    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(content)
    qr.make(fit=False)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    img = np.array(img)

    if img.shape[0] != qr_size:
        img = cv2.resize(img, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)

    return img  

def qr_chessboard(rows: int = 6,
                  cols: int = 8,
                  cell_size: int = 256,
                  margin: int = 40,
                  per_cell_payload: bool = True,
                  payload_prefix: str = "cell"):

    H = rows * cell_size
    W = cols * cell_size
    board = np.zeros((H, W), dtype=np.uint8)

    qr_size = cell_size - 2 * margin
    assert qr_size > 0, "margin trop grand par rapport à cell_size"

    for r in range(rows):
        for c in range(cols):
            cell_color = 255 if (r + c) % 2 == 0 else 0

            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size

            board[y0:y1, x0:x1] = cell_color

            payload = f"{payload_prefix}_{r}_{c}" if per_cell_payload else payload_prefix
            qr_img = make_qr(payload, qr_size)

            y_off = y0 + margin
            x_off = x0 + margin

            board[y_off:y_off + qr_size, x_off:x_off + qr_size] = qr_img

    return board



def cylindrical_projection_realistic(
    img: np.ndarray,
    f: float,
    R: float = None,
    num_tiles: int = 1,
    border_value: float = 0.0,
    cx: float = None,
    cy: float = None,
):
    if img.ndim == 2:
        H, W = img.shape
        channels = 1
    elif img.ndim == 3:
        H, W, channels = img.shape
    else:
        raise ValueError("img must be 2D or 3D")

    if R is None:
        R = f

    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    W_out = int(W * num_tiles)
    H_out = H

    j_out, i_out = np.meshgrid(np.arange(W_out), np.arange(H_out))
    cx_p = W_out / 2.0
    cy_p = H_out / 2.0

    u_p = j_out.astype(np.float64)
    v_p = i_out.astype(np.float64)

    theta = (u_p - cx_p) / R
    h = (v_p - cy_p) / R

    x = np.tan(theta)
    r = np.sqrt(1.0 + x**2)
    y = h * r

    u_src = f * x + cx
    v_src = f * y + cy

    map_x = u_src.astype(np.float32)
    map_y = v_src.astype(np.float32)

    if channels == 1:
        img_src = img.astype(np.float32)
        proj = cv2.remap(
            img_src, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(border_value),
        )
    else:
        proj = np.zeros((H_out, W_out, channels), dtype=np.float32)
        for c in range(channels):
            proj[..., c] = cv2.remap(
                img[..., c].astype(np.float32),
                map_x, map_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(border_value),
            )

    return proj


def project_points_cylindrical_realistic(
    points_uv: np.ndarray,
    f: float,
    R: float,
    src_shape,
    num_tiles: int = 1,
    cx: float = None,
    cy: float = None,
):
    H, W = src_shape[:2]

    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    W_out = int(W * num_tiles)
    H_out = H
    cx_p = W_out / 2.0
    cy_p = H_out / 2.0

    points_uv = np.asarray(points_uv, dtype=np.float64)
    u = points_uv[:, 0]
    v = points_uv[:, 1]

    x = (u - cx) / f
    y = (v - cy) / f

    theta = np.arctan(x)
    r = np.sqrt(1.0 + x**2)
    h = y / r

    u_p = R * theta + cx_p
    v_p = R * h + cy_p

    return np.stack([u_p, v_p], axis=-1)


def get_deformed_qr_centers(
    rows: int,
    cols: int,
    cell_size: int,
    margin: int,
    f: float,
    R: float,
    src_shape,
    num_tiles: int = 1,
    cx: float = None,
    cy: float = None,
):
    H, W = src_shape[:2]
    qr_size = cell_size - 2 * margin

    centers_array = np.zeros((rows, cols, 2), dtype=np.float64)
    centers_dict = {}

    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_size + margin
            x1 = x0 + qr_size
            y0 = r * cell_size + margin
            y1 = y0 + qr_size

            pts_src = np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ], dtype=np.float64)

            pts_proj = project_points_cylindrical_realistic(
                pts_src,
                f=f,
                R=R,
                src_shape=src_shape,
                num_tiles=num_tiles,
                cx=cx,
                cy=cy,
            )

            center = pts_proj.mean(axis=0)  # (u', v')

            centers_array[r, c, :] = center
            centers_dict[(r, c)] = center

    return centers_array, centers_dict



def compute_projected_qr_quads_for_chessboard(
    rows: int,
    cols: int,
    cell_size: int,
    margin: int,
    f: float,
    R: float,
    src_shape,
    num_tiles: int = 1,
):
    H, W = src_shape[:2]
    qr_size = cell_size - 2 * margin

    cell_quads = {}

    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_size + margin
            x1 = x0 + qr_size
            y0 = r * cell_size + margin
            y1 = y0 + qr_size

            pts_src = np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ], dtype=np.float64)

            pts_proj = project_points_cylindrical_realistic(
                pts_src,
                f=f,
                R=R,
                src_shape=src_shape,
                num_tiles=num_tiles,
            )

            cell_quads[(r, c)] = pts_proj  # (4,2)

    return cell_quads



def detect_and_classify_qr_in_projection(
    proj_img: np.ndarray,
    rows: int,
    cols: int,
    cell_quads: dict,
    per_cell_payload: bool = True,
    payload_prefix: str = "cell",
):
    if proj_img.ndim == 3:
        img_gray = cv2.cvtColor(proj_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        img_gray = proj_img.astype(np.uint8)

    detector = cv2.QRCodeDetector()

    # Initialisation de l'état par cellule
    cell_info = {}
    for r in range(rows):
        for c in range(cols):
            cell_info[(r, c)] = {
                "detected": False,
                "decoded": False,
                "message": None,
                "quad": cell_quads[(r, c)],
            }

    if hasattr(detector, "detectAndDecodeMulti"):
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img_gray)
    else:
        ok, decoded_info, points = False, [], None

    if not ok or points is None:
        return cell_info

    points = np.array(points, dtype=np.float32)  # (N,4,2)
    N = len(points)


    cell_centroids = {}
    for (r, c), info in cell_info.items():
        quad = info["quad"]
        cx = quad[:, 0].mean()
        cy = quad[:, 1].mean()
        cell_centroids[(r, c)] = np.array([cx, cy], dtype=np.float32)

    for i in range(N):
        pts = points[i]      # (4,2)
        msg = decoded_info[i] if i < len(decoded_info) else ""
        center = pts.mean(axis=0)  # (2,)

        best_cell = None
        best_dist2 = np.inf
        for cell, cc in cell_centroids.items():
            d2 = np.sum((center - cc) ** 2)
            if d2 < best_dist2:
                best_dist2 = d2
                best_cell = cell

        if best_cell is None:
            continue

        info = cell_info[best_cell]
        info["detected"] = True

        if msg:
            info["decoded"] = True
            info["message"] = msg

    return cell_info

def build_cell_info_from_detections(
    rows: int,
    cols: int,
    cell_quads: dict,
    detections: list,
):

    cell_info = {}
    for r in range(rows):
        for c in range(cols):
            cell_info[(r, c)] = {
                "detected": False,
                "decoded": False,
                "message": None,
                "quad": cell_quads[(r, c)],
            }

    cell_centroids = {}
    for (r, c), quad in cell_quads.items():
        cx = quad[:, 0].mean()
        cy = quad[:, 1].mean()
        cell_centroids[(r, c)] = np.array([cx, cy], dtype=np.float32)

    for d in detections:
        if not d["success"] or d["points"] is None:
            continue

        pts = d["points"].astype(np.float32)   # (4,2)
        msg = d["message"] if d["message"] is not None else ""
        decoded = bool(msg)

        cell = d.get("cell", None)

        if cell is None:
            center = pts.mean(axis=0)
            best_cell = None
            best_dist2 = np.inf
            for ckey, cc in cell_centroids.items():
                d2 = float(np.sum((center - cc)**2))
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_cell = ckey
            cell = best_cell

        if cell is None:
            continue

        info = cell_info[cell]

        if info["decoded"]:
            continue

        info["detected"] = True
        info["decoded"] = decoded
        if decoded:
            info["message"] = msg

    return cell_info


def detect_all_qr_corners(board_img: np.ndarray,
                          rows: int = None, cols: int = None,
                          cell_size: int = None, margin: int = None):
    detector = cv2.QRCodeDetector()

    if board_img.dtype != np.uint8:
        img = board_img.astype(np.float32)
        img_min = float(img.min())
        img_max = float(img.max())
        if img_max <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = board_img

    if hasattr(detector, "detectAndDecodeMulti"):
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img)

        if ok and points is not None:
            detections = []
            N = len(points)

            for i in range(N):
                pts = points[i].astype(np.float32)   # (4,2)
                msg = decoded_info[i] if i < len(decoded_info) else ""
                decoded = bool(msg)

                detections.append({
                    "success": True,
                    "decoded": decoded,
                    "points": pts,
                    "message": msg,
                    "cell": None
                })

            return detections

    if None in (rows, cols, cell_size, margin):
        raise ValueError(
            "detectAndDecodeMulti indisponible ou échec. "
            "Fournis rows, cols, cell_size et margin pour le fallback par cellule."
        )

    detections = []
    qr_size = cell_size - 2 * margin
    H, W = img.shape[:2]

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size

            yy0 = y0 + margin
            yy1 = yy0 + qr_size
            xx0 = x0 + margin
            xx1 = xx0 + qr_size

            if yy1 > H or xx1 > W:
                detections.append({
                    "success": False,
                    "decoded": False,
                    "points": None,
                    "message": None,
                    "cell": (r, c)
                })
                continue

            crop = img[yy0:yy1, xx0:xx1]

            if crop.ndim == 3:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                crop_gray = crop

            msg, pts, _ = detector.detectAndDecode(crop_gray)

            if pts is not None and len(pts) > 0:
                pts = pts.astype(np.float32)

                pts[:, 0] += xx0
                pts[:, 1] += yy0

                decoded = bool(msg)

                detections.append({
                    "success": True,
                    "decoded": decoded,
                    "points": pts,
                    "message": msg,
                    "cell": (r, c)
                })
            else:
                detections.append({
                    "success": False,
                    "decoded": False,
                    "points": None,
                    "message": None,
                    "cell": (r, c)
                })

    return detections



def draw_qr_overlay_on_projection_from_cells(
    proj_img: np.ndarray,
    cell_info: dict,
    alpha: float = 0.3,
    upscale: int = 2,  
):

    if proj_img.ndim == 2:
        base = cv2.cvtColor(proj_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        base = proj_img.astype(np.uint8).copy()

    H, W = base.shape[:2]

    H_up = H * upscale
    W_up = W * upscale

    base_up = cv2.resize(base, (W_up, H_up), interpolation=cv2.INTER_CUBIC)
    overlay_up = base_up.copy()


    GREEN = (0, 255, 0)
    ORANGE = (0, 165, 255)
    RED = (0, 0, 255)

    for cell, info in cell_info.items():
        quad = info["quad"]          # (4,2) en coords de proj_img
        detected = info["detected"]
        decoded = info["decoded"]

        if decoded:
            color = GREEN
        elif detected:
            color = ORANGE
        else:
            color = RED

        quad_up = quad * upscale
        pts_int = quad_up.reshape(-1, 1, 2).astype(np.int32)

        cv2.fillPoly(overlay_up, [pts_int], color)

    out_up = cv2.addWeighted(overlay_up, alpha, base_up, 1 - alpha, 0)

    for cell, info in cell_info.items():
        quad = info["quad"]
        detected = info["detected"]
        decoded = info["decoded"]

        if decoded:
            color = GREEN
        elif detected:
            color = ORANGE
        else:
            color = RED

        quad_up = quad * upscale
        pts_int = quad_up.reshape(-1, 1, 2).astype(np.int32)

        cv2.polylines(
            out_up,
            [pts_int],
            isClosed=True,
            color=color,
            thickness=2 * upscale,
            lineType=cv2.LINE_AA,
        )

    out = cv2.resize(out_up, (W, H), interpolation=cv2.INTER_AREA)

    return out





