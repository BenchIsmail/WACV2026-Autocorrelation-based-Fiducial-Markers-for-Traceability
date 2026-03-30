import numpy as np
import time as time
import cv2 as cv
import operations as oprt




def matrix_affinity_generator(lambdaa, theta, tilde, phi, translation):
    tx,ty = translation
    affinity = np.array([[lambdaa, 0], 
                           [0, lambdaa] ])@np.array([[np.cos(-theta*np.pi/180), -np.sin(-theta*np.pi/180)], 
                           [np.sin(-theta*np.pi/180), np.cos(-theta*np.pi/180)] ])@np.array([[tilde, 0], 
                           [0, 1] ])@np.array([[np.cos(theta*np.pi/180), -np.sin(theta*np.pi/180)], 
                           [np.sin(theta*np.pi/180), np.cos(theta*np.pi/180)] ])@np.array([[np.cos(phi*np.pi/180), -np.sin(phi*np.pi/180)], 
                           [np.sin(phi*np.pi/180), np.cos(phi*np.pi/180)] ])
    
    homography = np.pad(affinity, 1)[1:][:,1:]
    homography[2][2] = 1
    homography[0][2] = tx
    homography[1][2] = ty
    return homography




def border_remover(M):
    h,w = M.shape
    M[:1,:] =np.median(M[:4,:])
    M[:,:1] =np.median(M[:,:4])
    M[h-1:,:]=np.median(M[h-4:,:])
    M[:,w-1:]=np.median(M[:,w-4:])
    return M



def image_deformation_generator(image, transf):
    h,w = image.shape
    deformation, matrix = oprt.centralizer(image, transf, (w,h),int(np.mean(image)))
    #image_deformed = oprt.apply_homography(image,deformation, (w,h),int(np.mean(image)))
    return deformation, matrix



def affine_deformer(image,lambdaa, theta, tilt, phi, translation):
    image_befor = image
    affinity = matrix_affinity_generator(lambdaa, theta, tilt, phi, translation)
    image_deformed, matrix = image_deformation_generator(image_befor, affinity)
    return image_deformed, matrix



def projection_perspective(image, angle_vue_x=30, angle_vue_y=0, focal=1000):
    height, width = image.shape[:2]

    # Convertir les angles en radians
    ax = np.radians(angle_vue_x)
    ay = np.radians(angle_vue_y)

    # Coordonnées 3D normalisées des coins de l'image (plan Z=0)
    points_3d = np.float32([
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 1, 0]
    ])

    # Matrices de rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)]
    ])
    Ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)]
    ])
    R = Rx @ Ry

    # Appliquer la rotation
    rotated_points = (R @ points_3d.T).T

    # Projection perspective
    z_offset = 2.0
    projected = rotated_points.copy()
    projected[:, 0] = rotated_points[:, 0] * focal / (rotated_points[:, 2] + z_offset)
    projected[:, 1] = rotated_points[:, 1] * focal / (rotated_points[:, 2] + z_offset)

    # Mise à l'échelle pour s'adapter à l'image
    min_x, max_x = projected[:, 0].min(), projected[:, 0].max()
    min_y, max_y = projected[:, 1].min(), projected[:, 1].max()
    scale = min(width / (max_x - min_x), height / (max_y - min_y))
    offset_x = (width - scale * (max_x - min_x)) / 2 - scale * min_x
    offset_y = (height - scale * (max_y - min_y)) / 2 - scale * min_y

    projected[:, 0] = projected[:, 0] * scale + offset_x
    projected[:, 1] = projected[:, 1] * scale + offset_y

    # Points source de l'image
    src_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    # Points destination (après projection)
    dst_points = np.float32(projected[:, :2])

    # Matrice de transformation homographique
    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    # Appliquer la transformation
    result = cv.warpPerspective(
    image, matrix, (width, height),
    borderValue=int(np.mean(image)),
    flags=cv.INTER_CUBIC)

    return image, result, matrix




def general_deformer(image,scale, rot_z, tilt, tilt_orient, rot_x, rot_y):
    #affinity
    image_deformed, affinity = affine_deformer(
        image, scale, tilt_orient, tilt, rot_z, (0, 0)
    )
    
    #perspective
    _, deformm, perspective_matrix = projection_perspective(
                        image_deformed.copy(), angle_vue_x=rot_x, angle_vue_y=rot_y
                    )
    
    final_homography = perspective_matrix@affinity
    final_deformation = oprt.apply_homography(image, final_homography, image.shape, int(np.mean(image)))
    return final_deformation, final_homography





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
        raise ValueError("img must be 2D or 3D (grayscale or color)")

    if R is None:
        R = f

    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    # Taille de sortie (simplement W * num_tiles, même hauteur)
    W_out = int(W * num_tiles)
    H_out = H

    # Coordonnées de sortie (u', v') en pixels
    j_out, i_out = np.meshgrid(np.arange(W_out), np.arange(H_out))

    # Centre de l'image de sortie
    cx_p = W_out / 2.0
    cy_p = H_out / 2.0

    u_p = j_out.astype(np.float64)
    v_p = i_out.astype(np.float64)

    # 1) coords "cylindre" (theta, h) à partir de (u', v')
    theta = (u_p - cx_p) / R        # angle
    h = (v_p - cy_p) / R            # hauteur normalisée

    # 2) direction (x, y) dans le plan image
    x = np.tan(theta)
    r = np.sqrt(1.0 + x**2)
    y = h * r

    # 3) retour en coordonnées pixel dans l'image source
    u_src = f * x + cx
    v_src = f * y + cy

    map_x = u_src.astype(np.float32)
    map_y = v_src.astype(np.float32)

    if channels == 1:
        img_src = img.astype(np.float32)
        proj = cv.remap(
            img_src, map_x, map_y,
            interpolation=cv.INTER_CUBIC,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=float(border_value),
        )
    else:
        proj = np.zeros((H_out, W_out, channels), dtype=np.float32)
        for c in range(channels):
            proj[..., c] = cv.remap(
                img[..., c].astype(np.float32),
                map_x, map_y,
                interpolation=cv.INTER_CUBIC,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=float(border_value),
            )

    return proj


def cylindrical_jacobians_at_points_realistic(
    points_uv,
    f: float,
    R: float = None,
    cx: float = None,
    cy: float = None,
):
    results = []

    if R is None:
        R = f

    for (u, v) in points_uv:
        u = float(u)
        v = float(v)

        if cx is None or cy is None:
            raise ValueError("cx et cy doivent être fournis pour calculer le Jacobien réaliste.")

        # normalisation
        x = (u - cx) / f
        y = (v - cy) / f

        # dérivées
        denom1 = (1.0 + x**2)
        sqrt1 = np.sqrt(denom1)

        du_p_du = R / (f * denom1)
        du_p_dv = 0.0

        dv_p_du = - R * x * y / (f * denom1**1.5)
        dv_p_dv = R / (f * sqrt1)

        J = np.array([
            [du_p_du, du_p_dv],
            [dv_p_du, dv_p_dv]
        ], dtype=float)

        results.append({
            "point": (u, v),
            "x": x,
            "y": y,
            "J": J,
        })

    return results


