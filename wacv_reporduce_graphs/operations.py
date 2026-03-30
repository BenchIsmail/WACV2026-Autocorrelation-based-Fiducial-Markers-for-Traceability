from PIL import Image
import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment




def validate_homography_matrix(H):
    if H.shape != (3, 3):
        raise ValueError("La matrice de transformation doit être de dimension 3x3")
    if H.dtype not in [np.float32, np.float64]:
        H = H.astype(np.float32)
    return H


def apply_homography(src_img, H, shape,borderValue):
    #int(np.mean(src_img))
    if not isinstance(src_img, np.ndarray):
        raise ValueError("L'image source doit être un numpy array")
    
    if len(src_img.shape) > 2:
        src_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    
    if not isinstance(H, np.ndarray):
        H = np.array(H)
    if H.shape != (3, 3):
        raise ValueError("La matrice d'homographie doit être de taille 3x3")
    
    try:
        return cv.warpPerspective(src_img, H, shape, borderValue=borderValue, flags=2)
    except Exception as e:
        raise

def apply_homography_corners(H, points):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # Homogeneous coords
    transformed = (H @ points_h.T).T
    transformed /= transformed[:, 2:3]  # Normalize by last coord
    return transformed[:, :2]

def circ_mask(h, k, r, x, y):
    return (x-h)**2 + (y-k)**2 <= r**2



def draw_peaks(image, points):
    h,w = image.shape
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image")
    if image.dtype != np.uint8:
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    color = (255, 0, 0)  # Red color in BGR format
    thickness = 1
    radius = 0 # Radius of the circle     

    for i in range(len(points)):
        color_image[int(np.round(points[i][1])),int(np.round(points[i][0]))] = color

    return color_image


def order_points_clockwise(points, center):
    angles = []
    for point in points:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    
    points_with_angles = list(zip(points, angles))
    points_ordered = [p[0] for p in sorted(points_with_angles, key=lambda x: x[1])]
    
    return points_ordered



def order_points(points_unordered, ref_points,
                              max_match_dist=None):

    pu = np.asarray(points_unordered, dtype=float).reshape(-1, 2)
    rp = np.asarray(ref_points,      dtype=float).reshape(-1, 2)

    D = np.linalg.norm(pu[:, None, :] - rp[None, :, :], axis=2)

    rows, cols = linear_sum_assignment(D)

    ordered = [None] * rp.shape[0]
    for r, c in zip(rows, cols):
        if max_match_dist is not None and D[r, c] > max_match_dist:
            ordered[c] = None           
        else:
            ordered[c] = pu[r]

    return rp.tolist(), [tuple(p) if p is not None else None for p in ordered]

def algebraic_distance(point, ref_point):
    # f(p, r) = (x_p - x_r)^2 + (y_p - y_r)^2
    return (point[0] - ref_point[0])**2 + (point[1] - ref_point[1])**2



def find_closest_points_order(points_unordered, ref_points, center):
    points_unordered = np.array(points_unordered)
    ref_points = np.array(ref_points)

    angles_ref = np.arctan2(ref_points[:, 1] - center[1], ref_points[:, 0] - center[0])
    sorted_ref_indices = np.argsort(angles_ref)  
    sorted_ref_points = ref_points[sorted_ref_indices]

    distance_matrix = np.linalg.norm(points_unordered[:, np.newaxis, :] - sorted_ref_points[np.newaxis, :, :], axis=2)

    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    sorted_points_unordered = points_unordered[row_indices[np.argsort(col_indices)]]

    return sorted_ref_points.tolist(), sorted_points_unordered.tolist()



def calculate_symmetric_points(image, x1, y1, x2, y2, x3, y3):
    h, w = image.shape
    points_unordered = [
        (w//2 - x1, h//2 - y1),
        (w//2 - x2, h//2 - y2),
        (w//2 - x3, h//2 - y3),
        (w//2 + x1, h//2 + y1),
        (w//2 + x2, h//2 + y2),
        (w//2 + x3, h//2 + y3),
    ]
    return [(x,y) for x, y in points_unordered]


def calculate_3d_basis(points_current, points_ref, scale=10):
    
    diag1 = np.array(points_current[0]) - np.array(points_current[3])
    diag2 = np.array(points_current[1]) - np.array(points_current[4])
    
    diag1_ref = np.array(points_ref[0]) - np.array(points_ref[3])
    diag2_ref = np.array(points_ref[1]) - np.array(points_ref[4])
    
    ratio1 = np.linalg.norm(diag1) / np.linalg.norm(diag1_ref)
    ratio2 = np.linalg.norm(diag2) / np.linalg.norm(diag2_ref)
    
    z_component = scale * (1 - min(ratio1, ratio2))
    
    diag1_3d = np.array([diag1[0], diag1[1], z_component])
    diag2_3d = np.array([diag2[0], diag2[1], z_component])
    
    normal_3d = np.cross(diag1_3d, diag2_3d)
    
    if np.linalg.norm(normal_3d) > 1e-6:
        normal_3d = normal_3d / np.linalg.norm(normal_3d) * scale
        
    normal_2d = np.array([normal_3d[0], normal_3d[1]])
    
    x_vector = diag1
    if np.linalg.norm(x_vector) > 1e-6:
        x_vector = x_vector / np.linalg.norm(x_vector) * scale
        
    y_vector = np.array([-normal_2d[1], normal_2d[0]])
    if np.linalg.norm(y_vector) > 1e-6:
        y_vector = y_vector / np.linalg.norm(y_vector) * scale
        
    return x_vector, y_vector, normal_2d

def draw_peaks_with_normal(image, x1, y1, x2, y2, x3, y3,
                         x1_ref, y1_ref, x2_ref, y2_ref, x3_ref, y3_ref):
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image")
    
    h, w = image.shape
    
    if image.dtype != np.uint8:
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    point_color_ref = (33, 140, 15)    # Vert foncé
    point_color = (255, 255, 255)        # Bleu foncé
    line_color_ref = (148, 148, 148)      # Vert
    line_color = (255, 255, 255)           # Rouge
    
    center = (w//2, h//2)
    deformed_points = calculate_symmetric_points(image, x1, y1, x2, y2, x3, y3)
    reference_points = calculate_symmetric_points(image, x1_ref, y1_ref, x2_ref, y2_ref, x3_ref, y3_ref)
    

    points_ref,points_current =  find_closest_points_order(deformed_points, reference_points, np.array([w//2, h//2]))


    for i in range(len(points_current)):
        cv.circle(color_image, points_ref[i], 2, point_color_ref, 1)
        cv.circle(color_image, points_current[i], 3, point_color, 1)
        
        next_point_ref = points_ref[(i + 1) % len(points_ref)]
        next_point = points_current[(i + 1) % len(points_current)]
        cv.line(color_image, points_ref[i], next_point_ref, line_color_ref, 2)
        cv.line(color_image, points_current[i], next_point, line_color, 3)
    
    x_vector, y_vector, normal = calculate_3d_basis(points_current, points_ref)
    
    start_point = tuple(map(int, center))
    
    end_point_normal = (int(start_point[0] + normal[0]), 
                       int(start_point[1] + normal[1]))
    cv.arrowedLine(color_image, start_point, end_point_normal, (255, 0, 0), 4)
    
    end_point_y = (int(start_point[0] + y_vector[0]), 
                  int(start_point[1] + y_vector[1]))
    cv.arrowedLine(color_image, start_point, end_point_y, (0, 0, 255), 4)
    
    
    end_point_x = (int(start_point[0] + x_vector[0]), 
                  int(start_point[1] + x_vector[1]))
    cv.arrowedLine(color_image, start_point, end_point_x, (0, 255, 0), 4)


    points_current = np.array(points_current)
    points_ref = np.array(points_ref)
    for i in range(3):
        start_point = tuple(points_ref[i].astype(int))
        end_point = tuple(points_current[i].astype(int))
        cv.arrowedLine(color_image, start_point, end_point, (0, 0, 0), 1, tipLength=0.2)
    
    
    return color_image




def image_centralizer(M, transf):
    h,w = M.shape
    x,y = w/2,h/2
    rot_centre_i_red = np.array([[x],[y]])@np.array([[x,y]])@([[transf[2][0]],[transf[2][1]]])+np.array([[transf[2][2]-transf[0][0],-transf[0][1]],[-transf[1][0],transf[2][2]-transf[1][1]]])@np.array([[x],[y]])
    transf[0][2] = rot_centre_i_red[0] 
    transf[1][2] = rot_centre_i_red[1] 
    return transf
    
def renormalize(image, point):
    h, w = image.shape
    x_new = point[0] - w // 2
    y_new = point[1] - h // 2
    return (x_new, y_new)
    


def transf_clac(image, matrice, vect) :
    
    transf = np.linalg.solve(matrice, vect).reshape((2,2))
    
    if np.linalg.cond(transf) > 1e12:
        print("[WARNING] Invalid or ill-conditioned transform matrix. Using identity fallback.")
        return None

    matrice_redressment = np.linalg.inv(transf)
    matrice_redressment = np.pad(matrice_redressment, 1)[1:][:,1:]
    matrice_redressment[2][2] = 1
    _, transformation = centralizer(image, matrice_redressment, image.shape,0)
    return transformation



def rescale_contrast(image_np, min_val, max_val):
    rescaled_image = image_np * (max_val - min_val) + min_val
    return rescaled_image


def resize_to_match(image1, image2):
    image1 = image1
    if len(image1.shape) == 2:
        h1, w1 = image1.shape
        ch1 = 1
    else:
        h1, w1, ch1 = image1.shape

    if len(image2.shape) == 2:
        h2, w2 = image2.shape
        ch2 = 1
    else:
        h2, w2, ch2 = image2.shape
    
    new_width = max(w1, w2)
    new_height = max(h1, h2)
    
    channels = max(ch1, ch2)
    if channels == 1:
        result = np.full((new_height, new_width), 0, dtype=np.uint8)
    else:
        result = np.full((new_height, new_width, channels), 0, dtype=np.uint8)
    
    x_center = (new_width - w1) // 2
    y_center = (new_height - h1) // 2
    
    if ch1 == 1 and channels > 1:
        for c in range(channels):
            result[y_center:y_center + h1, x_center:x_center + w1, c] = image1
    else:
        if channels == 1:
            result[y_center:y_center + h1, x_center:x_center + w1] = image1
        else:
            result[y_center:y_center + h1, x_center:x_center + w1] = image1
    h_,w_ = image1.shape
    h,w = image2.shape
    image2[int(h/2 - h_/2): int(h/2 + h_/2), int(w/2 - w_/2): int(w/2 + w_/2)] = 1
    return result, image1, image2




    
def reshape_to_match(image1, image2):
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    new_image_width = max(w1, w2)
    new_image_height = max(h1, h2)

    return cv.resize(image1, (new_image_width, new_image_height)),cv.resize(image2, (new_image_width, new_image_height))
    

def gray_to_png(image):
    normalized = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return cv.cvtColor(normalized, cv.COLOR_GRAY2BGR)



def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        #row,col,ch = image.shape
        s_vs_p = 0.005
        amount = 0.00004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    



def padding(image_1, image_2) :
    h, w = image_1.shape
    h_, w_ = image_2.shape    
    top_pad = (h - h_) // 2
    bottom_pad = h - h_ - top_pad
    left_pad = (w - w_) // 2
    right_pad = w - w_ - left_pad

    zone_resized = np.pad(
        image_2, 
        ((top_pad, bottom_pad), (left_pad, right_pad)), 
        mode='constant', 
        constant_values=0
    )
    return image_1, zone_resized

def padding_zone_corr(image_ref, zone):
    if image_ref.size >= zone.size :
        image_ref, zone_resized = padding(image_ref, zone)
        return image_ref, zone_resized
    else :
        zone_resized, image_ref = padding(zone,image_ref)
        return image_ref, zone_resized
    





def centralizer(patch, H_affine_3x3, canvas_shape_hw,border_val):
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
    border_val = int(np.round(np.mean(patch))) if border_val is None else border_val
    out = cv.warpPerspective(patch, H_final, (Wc, Hc),
                             flags=cv.INTER_LANCZOS4,
                             borderMode=cv.BORDER_CONSTANT,
                             borderValue=border_val)
    return out, H_final





from typing import List, Tuple, Optional

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



from typing import List, Tuple
import numpy as np
import cv2 as cv

def warped_corner_centers_from_matrix(
    matrix: np.ndarray,
    image_shape: Tuple[int, int],
    margin: int,
    *,
    keep_only_inside: bool = True,
) -> List[Tuple[int, int]]:
    H, W = image_shape
    if matrix.shape != (3, 3):
        raise ValueError("matrix doit être 3x3")
    if margin < 0:
        raise ValueError("margin doit être >= 0")

    m = int(margin)

    if (W - 1 - m) < m or (H - 1 - m) < m:
        return []

    src_xy = np.array([
        [m,       m],         # TL
        [W-1-m,   m],         # TR
        [m,       H-1-m],     # BL
        [W-1-m,   H-1-m],     # BR
        [(W-1)/2, (H-1)/2],   # CENTER
    ], dtype=np.float32)

    pts_cv = src_xy.reshape(-1, 1, 2)
    warped_xy = cv.perspectiveTransform(pts_cv, matrix).reshape(-1, 2)

    centers_rc: List[Tuple[int, int]] = []
    for xw, yw in warped_xy:
        if not np.isfinite(xw) or not np.isfinite(yw):
            continue

        if keep_only_inside and not (0 <= xw <= W - 1 and 0 <= yw <= H - 1):
            continue

        x = int(np.clip(round(xw), 0, W - 1))
        y = int(np.clip(round(yw), 0, H - 1))
        centers_rc.append((y, x))

    return centers_rc



def warped_grid_centers_from_matrix(
    matrix: np.ndarray,
    image_shape: Tuple[int, int],
    margin: int,
    *,
    keep_only_inside: bool = True,
) -> List[Tuple[int, int]]:
    H, W = image_shape
    if matrix.shape != (3, 3):
        raise ValueError("matrix doit être 3x3")
    if margin < 0:
        raise ValueError("margin doit être >= 0")

    m = int(margin)

    if (W - 1 - m) < m or (H - 1 - m) < m:
        return []

    xs = np.linspace(m, W - 1 - m, 3, dtype=np.float32)
    ys = np.linspace(m, H - 1 - m, 3, dtype=np.float32)

    src_xy = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)  # row-major

    pts_cv = src_xy.reshape(-1, 1, 2)
    warped_xy = cv.perspectiveTransform(pts_cv, matrix).reshape(-1, 2)

    centers_rc: List[Tuple[int, int]] = []
    for xw, yw in warped_xy:
        if not np.isfinite(xw) or not np.isfinite(yw):
            continue

        if keep_only_inside and not (0 <= xw <= W - 1 and 0 <= yw <= H - 1):
            continue

        x = int(np.clip(round(xw), 0, W - 1))
        y = int(np.clip(round(yw), 0, H - 1))
        centers_rc.append((y, x))

    return centers_rc
