import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter
import operations as oprt





def find_local_maxima(image):
    neighborhood = np.ones((50, 50), bool)  
    local_max = maximum_filter(image, footprint=neighborhood) == image
    return local_max

def calculate_prominence(image, local_max):
    prominence = np.zeros_like(image, dtype=float)
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            if local_max[i, j]:
                min_value = np.min(image[max(0, i-1):min(rows, i+2), max(0, j-1):min(cols, j+2)])
                prominence[i, j] = image[i, j] - min_value

    return prominence


def generate_hexagon_from_vectors(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    return np.array([
        v1,
        v2,
        v2 - v1,
        -v1,
        -v2,
        v1 - v2
    ], dtype=float)

def is_convex(polygon):
    polygon = np.array(polygon)
    n = len(polygon)
    if n < 4:
        return True  # Triangle is always convex

    signs = []
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        c = polygon[(i + 2) % n]
        ab = b - a
        bc = c - b
        cross = np.cross(ab, bc)
        signs.append(np.sign(cross))
    
    signs = np.array(signs)
    return np.all(signs > 0) or np.all(signs < 0)

def sort_points_cyclically(points):
    points = np.array(points)
    center = np.mean(points, axis=0)
    shifted = points - center
    angles = np.arctan2(shifted[:,1], shifted[:,0])
    return points[np.argsort(angles)]


def check_affine_hexagon(points, v1, v2, tol=3):
    if len(points) != 6:
        return False

    # Générer hexagone de référence
    ref_hex = generate_hexagon_from_vectors(v1, v2)
    ref_hex = sort_points_cyclically(ref_hex)
    ref_hex = ref_hex - np.mean(ref_hex, axis=0)

    # Centrer et trier les points détectés
    pts = np.array(points, dtype=float)
    pts = sort_points_cyclically(pts)
    pts = pts - np.mean(pts, axis=0)
    # Vérifier convexité
    if not is_convex(pts):
        return False

    # Ajustement affine (M @ ref = pts)
    X = ref_hex.T
    Y = pts.T
    M = Y @ np.linalg.pinv(X)
    pred_pts = (M @ X).T

    error = np.linalg.norm(pred_pts - pts, axis=1).mean()
    return error < tol



def subpixel_peak_position_quadratic(image, x, y):
    x_min = max(0, x-1)
    x_max = min(image.shape[0], x+2)
    y_min = max(0, y-1)
    y_max = min(image.shape[1], y+2)
    
    region = image[x_min:x_max, y_min:y_max]
    pad_x = 3 - region.shape[0]
    pad_y = 3 - region.shape[1]
    if pad_x > 0 or pad_y > 0:
        region = np.pad(region, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=0)

    dx = (region[2, 1] - region[0, 1]) / (2 * (2 * region[1, 1] - region[2, 1] - region[0, 1]))
    dy = (region[1, 2] - region[1, 0]) / (2 * (2 * region[1, 1] - region[1, 2] - region[1, 0]))
    
    return y + dy, x + dx




def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coords
    x0, y0 = float(x0), float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return A * np.exp( - (a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) + offset

def subpixel_peak_position_gaussian_fit(image, x, y):
    x = int(round(x))
    y = int(round(y))
    
    x_min = max(0, x-1)
    x_max = min(image.shape[0], x+2)
    y_min = max(0, y-1)
    y_max = min(image.shape[1], y+2)
    
    region = image[x_min:x_max, y_min:y_max]
    
    if region.shape != (3, 3):
        region = np.pad(region, ((0, 3 - region.shape[0]), (0, 3 - region.shape[1])), mode='constant')

    # Create grid for fitting
    x_grid, y_grid = np.meshgrid(np.arange(3), np.arange(3))
    coords = (x_grid.ravel(), y_grid.ravel())
    values = region.ravel()

    # Initial guess: A, x0, y0, sigma_x, sigma_y, theta, offset
    A0 = np.max(values) - np.min(values)
    offset0 = np.min(values)
    x0 = y0 = 1.0  # center of 3x3
    initial_guess = [A0, x0, y0, 1.0, 1.0, 0.0, offset0]

    def error(params):
        return np.sum((gaussian_2d(coords, *params) - values)**2)

    result = minimize(error, initial_guess, method='L-BFGS-B')
    
    if result.success:
        _, x0_fit, y0_fit, *_ = result.x
        subpixel_x = y + (x0_fit - 1)
        subpixel_y = x + (y0_fit - 1)
        return subpixel_x, subpixel_y
    else:
        # fallback
        return float(y), float(x)




def subpixel_peak_position_centroid(image, x, y):
    x_min = max(0, x-1)
    x_max = min(image.shape[0], x+2)
    y_min = max(0, y-1)
    y_max = min(image.shape[1], y+2)
    
    region = image[x_min:x_max, y_min:y_max]
    region = np.pad(region, ((0, 3 - region.shape[0]), (0, 3 - region.shape[1])), mode='constant')

    total_intensity = np.sum(region)
    if total_intensity == 0:
        return y, x
    
    x_coords, y_coords = np.meshgrid(np.arange(region.shape[1]), np.arange(region.shape[0]))
    x_weighted = np.sum(x_coords * region) / total_intensity
    y_weighted = np.sum(y_coords * region) / total_intensity

    return y + (x_weighted - 1), x + (y_weighted - 1)



def subpixel_peak_position_fit2D(image, x, y):
    from numpy.linalg import lstsq
    
    x_min = max(0, x-1)
    x_max = min(image.shape[0], x+2)
    y_min = max(0, y-1)
    y_max = min(image.shape[1], y+2)
    
    region = image[x_min:x_max, y_min:y_max]
    region = np.pad(region, ((0, 3 - region.shape[0]), (0, 3 - region.shape[1])), mode='constant')

    X, Y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    Z = region.flatten()
    
    A = np.column_stack((X.flatten()**2, Y.flatten()**2, X.flatten()*Y.flatten(), X.flatten(), Y.flatten(), np.ones(9)))
    coeffs, _, _, _ = lstsq(A, Z, rcond=None)
    a, b, c, d, e, f = coeffs

    # Calcul du sommet de la parabole 2D
    denom = 4*a*b - c**2
    if denom == 0:
        return y, x

    x0 = (c*e - 2*b*d) / denom
    y0 = (c*d - 2*a*e) / denom

    return y + x0, x + y0


def peaks_and_shifts_subpixelic(image, v1, v2, method):
    local_max = find_local_maxima(image)
    prominence = calculate_prominence(image, local_max)
    flat_prominence = prominence.flatten()
    sorted_indices = np.argsort(flat_prominence)[-7:]
    peak_positions = np.unravel_index(sorted_indices, image.shape)
    method_dict = {
        'quadratic': subpixel_peak_position_quadratic,
        'gaussian': subpixel_peak_position_gaussian_fit,
        'centroid': subpixel_peak_position_centroid,
        'fit2d': subpixel_peak_position_fit2D
    }

    subpixel_func = method_dict.get(method)
    if not subpixel_func:
        raise ValueError("Méthode inconnue : choisir parmi 'quadratic', 'gaussian', 'centroid', 'fit2d'")

    subpixel_positions = []
    for x, y in zip(peak_positions[0], peak_positions[1]):
        sub_x, sub_y = subpixel_func(image, x, y)
        subpixel_positions.append((sub_x, sub_y))

    #return subpixel_positions[:-1]
    if check_affine_hexagon(subpixel_positions[:6], v1, v2, tol=1):# and check_min_distance(subpixel_positions[:-1], 2):
        return subpixel_positions[:-1]
    
    else:
        return []
    

def find_centred_peaks(image, v1, v2, method) :
    peaks = peaks_and_shifts_subpixelic(image, v1, v2, method)
    if peaks is None or (hasattr(peaks, "size") and peaks.size == 0) or (hasattr(peaks, "__len__") and len(peaks) == 0):
        return []


    pts_cur = []
    for p in peaks:
        q = oprt.renormalize(image, p)
        if q is not None and len(q) == 2:
            pts_cur.append(q)


    x1u, y1u = v1
    x3u, y3u = v2
    x2u, y2u = x1u - x3u, y1u - y3u
    x4u, y4u = -x1u, -y1u
    x5u, y5u = -x2u, -y2u
    x6u, y6u = -x3u, -y3u
    renorm_ref = [(x1u, y1u), (x2u, y2u), (x3u, y3u),
                    (x4u, y4u), (x5u, y5u), (x6u, y6u)]

    ref, picked = oprt.order_points(pts_cur, renorm_ref)
    return picked


def find_peak(arr, excluded_regions, h, w):
    work_arr = arr.copy()
    
    for (cy, cx, radius) in excluded_regions:
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius + 1)
        work_arr[y_min:y_max, x_min:x_max] = -np.inf
        
        sy = h - cy
        sx = w - cx
        y_min = max(0, sy - radius)
        y_max = min(h, sy + radius + 1)
        x_min = max(0, sx - radius)
        x_max = min(w, sx + radius + 1)
        work_arr[y_min:y_max, x_min:x_max] = -np.inf
    #Image.fromarray(work_arr).save("/Users/i.bencheikh/Desktop/fp.tif")
    return np.unravel_index(np.argmax(work_arr), work_arr.shape)




def find_peak_position(image, method):
    local_max = find_local_maxima(image)
    prominence = calculate_prominence(image, local_max)
    flat_prominence = prominence.flatten()
    sorted_indices = np.argsort(flat_prominence)[-1:]
    peak_positions = np.unravel_index(sorted_indices, image.shape)
    method_dict = {
        'quadratic': subpixel_peak_position_quadratic,
        'gaussian': subpixel_peak_position_gaussian_fit,
        'centroid': subpixel_peak_position_centroid,
        'fit2d': subpixel_peak_position_fit2D
    }

    subpixel_func = method_dict.get(method)
    if not subpixel_func:
        raise ValueError("Méthode inconnue : choisir parmi 'quadratic', 'gaussian', 'centroid', 'fit2d'")

    sub_x, sub_y = subpixel_func(image, peak_positions[0][0], peak_positions[1][0])

    return sub_x, sub_y

