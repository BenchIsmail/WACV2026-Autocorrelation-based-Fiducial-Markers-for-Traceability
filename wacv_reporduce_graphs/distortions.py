import cv2
import numpy as np
from deformation_generator import projection_perspective


def add_noise(image, intensity=25):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)


    image_float = image.astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=intensity, size=image.shape)

    noisy = image_float + noise
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def add_blur(image, ksize=5):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def add_rotation(image, angle):
    h, w = image.shape
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=255)


def add_zoom(image, factor):
    h, w = image.shape
    zoomed = cv2.resize(image, None, fx=factor, fy=factor)
    if factor > 1:
        return zoomed[:h, :w]
    else:
        out = np.ones((h, w), dtype=image.dtype) * 255
        zh, zw = zoomed.shape
        out[:zh, :zw] = zoomed
        return out


def add_occlusion(image, percentage_of_occlusion=0.3, max_attempts=1000):
    h, w = image.shape
    total_pixels = h * w
    target_pixels = int(total_pixels * percentage_of_occlusion)

    occlusion_mask = np.zeros((h, w), dtype=bool)
    current_occluded = 0
    attempts = 0

    while current_occluded < target_pixels and attempts < max_attempts:
        band_w = np.random.randint(w // 20, w // 4)
        band_h = np.random.randint(h // 20, h // 4)
        x = np.random.randint(0, w - band_w)
        y = np.random.randint(0, h - band_h)

        # Extract region
        region = occlusion_mask[y:y + band_h, x:x + band_w]
        new_occlusion = np.count_nonzero(~region)  # how many new pixels to occlude
        if new_occlusion == 0:
            attempts += 1
            continue

        occlusion_mask[y:y + band_h, x:x + band_w] = True
        current_occluded += new_occlusion
        attempts = 0  # reset if progress made

    # Apply occlusion: set occluded pixels to white (or black if preferred)
    image[occlusion_mask] = 255  # white occlusion (or 0 for black)
    return image




def gblur(x, σ):
	from numpy.fft import fft2, ifft2, fftfreq
	from numpy import meshgrid, exp, pi as π
	h,w = x.shape                              # shape of the rectangle
	p,q = meshgrid(fftfreq(w), fftfreq(h))     # build frequency abscissae
	X = fft2(x)                                # move to frequency domain
	F = exp(-2 * π**2 * σ**2 * (p**2 + q**2) ) # filter in frequency domain
	Y = F*X                                    # apply filter
	y = ifft2(Y).real                          # go back to spatial domain
	return y

__global_random = 0 # seed protect
def randg(s):
	import numpy
	global __global_random
	if not __global_random:
		__global_random = numpy.random.default_rng(0)
	if isinstance(s, numpy.ndarray):
		s = s.shape
	return __global_random.standard_normal(s)


def omask(s, σ, q):
    from numpy import quantile
    x = randg(s)            # white noise
    y = gblur(x, σ)         # blur the noise by σ
    t = quantile(y, q/100)  # find binarization threshold
    z = 255.0*(y > t)       # binarize
    return z


def apply_occlusion(img, σ=16, q=50):
    mask = omask(img.shape, σ, q)
    #return np.where(mask == 255, np.mean(img), img)
    occluded = img * (mask)                # supprime les pixels masqués
    return occluded


def add_homography(image, angle_x=0, angle_y=0):
    img, projected, matrix = projection_perspective(image, angle_vue_x=angle_x, angle_vue_y=angle_y, focal=1000)
    return img, projected, matrix



def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    q = int(max(0, min(100, quality)))

    # 1) Mise à l'échelle si ce n'est pas de l'uint8
    if img.dtype != np.uint8:
        imin = float(img.min())
        imax = float(img.max())
        if imax > imin:
            img_norm = (img - imin) / (imax - imin)   # -> [0, 1]
            img8 = (img_norm * 255.0).astype(np.uint8)
        else:
            # image constante : on fait juste un tableau de 0
            img8 = np.zeros_like(img, dtype=np.uint8)
    else:
        img8 = img

    # 2) Encodage JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, q]
    ok, enc = cv2.imencode(".jpg", img8, encode_params)
    if not ok:
        return img8

    # 3) Décodage JPEG (on garde le type d'image : niveaux de gris si 2D, BGR si 3 canaux)
    dec = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return dec if dec is not None else img8