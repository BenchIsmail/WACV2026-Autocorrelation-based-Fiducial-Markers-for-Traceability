import numpy as np
from scipy.ndimage import binary_dilation

import operations as oprt
import deformation_generator as dg


#Ghostseal_noisy
def generate_white_noise_and_shifts(h, w, shift1, shift2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    #bruit blanc
    base = np.random.randn(h, w)

    #doubly_shift
    copy1 = np.roll(base, shift=shift1, axis=(0, 1))
    copy2 = np.roll(base, shift=shift2, axis=(0, 1))

    #Superposition
    combined = base+copy1+ copy2

    return combined


def generate_gs3d_noise_deformation(h, w, shift1, shift2, scale, rot_z, tilt, tilt_orient, rot_x, rot_y, seed=None):
    large_noise = generate_white_noise_and_shifts(h, w, shift1, shift2, seed=seed)
    final_deformation, final_homography = dg.general_deformer(large_noise,scale, rot_z, tilt, tilt_orient, rot_x, rot_y)

    return large_noise,final_deformation, final_homography








#random biliniar ghostseal
class ghostseal:
    def __init__(self, W, H, densite):
        self.W = W
        self.H = H
        self.status = "unexpanded"

        self.matrix = np.random.rand(self.H, self.W)
        
        threshold = densite
        self.binary_matrix = (self.matrix > threshold).astype(int)


    @staticmethod
    def generate(oligo1, oligo2, oligo3, dx2, dy2, dx3, dy3, densite):
        new_H = int(max(oligo1.H, oligo2.H + dy2, oligo3.H + dy3))
        new_W = int(max(oligo1.W, oligo2.W + dx2, oligo3.W + dx3))
    
        new_matrix = np.ones((3*new_H, 3*new_W))

        new_matrix[2*oligo1.H:3*oligo1.H, 2*oligo1.W:3*oligo1.W] += oligo1.binary_matrix

        new_matrix[2*oligo2.H + dy2:dy2 + 3*oligo2.H, 2*oligo2.W + dx2:dx2 + 3*oligo2.W] += oligo2.binary_matrix
        new_matrix[2*oligo3.H + dy3:dy3 + 3*oligo3.H, 2*oligo3.W + dx3:dx3 + 3*oligo3.W] += oligo3.binary_matrix
        new_oligo = ghostseal(new_W, new_H, densite)
        new_oligo.binary_matrix = new_matrix*(-1)
        new_oligo.binary_matrix = np.where(new_oligo.binary_matrix>-4, 0, 255)

        new_oligo.binary_matrix = new_oligo.binary_matrix.astype(np.float32)

        image_cropped_net = new_oligo.binary_matrix[2*oligo1.H + max(int(dy2),int(dy3)):3*oligo1.H+min(0,dy2,dy3),2*oligo1.W + max(int(dx2),int(dx3)):3*oligo1.W+min(0,dx2,dx3)]

        return image_cropped_net 

    
    @staticmethod
    def add_fusion_2(oligo1, oligo2, dx2, dy2):
        h,w = oligo1.shape
        h_,w_ = oligo2.shape
        new_H = max(h, h_ + dy2, )
        new_W = max(w, w_ + dx2)
        
    
        new_matrix = np.ones((3*new_H, 3*new_W))

        new_matrix[2*h:3*h, 2*w:3*w] += oligo1

        new_matrix[2*h_ + dy2:dy2 + 3*h_, 2*w_ + dx2:dx2 + 3*w_] += oligo2

        image_cropped_net = new_matrix[2*h + dy2:3*h,2*w + dx2:3*w]

        return image_cropped_net
    @staticmethod
    def add_fusion_3(oligo1, oligo2, oligo3,dx2, dy2,dx3, dy3):
        h,w = oligo1.shape
        h_,w_ = oligo2.shape
        h__,w__ = oligo3.shape

        new_H = max(h, h_ + dy2, h__ + dy3)
        new_W = max(w, w_ + dx2, w__ + dx3)

        new_matrix = np.ones((3*new_H, 3*new_W))

        new_matrix[2*h:3*h, 2*w:3*w] += oligo1

        new_matrix[2*h_ + dy2:dy2 + 3*h_, 2*w_ + dx2:dx2 + 3*w_] += oligo2
        new_matrix[2*h__ + dy3:dy3 + 3*h__, 2*w__ + dx3:dx3 + 3*w__] += oligo3
        image_cropped_net = new_matrix[2*h + max(dy2,dy3):3*h,2*w + max(dx2,dx3):3*w]

        return image_cropped_net
    

def gen_random_binary_texture(h, w, dilation_size_i,density, angle_shift, norm_shift):

    k = angle_shift * np.pi / 180
    cos_k, sin_k = np.cos(k), np.sin(k)

    shift1_x_i, shift1_y_i = 0, norm_shift
    shift2_x_i = shift1_x_i * cos_k + shift1_y_i * sin_k
    shift2_y_i = -shift1_x_i * sin_k + shift1_y_i * cos_k

    structuring_element_i = np.ones((2 * dilation_size_i + 1, 2 * dilation_size_i + 1), dtype=bool)


    ghostss_i = ghostseal(h, w, density)
    ghostsseal_i = ghostseal.generate(
        ghostss_i, ghostss_i, ghostss_i,
        int(shift1_x_i), int(shift1_y_i),
        int(shift2_x_i), int(shift2_y_i),
        density
    )

    transformed_image_i = np.ones_like(ghostsseal_i, dtype=np.uint8) * 255
    dilated_image_i = binary_dilation(ghostsseal_i == 0, structure=structuring_element_i)
    transformed_image_i[dilated_image_i] = 0

    return transformed_image_i


