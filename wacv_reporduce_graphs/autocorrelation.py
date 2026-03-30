import numpy as np
from numpy.fft import fft2,ifft2, fftshift
import periodic_smooth as ps
import cv2 as cv


def autocorrelation(arg):
    u = arg
    p = ps.composante_periodique(u)
    fimg1 = fft2(p)
    fimgnorm = fimg1*np.conjugate(fimg1)
    fout1 = ifft2(fimgnorm)
    fout = fout1.real
    return fout




def autocorrelation_display(arg):
    u = arg
    p = ps.composante_periodique(u)
    
    fimg1 = fft2(p)
    fimg1[0, 0] = 0 
    
    fimgnorm = fimg1 * np.conjugate(fimg1)
    
    fout1 = ifft2(fimgnorm)
    autocorr = fftshift(fout1).real
    
    # Normalisation par l'énergie
    energy = np.sum(np.abs(p) ** 2)
    autocorr_normalized = autocorr / energy if energy != 0 else autocorr
    
    return autocorr_normalized


def correlation_nomalisee(arg1, arg2):
    u = arg1
    v = arg2
    p1 = ps.composante_periodique(u)
    p2 = ps.composante_periodique(v)
    fimg1 = fft2(p1)
    fimg2 = fft2(p2)
    fimgnorm = (fimg1/np.linalg.norm(fimg1))*(np.conjugate(fimg2)/np.linalg.norm(fimg2))
    fout1 = ifft2(fimgnorm)
    return fftshift(fout1).real


def phase_corr(im1,im2):
    image = np.float32(im1)
    template = np.float32(im2)

    # Use OpenCV phase correlation
    shift = cv.phaseCorrelate(image, template)
    return shift














