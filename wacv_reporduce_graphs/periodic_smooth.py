import numpy as np
from numpy.fft import fft2, ifft2



'''Periodic_smooth'''
def construit_indices_centres(M,N):
    i = M//2 - (M//2 - np.arange(0,M)) % M  # (0, 1, ..., m/2, -m/2, ..., -1)
    j = N//2 - (N//2 - np.arange(0,N)) % N  # (0, 1, ..., n/2, -n/2, ..., -1)
    return np.meshgrid(j, i)


def miroir(u):
    return np.vstack((np.hstack((u,np.fliplr(u))),np.hstack((np.flipud(u),np.flipud(np.fliplr(u))))))


def laplacien_per_dft2(u):
    M,N = u.shape
    U = fft2(u)
    (ii, jj) = construit_indices_centres(M, N)
    U *= -4 * np.pi**2 * ( jj**2/(M**2) + ii**2/(N**2))
    return np.real(ifft2(U))


def laplacien_sym_dft2(u):
    M,N = u.shape
    v = laplacien_per_dft2(miroir(u))
    return v[:M,:N]


def composante_periodique(u):    
    M,N = u.shape
    v = laplacien_sym_dft2(u)
    V = fft2(v)
    
    (ii, jj) = construit_indices_centres(M, N)
    
    ii[0,0] = 1 # to prevent the division by zero at 0,0
    V /= -4* np.pi**2 * ( jj **2/(M**2) + ii **2/(N**2) )
    V[0,0] = np.sum(u)
    
    return np.real(ifft2(V))
