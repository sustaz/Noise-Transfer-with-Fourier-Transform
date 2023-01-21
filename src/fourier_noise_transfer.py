from scipy import fftpack
from tqdm import tqdm
from skimage.util import random_noise
import skimage.io
import skimage.filters
import cv2
from cv2 import GaussianBlur
import numpy as np
import os

normalize = lambda x: (x - x.min())/(x.max() - x.min())

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        return cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
    
    
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy


def get_noise_fourier(img_fft, treshold = 0.1 ):  
    r, c = img_fft.shape
    
    img_fft[int(r*(treshold)):int(r*(1-treshold))]  = 0
    img_fft[:, int(c*(treshold)):int(c*(1-treshold))]  = 0
    
    img_fft[int(r*(-1*treshold)):int(r*(1+treshold))]  = 0
    img_fft[:, int(c*(-1*treshold)):int(c*(1+treshold))]  = 0
   
    return img_fft


def noise_transfer_single_img(img_perturbed, img_to_perturb, external_noise = False):
    img_perturbed_fft = fftpack.fft2(img_perturbed)
    noise_fft = get_noise_fourier(img_perturbed_fft, treshold = 0.03)
    noise_real = fftpack.ifft2(noise_fft).real 
    noise_interpolated = cv2.resize(noise_real, dsize=(img_to_perturb.shape[1], img_to_perturb.shape[0]), 
                                                                            interpolation=cv2.INTER_CUBIC)
    if external_noise == True:
        img_to_perturb = noisy(img_to_perturb)
    
    noise_interpolated_norm = normalize(noise_interpolated)
    img_to_perturb_norm = normalize(img_to_perturb)
    
    return img_to_perturb_norm + noise_interpolated_norm
