import numpy as np
import PIL.Image as Image
import cv2
import sys, os, io
import PIL
import gcsfs
from cv2 import GaussianBlur
import random
from skimage.util import random_noise
import skimage.io
import skimage.filters
from scipy import fftpack
import pandas as pd
import argparse
from tqdm import tqdm

normalize = lambda x: (x - x.min())/(x.max() - x.min())
adapt_to_pil = lambda x:((x - x.min()) * (1/(x.max() - x.min()) * 255)).astype('uint8')
noise_type = ['gauss','poisson', 'speckle']


def noisy(noise_typ,image):
    if noise_typ == "gauss":
        return cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
    
    
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals)) 
        return np.random.poisson(image * vals) / float(vals)
    
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        return image + (image * gauss * 0.08)



def gcs_ls(uri):
    return ['gs://' + content for content in gcsfs.GCSFileSystem().ls(uri)]


def read_image(uri): 
    
    with gcsfs.GCSFileSystem().open(uri, "rb") as f:
         byte = f.read()
            
    pil_img = Image.open(io.BytesIO(byte)).convert('L')

    return np.array(pil_img)


def write_image(uri, img):
    
    img = Image.fromarray( img ,'L')
    
    with gcsfs.GCSFileSystem().open(uri, "wb") as f:
    
        img.save(f,"JPEG")


def get_noise_fourier(img_fft, treshold = 0.1 ):  
    r, c = img_fft.shape
    
    img_fft[int(r*(treshold)):int(r*(1-treshold))]  = 0
    img_fft[:, int(c*(treshold)):int(c*(1-treshold))]  = 0
    
    img_fft[int(r*(-1*treshold)):int(r*(1+treshold))]  = 0
    img_fft[:, int(c*(-1*treshold)):int(c*(1+treshold))]  = 0
   
    return img_fft


def main(gcs_source_folder, query, gcs_destination_folder, output_counter, treshold, alpha):
    
    uris_target = gcs_ls(gcs_source_folder) 
    pd_query = pd.read_gbq(query)
    uris_to_perturbe = pd_query.crop_image_path
    
    noise_real = []
    imgs_to_perturb = []
    imgs_perturbed = []
    
    
    for uri in tqdm(uris_target): 

        img_target = read_image(uri)
        img_fft = fftpack.fft2(img_target)
        noise_from_targets = get_noise_fourier(img_fft, treshold = 0.03)
        noise_real.append(fftpack.ifft2(noise_from_targets).real)

    for idx, uri in tqdm(enumerate(uris_to_perturbe)):

        if(idx <= int(output_counter)):
            img_to_perturb = read_image(uri)

            noise_real_interpolated=cv2.resize(noise_real[random.randint(0,len(noise_real)-1)], dsize=(img_to_perturb.shape[1], 
                                            img_to_perturb.shape[0]), interpolation=cv2.INTER_CUBIC)
            img_perturbed = (normalize(img_to_perturb)*alpha + normalize(noise_real_interpolated)*(1-alpha))
            
            random_noise_type = random.randint(0,1)
            if random_noise_type == 1:
                img_perturbed = adapt_to_pil(img_perturbed)
            else:
                try:
                    img_perturbed = adapt_to_pil(noisy(noise_type[random.randint(0, len(noise_type)-1)], img_perturbed))
                except:
                    print('too large values')

            img_perturbed_name = uri.split('/')[-1]                                      
            uri_destination_perturbed = os.path.join(gcs_destination_folder, img_perturbed_name)

            write_image(uri_destination_perturbed, img_perturbed)



def parse_args(args):
    parser = argparse.ArgumentParser()
      
    parser.add_argument('--gcs_source_folder', type=str, required=True, help="gcs folder containing images")
    parser.add_argument('--query', type=str, required=False, default = "SELECT * FROM formazione-marco-aspromonte.denoising.platesmania_dataset", help="query to get the images with BigQuery")
    parser.add_argument('--gcs_destination_folder', type=str, required=True, help="gcs folder containing noised output")
    parser.add_argument('--output_counter', type=str, required=True, help="int defining the number of outputs")
    parser.add_argument('--treshold', type=str, required=False, default = 0.03 , help="float defining the treshold for noise filtering")
    parser.add_argument('--alpha', type=str, required=False, default = 0.5 ,  help="float that define the weight of the noise wrt the good signal")
    
    return parser.parse_known_args(args)


if __name__ == "__main__":
    known_args, unknown_args = parse_args(sys.argv[1:])
    
    main(**vars(known_args))
