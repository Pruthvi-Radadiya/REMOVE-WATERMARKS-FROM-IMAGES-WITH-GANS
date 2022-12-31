#test on a batch of image and check the performance
import os
import tensorflow as tf
import numpy as np 
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import vstack
from numpy import expand_dims
import matplotlib.pyplot as plt
import random
from keras.callbacks import CSVLogger
from collections import OrderedDict
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import cv2
from math import log10, sqrt
from PIL import Image as im
 
dir_path_test_wm = './wm-nowm/valid/watermark/'
dir_path_test_nwm = './wm-nowm/valid/no-watermark/'

# load an image
def load_image(filename, size=(256,256)):
 # load image with given size
 img = load_img(filename, target_size=size)
 # convert to numpy array
 img_arr = img_to_array(img)
 # scale from [0,255] to [-1,1]
 img_arr = (img_arr - 127.5) / 127.5
 # reshape to 1 sample
 img_arr = expand_dims(img_arr, 0)
 return img_arr


# plot input, generated and target images
def plot_images(step,inp_img, gen_img, tar_img):
 images = vstack((inp_img, gen_img, tar_img))
 # scale from [-1,1] to [0,1]
 images = (images + 1) / 2.0
 titles = ['Input', 'Generated', 'Target']
 # plot images row by row
 for i in range(len(images)):
    plt.subplot(1, 1+len(images), 1 + i)
    plt.axis('off')
    plt.imshow(images[i])
    plt.title(titles[i])
    filename = './test_loss_output/result_%03d.png' % (step+1)
    plt.savefig(filename)


# store and arrange name of all image file in a list
def GetFileName(dir_path):
  files = []
  for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        files.append(path)
  files.sort(key=str.casefold)
  return files

wm_list = GetFileName(dir_path_test_wm)
nwm_list = GetFileName(dir_path_test_nwm)

# Make a seperate list for same images in watermarker and non watermarked folder
# store full directory path of image in a list

def matchFileNames(wm_list, nwm_list, wm_dir, nwm_dir):
    sorted_wm = []
    sorted_nwm = []     
    for i in wm_list:
        if i in nwm_list:
            sorted_wm.append(wm_dir + i)
            sorted_nwm.append(nwm_dir + i)
        else:
            continue
    return sorted_wm, sorted_nwm
input_img_set, target_img_set = matchFileNames(wm_list, nwm_list, dir_path_test_wm, dir_path_test_nwm)

rand_num = random.sample(range(0, len(input_img_set)-1), 50)

def PSNR(imageA, imageB):
 gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
 mae_error = np.sum(np.absolute(gray1.astype("float") - gray2.astype("float")))
 mae_error /= float(imageA.shape[0] * imageA.shape[1])	
 max_pixel = 255.0
 psnr = 20 * log10(max_pixel / sqrt(mae_error))
 return psnr

def dssim_loss_fun(imageA, imageB):
 gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
 s = ssim(gray1, gray2)
 return (1-s)/2


list_psnr_loss=[]
list_dssim_loss=[]

list_psnr_loss_l1=[]
list_dssim_loss_l1=[]

list_psnr_loss_cg=[]
list_dssim_loss_cg=[]

for i in range(len(rand_num)):
    # load test image
    test_image = load_image(input_img_set[rand_num[i]])
    target_image = load_image(target_img_set[rand_num[i]])
    # load all 3 models
    model = load_model('./loss_output_psnr_dssim/gen_model/gen_model_000150.h5', compile=False)
    model_cg = load_model('./saved_output_cGAN/gen_model_000150.h5', compile=False)
    model_l1 = load_model('./saved_output_L1/gen_model_000150.h5', compile=False)
    # generate images for all 3 models
    gen_image = model.predict(test_image)
    gen_image_cg = model_cg.predict(test_image)
    gen_image_l1 = model_l1.predict(test_image)


    psnr_loss = []
    dssim_loss = []

    psnr_loss_l1=[]
    dssim_loss_l1=[]

    psnr_loss_cg=[]
    dssim_loss_cg=[]

    psnr_loss = PSNR(gen_image[0],target_image[0])
    dssim_loss = dssim_loss_fun(gen_image[0], target_image[0])

    psnr_loss_l1 = PSNR(gen_image_l1[0],target_image[0])
    dssim_loss_l1 = dssim_loss_fun(gen_image_l1[0], target_image[0])

    psnr_loss_cg = PSNR(gen_image_cg[0],target_image[0])
    dssim_loss_cg = dssim_loss_fun(gen_image_cg[0], target_image[0])
    print('>%d' % (i+1))
    
    list_psnr_loss.append(psnr_loss)
    list_dssim_loss.append(dssim_loss)
    list_psnr_loss_l1.append(psnr_loss_l1)
    list_dssim_loss_l1.append(dssim_loss_l1)
    list_psnr_loss_cg.append(psnr_loss_cg)
    list_dssim_loss_cg.append(dssim_loss_cg)
    od = OrderedDict()
    od['psnr_loss']=list_psnr_loss
    od['dssim_loss']=list_dssim_loss
    od['psnr_loss_l1']=list_psnr_loss_l1
    od['dssim_loss_l1']=list_dssim_loss_l1
    od['psnr_loss_cg']=list_psnr_loss_cg
    od['dssim_loss_cg']=list_dssim_loss_cg

    df = pd.DataFrame(od, columns=od.keys())
    df.to_csv('./test_loss_output/loss.csv',index=False)
    
    





