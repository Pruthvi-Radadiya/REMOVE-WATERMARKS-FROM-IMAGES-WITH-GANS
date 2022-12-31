
import tensorflow as tensorflow
from tensorflow import keras
import os
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from numpy import load

#directory path for training data for watermarked and non watermarekd images
dir_path_train_wm = './wm-nowm/train/watermark/'
dir_path_train_nwm = './wm-nowm/train/no-watermark/'

# store and arrange name of all image file in a list
def GetFileName(dir_path):
  files = []
  for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        files.append(path)
  files.sort(key=str.casefold)
  return files

wm_list = GetFileName(dir_path_train_wm)
nwm_list = GetFileName(dir_path_train_nwm)

print((wm_list[0:5]))
print(len(nwm_list[0:5]))
type(nwm_list)

# Make a seperate list for same images in watermarker and non watermarked folder
# store fill directory path of image in a list

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


# Input Data lists: input_img_set = watermarked images & target_img_set = non watermarked images
input_img_set, target_img_set = matchFileNames(wm_list, nwm_list, dir_path_train_wm, dir_path_train_nwm)

print(len(input_img_set))
print(len(target_img_set))
print(input_img_set[1:5])
print(target_img_set[1:5])

# load and convert dataset into np array for training

def load_images(wm_img_set,nwm_img_set, size=(256,256)):
    wmlist, nwmlist = list(), list()
    # enumerate filenames in directory
    for filename in wm_img_set:
        # load and resize the image
        img = load_img(filename, target_size=size)
        # convert to numpy array
        img_arr = img_to_array(img)
        wmlist.append(img_arr)

    for filename in nwm_img_set:   
        img = load_img(filename, target_size=size)  
        img_arr = img_to_array(img)
        nwmlist.append(img_arr)  
        
    return [asarray(wmlist), asarray(nwmlist)]
 

# load dataset
[wm_images, nwm_images] = load_images(input_img_set, target_img_set)
print('Loaded: ', wm_images.shape, nwm_images.shape)
print(type(wm_images))
# save as compressed numpy array
Comp_data_filename = 'Train_data.npz'
savez_compressed(Comp_data_filename, wm_images, nwm_images)
print('Saved: ', Comp_data_filename)




