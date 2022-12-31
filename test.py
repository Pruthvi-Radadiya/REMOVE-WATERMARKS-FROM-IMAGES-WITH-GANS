# Example to load model and test it on test image

import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import vstack
from numpy import expand_dims
import matplotlib
import matplotlib.pyplot as plt
import random
 
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
 

test_dir = './wm-nowm/valid/watermark/'
# store and arrange name of all image file in a list
def GetFileName(dir_path):
  files = []
  for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        files.append(dir_path + path)
  return files

#select random test image
test_image_list = GetFileName(test_dir)
rand_test_img = test_image_list[random.randint(0, len(test_image_list)-1)]
print("..................................................\n")
print(rand_test_img)
print("\n..................................................\n")



# load test image
test_image = load_image(rand_test_img)
print('Loaded', test_image.shape)
# load model
model = load_model('./saved_model_150epc/gen_model_261150.h5')
# generate image
gen_image = model.predict(test_image)
images = vstack((test_image, gen_image))
# scale from [-1,1] to [0,1]
images = (images + 1) / 2.0
titles = ['Input', 'Generated']

for i in range(len(images)): 
    plt.subplot(1, 1+len(images), 1 + i)
    plt.axis('off')
    plt.imshow(images[i])
    plt.title(titles[i])
    filename = 'test_result'
    plt.savefig(filename)
    