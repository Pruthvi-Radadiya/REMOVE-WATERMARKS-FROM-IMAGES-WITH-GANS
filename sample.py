
# Example of loading model and use it for other sample of training image 
from keras.models import load_model
from numpy import load
from numpy import vstack
import matplotlib.pyplot as plt 
from numpy.random import randint
import tensorflow as tf



# load and prepare training images
def load_real_samples(file_name):
 training_dataset = load(file_name)
 arr0, arr1 = training_dataset['arr_0'], training_dataset['arr_1']
 # scale from [0,255] to [-1,1]
 arr0 = (arr0 - 127.5) / 127.5
 arr1 = (arr1 - 127.5) / 127.5
 return [arr0, arr1]
 
# plot input, generated and target images
def plot_images(inp_img, gen_img, tar_img):
 images = vstack((inp_img, gen_img, tar_img))
 # scale from [-1,1] to [0,1]
 images = (images + 1) / 2.0
 titles = ['Input', 'Generated', 'Target']
 for i in range(len(images)):
    plt.subplot(1, 1+len(images), 1 + i)
    plt.axis('off')
    plt.imshow(images[i])
    plt.title(titles[i])
    filename = 'model_example'
    plt.savefig(filename)
    
 
# load dataset
[inp_arr, tar_arr] = load_real_samples('Training_data.npz')
print('Loaded', inp_arr.shape, tar_arr.shape)
# load model
model = load_model('./loss_output_psnr_dssim/gen_model/gen_model_000150.h5')
# select random example
rand_ex = randint(0, len(inp_arr), 1)
input_image, target_image = inp_arr[rand_ex], tar_arr[rand_ex]
# generate image 
gen_image = model.predict(input_image)
# plot all three images
plot_images(input_image, gen_image, target_image)



