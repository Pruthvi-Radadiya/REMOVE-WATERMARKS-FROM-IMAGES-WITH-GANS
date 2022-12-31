# GAN network for remove watermark from images
import numpy as np 
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import time
from keras.callbacks import CSVLogger
from collections import OrderedDict
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import cv2
from math import log10, sqrt
from PIL import Image as im

# discriminator model
def discriminator(img_shape):
 # weight initialization
 initializer = RandomNormal(mean=0.0, stddev=0.02)
 # image input
 input_image = Input(shape=img_shape)
 target_image = Input(shape=img_shape)
 # concatenate input and traget images
 merged_image = Concatenate()([input_image, target_image])
 
 disc = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(merged_image)
 disc = LeakyReLU(alpha=0.2)(disc)
 
 disc = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(disc)
 disc = BatchNormalization()(disc)
 disc = LeakyReLU(alpha=0.2)(disc)

 disc = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(disc)
 disc = BatchNormalization()(disc)
 disc = LeakyReLU(alpha=0.2)(disc)
 
 disc = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(disc)
 disc = BatchNormalization()(disc)
 disc = LeakyReLU(alpha=0.2)(disc)
 # second last layer
 disc = Conv2D(512, (4,4), padding='same', kernel_initializer=initializer)(disc)
 disc = BatchNormalization()(disc)
 disc = LeakyReLU(alpha=0.2)(disc)
 # output layer
 disc = Conv2D(1, (4,4), padding='same', kernel_initializer=initializer)(disc)
 patch_out = Activation('sigmoid')(disc)
 # define and compile model
 model = Model([input_image, target_image], patch_out)
 opt = Adam(learning_rate=0.0002, beta_1=0.5)
 model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
 return model
 


# Encoder block
def encoder_block(input_img, filters_no, batchnormalization=True):
 # weight initialization
 initializer = RandomNormal(mean=0.0, stddev=0.02)
 #downsampling 
 enc = Conv2D(filters_no, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(input_img)
 # add batch normalization conditionally
 if batchnormalization:
        enc = BatchNormalization()(enc, training=True)
 #  activation function (LeakyReLU)
 enc = LeakyReLU(alpha=0.2)(enc)
 return enc
 
# Decoder block
def decoder_block(input_layer, skip_connection, filters_no, dropout=False):
 # weight initialization
 initializer = RandomNormal(stddev=0.02)
 #upsampling
 dec = Conv2DTranspose(filters_no, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(input_layer)
 # batch normalization
 dec = BatchNormalization()(dec, training=True)
 # add dropout conditionally
 if dropout:      
        dec = Dropout(0.5)(dec, training=True)
 # merge with skip connection
 dec = Concatenate()([dec, skip_connection])
 # activation function (relu)
 dec = Activation('relu')(dec)
 return dec
 
# generator model
def generator(img_shape=(256,256,3)):
 # weight initialization
 initializer = RandomNormal(mean=0.0, stddev=0.02)
 # image input
 input_image = Input(shape=img_shape)
 # encoder model
 enc1 = encoder_block(input_image, 64, batchnormalization=False)
 enc2 = encoder_block(enc1, 128)
 enc3 = encoder_block(enc2, 256)
 enc4 = encoder_block(enc3, 512)
 enc5 = encoder_block(enc4, 512)
 enc6 = encoder_block(enc5, 512)
 enc7 = encoder_block(enc6, 512)
 # bottleneck
 btl_nc = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(enc7)
 btl_nc = Activation('relu')(btl_nc)
 # decoder model
 disc1 = decoder_block(btl_nc, enc7, 512, dropout=True)
 disc2 = decoder_block(disc1, enc6, 512, dropout=True)
 disc3 = decoder_block(disc2, enc5, 512, dropout=True)
 disc4 = decoder_block(disc3, enc4, 512)
 disc5 = decoder_block(disc4, enc3, 256)
 disc6 = decoder_block(disc5, enc2, 128)
 disc7 = decoder_block(disc6, enc1, 64)
 # output
 gen = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(disc7)
  # activation function (tanh)
 output_image = Activation('tanh')(gen)
 # define model
 model = Model(input_image, output_image)
 return model
 
# combined generator and discriminator model, for updating the generator
def gan(gen_model, disc_model, img_shape):
 # the discriminator weights not trainable
 for layer in disc_model.layers:
     if not isinstance(layer, BatchNormalization):
         layer.trainable = False
 # define the input(watermarked) image
 input_image = Input(shape=img_shape)
 # input image as the generator input
 gen_out = gen_model(input_image)
 # input image and generator output as discriminator input
 disc_out = disc_model([input_image, gen_out])
 # input image as input, classification and generated image output
 model = Model(input_image, [disc_out, gen_out])
 # Optimizer (Adam) and compile model
 opt = Adam(learning_rate=0.0002, beta_1=0.5)
 model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
 return model
 
# load and prepare training images
def load_real_samples(file_name):
 # load compressed arrays
 training_dataset = load(file_name)
 arr0, arr1 = training_dataset['arr_0'], training_dataset['arr_1']

 # scale from [0,255] to [-1,1]
 arr0 = (arr0 - 127.5) / 127.5
 arr1 = (arr1 - 127.5) / 127.5
 return [arr0, arr1]
 
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, samples, patch_shape):
 train_A, train_B = dataset
 # choose random instances
 rand_num = randint(0, train_A.shape[0], samples)
 # retrieve selected images
 img1, img2 = train_A[rand_num], train_B[rand_num]
 # generate 'real' class labels (ones)
 real_lable = ones((samples, patch_shape, patch_shape, 1))
 return [img1, img2], real_lable
 

def generate_fake_samples(gen_model, samples, patch_shape):
 # generate fake image
 fake_img = gen_model.predict(samples)
 # create 'fake' class labels (zeros)
 fake_lab = zeros((len(fake_img), patch_shape, patch_shape, 1))
 return fake_img, fake_lab
 

# PSNR with MAE for evaluation of GAN
def PSNR(imageA, imageB):
 #convert to gray scale
 gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
 # calculate MAE error
 mae_error = np.sum(np.absolute(gray1.astype("float") - gray2.astype("float")))
 mae_error /= float(imageA.shape[0] * imageA.shape[1])	
 max_pixel = 255.0
 psnr = 20 * log10(max_pixel / sqrt(mae_error))
 return psnr

#DSSIM to check simillarity of generated image
def dssim_loss_fun(imageA, imageB):
 gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
 s = ssim(gray1, gray2)
 return (1-s)/2


# generate samples and save as a plot and save the model
def visualization_output(step, gen_model, dataset, samples=3):
 # select a sample of input images
 [real_input, real_target], _ = generate_real_samples(dataset, samples, 1)
 # generate a fake samples
 fake_gen, _ = generate_fake_samples(gen_model, real_input, 1)
 # scale all pixels from [-1,1] to [0,1]
 real_input = (real_input + 1) / 2.0
 real_target = (real_target + 1) / 2.0
 fake_gen = (fake_gen + 1) / 2.0
 plt.figure(figsize=(12,12))
 # plot real input images
 for i in range(samples):
     plt.subplot(3, samples, 1 + i)
     plt.axis('off')
     plt.imshow(real_input[i])
 # plot generated image
 for i in range(samples):
     plt.subplot(3, samples, 1 + samples + i)
     plt.axis('off')
     plt.imshow(fake_gen[i])
 # plot real target image
 for i in range(samples):
     plt.subplot(3, samples, 1 + samples*2 + i)
     plt.axis('off')
     plt.imshow(real_target[i])
 # save plot as a image file
 filename1 = './loss_output/result_%04d.png' % (step+1)
 plt.savefig(filename1)
 plt.close()
 # save the generator model
 filename2 = './loss_output/gen_model_%04d.h5' % (step+1)
 gen_model.save(filename2)
 print('>Saved: %s and %s' % (filename1, filename2))
 
# train pix2pix model
def train(disc_model, gen_model, gan_model, dataset, epochs=150, batch=1):

    print('training the model')
    epoch_start=0
    epoch_end=epochs
    #list to save performance
    list_epoch=[]
    list_epoch_duration=[]
    #loss output
    list_dis_loss_real=[]
    list_dis_loss_fake=[]
    list_dis_loss_total=[]
    list_gan_loss_bce=[]
    list_gan_loss_mae=[]
    list_gan_loss_total=[]
    list_psnr_loss=[]
    list_dssim_loss=[]
    print(gen_model.summary())
    print(disc_model.summary())
    print(gan_model.summary())
    # determine the output square shape of the discriminator
    patch = disc_model.output_shape[1]
    # unpack dataset
    input_img, target_img = dataset
    # calculate the number of batches per training epoch
    batches_per_epoch = int(len(input_img) / batch)
    # enumerate epochs
    for epc in np.arange(epoch_start,epoch_end):
        start=time.time()
        print()
        print('epoch',epc+1,'of',epoch_start+epochs)

        #initialization of performance output
        dis_loss_real=np.zeros(batches_per_epoch)
        dis_loss_fake=np.zeros(batches_per_epoch)
        gan_loss_total=np.zeros(batches_per_epoch)
        gan_loss_bce=np.zeros(batches_per_epoch)
        gan_loss_mae=np.zeros(batches_per_epoch)
        psnr_loss=np.zeros(batches_per_epoch)
        dssim_loss=np.zeros(batches_per_epoch)
        #train on the training image data
        for i in range(batches_per_epoch): 
            # batch of real samples
            [real_input, real_target], real_lable = generate_real_samples(dataset, batch, patch)
            # generate fake samples batch
            fake_image, fake_lable = generate_fake_samples(gen_model, real_input, patch)
            # update discriminator for real image
            dis_loss_real[i] = disc_model.train_on_batch([real_input, real_target], real_lable)
            # update discriminator for generated image
            dis_loss_fake[i] = disc_model.train_on_batch([real_input, fake_image], fake_lable)
            # update the generator
            #the generator is indirectly trained through the gan
            gan_loss_total[i], gan_loss_bce[i], gan_loss_mae[i] = gan_model.train_on_batch(real_input, [real_lable, real_target])
            #calculation of evaluation matrics
            psnr_loss[i] = PSNR(fake_image[0],real_target[0])
            dssim_loss[i] = dssim_loss_fun(fake_image[0], real_target[0])
            end=time.time()

            # model performance output
            print('>%d, dLreal[%.3f] dLfake[%.3f] gLmae[%.3f] gLtotal[%.3f] psnr_l1[%.3f] dssim_l[%.3f]' % (i+1, dis_loss_real[i], dis_loss_fake[i], gan_loss_mae[i], gan_loss_total[i]
            , psnr_loss[i], dssim_loss[i]))
        # save the model and visualize generated image 
        if (epc+1) % (10) == 0:
            visualization_output(epc, gen_model, dataset)
        end=time.time()
        # save the output performance
        list_epoch.append(epc+1)
        list_epoch_duration.append(end-start)
        list_dis_loss_real.append(dis_loss_real.mean())
        list_dis_loss_fake.append(dis_loss_fake.mean())
        list_dis_loss_total.append(dis_loss_real.mean()+dis_loss_fake.mean())
        list_gan_loss_total.append(gan_loss_total.mean())
        list_gan_loss_bce.append(gan_loss_bce.mean())
        list_gan_loss_mae.append(gan_loss_mae.mean())
        list_psnr_loss.append(psnr_loss.mean())
        list_dssim_loss.append(dssim_loss.mean())


        od = OrderedDict()
        od['epoch']=list_epoch
        od['epoch_duration']=list_epoch_duration
        od['dis_loss_real']=list_dis_loss_real
        od['dis_loss_fake']=list_dis_loss_fake
        od['dis_loss_total']=list_dis_loss_total
        od['gan_loss_bce']=list_gan_loss_bce
        od['gan_loss_mae']=list_gan_loss_mae
        od['gan_loss_total']=list_gan_loss_total
        od['psnr_loss']=list_psnr_loss
        od['dssim_loss']=list_dssim_loss

        df = pd.DataFrame(od, columns=od.keys())
        df.to_csv('./loss_output/loss.csv',index=False)

 
# load image data
dataset = load_real_samples('Training_data.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

# define input shape
image_shape = dataset[0].shape[1:]
# define the models
disc_model = discriminator(image_shape)
gen_model = generator(image_shape)
# define the gan model
gan_model = gan(gen_model, disc_model, image_shape)
# train model
train(disc_model, gen_model, gan_model, dataset)