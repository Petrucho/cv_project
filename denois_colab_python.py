import numpy as np
import cv2
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import streamlit as st
from PIL import Image
import PIL  

import os
import glob

# print(tf.version.VERSION)

import torch

# Create model
def autoencoder():
    model = Sequential()

    # input layer
    model.add(layers.Input(shape=(400,400, 1)))

    # encoder section
    model.add(layers.Conv2D(32, (3, 3), activation='relu',strides=2,padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',strides=2,padding='same'))
    model.add(layers.BatchNormalization())
    
    # decoder section
    model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu',strides=2,padding='same'))
    model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu',strides=2,padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(1, (3, 3), activation='sigmoid',strides=1, padding='same'))

    # compile model
    model.compile(optimizer='adam' , loss='mean_squared_error', metrics=['mae'])

    #print model summary
    model.summary()

    return model

# create model
model = autoencoder()
# print(f'created model: autoencoder\n')

# download weigths from learned model
model.load_weights('./data/denoising.h5')
# print(f'loaded model weights\n')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # print(f'torch.cuda.empty_cache\n')

st.title('Unnoising images')

# determine current_directory to store file in it
current_directory = os.getcwd()
print(f'\ncurrent_directory: {current_directory}\n')

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    # print(f'\ntype(im): {type(im)}\n')
    # print(f'img.name: {img.name}')
    # print(f'img.type: {img.type}')
    
    # print(f'\nim.dir(): {im.dir()}\n')
    print(f'\nthe path is:\n{current_directory + "/" + img.name}\n')
    im.save(current_directory + "/" + img.name)
    image = np.array(im)
    return im, image

def preprocess(img):    
    img = np.asarray(img, dtype="float32")
    img = img/255.0 #Scaling the pixel values    
    return img.reshape(400,400,1)


# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
if uploadFile is not None:
    img, image = load_image(uploadFile)
    # print(f'\nuploadFile: {uploadFile}\n')
    # print(f'\ntype(uploadFile): {type(uploadFile)}\n')
    # print(f'\nimg: {img}\n')
    # print(f'\nimage: {image}\n')
    
    # showing original image
    st.write('Noising image')    
    st.image(img)
    # print(f'type(img): {type(img)}\n')
        
    test_imgs = []    
    print('\nbefore preprocess\n')
    test_imgs.append(preprocess(img))
    print('\nafter preprocess\n')
    test_imgs = np.asarray(test_imgs)

    # # get cleaned images using trained model
    # img_predicted = model.predict(test_imgs, batch_size=2)
    # print(f'type(img_predicted): {type(img_predicted)}\n')    

    # # get cleaned images using trained model
    # img_predicted = model.predict(test_imgs, batch_size=2)
    # for i, (predicted, testing_path) in enumerate(zip(img_predicted, uploadFile)):
    #     predicted_sequeeze = (np.squeeze(predicted) * 255).astype("uint8")
    #     st.image(predicted_sequeeze)