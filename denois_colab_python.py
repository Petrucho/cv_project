import numpy as np
import cv2
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import glob

print(tf.version.VERSION)

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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
print(f'created model: autoencoder\n')

# download weigths from learned model
model.load_weights('./data/denoising.h5')
print(f'loaded model weights\n')

torch.cuda.empty_cache()
# print(f'torch.cuda.empty_cache\n')


test_images = '/home/petrucho/cv_project/data/Datasets/val/input_one_image/*'
test_target_path = '/home/petrucho/cv_project/data/Datasets/val/target_one_image/'

# get testing image
def preprocess(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img, dtype="float32")
    img = img/255.0 #Scaling the pixel values
    
    return img.reshape(400,400,1)

img_test_path = sorted(glob.glob(test_images))
print(f'img_test_path: {img_test_path}\n')

test_imgs = []
for file_path in img_test_path:
    print(f'file_path: {file_path}')
    test_imgs.append(preprocess(file_path))
test_imgs = np.asarray(test_imgs)

if len(test_imgs)>0:
    # get cleaned images using trained model
    img_predicted = model.predict(test_imgs, batch_size=2)
    for i, (predicted, testing_path) in enumerate(zip(img_predicted, img_test_path)):
        predicted_sequeeze = (np.squeeze(predicted) * 255).astype("uint8")
        cv2.imwrite(test_target_path+os.path.basename(testing_path), predicted_sequeeze)
else:
    print(f'len(test_imgs): {len(test_imgs)}\nNo one file was read!\n')