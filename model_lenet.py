# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:15:06 2017

@author: Delafu
"""
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

lines = []
with open('../driving-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
for line in lines:
    source_path = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    measurement = float(line[3])
    #center
    filename = source_path.split('\\')[-1]
    current_path = '../driving-data/IMG/' + filename
    image = cv2.imread(current_path)
    imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images.append(imgRGB)
    measurements.append(measurement)
    #left
    filename = source_path_left.split('\\')[-1]    
    current_path = '../driving-data/IMG/' + filename
    image = cv2.imread(current_path)
    imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images.append(imgRGB)
    measurements.append(measurement+0.25)
    #right
    filename = source_path_right.split('\\')[-1]    
    current_path = '../driving-data/IMG/' + filename
    image = cv2.imread(current_path)
    imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images.append(imgRGB)
    measurements.append(measurement-0.25)
    
augmented_images, augmented_measurements= [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1)
    
X_train = np.array(augmented_images)
print(X_train.shape)
y_train = np.array(augmented_measurements)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')





    
    
