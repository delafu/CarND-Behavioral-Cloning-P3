# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:15:06 2017

@author: Delafu
"""
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout
from sklearn.model_selection import train_test_split


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                current_path = batch_sample[0]
                image = cv2.imread(current_path)
                image = image [55:160, 0:320]
                image = cv2.resize(image,(200,66))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train
def loadImages(samples):
    images = []
    measurements = []
    X_train=[]
    y_train=[]
    for sample in samples:
        current_path = sample[0]
        image = cv2.imread(current_path)
        image = image [55:160, 0:320]
        image = cv2.resize(image,(200,66))
        imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(imgRGB)
        measurements.append(sample[1])    
        augmented_images, augmented_measurements= [], []
    for image, measurement in zip(images, measurements):
        if measurement != 0:
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            augmented_images.append(cv2.flip(image,1))
            augmented_measurements.append(measurement*-1)
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    return X_train,y_train
def calc_ecpoch_size(samples):
    NOT_ZERO_FACTOR=2
    ZERO_FACTOR=1
    not_zero=0
    zero=0
    for sample in samples:
        angle=sample[1]
        if angle==0:
            zero=zero+1
        else:
            not_zero=not_zero+1
    return (ZERO_FACTOR*zero+NOT_ZERO_FACTOR*not_zero)
usegenerator=False
lines = []
with open('../driving-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
batch_samples=[]

for line in lines:
    source_path = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    measurement = float(line[3])
    #center
    filename = source_path.split('\\')[-1]
    current_path = '../driving-data/IMG/' + filename
    batch_samples.append([current_path,measurement])
    #left
    filename = source_path_left.split('\\')[-1]    
    current_path = '../driving-data/IMG/' + filename
    batch_samples.append([current_path,measurement+0.3])
    #right
    filename = source_path_right.split('\\')[-1]    
    current_path = '../driving-data/IMG/' + filename
    batch_samples.append([current_path,measurement-0.3])
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,3)))
model.add(Convolution2D(24,5,5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64,3,3, subsample=(1, 1), activation="relu"))
model.add(Convolution2D(64,3,3, subsample=(1, 1), activation="relu"))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
if (usegenerator):
    train_samples, validation_samples = train_test_split(batch_samples, test_size=0.2)
    train_size=calc_ecpoch_size(train_samples)
    val_size=calc_ecpoch_size(validation_samples)
    train_generator=generator(train_samples)
    valid_generator=generator(validation_samples)
    model.fit_generator(train_generator, samples_per_epoch=train_size, \
                        validation_data=valid_generator, nb_val_samples=val_size, \
                        nb_epoch=3)
else:
    X_train,y_train=loadImages(batch_samples)
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')





    
    
