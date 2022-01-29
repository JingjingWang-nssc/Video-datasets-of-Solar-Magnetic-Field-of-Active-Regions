#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:02:10 2020

@author: dell
"""
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model

def model_c3d_modified():
    weight_decay = 0.005
    nb_classes = 1
    input_shape = (112,112,7,1)
    inputs = Input(input_shape)
    # print("inputs = ",inputs.shape)
    x = Conv3D(32,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    # print("Conv1a = ",x.shape)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)
    # print("pool1 = ",x.shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("Conv2a = ",x.shape)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # print("pool2 = ",x.shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("Conv3a = ",x.shape)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # print("pool3 = ",x.shape)
    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("Conv4a = ",x.shape)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # print("pool4 = ",x.shape)
    x = Conv3D(128,(3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("Conv5a = ",x.shape)
    x = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(x)
    # print("pool5 = ",x.shape)
    x = Flatten()(x)
    # print("fc6a = ",x.shape)
    x = Dense(1024,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("fc7 = ",x.shape)
    x = Dropout(0.5)(x)
    # print("fc8a = ",x.shape)
    x = Dense(1024,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("fc9 = ",x.shape)
    x = Dropout(0.5)(x)
    # print("fc10a = ",x.shape)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    # print("fc11 = ",x.shape)
    x = Activation('sigmoid')(x) # x = Activation('softmax')(x)
    # print("fc12 = ",x.shape)
    model = Model(inputs,x)
    return model

def model_c3d_origin():
    weight_decay = 0.005
    nb_classes=101
    input_shape = (112,112,16,3)
    inputs = Input(input_shape)
    # print("inputs = ",inputs.shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
                activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    # print("Conv1a = ",x.shape)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)
    # print("pool1 = ",x.shape)
    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
                activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("Conv2a = ",x.shape)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # print("pool2 = ",x.shape)
    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
                activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("Conv3a = ",x.shape)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # print("pool3 = ",x.shape)
    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
                activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("Conv4a = ",x.shape)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # print("pool4 = ",x.shape)
    x = Conv3D(256,(3,3,3), strides=(1,1,1), padding='same',
                activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("Conv5a = ",x.shape)
    x = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(x)
    # print("pool5 = ",x.shape)
    x = Flatten()(x)
    # print("fc6a = ",x.shape)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("fc7 = ",x.shape)
    x = Dropout(0.5)(x)
    # print("fc8a = ",x.shape)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # print("fc9 = ",x.shape)
    x = Dropout(0.5)(x)
    # print("fc10a = ",x.shape)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    # print("fc11 = ",x.shape)
    x = Activation('softmax')(x)
    # print("fc12 = ",x.shape)
    model = Model(inputs, x)
    # inputs =  (None, 112, 112, 16, 3)
    # Conv1a =  (None, 112, 112, 16, 64)
    # pool1 =  (None, 56, 56, 16, 64)
    # Conv2a =  (None, 56, 56, 16, 128)
    # pool2 =  (None, 28, 28, 8, 128)
    # Conv3a =  (None, 28, 28, 8, 128)
    # pool3 =  (None, 14, 14, 4, 128)
    # Conv4a =  (None, 14, 14, 4, 256)
    # pool4 =  (None, 7, 7, 2, 256)
    # Conv5a =  (None, 7, 7, 2, 256)
    # pool5 =  (None, 4, 4, 1, 256)
    # fc6a =  (None, 4096)
    # fc7 =  (None, 2048)
    # fc8a =  (None, 2048)
    # fc9 =  (None, 2048)
    # fc10a =  (None, 2048)
    # fc11 =  (None, 101)
    # fc12 =  (None, 101)
    return model
