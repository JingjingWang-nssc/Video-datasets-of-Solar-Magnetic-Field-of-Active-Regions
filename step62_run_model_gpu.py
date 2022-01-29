# -*- coding: utf-8 -*-
"""
Convolutional autoencoder from https://blog.keras.io/building-autoencoders-in-keras.html
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
from keras import backend as K
from sklearn.cluster import KMeans
import pickle,time,itertools
from keras.utils.vis_utils import plot_model
import os,random,imageio
import keras_metrics as km

import step60_model
time_start = time.time()

pathtitle = "wang/sharpdata/"
gif_dir = pathtitle+'gif_prepare/'
tadvance = "0h_advance"
tsize = (112,112,7)
nb_classes = 1

boardpath = "wang/sharpdata/results20220127/"

checkpointfile = boardpath+tadvance+"_gpu_"+'c3d_model_weights_best.hdf5'
historyfile = boardpath+tadvance+"_gpu_"+'c3d_model_history.pickle'
historyfigfile = boardpath+tadvance+"_gpu_"+'c3d_model_history.png'
loss,val_loss,acc,val_acc,mse,val_mse,mae,val_mae = list(),list(),list(),list(),list(),list(),list(),list()
model = step60_model.model_c3d_modified()
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc','binary_accuracy','categorical_accuracy'])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc','mae','mse',km.binary_precision(),km.binary_recall(),km.f1_score()])
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model,to_file=boardpath+'c3d_model.png',show_shapes=True)
checkpoint = ModelCheckpoint(checkpointfile,monitor='val_loss',mode='min',verbose=1,save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=2)

for tryi in range(500):
    print("Now trying times = ",tryi)
    epochs = 500
    batchsize = 50
    validrate = 0.4
    dirs = sorted(os.listdir(os.path.join(gif_dir,'XM',tadvance)))
    print("XM cases: = ",len(dirs))
    setn = int(len(dirs)/4.)
    train_x_class0 = np.zeros((setn*2,tsize[0],tsize[1],tsize[2],1),dtype=np.float32)
    train_y_class0 = np.zeros((setn*2,nb_classes),dtype=np.float32)
    train_x_class1 = np.zeros((setn,tsize[0],tsize[1],tsize[2],1),dtype=np.float32)
    train_y_class1 = np.zeros((setn,nb_classes),dtype=np.float32)
    train_x_class2 = np.zeros((setn,tsize[0],tsize[1],tsize[2],1),dtype=np.float32)
    train_y_class2 = np.zeros((setn,nb_classes),dtype=np.float32)
    dirs = os.listdir(os.path.join(gif_dir,'XM',tadvance))
    random.shuffle(dirs[0:int(len(dirs)/2.)])
    for k in range(0,setn*2):
        filegif = os.path.join(gif_dir,'XM',tadvance,dirs[k])
        gifk = imageio.mimread(filegif,memtest=False)
        for ii in range(0,tsize[2]):
            train_x_class0[k,:,:,0] = gifk[ii].reshape((tsize[0],tsize[1],1))
        train_y_class0[k,:] = 1
    dirs = os.listdir(os.path.join(gif_dir,'C',tadvance))
    random.shuffle(dirs[0:int(len(dirs)/2.)])
    for k in range(0,setn):
        filegif = os.path.join(gif_dir,'C',tadvance,dirs[k])
        gifk = imageio.mimread(filegif,memtest=False)
        for ii in range(0,tsize[2]):
            train_x_class1[k,:,:,0] = gifk[ii].reshape((tsize[0],tsize[1],1))
        train_y_class1[k,:] = 0
    dirs = os.listdir(os.path.join(gif_dir,'N',tadvance))
    random.shuffle(dirs[0:int(len(dirs)/2.)])
    for k in range(0,setn):
        filegif = os.path.join(gif_dir,'N',tadvance,dirs[k])
        gifk = imageio.mimread(filegif,memtest=False)
        for ii in range(0,tsize[2]):
            train_x_class2[k,:,:,0] = gifk[ii].reshape((tsize[0],tsize[1],1))
        train_y_class2[k,:] = 0
    print(train_x_class0.shape,train_x_class1.shape,train_x_class2.shape)
    print(train_y_class0.shape,train_y_class1.shape,train_y_class2.shape)

    sp = [0,int(setn*2.*(1.0-validrate)),setn*2]
    sp1 = [0,int(setn*(1.0-validrate)),setn]
    train_x = np.concatenate([train_x_class0[sp[0]:sp[1]],train_x_class1[sp1[0]:sp1[1]],train_x_class2[sp1[0]:sp1[1]]])
    train_y = np.concatenate([train_y_class0[sp[0]:sp[1]],train_y_class1[sp1[0]:sp1[1]],train_y_class2[sp1[0]:sp1[1]]])
    valid_x = np.concatenate([train_x_class0[sp[1]:sp[2]],train_x_class1[sp1[1]:sp1[2]],train_x_class2[sp1[1]:sp1[2]]])
    valid_y = np.concatenate([train_y_class0[sp[1]:sp[2]],train_y_class1[sp1[1]:sp1[2]],train_y_class2[sp1[1]:sp1[2]]])
    if os.path.exists(checkpointfile):
        model.load_weights(checkpointfile)
        model.evaluate(x=train_x,y=train_y,verbose=1)
        print('now begin retun times: ',tryi)
    history = model.fit(x=train_x,y=train_y,validation_data=(valid_x,valid_y),
                        epochs=epochs,batch_size=batchsize,callbacks=[checkpoint,earlystop])
time_end = time.time()
print('time cost :',time_end-time_start,' s')
