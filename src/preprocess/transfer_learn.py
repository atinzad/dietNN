#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:40:10 2018

@author: atinzad
"""

#Transfer Learning
#followed many of https://cv-tricks.com/keras/fine-tuning-tensorflow/

from argparse import ArgumentParser
from keras.models import model_from_json
from keras.preprocessing import image
import os
from keras import Model
from keras.layers import Dense
from keras import optimizers 
from keras.utils.vis_utils import plot_model



def build_parser():
    par = ArgumentParser(fromfile_prefix_chars='@')
    par.add_argument('--t', type=str, dest = 'train_dir', required=True, help="(required) the train data directory")
    par.add_argument('--n', type=int, dest = 'nclass', required=True, help="(required) number of classes to be trained")
    par.add_argument('--v', type=str, dest = 'val_dir', required=True, help="(required) the validation data directory")
    par.add_argument('--m', type=str, dest='model_load_path', help='filepath to load model in json format', required=True)
    par.add_argument('--w', type=str, dest='weights_load_path', help='filepath to load weights in h5 format', required=True)
    
    return par


if __name__ == "__main__":
    
#    parser = build_parser()
#    options = parser.parse_args()
#    
#    train_dir = options.train_dir
#    val_dir = options.val_dir
#    nclass = options.nclass
#    model_load_path = options.model_load_path
#    weights_load_path = options.weights_load_path
    
    
    train_dir='~/dietNN/data/raw/catdog/train'
    val_dir='~/dietNN/data/raw/catdog/test'
    nclass=2
    model_load_path='~/dietNN/data/raw/model.json'
    weights_load_path='~/dietNN/data/raw/model.h5'
    
    if model_load_path[0]=="~":
        model_load_path = os.path.expanduser(model_load_path)
        
    if weights_load_path[0]=="~":
        weights_load_path = os.path.expanduser(weights_load_path)
    
    if train_dir[0]=="~":
        train_dir = os.path.expanduser(train_dir)
        
    if val_dir[0]=="~":
        val_dir = os.path.expanduser(val_dir)
    
    img_size=224
    batch_size=10
    
    train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = image.ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')
    

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical')
    
    
    # load json and create model
    with open(model_load_path, 'r')as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_load_path)
    print("Loaded model from disk")
    
    print (loaded_model.summary())
    
    i=0
    for layer in loaded_model.layers:
        layer.trainable = False
        i = i+1
        print(i,layer.name)
    
    loaded_model.layers.pop()
    
    x = loaded_model.layers[-1].output
    predictions = Dense(nclass, activation='softmax')(x)
    saved_model = Model(inputs=loaded_model.input, outputs=predictions)
    
    saved_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])
    
    num_training_img=1000
    num_validation_img=400
    stepsPerEpoch = num_training_img/batch_size
    validationSteps= num_validation_img/batch_size
    
    saved_model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=10)
    
    
    saved_model.evaluate_generator(generator=validation_generator)
    
    model_json = saved_model.to_json()
    with open("catdog_model.json", "w") as json_file:
        json_file.write(model_json)
    saved_model.save_weights("catdog_model.h5")
    print("Saved model to disk")
    try:
        plot_model(saved_model, to_file='catdog_vgg.png')
    except:
        pass
