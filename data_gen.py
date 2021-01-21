from eye_dataset_gen import *
from yawn_dataset_gen import *
from model_gen import *
import pandas as pd
import random
import numpy as np
from keras.utils.np_utils import to_categorical as tcg

def convert_image(pixels_1d):
    pixels_2d = np.reshape(pixels_1d, (24,24))
    pixels_2d = np.array(pixels_2d, dtype=np.uint8)
    return pixels_2d

def read_data(filepath):
    dataset = []
    df = pd.read_csv(filepath, header=None)
    for i in df.index:                      
        string_pixels = df.iloc[i][0].replace('[','').replace(']','').replace(' ','')
        pixels_1d = string_pixels.split(',')
        pixels_2d = convert_image(pixels_1d)
        category = df.iloc[i][1]
        image_bundle = (pixels_2d, category)
        dataset.append(image_bundle)
    return dataset

def split_train_test(dataset):
    length = len(dataset)
    breakpoint_length = int(0.8*length)
    random.shuffle(dataset)
    train = dataset[0:breakpoint_length]
    test = dataset[breakpoint_length+1:length]
    xtr = []
    xte = []
    ytr = []
    yte = []
    for i, j in train:
        xtr.append(i)
        ytr.append(j)
    for i, j in test:
        xte.append(i)
        yte.append(j)
    xtr = np.array(xtr)
    xte = np.array(xte)
    ytr = np.array(ytr)
    yte = np.array(yte)

    xtr = xtr.reshape(xtr.shape[0], xtr.shape[1], xtr.shape[2], 1).astype('float32')/32
    xte = xte.reshape(xte.shape[0], xte.shape[1], xte.shape[2], 1).astype('float32')/32
    ytr = tcg(ytr)
    yte = tcg(yte)
    print(xtr.shape, ytr.shape, xte.shape, yte.shape)
    return (xtr, ytr, xte, yte)

def get_data(category):
    if category == "Eye":
        try:
            with open(r"Dataset\eye_dataset.csv","r"):
                print("<------Eye Dataset Exists------>")
                eye_dataset = read_data(r"Dataset\eye_dataset.csv")
                eye_model_data = split_train_test(eye_dataset)
                create_eye_model(eye_model_data)
        except FileNotFoundError:
            gen_eye_dataset()
            eye_dataset = read_data(r"Dataset\eye_dataset.csv")
            eye_model_data = split_train_test(eye_dataset)
            create_eye_model(eye_model_data)

    else:
        try:
            with open(r"Dataset\yawn_dataset.csv","r"):
                print("<------Yawn Dataset Exists------>")
                yawn_dataset = read_data(r"Dataset\yawn_dataset.csv")
                yawn_model_data = split_train_test(yawn_dataset)
                create_yawn_model(yawn_model_data)
        except FileNotFoundError:
            gen_yawn_dataset()
            yawn_dataset = read_data(r"Dataset\yawn_dataset.csv")
            yawn_model_data = split_train_test(yawn_dataset)
            create_yawn_model(yawn_model_data)