import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D
import matplotlib.pyplot as plt
from model_pred import *


def plot_graph(history, category):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["Train", "Test"], loc="upper left")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    text = category+"_Loss.png"
    plt.savefig(r"Model\\"+text)
    plt.close()

def create_eye_model(data):
    xtr, ytr, xte, yte = data
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu",padding="same"))
    model.add(MaxPooling2D(1, 1))
    model.add(Conv2D(32, (3, 3), activation="relu",padding="same"))
    model.add(MaxPooling2D(1, 1))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation="relu",padding="same"))
    model.add(MaxPooling2D(1, 1))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation="relu",padding="same"))
    model.add(MaxPooling2D(1, 1))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(x=xtr, y=ytr, validation_data=(xte, yte), epochs=1, batch_size=64)
    model.summary() 

    model_json = model.to_json()
    with open(r"Model\Eye_CNN_Model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(r"Model\Eye_CNN_Model.h5")
    plot_graph(history, "Eye")


####################################################################################################
                        #Yawn model



def create_yawn_model(data):
    
    xtr, ytr, xte, yte = data
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu",padding="same"))
    model.add(MaxPooling2D(1, 1))
    model.add(Conv2D(32, (3, 3), activation="relu",padding="same"))
    model.add(MaxPooling2D(1, 1))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation="relu",padding="same"))
    model.add(MaxPooling2D(1, 1))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation="relu",padding="same"))
    model.add(MaxPooling2D(1, 1))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(x=xtr, y=ytr, validation_data=(xte, yte), epochs=50, batch_size=64)
    model.summary() 

    model_json = model.to_json()
    with open(r"Model\Yawn_CNN_Model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(r"Model\Yawn_CNN_Model.h5")
    plot_graph(history, "Yawn")