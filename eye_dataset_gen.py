import numpy as np
import os
import cv2
import csv

def gen_eye_dataset():

    DATADIR = r"Data\dataset_B_Eye_Images\Eyes"

    CATEGORIES = ["Closed", "open"]
    count=0
    for category in CATEGORIES:  # do closed and open
        path = os.path.join(DATADIR,category)  # create path to closed and open
        for img in os.listdir(path):  # iterate over each image per closed and open
            img_1 = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
            with open (r"Dataset\eye_dataset.csv",'a',newline='') as df:
                count+=1
                print(count)
                img_1 = img_1.reshape([1, img_1.size])
                img_1 = img_1.tolist()
                wr = csv.writer(df, dialect="excel")
                wr.writerow([img_1, CATEGORIES.index(category)])