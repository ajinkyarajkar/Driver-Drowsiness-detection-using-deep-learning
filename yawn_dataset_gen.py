import numpy as np
import os
import cv2
import csv

def gen_yawn_dataset():

    DATADIR = r"Data\datasets\Mouth"

    CATEGORIES = ["yawn", "no_yawn"]
    count=0
    for category in CATEGORIES:  
        path = os.path.join(DATADIR,category)  
        for img in os.listdir(path):  
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
            haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
            cv2.ocl.setUseOpenCL(False)
            face_cascade = cv2.CascadeClassifier(haar_model)
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.2, minNeighbors=1, minSize=(20, 20))
            for (x, y, w, h) in faces:
                face = img_array[y:y+h, x:x+w]

                img_1= cv2.resize(face, (24,24))
                with open (r"Dataset\yawn_dataset.csv",'a',newline='') as df:
                    count+=1
                    print(count)
                    img_1 = img_1.reshape([1, img_1.size])
                    img_1 = img_1.tolist()
                    wr = csv.writer(df, dialect="excel")
                    wr.writerow([img_1, CATEGORIES.index(category)])