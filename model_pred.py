from keras.models import model_from_json
import numpy as np
import cv2
import pyttsx3
from threading import *

#Closed is 0 and Eye is 1

def load_eye_model():
    json_file = open(r"Model\Eye_CNN_Model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(r"Model\Eye_CNN_Model.h5")
    return loaded_model

def load_yawn_model():
    json_file = open(r"Model\Yawn_CNN_Model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(r"Model\Yawn_CNN_Model.h5")
    return loaded_model

def speak():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume',1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say("ALERT ALERT ALERT ALERT!!!!!!!")
    engine.runAndWait()

def speak2():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume',1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say("Take a coffe break I think you are drowsy")
    engine.runAndWait()

def predict(eye_model, yawn_model):
    cv2.ocl.setUseOpenCL(False)

    eye_array = {"Closed":0, "Open":0}
    yawn_array = {"Yawn":0, "No Yawn":0}

    live_video = cv2.VideoCapture(0)
    while True:
        _, image = live_video.read()
        image = cv2.flip(image, flipCode=1)

        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=2, minSize=(20, 20))
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=2, minSize=(20, 20))


        for (x, y, w, h) in eyes:
            single_eye = image[y:y+h, x:x+w]
            
            single_eye = cv2.cvtColor(single_eye, cv2.COLOR_BGR2GRAY)
            requiredsize = (24, 24)
            single_eye = cv2.resize(single_eye, requiredsize, interpolation=cv2.INTER_AREA)
            single_eye = single_eye.reshape(1, single_eye.shape[0], single_eye.shape[1], 1)
            single_eye = single_eye.astype('float32')/32
            output = eye_model.predict(single_eye)
            index = np.argmax(output)
            if index == 0:
                text = "Closed"
                eye_array["Closed"]+=1
            elif index == 1:
                text = "Open"
                eye_array["Closed"]=0

            if eye_array["Closed"]>6:
                try:
                    child_Thread = Thread(target=speak)
                    print("Warning!!!!!!!!!")
                    child_Thread.start()
                    child_Thread.join()
                except:
                    print("Closed")
                eye_array["Closed"]=0

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(image, text, (x+w-50, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        for (x, y, w, h) in faces:
                single_face = image[y:y+h, x:x+w]

                single_face = cv2.cvtColor(single_face, cv2.COLOR_BGR2GRAY)
                requiredsize = (24, 24)
                single_face = cv2.resize(single_face, requiredsize, interpolation=cv2.INTER_AREA)
                single_face = single_face.reshape(1, single_face.shape[0], single_face.shape[1], 1)
                single_face = single_face.astype('float32')/32
                output = yawn_model.predict(single_face)
                index = np.argmax(output)
                
                if index == 1:
                    text = "Yawn"
                    yawn_array["Yawn"]+=1
                elif index == 0:
                    text = "No Yawn"
                    yawn_array["Yawn"]=0

                if yawn_array["Yawn"]>8:
                    try:
                        child_Thread = Thread(target=speak2)
                        print("Warning!!!!!!!!!")
                        child_Thread.start()
                        child_Thread.join()
                    except:
                        print("Yawn")
                    yawn_array["Yawn"]=0

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(image, text, (x+w-50, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow("Feed", image)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    live_video.release()
    cv2.destroyAllWindows()
