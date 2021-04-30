#!/usr/bin/env python
# coding: utf-8

# In[1]:

# importing important packages
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import warnings
import time
import os
import tkinter
import tkinter as tk
from tkinter import *
from tkinter import Tk
from tkinter import ttk
import random
from pygame import mixer
from utils.array import scale
from utils.image import crop_bounding_box, draw_bounding_box_with_label
warnings.filterwarnings('ignore')


# Face image size
face_size = 64

FACE_MODEL_FILE = "models\\haarcascade_frontalface_alt.xml"


EM_MODEL_FILE = 'models\\emotion_model.hdf5'

#paths for songs directory
sad_path='E:\\ipy_notebooks\\projects\\emotion_song\\songs\\sad\\'
angry_path='E:\\ipy_notebooks\\projects\\emotion_song\\songs\\disgust_angry\\'
happy_path='E:\\ipy_notebooks\\projects\\emotion_song\\songs\\happy\\'
fear_path='E:\\ipy_notebooks\\projects\\emotion_song\\songs\\fear\\'
neutral_path='E:\\ipy_notebooks\\projects\\emotion_song\\songs\\neutral\\'
surprise_path='E:\\ipy_notebooks\\projects\\emotion_song\\songs\\surprise\\'

mixer.init()


# creating a root window for gui in tkinter
root = Tk()
root.title("song on emotion ")
root.configure(bg='#ADD8E6')
root.configure()
root.geometry("610x452")
label_0 = Label(root, text="Emotion Based Song Playback     ",width=30,font=("bold", 20),bg="#ADD8E6")
label_0.place(x=70,y=23)
label_2 = Label(root, text="__________________________________________________________________________",bg="#ADD8E6")
label_2.place(x=95,y=58)






# function to get emotion from face detected
def get_emotion(face_image):
   
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_face = scale(gray_face)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)

    # Get EMOTION
    emotion_pred = emotion_classifier.predict(gray_face)
    emotion_probab = np.max(emotion_pred)
    emotion_label_arg = np.argmax(emotion_pred)
    return emotion_labels[emotion_label_arg]


# function to play song based on emotion detected
def play_song(emotion):
    
    if emotion=='sad':
        flies=os.listdir(sad_path)
        name=random.choice(flies)
        song='{}{}'.format(sad_path,name)
        mixer.music.load(song)
        mixer.music.play()
        
        
    elif emotion=='happy':
        flies=os.listdir(happy_path)
        name=random.choice(flies)
        song='{}{}'.format(happy_path,name)
        mixer.music.load(song)
        mixer.music.play()
        
        
    elif emotion=='disgust' or 'angry':
        flies=os.listdir(angry_path)
        name=random.choice(flies)
        song='{}{}'.format(angry_path,name)
        mixer.music.load(song)
        mixer.music.play()
        
        
    elif emotion=='surprise':
        flies=os.listdir(surprise_path)
        name=random.choice(flies)
        song='{}{}'.format(surprise_path,name)
        mixer.music.load(song)
        mixer.music.play()
        
        
    elif emotion=='neutral':
        flies=os.listdir(neutral_path)
        name=random.choice(flies)
        song='{}{}'.format(neutral_path,name)
        mixer.music.load(song)
        mixer.music.play()
        
        
    elif emotion=='fear':
        flies=os.listdir(fear_path)
        name=random.choice(flies)
        song='{}{}'.format(fear_path,name)
        mixer.music.load(song)
        mixer.music.play()
        


# VGG model for emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad',
                  5: 'Surprise', 6: 'Neutral'}


emotion_classifier = load_model(EM_MODEL_FILE)
emotion_target_size = emotion_classifier.input_shape[1:3]


# Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

global emotion_detected
emotion_detected=''

# Select video or webcam feed
def emotion_det_song():
    capture = cv2.VideoCapture(0)
    count=0 # using as buffer for camera 
    while capture.isOpened():
        success, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for face in faces:
            # Get face image, cropped to the size accepted by the WideResNet
            face_img, cropped = crop_bounding_box(frame, face, margin=.4, size=(face_size, face_size))
            (x, y, w, h) = cropped
            count=count+1
        
            emotion_detected = get_emotion(face_img)

            # Add box and label to image
            label = "{}".format(emotion_detected)
            draw_bounding_box_with_label(frame, x, y, w, h, label)

    # Display the resulting image
        cv2.imshow('Video', frame)
        if mixer.music.get_busy():
            pass
        
        else:        
            # waiting for 20 frames, giving time to camera to start
            if count>20:
                play_song(emotion_detected)
            

    # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # stopping music play
            mixer.music.stop()
            # releasing camera resource
            capture.release()
            # destroying opened cv window
            cv2.destroyAllWindows()
            # closing tkinter gui window
            #root.destroy()
            #root.quit()
            break

     
text2='This project Detects the emotion of the user and plays a song according to user\'s mood.\n\nDeveloped by-- \n Sai Prasad \n Vinay \n Anirudh \n Mohan'
    
T2=Text(root,font=('bold',12))
T2.insert(tkinter.END,text2)
T2.place(x=0,y=320)
T2.configure(bg='grey')

songbut = tkinter.Button(root, text="Detect Emotion And Play Song", 
                        command=lambda:emotion_det_song(),width=35,height=2)
songbut.place(x=155,y=100)

root.mainloop()


# In[ ]:




