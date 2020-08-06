
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import cv2 as cv
import openface
import dlib
import face_recognition as fr
import cv2 as cv
import numpy as npy
from numpy import load
from sklearn.svm import LinearSVC as svc
from scipy import spatial
import os
import sys
import face_recognition_models
import openface

import sklearn.utils._cython_blas

aligner = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")

det=dlib.get_frontal_face_detector()

def train(path):
    images=[]
    label=[]
    database=[]
    for j in os.listdir(path):
        for i in os.listdir(path+"/"+j):
            frame=cv.imread(path+"/"+j+"/"+i)
            face=det(frame[:,:,::-1],1)
            if face[0]:
                final=aligner.align(534,frame,face[0], landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                enc=fr.face_encodings(final)
                if len(enc)>0:
                    enc=enc[0]
                    images.append(enc)
                    label.append(j)
                    database.append((frame,enc,j))
    np.save("database",database)
    return database
def no_image():
    global ih
    image_data="no_image.jpg"
    for img_display in frame.winfo_children():
        img_display.destroy()
    ih = Image.open(image_data)
    basewidth = 150
    wpercent = (basewidth / float(ih.size[0]))
    hsize = int((float(ih.size[1]) * float(wpercent)))
    ih = ih.resize((basewidth, hsize), Image.ANTIALIAS)
    ih = ImageTk.PhotoImage(ih)
    panel = tk.Label(frame, text= "Sorry , No Image Detected").pack()
    panel_image = tk.Label(frame, image=ih).pack()  

def top5(image):
    database=np.load("database.npy",allow_pickle=True)

    frame=image
    face=det(frame[:,:,::-1],1)
    if len(face)==1:
        final=aligner.align(534,frame,face[0], landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        enc=fr.face_encodings(final)
        if len(enc)>0:
             enc=enc[0]
        else:
            return [0]
    else:
        return [0]
    sim=[]
    final=None
    for i,x in enumerate(database):
        result = 1 - spatial.distance.cosine(x[1], enc)
        sim.append(result)
    final=np.argsort(sim,)[-5:][::-1]
    return database[final[0]][0]
        
 

        
def predict(image=None,model=None):

    frame=image
    face=det(frame[:,:,::-1],1)
    if face[0]:
        final=aligner.align(534,frame,face[0], landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        enc=fr.face_encodings(final)[0]
        return model.predict(np.array(enc).reshape((1,128)))
        



def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 150 # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()
    
def finder():
    global iv
    for img_display in frame.winfo_children():
        img_display.destroy()
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)
    numpy_image = cv.imread(image_data)
    im=top5(numpy_image)
    if len(im)!=1:
        iv=cv.cvtColor(im,cv.COLOR_BGR2RGB)
        iv=Image.fromarray(( iv).astype(np.uint8))
        basewidth = 150
        wpercent = (basewidth / float(iv.size[0]))
        hsize = int((float(iv.size[1]) * float(wpercent)))
        iv = iv.resize((basewidth, hsize), Image.ANTIALIAS)
        iv = ImageTk.PhotoImage(iv)
        panel_image = tk.Label(frame, image=iv).pack() 
    else:
        no_image()
   
def train_path():
    global path
    path=filedialog.askdirectory()
    db=train(path)
    
root = tk.Tk()
root.title('Portable Criminal Finder')
root.resizable(False, False)
tit = tk.Label(root, text="Portable Criminal Finder", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Upload sketch',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Find similar images',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=finder)
class_image.pack(side=tk.RIGHT)
train_image=tk.Button(root, text='Train on images',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=train_path)
train_image.pack(side=tk.BOTTOM)
root.mainloop()