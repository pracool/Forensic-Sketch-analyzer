# Forensic-Sketch-analyzer
A cool GUI based app where you could upload a sketch and it will give the most relevant picture from the database provided by the user.

It is based on siamese network where the model is trained on images using the siamese methodology.

This app uses Face_recognition library as an dependency in which the model is already trained using siamese network and the model generates encoding of the input image.

It uses dlib and openface for face detection and aligning purpose.

# Installation 
App can be built using pyinstaller library.

1. pip install pyinstaller
2. pip install face_recognition_models ( put this files into the same directory as that of script.spec)
3. pyinstaller script.spec

App can then be accessed from dist->script.py.exe



