# Mengimport package yang diperlukan
import cv2, os
import numpy as np
from PIL import Image
# Membuat variabel recognizer

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Untuk detector menggunakan file haarcascade_frontalface_default.xml
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Membuat fungsi dengan  getImagesWithLabels parameter path
def getImagesWithLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    # for untuk perulangan imagePath yang ada pada imagePaths
    for imagePath in imagePaths:
        # Image
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    # return untuk mengembalikan nilai
    return faceSamples, Ids
faces, Ids = getImagesWithLabels('Dataset')
recognizer.train(faces, np.array(Ids))

# Data training disimpan di folder Dataset dengan nama file training.xml
recognizer.save('Dataset/training.xml')