# Mengimport package yg diperlukan
import cv2, time
import os
from PIL import Image

# camera = 0 berarti menggunakan web cam bawaan perangkat. Ubah 0 jika menggunakan webcam external
camera = 0

# Inisialisasi video capture
# cv2 -> modul open-cv
#videoCapture() -> object dari opencv dengan parameter (source, CAP_DSHOW = DirectShow sebagai video input)
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# cascade classifier menggunakan file haarcascade yang ada
# CascadeClassifier(source) -> objet dari opencv yang membaca classifier yang akan digunakan 
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# face.LBPHFaceRecognizer_create() -> membuat pengenalan dengan menggunakan algoritma LBPH(Local Binary Pattern)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# recognizer membaca file training.xml
recognizer.read('Dataset/training.xml')

# deklarasi variabel a
a = 0
# Program webcam akan terus berjalan selama bernilai TRUE
while True:
    # Iterasi variabel a (a+1)
    a = a + 1
    # Membuat cam di frame windows
    check, frame = video.read()
    # cvtColor(frame, mode warna) -> object dalam cv2 untuk menentukan mode warna dalam frame
    # COLOR_BGR2GRAY -> mengubah mode warna (Blue Green Red) menjadi Gray
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecMultiScale() -> object untuk Mendeteksi wajah, dengan paramter (mode gambar, faktor scala, spesifik berapa Neightboors kandidat)
    wajah = faceDeteksi.detectMultiScale(abu,1.3,5)
    for(x,y,w,h) in wajah :
        # Membuat kotak hijau di wajah
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        id, conf = recognizer.predict(abu[y:y+h, x:x+w])

        # Seleksi Id
        if (id == 1):
            id = 'Alan'
        elif (id == 2):
            id = 'Natan'
        elif (id == 3):
            id = 'Yogi'
        elif (id == 4):
            id = 'Oswal'
        elif (id == 5):
            id= 'Andi'
        elif (id == 13):
            id ='Gunawan'
        elif (id == '6'):
            id = 'Keyzia'
        else:
            id = 'Unknown'

        # Menambahkan teks sesuai Id ke wajah didalam frame
        cv2.putText(frame, str(id),(x+40, y-10), cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
    # imshow untuk memberikan label pada frame saat window terbuka
    cv2.imshow("Face Recognation", frame)
    # menentukan keyboard event
    key = cv2.waitKey(1)

    # Cam berhenti saat menekan tombol q pada keyboard
    if key == ord('q'):
        break

# Camera berhenti
video.release()
cv2.destroyAllWindows()