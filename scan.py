# Mengimport package yg diperlukan
import cv2, time
import os
from PIL import Image

# camera = 0 berarti menggunakan web cam bawaan perangkat. Ubah 0 jika menggunakan webcam external
camera = 0
# Inisialisasi video capture
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# cascade classifier menggunakan file haarcascade yang ada
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Membaca file training.xml
recognizer.read('Dataset/training.xml')

a = 0
# Program webcam akan terus berjalan selama bernilai TRUE
while True:
    a = a + 1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    cv2.imshow("Face Recognation", frame)
    key = cv2.waitKey(1)

    # Cam berhenti saat menekan tombol q pada keyboard
    if key == ord('q'):
        break

# Camera berhenti
video.release()
cv2.destroyAllWindows()