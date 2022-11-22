import cv2, time
camera = 0
# membuka webcam
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
# algoritma FR
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# mengambil id
id = input('Id : ')
a = 0
while True: 
    a = a + 1
    check, frame = video.read() 
    # membuat mode pengambilan gambar pada scan menjadi Gray (abu-abu)
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah
    wajah = faceDeteksi.detectMultiScale(abu,1.3,5)
    print(wajah)
    for(x,y,w,h) in wajah:
        # Membuat file foto ke folder Dataset/ dengan identifikasi Id dan perulangan a
        cv2.imwrite('Dataset/User.'+str(id)+'.'+str(a)+'.jpg', abu[y:y+h,x:x+w])
        # Mengenali bentuk wajah (kotak warna hijau di wajah)
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)
    # Nama Window 
    cv2.imshow("Face Recognation Window", frame)
    # Perulangan dilakukan hingga 30 pengambilan foto
    if (a > 29):
        break
# Cam berhenti
video.release()
cv2.destroyAllWindows()