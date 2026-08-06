"""Deteksi wajah dengan Haar Cascade + praproses gambar untuk LBPH."""

import cv2

from .paths import CASCADE_PATH, SAMPLE_SIZE


class FaceDetector:
    def __init__(self, cascade_path=CASCADE_PATH):
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise FileNotFoundError(f"Haar cascade tidak ditemukan: {cascade_path}")

    def detect(self, gray, scale_factor=1.1, min_neighbors=5, min_size=(80, 80)):
        return self.cascade.detectMultiScale(
            gray, scale_factor, min_neighbors, minSize=min_size
        )

    def largest(self, gray, **kwargs):
        faces = self.detect(gray, **kwargs)
        if len(faces) == 0:
            return None
        return max(faces, key=lambda r: r[2] * r[3])

    @staticmethod
    def preprocess(gray, size=(SAMPLE_SIZE, SAMPLE_SIZE)):
        equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.resize(equalized, size, interpolation=cv2.INTER_AREA)
