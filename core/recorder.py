"""Perekam sampel wajah: menyimpan crop wajah yang tajam ke folder Dataset."""

from pathlib import Path

import cv2

from .detector import FaceDetector
from .paths import DATASET_DIR


class Recorder:
    def __init__(self, user_id, detector=None):
        self.user_id = int(user_id)
        self.detector = detector or FaceDetector()
        self.count = 0

    @staticmethod
    def sharpness(gray):
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def clean_slate(self):
        prefix = f"User.{self.user_id}."
        for f in Path(DATASET_DIR).glob(prefix + "*.jpg"):
            f.unlink(missing_ok=True)
        self.count = 0

    def capture(self, frame_bgr, min_sharp=60.0):
        """Ambil wajah terbesar di frame.

        Mengembalikan (face_preprocessed, rect, info). info['reason'] bisa
        'ok', 'blur', atau 'noface'.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rect = self.detector.largest(gray)
        if rect is None:
            return None, None, {"reason": "noface", "sharp": 0.0}

        x, y, w, h = rect
        face = gray[y:y + h, x:x + w]
        sharp = self.sharpness(face)
        if sharp < min_sharp:
            return None, rect, {"reason": "blur", "sharp": round(sharp, 1)}

        prep = self.detector.preprocess(face)
        self.count += 1
        path = DATASET_DIR / f"User.{self.user_id}.{self.count}.jpg"
        ok, buf = cv2.imencode(".jpg", prep)
        if ok:
            buf.tofile(str(path))
        return prep, rect, {"reason": "ok", "sharp": round(sharp, 1), "path": path}
