"""Pengenalan wajah real-time dengan model LBPH yang telah dilatih."""

from pathlib import Path

import cv2

from .detector import FaceDetector
from .paths import MODEL_PATH


class ModelMissingError(FileNotFoundError):
    pass


class FaceRecognizer:
    def __init__(self, model_path=MODEL_PATH, threshold=100.0):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.detector = FaceDetector()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.loaded = False

    def load(self):
        if not self.model_path.exists():
            raise ModelMissingError(
                f"Model tidak ditemukan di {self.model_path}. Latih model terlebih dahulu."
            )
        self.recognizer.read(str(self.model_path))
        self.loaded = True

    def recognize(self, face_gray):
        face = self.detector.preprocess(face_gray)
        uid, distance = self.recognizer.predict(face)
        matched = distance < self.threshold
        return {
            "id": int(uid),
            "distance": round(float(distance), 1),
            "confidence": round(max(0.0, 100.0 - distance), 1),
            "matched": bool(matched),
        }

    def process_frame(self, frame_bgr, labeler, on_result=None):
        """Deteksi wajah di frame, gambar kotak + nama, dan kembalikan frame hasil.

        labeler(uid, distance) -> (name, matched)
        on_result(x, y, w, h, result, name, matched) dipanggil untuk tiap wajah.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect(gray)
        annotated = frame_bgr.copy()

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            result = self.recognize(face) if self.loaded else None
            if result and result["id"] is not None:
                name, matched = labeler(result["id"], result["distance"])
            else:
                name, matched = "Unknown", False
                result = {"id": None, "distance": -1.0, "confidence": 0.0, "matched": False}

            color = (68, 220, 132) if matched else (242, 109, 109)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            label = f"{name}  {result['confidence']:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            ty = y - 10 if y - 10 > th + 8 else y + h + th + 8
            cv2.rectangle(
                annotated,
                (x, ty - th - 6),
                (x + tw + 10, ty + 4),
                color,
                -1,
            )
            cv2.putText(
                annotated, label, (x + 5, ty - 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (18, 18, 30), 1, cv2.LINE_AA,
            )

            if on_result:
                on_result(x, y, w, h, result, name, matched)

        return annotated
