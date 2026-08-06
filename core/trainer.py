"""Pelatihan model LBPH dari sampel wajah di folder Dataset."""

import time
from pathlib import Path

import cv2
import numpy as np

from .detector import FaceDetector
from .paths import DATASET_DIR, MODEL_PATH, SAMPLE_SIZE


class TrainingError(Exception):
    pass


def _read_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


class Trainer:
    def __init__(self, detector=None):
        self.detector = detector or FaceDetector()

    def collect(self, progress=None):
        image_paths = sorted(Path(DATASET_DIR).glob("User.*.jpg"))
        samples = []
        labels = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            try:
                uid = int(path.name.split(".")[1])
            except (IndexError, ValueError):
                continue

            img = _read_gray(path)
            if img is None:
                continue

            equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
            faces = self.detector.detect(equalized, min_size=(40, 40))
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                crop = equalized[fy:fy + fh, fx:fx + fw]
            else:
                crop = equalized

            crop = cv2.resize(crop, (SAMPLE_SIZE, SAMPLE_SIZE), interpolation=cv2.INTER_AREA)

            # Label = ID asli pengguna, supaya prediksi mengembalikan ID yang sama.
            samples.append(crop)
            labels.append(uid)

            if progress and (i % 10 == 0 or i == total - 1):
                progress(i + 1, total, f"Memproses sampel... {i + 1}/{total}")

        return samples, np.array(labels, dtype=np.int32)

    def train(self, progress=None):
        t0 = time.time()
        samples, labels = self.collect(progress)
        if len(samples) < 2:
            raise TrainingError(
                "Data sampel wajah tidak cukup untuk dilatih (minimal 2 sampel). "
                "Daftarkan wajah terlebih dahulu."
            )

        if progress:
            progress(0, 1, "Melatih model LBPH...")
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer.train(samples, labels)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        recognizer.write(str(MODEL_PATH))

        return {
            "model_path": MODEL_PATH,
            "samples": len(samples),
            "users": len(set(int(i) for i in labels)),
            "time": time.time() - t0,
        }
