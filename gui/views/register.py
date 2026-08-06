"""Daftar Wajah: rekam sampel wajah pengguna baru dengan pratinjau webcam."""

import tkinter as tk

import cv2

from core.detector import FaceDetector
from core.recorder import Recorder
from core.users import UserStore
from ..theme import *
from ..widgets import (
    CameraCanvas,
    CameraError,
    Card,
    ModernButton,
    ProgressBar,
    StatusLine,
    ViewHeader,
)


class RegisterView(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        self.users = UserStore()
        self.detector = FaceDetector()
        self.recorder = None
        self.recording = False
        self.target = 30
        self._uid = None

        self.name_var = tk.StringVar()
        self.target_var = tk.StringVar(value="30")
        self.uid_var = tk.StringVar()
        self.name_var.trace_add("write", self._on_name_change)

        self._build()

    def _build(self):
        ViewHeader(
            self,
            "Daftar Wajah",
            "Rekam sampel wajah baru untuk melatih model LBPH",
        ).pack(fill="x", padx=30, pady=(26, 10))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=(6, 24))

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self.camera = CameraCanvas(left, width=680, height=500, on_frame=self._on_frame)
        self.camera.pack(fill="both", expand=True)

        right = tk.Frame(body, bg=BG, width=330)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        card = Card(right, title="Informasi Pengguna")
        card.pack(fill="x")

        tk.Label(card, text="Nama Lengkap", font=FONT_SMALL_BOLD, bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        self.entry_name = tk.Entry(
            card, textvariable=self.name_var, font=FONT, bg=BG_CARD2, fg=FG,
            insertbackground=FG, relief="flat", highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT,
        )
        self.entry_name.pack(fill="x", pady=(4, 10))

        tk.Label(card, text="ID Wajah", font=FONT_SMALL_BOLD, bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        tk.Label(card, textvariable=self.uid_var, font=FONT, bg=BG_CARD, fg=ACCENT_LT, anchor="w").pack(
            fill="x", pady=(2, 10)
        )

        tk.Label(card, text="Jumlah Sampel", font=FONT_SMALL_BOLD, bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        self.spin_target = tk.Spinbox(
            card, from_=10, to=100, textvariable=self.target_var, width=6,
            font=FONT, bg=BG_CARD2, fg=FG, insertbackground=FG, relief="flat",
            buttonbackground=BG_CARD2, highlightthickness=1, highlightbackground=BORDER,
        )
        self.spin_target.pack(anchor="w", pady=(4, 14))

        self.btn_start = ModernButton(card, "Mulai Merekam", command=self._start, accent=True)
        self.btn_start.pack(fill="x", pady=(0, 8))
        self.btn_stop = ModernButton(card, "Berhenti", command=self._stop, accent=False)
        self.btn_stop.pack(fill="x", pady=(0, 14))
        self.btn_stop.configure(state="disabled")

        self.progress = ProgressBar(card, width=300)
        self.progress.pack(fill="x", pady=(0, 12))

        self.status = StatusLine(card)
        self.status.pack(fill="x")

        self.btn_train = ModernButton(card, "Latih Model Sekarang", command=lambda: self.app.show("train"))
        self.btn_train.configure(bg=GREEN, activebackground="#2cb96a")

    def refresh(self):
        self._update_uid()

    def _on_name_change(self, *_):
        self._update_uid()

    def _update_uid(self):
        name = self.name_var.get().strip()
        if not name:
            self.uid_var.set(f"ID otomatis: {self.users.next_id()}")
            return
        uid = self.users.id_for_name(name)
        if uid is not None:
            self.uid_var.set(f"ID {uid} — sampel {name} akan diperbarui")
        else:
            self.uid_var.set(f"ID otomatis: {self.users.next_id()} (pengguna baru)")

    def _start(self):
        name = self.name_var.get().strip()
        if not name:
            self.status.show("Masukkan nama terlebih dahulu.", RED)
            return
        try:
            self.target = int(self.target_var.get())
        except ValueError:
            self.target = 30

        self._uid = self.users.id_for_name(name)
        if self._uid is None:
            self._uid = self.users.add(name)

        self.recorder = Recorder(self._uid)
        self.recorder.clean_slate()
        self.recording = True
        self._set_controls(True)
        self.progress.set(0.0, f"0 / {self.target}")
        self.status.show("Hadapkan wajah ke kamera dan tetaplah diam.", ACCENT_LT)

        try:
            self.camera.start()
        except CameraError as exc:
            self.status.show(str(exc), RED)
            self.recording = False
            self._set_controls(False)

    def _stop(self):
        self.recording = False
        self.camera.stop()
        if self.recorder and self.recorder.count > 0:
            self.status.show(f"Dihentikan. {self.recorder.count} sampel tersimpan.", AMBER)
        else:
            self.status.show("Perekaman dihentikan.", FG_MUTED)
        self._set_controls(False)

    def _on_frame(self, frame):
        if not self.recording or self.recorder is None:
            return None
        _, rect, info = self.recorder.capture(frame, min_sharp=60.0)
        annotated = frame.copy()
        if rect is not None:
            x, y, w, h = rect
            color = GREEN if info["reason"] == "ok" else AMBER
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            if info["reason"] == "ok":
                cv2.putText(
                    annotated, f"Sampel {self.recorder.count}",
                    (x + 6, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA,
                )
        self._handle_info(info)
        return annotated

    def _handle_info(self, info):
        reason = info["reason"]
        if reason == "ok":
            self.progress.set(self.recorder.count / self.target, f"{self.recorder.count} / {self.target}")
            self.status.show(f"Sampel {self.recorder.count} tersimpan.", GREEN)
            if self.recorder.count >= self.target:
                self._finish()
        elif reason == "blur":
            self.status.show(f"Wajah kurang jelas (skor {info['sharp']}) — tetaplah diam.", AMBER)
        else:
            self.status.show("Wajah tidak terdeteksi — hadapkan wajah ke kamera.", FG_MUTED)

    def _finish(self):
        self.recording = False
        self.camera.stop()
        self._set_controls(False)
        name = self.users.display_for(self._uid)
        self.status.show(f"Selesai! {self.target} sampel {name} tersimpan.", GREEN)
        self.btn_train.pack(fill="x", pady=(14, 0))

    def _set_controls(self, recording):
        self.btn_start.configure(state="disabled" if recording else "normal")
        self.btn_stop.configure(state="normal" if recording else "disabled")
        self.entry_name.configure(state="disabled" if recording else "normal")
        self.spin_target.configure(state="disabled" if recording else "normal")
