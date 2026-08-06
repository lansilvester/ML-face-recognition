"""Pengenalan: pengenalan wajah real-time dengan model LBPH."""

from datetime import datetime
import tkinter as tk

from core.paths import MODEL_PATH
from core.recognizer import FaceRecognizer, ModelMissingError
from core.users import UserStore
from ..theme import *
from ..widgets import (
    CameraCanvas,
    CameraError,
    Card,
    ModernButton,
    StatusLine,
    ViewHeader,
)


class RecognizeView(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        self.users = UserStore()
        self.recognizer = FaceRecognizer()
        self.model_loaded = False
        self.threshold = 100
        self.running = False
        self._info_job = None
        self._build()

    def _build(self):
        ViewHeader(
            self,
            "Pengenalan",
            "Kenali wajah secara real-time menggunakan model LBPH",
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

        card = Card(right, title="Kontrol")
        card.pack(fill="x")

        self.btn = ModernButton(card, "Mulai Pengenalan", command=self._toggle, accent=True)
        self.btn.pack(fill="x", pady=(0, 12))

        tk.Label(card, text="Sensitivitas", font=FONT_SMALL_BOLD, bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        scale_row = tk.Frame(card, bg=BG_CARD)
        scale_row.pack(fill="x", pady=(2, 2))
        self.slider = tk.Scale(
            scale_row, from_=40, to=140, orient="horizontal",
            bg=BG_CARD, fg=FG, troughcolor=BG_CARD2, highlightthickness=0,
            bd=0, activebackground=ACCENT, font=FONT_SMALL,
            showvalue=False, command=self._on_slider,
        )
        self.slider.set(self.threshold)
        self.slider.pack(side="left", fill="x", expand=True)
        self.slider_val = tk.Label(scale_row, text="100", font=FONT_SMALL_BOLD, bg=BG_CARD, fg=ACCENT_LT, width=4)
        self.slider_val.pack(side="right")
        tk.Label(
            card, text="Semakin kecil nilai = semakin ketat (hanya wajah yang sangat cocok dikenali)",
            font=FONT_SMALL, bg=BG_CARD, fg=FG_MUTED, justify="left", wraplength=280,
        ).pack(anchor="w", pady=(0, 12))

        self.fps_label = tk.Label(card, text="FPS: 0", font=FONT, bg=BG_CARD, fg=FG_MUTED, anchor="w")
        self.fps_label.pack(fill="x", pady=(0, 4))
        self.model_label = tk.Label(card, text="Model: belum dimuat", font=FONT, bg=BG_CARD, fg=AMBER, anchor="w")
        self.model_label.pack(fill="x", pady=(0, 4))
        self.status = StatusLine(card)
        self.status.pack(fill="x", pady=(8, 0))

        log_card = Card(right, title="Riwayat Pengenalan")
        log_card.pack(fill="both", expand=True, pady=(16, 0))
        self.log = tk.Text(
            log_card, font=FONT_SMALL, bg=BG_CARD2, fg=FG, relief="flat",
            highlightthickness=1, highlightbackground=BORDER, wrap="word",
            state="disabled", height=8,
        )
        self.log.pack(side="left", fill="both", expand=True)
        clear_btn = ModernButton(log_card, "Bersihkan", command=self._clear_log, accent=False)
        clear_btn.configure(padx=10, pady=4)
        clear_btn.pack(side="right", padx=(8, 0))

        self._info_job = self.after(500, self._tick_info)

    def refresh(self):
        if not self.model_loaded and MODEL_PATH.exists():
            try:
                self.recognizer.load()
                self.model_loaded = True
            except Exception:
                self.model_loaded = False
        self._update_model_status()

    def _update_model_status(self):
        if self.model_loaded:
            self.model_label.configure(text="Model: terlatih", fg=GREEN)
        else:
            self.model_label.configure(text="Model: belum dilatih", fg=AMBER)

    def _on_slider(self, value):
        self.threshold = int(float(value))
        self.slider_val.configure(text=str(self.threshold))

    def _toggle(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        if not self.model_loaded:
            try:
                self.recognizer.load()
                self.model_loaded = True
                self._update_model_status()
            except ModelMissingError as exc:
                self.status.show(str(exc), RED)
                return
            except Exception as exc:
                self.status.show(f"Gagal memuat model: {exc}", RED)
                return
        try:
            self.camera.start()
        except CameraError as exc:
            self.status.show(str(exc), RED)
            return
        self.running = True
        self.btn.configure(text="Berhenti", command=self._toggle)
        self.status.show("Pengenalan aktif.", GREEN)

    def _stop(self):
        self.running = False
        self.camera.stop()
        self.btn.configure(text="Mulai Pengenalan", command=self._toggle)
        self.status.show("Pengenalan dihentikan.", FG_MUTED)

    def _on_frame(self, frame):
        if not self.running:
            return None
        return self.recognizer.process_frame(
            frame, self._labeler, on_result=self._log_result
        )

    def _labeler(self, uid, distance):
        name = self.users.display_for(uid)
        matched = distance < self.threshold
        return name, matched

    def _log_result(self, x, y, w, h, result, name, matched):
        tag = "✓" if matched else "?"
        conf = result["confidence"]
        self.status.show(f"{tag} {name} — kesamaan {conf:.0f}%", GREEN if matched else RED)
        self._log(f"{tag} {name} — {conf:.0f}%")

    def _log(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{ts}] {message}\n")
        if int(self.log.index("end-1c").split(".")[0]) > 60:
            self.log.delete("1.0", "2.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _tick_info(self):
        self.fps_label.configure(text=f"FPS: {self.camera.fps:.0f}")
        self._info_job = self.after(500, self._tick_info)

    def destroy(self):
        if self._info_job:
            self.after_cancel(self._info_job)
        super().destroy()
