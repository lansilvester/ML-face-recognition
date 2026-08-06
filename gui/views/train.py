"""Latih Model: melatih model LBPH dari Dataset dengan progress real-time."""

import queue
import threading
import tkinter as tk

from core.paths import MODEL_PATH
from core.trainer import Trainer
from ..theme import *
from ..widgets import Card, ModernButton, ProgressBar, StatusLine, ViewHeader


class TrainView(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        self.training = False
        self.q = queue.Queue()
        self._poll_job = None
        self._result_rows = []
        self._build()

    def _build(self):
        ViewHeader(
            self,
            "Latih Model",
            "Latih model LBPH dari semua sampel wajah di folder Dataset",
        ).pack(fill="x", padx=30, pady=(26, 10))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=(6, 24))

        left = Card(body, title="Mulai Pelatihan", padx=20, pady=20)
        left.pack(side="left", fill="both", expand=True, padx=(0, 16))

        self.btn = ModernButton(left, "Mulai Latih", command=self._start, accent=True)
        self.btn.pack(fill="x", pady=(0, 16))

        self.progress = ProgressBar(left, width=460)
        self.progress.pack(fill="x", pady=(0, 14))
        self.status = StatusLine(left)
        self.status.pack(fill="x")

        tk.Label(
            left,
            text=(
                "Model tersimpan di: models/trainer.yml\n"
                "Gunakan tombol 'Latih Ulang' setiap kali menambahkan sampel wajah baru."
            ),
            font=FONT_SMALL, bg=BG_CARD, fg=FG_MUTED, justify="left", anchor="w",
        ).pack(fill="x", pady=(20, 0))

        self.right = Card(body, title="Hasil Latihan", padx=20, pady=20)
        self.right.pack(side="right", fill="y")

        self._waiting_text = tk.Label(
            self.right,
            text="Belum ada hasil latihan.\nLatih model untuk melihat ringkasan di sini.",
            font=FONT, bg=BG_CARD, fg=FG_MUTED, justify="left", anchor="w",
        )
        self._waiting_text.pack(anchor="w")

        self._done_btn = ModernButton(self.right, "Buka Pengenalan", command=lambda: self.app.show("recognize"))
        self._done_btn.configure(bg=GREEN, activebackground="#2cb96a")

    def _row(self, parent, caption, value="–"):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=caption, font=FONT_SMALL, bg=BG_CARD, fg=FG_MUTED).pack(side="left")
        value_lbl = tk.Label(row, text=value, font=FONT_SMALL_BOLD, bg=BG_CARD, fg=FG)
        value_lbl.pack(side="right")
        return value_lbl

    def refresh(self):
        pass

    def _start(self):
        if self.training:
            return
        self.training = True
        self.btn.set_busy(True, "Melatih...")
        self.status.show("Menghitung sampel...", ACCENT_LT)

        def worker():
            try:
                result = Trainer().train(progress=self._push_progress)
                self.q.put(("done", result))
            except Exception as exc:
                self.q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_job = self.after(80, self._drain)

    def _push_progress(self, i, n, msg):
        self.q.put(("progress", (i, n, msg)))

    def _drain(self):
        finished = False
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "progress":
                    i, n, msg = payload
                    frac = (i / n) if n else 0.0
                    self.progress.set(frac, msg)
                elif kind == "done":
                    self._done(payload)
                    finished = True
                elif kind == "error":
                    self._error(payload)
                    finished = True
        except queue.Empty:
            pass
        if not finished and self.training:
            self._poll_job = self.after(80, self._drain)
        else:
            self._poll_job = None

    def _done(self, result):
        self.training = False
        self.btn.set_busy(False, "Latih Ulang")
        self.progress.set(1.0, "Selesai")
        self.status.show(
            f"Model berhasil dilatih dan disimpan. ({result['time']:.1f} detik)",
            GREEN,
        )
        self._waiting_text.pack_forget()
        for lbl in self._result_rows:
            lbl.master.destroy()
        self._result_rows = [
            self._row(self.right, "Jumlah pengguna", str(result["users"])),
            self._row(self.right, "Total sampel", str(result["samples"])),
            self._row(self.right, "Waktu latih", f"{result['time']:.1f} dtk"),
            self._row(self.right, "Lokasi model", str(MODEL_PATH)),
        ]
        self._done_btn.pack(fill="x", pady=(16, 0))

    def _error(self, message):
        self.training = False
        self.btn.set_busy(False, "Coba Lagi")
        self.status.show(message, RED)
        self.progress.set(0.0, "")
