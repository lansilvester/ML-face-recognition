"""Widget reusable bertema gelap modern."""

import threading
import time

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .theme import *


class CameraError(Exception):
    pass


class ModernButton(tk.Button):
    def __init__(self, master, text, command=None, accent=True, **kwargs):
        bg = ACCENT if accent else BG_CARD2
        hover = ACCENT_DK if accent else BG_CARD
        fg = "#ffffff" if accent else FG
        super().__init__(
            master,
            text=text,
            command=command,
            relief="flat",
            bd=0,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            font=FONT_BOLD,
            padx=18,
            pady=8,
            cursor="hand2",
            highlightthickness=0,
            **kwargs,
        )
        self._hover_bg = hover
        self._base_bg = bg
        self._base_fg = fg

    def set_busy(self, busy, text=None):
        if text:
            self.configure(text=text)
        if busy:
            self.configure(state="disabled", bg=BG_CARD, fg=FG_MUTED)
        else:
            self.configure(state="normal", bg=self._base_bg, fg=self._base_fg)


class NavButton(tk.Button):
    def __init__(self, master, text, command=None):
        super().__init__(
            master,
            text=text,
            command=command,
            anchor="w",
            relief="flat",
            bd=0,
            padx=16,
            pady=9,
            cursor="hand2",
            font=FONT,
            bg=BG_SIDEBAR,
            fg=FG_MUTED,
            activebackground=BG_CARD,
            activeforeground=FG,
            highlightthickness=0,
        )
        self._active = False

    def set_active(self, active):
        self._active = active
        if active:
            self.configure(bg=ACCENT, fg="#ffffff", activebackground=ACCENT_DK)
        else:
            self.configure(bg=BG_SIDEBAR, fg=FG_MUTED, activebackground=BG_CARD, activeforeground=FG)


class Card(tk.Frame):
    def __init__(self, master, title=None, bg=BG_CARD, padx=18, pady=16, **kwargs):
        super().__init__(master, bg=bg, **kwargs)
        if title:
            tk.Label(
                self, text=title, font=FONT_H, bg=bg, fg=FG, anchor="w"
            ).pack(fill="x", pady=(0, 10))


class ViewHeader(tk.Frame):
    def __init__(self, master, title, subtitle=""):
        super().__init__(master, bg=BG)
        tk.Label(self, text=title, font=FONT_TITLE, bg=BG, fg=FG).pack(anchor="w")
        if subtitle:
            tk.Label(self, text=subtitle, font=FONT, bg=BG, fg=FG_MUTED).pack(anchor="w", pady=(2, 0))


class StatusLine(tk.Label):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            text="",
            font=FONT,
            bg=BG_CARD,
            fg=FG_MUTED,
            anchor="w",
            justify="left",
            wraplength=400,
            **kwargs,
        )

    def show(self, text, color=FG_MUTED):
        self.configure(text=text, fg=color)


class Badge(tk.Label):
    def __init__(self, master, text, color=GREEN, bg=BG_CARD, **kwargs):
        super().__init__(
            master,
            text=text,
            font=FONT_SMALL_BOLD,
            bg=color,
            fg="#0e0e16",
            padx=10,
            pady=3,
            **kwargs,
        )

    def update_text(self, text, color=None):
        self.configure(text=text)
        if color:
            self.configure(bg=color)


class ProgressBar(tk.Canvas):
    def __init__(self, master, width=320, height=14, fg=ACCENT, bg=BG_CARD2):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=BG_CARD,
            highlightthickness=0,
            bd=0,
        )
        self._width = width
        self._height = height
        self._fg = fg
        self._bg = bg
        self.set(0.0)

    def set(self, frac, text=None):
        self.delete("all")
        self.create_rectangle(0, 0, self._width, self._height, fill=self._bg, outline="")
        fw = max(self._height, int(self._width * min(max(frac, 0.0), 1.0)))
        if fw > 0:
            self.create_rectangle(0, 0, fw, self._height, fill=self._fg, outline="")
        if text:
            self.create_text(
                self._width / 2, self._height / 2, text=text,
                fill="#ffffff", font=FONT_SMALL_BOLD,
            )


class CameraCanvas(tk.Frame):
    """Canvas menampilkan webcam dari thread latar dengan polling via after()."""

    def __init__(self, master, width=640, height=480, on_frame=None, bg=BG_CARD):
        super().__init__(master, bg=bg)
        self.width = width
        self.height = height
        self.on_frame = on_frame  # def(frame_bgr) -> annotated_bgr atau None
        self.fps = 0.0

        self.canvas = tk.Canvas(
            self, width=width, height=height,
            bg="#000000", highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self._cap = None
        self._thread = None
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._photo = None
        self._job = None
        self._fps_counter = 0
        self._fps_t0 = time.time()

        self._draw(self._placeholder())

    def _placeholder(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (28, 28, 40)
        cv2.putText(
            img, "Kamera tidak aktif",
            (self.width // 2 - 150, self.height // 2 - 8),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (160, 160, 200), 1, cv2.LINE_AA,
        )
        return img

    @property
    def is_running(self):
        return self._running

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._latest = frame

    def start(self, index=0):
        if self._running:
            return
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Webcam {index} tidak dapat dibuka.")
        self._cap = cap
        self._running = True
        self._fps_counter = 0
        self._fps_t0 = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._job = self.after(15, self._tick)

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest = None
        self.fps = 0.0
        self._draw(self._placeholder())

    def _tick(self):
        if not self._running:
            return
        frame = None
        with self._lock:
            if self._latest is not None:
                frame = self._latest.copy()
        if frame is not None:
            frame = cv2.flip(frame, 1)
            annotated = self.on_frame(frame) if self.on_frame else None
            self._draw(annotated if annotated is not None else frame)
            self._fps_counter += 1
            now = time.time()
            if now - self._fps_t0 >= 1.0:
                self.fps = self._fps_counter / (now - self._fps_t0)
                self._fps_counter = 0
                self._fps_t0 = now
        self._job = self.after(15, self._tick)

    def _draw(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(self.width / w, self.height / h)
        nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
        disp = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(self.width // 2, self.height // 2, image=self._photo)

    def destroy(self):
        self.stop()
        super().destroy()
