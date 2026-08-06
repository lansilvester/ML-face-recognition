"""Jendela utama: sidebar navigasi + konten yang bisa berpindah view."""

import tkinter as tk

from .theme import *
from .widgets import NavButton
from .views import VIEWS


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition — Haar Cascade & LBPH")
        self.geometry("1200x740")
        self.minsize(1040, 660)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_sidebar()
        self._build_content()
        self.show("dashboard")

    def _build_sidebar(self):
        sidebar = tk.Frame(self, bg=BG_SIDEBAR, width=SIDEBAR_W)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self.sidebar = sidebar

        logo = tk.Frame(sidebar, bg=BG_SIDEBAR)
        logo.pack(fill="x", pady=(28, 14))
        tk.Label(
            logo, text="FR", font=("Segoe UI", 20, "bold"),
            bg=ACCENT, fg="#ffffff", padx=14, pady=4,
        ).pack()

        tk.Label(sidebar, text="Face Recognition", font=FONT_H, bg=BG_SIDEBAR, fg=FG).pack()
        tk.Label(
            sidebar, text="Haar Cascade + LBPH",
            font=FONT_SMALL, bg=BG_SIDEBAR, fg=FG_MUTED,
        ).pack(pady=(2, 22))

        self.nav_btns = []
        for key, label in [
            ("dashboard", "Beranda"),
            ("register", "Daftar Wajah"),
            ("train", "Latih Model"),
            ("recognize", "Pengenalan"),
        ]:
            btn = NavButton(sidebar, label, command=lambda k=key: self.show(k))
            btn.pack(fill="x", padx=10, pady=3)
            self.nav_btns.append((key, btn))

        tk.Frame(sidebar, bg=BG_SIDEBAR).pack(expand=True)
        tk.Label(sidebar, text="v2.0 — 2026", font=FONT_SMALL, bg=BG_SIDEBAR, fg=FG_MUTED).pack(pady=14)

    def _build_content(self):
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)
        self._views = {}
        self._current = None

    def show(self, key):
        if key not in VIEWS:
            raise ValueError(f"View tidak dikenal: {key}")

        for k, btn in self.nav_btns:
            btn.set_active(k == key)

        if self._current in self._views:
            self._views[self._current].pack_forget()

        if key not in self._views:
            self._views[key] = VIEWS[key](self.content, self)

        self._current = key
        self._views[key].refresh()
        self._views[key].pack(fill="both", expand=True)

    def _on_close(self):
        self.destroy()
