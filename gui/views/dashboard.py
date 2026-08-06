"""Beranda: ringkasan status aplikasi + aksi cepat."""

import time
import tkinter as tk

from core.paths import DATASET_DIR, MODEL_PATH
from core.users import UserStore
from ..theme import *
from ..widgets import Card, ModernButton, ViewHeader


def _count_samples():
    return len(list(DATASET_DIR.glob("User.*.jpg")))


class DashboardView(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        ViewHeader(
            self,
            "Beranda",
            "Dashboard pengenalan wajah — Haar Cascade (deteksi) & LBPH (pengenalan)",
        ).pack(fill="x", padx=30, pady=(26, 10))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=(6, 22))

        self._stats_row = tk.Frame(body, bg=BG)
        self._stats_row.pack(fill="x", pady=(0, 18))
        self._stat_users = self._stat_card(self._stats_row, 0, "Pengguna Terdaftar", ACCENT)
        self._stat_samples = self._stat_card(self._stats_row, 1, "Sampel Wajah", ACCENT_LT)
        self._stat_model = self._stat_card(self._stats_row, 2, "Model LBPH", AMBER)
        self._stat_fps = self._stat_card(self._stats_row, 3, "Metode", GREEN)

        actions = Card(body, title="Aksi Cepat")
        actions.pack(fill="x", pady=(0, 18))
        row = tk.Frame(actions, bg=BG_CARD)
        row.pack(fill="x")
        self._action_btn(row, "Daftar Wajah", "Rekam sampel wajah baru", ACCENT, "register")
        self._action_btn(row, "Latih Model", "Latih model dari dataset", ACCENT_DK, "train")
        self._action_btn(row, "Mulai Pengenalan", "Pengenalan wajah real-time", GREEN, "recognize")

        info = Card(body, title="Cara Pakai")
        info.pack(fill="x")
        tk.Label(
            info,
            text=(
                "1. Daftarkan wajah Anda (30 sampel otomatis diambil secara otomatis)\n"
                "2. Latih model LBPH dari semua sampel yang tersimpan\n"
                "3. Mulai pengenalan wajah secara real-time"
            ),
            font=FONT, bg=BG_CARD, fg=FG_MUTED, justify="left", anchor="w",
        ).pack(fill="x")

    def _stat_card(self, parent, col, caption, color):
        card = tk.Frame(parent, bg=BG_CARD, padx=18, pady=14)
        card.grid(row=0, column=col, sticky="nsew", padx=(0, 14))
        parent.grid_columnconfigure(col, weight=1)
        value = tk.Label(card, text="–", font=FONT_BIG, bg=BG_CARD, fg=color)
        value.pack(anchor="w")
        tk.Label(card, text=caption, font=FONT, bg=BG_CARD, fg=FG_MUTED).pack(anchor="w", pady=(4, 0))
        return value

    def _action_btn(self, parent, title, caption, color, target):
        box = tk.Frame(parent, bg=BG_CARD2, padx=16, pady=14)
        box.pack(side="left", fill="both", expand=True, padx=(0, 14))
        btn = ModernButton(box, title, command=lambda: self.app.show(target))
        btn.configure(bg=color, activebackground=color)
        btn.pack(anchor="w")
        tk.Label(
            box, text=caption, font=FONT_SMALL, bg=BG_CARD2, fg=FG_MUTED,
        ).pack(anchor="w", pady=(6, 0))

    def refresh(self):
        users = UserStore().count()
        samples = _count_samples()
        self._stat_users.configure(text=str(users))
        self._stat_samples.configure(text=str(samples))
        self._stat_fps.configure(text="Haar+LBPH")
        if MODEL_PATH.exists():
            mtime = time.localtime(MODEL_PATH.stat().st_mtime)
            self._stat_model.configure(
                text=time.strftime("%d %b %Y", mtime), fg=GREEN
            )
        else:
            self._stat_model.configure(text="Belum", fg=AMBER)
