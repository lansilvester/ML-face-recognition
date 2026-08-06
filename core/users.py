"""Database pengguna sederhana berbasis JSON: memetakan ID wajah ke nama."""

import json
from pathlib import Path

from .paths import USERS_PATH

# Nama awal dari dataset asli (2021) supaya data lama langsung terbaca.
DEFAULT_USERS = {
    "1": "Alan",
    "2": "Natan",
    "3": "Yogi",
    "4": "Oswal",
    "5": "Andi",
    "6": "Keyzia",
    "13": "Gunawan",
}


class UserStore:
    def __init__(self, path=USERS_PATH):
        self.path = Path(path)
        self.users = {}
        self.load()

    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.users = {str(k): v for k, v in data.items()}
        else:
            self.users = dict(DEFAULT_USERS)
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.users, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def next_id(self):
        ids = {int(i) for i in self.users if str(i).isdigit()}
        i = 1
        while i in ids:
            i += 1
        return i

    def add(self, name):
        uid = self.next_id()
        self.users[str(uid)] = name.strip()
        self.save()
        return uid

    def rename(self, uid, name):
        self.users[str(uid)] = name.strip()
        self.save()

    def remove(self, uid):
        self.users.pop(str(uid), None)
        self.save()

    def name_for(self, uid):
        return self.users.get(str(uid))

    def id_for_name(self, name):
        target = name.strip().lower()
        for uid, value in self.users.items():
            if value.strip().lower() == target:
                return int(uid)
        return None

    def display_for(self, uid):
        name = self.users.get(str(uid))
        return name if name else f"User {uid}"

    def count(self):
        return len(self.users)
