"""Entry point aplikasi GUI pengenalan wajah.

Jalankan:  python app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.paths import ensure_dirs
from gui import MainWindow


def main():
    ensure_dirs()
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
