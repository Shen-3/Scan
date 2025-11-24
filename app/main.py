from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QApplication

from app.ui.main_window import MainWindow

#check that
def main() -> None:
    app = QApplication([])
    window = MainWindow(Path("app/settings.json"))
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

