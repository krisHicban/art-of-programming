# North Star Premium Imobiliare

from PySide6 import QtWidgets, QtCore, QtGui
import sys, os
from pathlib import Path

GOLD = "#C9A227"
GOLD_SOFT = "#E8D07A"
BG = "#0b0b0b"

# Handle __file__ safely when running interactively (like in IDE)
if getattr(sys, 'frozen', False):
    BG_IMG = Path(sys.executable).with_name("bg.jpg")
else:
    BG_IMG = Path(__file__).with_name("bg.jpg")


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("North Star Premium Imobiliare")
        self.resize(1000, 650)

        c = QtWidgets.QWidget()
        self.setCentralWidget(c)
        s = QtWidgets.QStackedLayout(c)

        bg = QtWidgets.QLabel()
        bg.setScaledContents(True)
        if BG_IMG.exists():
            bg.setPixmap(QtGui.QPixmap(str(BG_IMG)))
        else:
            bg.setStyleSheet(f"background:{BG};")
        s.addWidget(bg)

        layer = QtWidgets.QWidget()
        s.addWidget(layer)
        v = QtWidgets.QVBoxLayout(layer)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)

        self.title = QtWidgets.QLabel("🌟 NORTH STAR PREMIUM IMOBILIARE 🌟")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet(f"font-size:26px; font-weight:800; color:{GOLD};")

        self.slogan = QtWidgets.QLabel("Unde rafinamentul intalneste viziunea")
        self.slogan.setAlignment(QtCore.Qt.AlignCenter)
        self.slogan.setStyleSheet(f"font-size:15px; color:{GOLD}; font-weight:600;")

        v.addWidget(self.title)
        v.addWidget(self.slogan)
        v.addSpacing(10)

        grid = QtWidgets.QGridLayout()
        v.addLayout(grid)
        v.addStretch(1)

        btns = [
            "CRM", "HR", "Contabilitate", "AURA Asistent AI",
            "Promovare", "Administrare", "Analize & Rapoarte",
            "Design Grafic", "Securitate Cibernetica", "Recenzii",
            "Juridic", "Agentii RO"
        ]

        style = (
            f"QPushButton{{background:{GOLD};color:#000;"
            "font-weight:700;font-size:15px;border-radius:10px;padding:12px;}}"
            f"QPushButton:hover{{background:{GOLD_SOFT};}}"
        )

        for i, text in enumerate(btns):
            b = QtWidgets.QPushButton(text)
            b.setStyleSheet(style)
            b.setMinimumHeight(60)
            b.clicked.connect(lambda _, t=text: self.msg(t))
            r, c = divmod(i, 3)
            grid.addWidget(b, r, c)

        self.glow = False
        t = QtCore.QTimer(self)
        t.timeout.connect(self._anim)
        t.start(1300)

    def msg(self, modul):
        QtWidgets.QMessageBox.information(self, modul, f"Modul «{modul}» deschis (demo).")

    def _anim(self):
        self.glow = not self.glow
        c = GOLD_SOFT if self.glow else GOLD
        self.title.setStyleSheet(f"font-size:26px; font-weight:800; color:{c};")
        self.slogan.setStyleSheet(f"font-size:15px; font-weight:600; color:{c};")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec())
