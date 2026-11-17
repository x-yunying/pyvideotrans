from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog

from videotrans.configure.config import ROOT_DIR, params, tr
from videotrans.util.help_misc import open_url


class Ui_deepseekasr(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(f"{ROOT_DIR}/videotrans/styles/icon.ico"))
        self.setupUi(self)

    def setupUi(self, form):
        self.has_done = False
        form.setObjectName("form")
        form.resize(560, 220)
        form.setWindowTitle("DeepSeek ASR")

        v1 = QtWidgets.QVBoxLayout(form)
        v1.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        tip = QtWidgets.QLabel()
        tip.setWordWrap(True)
        tip.setObjectName("tip")
        tip.setText(tr("Reuses the DeepSeek API Key configured in the translation settings."))
        v1.addWidget(tip)

        h2 = QtWidgets.QHBoxLayout()
        label_key = QtWidgets.QLabel()
        label_key.setMinimumSize(QtCore.QSize(0, 35))
        label_key.setText(tr("SK"))

        self.deepseek_key = QtWidgets.QLineEdit()
        self.deepseek_key.setMinimumSize(QtCore.QSize(0, 35))
        self.deepseek_key.setObjectName("deepseek_key")
        h2.addWidget(label_key)
        h2.addWidget(self.deepseek_key)
        v1.addLayout(h2)

        h4 = QtWidgets.QHBoxLayout()
        self.set = QtWidgets.QPushButton()
        self.set.setMinimumSize(QtCore.QSize(0, 35))
        self.set.setObjectName("set")
        self.set.setText(tr("Save"))

        self.test = QtWidgets.QPushButton()
        self.test.setMinimumSize(QtCore.QSize(0, 35))
        self.test.setObjectName("test")
        self.test.setText(tr("Test"))

        help_btn = QtWidgets.QPushButton()
        help_btn.setMinimumSize(QtCore.QSize(0, 35))
        help_btn.setStyleSheet("background-color: rgba(255, 255, 255,0)")
        help_btn.setObjectName("help_btn")
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.setText(tr("Fill out the tutorial"))
        help_btn.clicked.connect(lambda: open_url(url='https://api-docs.deepseek.com/guides/asr'))

        h4.addWidget(self.set)
        h4.addWidget(self.test)
        h4.addWidget(help_btn)
        v1.addLayout(h4)
        QtCore.QMetaObject.connectSlotsByName(form)

    def update_ui(self):
        self.deepseek_key.setText(str(params.get("deepseek_key", "")))
