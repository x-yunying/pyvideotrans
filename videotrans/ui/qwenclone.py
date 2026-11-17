# 阿里云百炼 千问3-TTS 声音复刻（在线）
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog

from videotrans.configure.config import tr, params, ROOT_DIR
from videotrans.util.help_misc import open_url


class Ui_qwenclone(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(f"{ROOT_DIR}/videotrans/styles/icon.ico"))
        self.setupUi(self)

    def setupUi(self, form):
        self.has_done = False
        form.setObjectName("form")
        form.resize(620, 330)

        self.verticalLayout = QtWidgets.QVBoxLayout(form)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.formLayout.setObjectName("formLayout")

        self.label_key = QtWidgets.QLabel()
        self.label_key.setMinimumSize(QtCore.QSize(140, 35))
        self.qwenclone_key = QtWidgets.QLineEdit()
        self.qwenclone_key.setMinimumSize(QtCore.QSize(280, 35))
        self.qwenclone_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.formLayout.addRow(self.label_key, self.qwenclone_key)

        self.label_spaceid = QtWidgets.QLabel()
        self.label_spaceid.setMinimumSize(QtCore.QSize(140, 35))
        self.qwenclone_spaceid = QtWidgets.QLineEdit()
        self.qwenclone_spaceid.setMinimumSize(QtCore.QSize(280, 35))
        self.qwenclone_spaceid.setPlaceholderText(tr("Optional, business space ID or full API URL"))
        self.formLayout.addRow(self.label_spaceid, self.qwenclone_spaceid)

        self.label_model = QtWidgets.QLabel()
        self.label_model.setMinimumSize(QtCore.QSize(140, 35))
        self.qwenclone_model = QtWidgets.QLineEdit()
        self.qwenclone_model.setMinimumSize(QtCore.QSize(280, 35))
        self.formLayout.addRow(self.label_model, self.qwenclone_model)

        self.label_roles = QtWidgets.QLabel()
        self.label_roles.setMinimumSize(QtCore.QSize(140, 35))
        self.qwenclone_roles = QtWidgets.QLineEdit()
        self.qwenclone_roles.setMinimumSize(QtCore.QSize(280, 35))
        self.qwenclone_roles.setPlaceholderText(tr("Created voice IDs, separated by commas"))
        self.formLayout.addRow(self.label_roles, self.qwenclone_roles)

        self.verticalLayout.addLayout(self.formLayout)

        h4 = QtWidgets.QHBoxLayout()
        self.set = QtWidgets.QPushButton()
        self.set.setMinimumSize(QtCore.QSize(0, 35))
        self.test = QtWidgets.QPushButton()
        self.test.setMinimumSize(QtCore.QSize(0, 35))
        help_btn = QtWidgets.QPushButton()
        help_btn.setMinimumSize(QtCore.QSize(0, 35))
        help_btn.setStyleSheet("background-color: rgba(255, 255, 255,0)")
        help_btn.setCursor(QtCore.Qt.PointingHandCursor)
        help_btn.clicked.connect(lambda: open_url(url='https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning'))
        h4.addWidget(self.set)
        h4.addWidget(self.test)
        h4.addWidget(help_btn)
        self.verticalLayout.addLayout(h4)

        self.retranslateUi(form)
        QtCore.QMetaObject.connectSlotsByName(form)

    def update_ui(self):
        self.qwenclone_key.setText(str(params.get("qwenclone_key", '')))
        self.qwenclone_spaceid.setText(str(params.get("qwenclone_spaceid", '')))
        self.qwenclone_model.setText(str(params.get("qwenclone_model", 'qwen-audio-3.0-tts-flash')))
        self.qwenclone_roles.setText(str(params.get("qwenclone_roles", '')))

    def retranslateUi(self, form):
        form.setWindowTitle(tr("Qwen Online Voice Cloning"))
        self.label_key.setText(tr("API Key"))
        self.label_spaceid.setText(tr("Workspace ID"))
        self.label_model.setText(tr("Voice cloning model"))
        self.label_roles.setText(tr("Created voice IDs"))
        self.set.setText(tr("Save"))
        self.test.setText(tr("Test"))
