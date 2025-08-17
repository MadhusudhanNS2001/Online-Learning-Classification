if True:
    from reset_random import reset_random

    reset_random()

import sys

import pandas as pd
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QFont, QTextCursor, QTextOption
from PyQt5.QtWidgets import (QWidget, QApplication, QGridLayout, QGroupBox, QVBoxLayout, QPushButton,
                             QScrollArea, QMessageBox, QPlainTextEdit, QFrame, QTableView,
                             QAbstractItemView, QLabel)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

import train
from data_handler import load_data, preprocess_data, merge_data
from utils import Stream, Worker, PandasDfToPyqtTable


class MainGUI(QWidget):
    def __init__(self):
        super(MainGUI, self).__init__()

        self.setWindowTitle('Online Learning Classification')
        self.width_p = (QApplication.desktop().availableGeometry().width() // 100)
        self.height_p = (QApplication.desktop().availableGeometry().height() // 100)

        app.setFont(QFont('JetBrains Mono', 9))
        self.setWindowFlags(Qt.WindowMinimizeButtonHint
                            | Qt.WindowCloseButtonHint)

        self.main_layout = QGridLayout()
        self.main_layout.setAlignment(Qt.AlignTop)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.gb_1 = QGroupBox('Input Data')
        self.gb_1.setFixedHeight(self.height_p * 15)
        self.gb_1.setFixedWidth(self.width_p * 49)
        self.grid_1 = QGridLayout()
        self.gb_1.setLayout(self.grid_1)

        self.load_btn = QPushButton('Load Data')
        self.load_btn.setFixedHeight(self.height_p * 5)
        self.load_btn.clicked.connect(self.load_thread)
        self.grid_1.addWidget(self.load_btn, 0, 0)

        self.mg_btn = QPushButton('Merge Data')
        self.mg_btn.setFixedHeight(self.height_p * 5)
        self.mg_btn.clicked.connect(self.mg_thread)
        self.grid_1.addWidget(self.mg_btn, 0, 1)

        self.pp_btn = QPushButton('Preprocess Data')
        self.pp_btn.setFixedHeight(self.height_p * 5)
        self.pp_btn.clicked.connect(self.pp_thread)
        self.grid_1.addWidget(self.pp_btn, 0, 2)

        self.train_btn = QPushButton('Train Deep Stacked\nSparse AutoEncoder')
        self.train_btn.setFixedHeight(self.height_p * 5)
        self.train_btn.clicked.connect(self.train_thread)
        self.grid_1.addWidget(self.train_btn, 1, 0, 1, 2)

        self.reset_btn = QPushButton('Reset')
        self.reset_btn.setFixedHeight(self.height_p * 5)
        self.reset_btn.clicked.connect(self.reset)
        self.grid_1.addWidget(self.reset_btn, 1, 2)

        self.gb_2 = QGroupBox('Data Table')
        self.gb_2.setFixedHeight(self.height_p * 50)
        self.gb_2.setFixedWidth(self.width_p * 49)
        self.grid_2_scroll = QScrollArea()
        self.grid_2_scroll.setFrameShape(False)
        self.gb_2_v_box = QVBoxLayout()
        self.grid_2_widget = QWidget()
        self.grid_2_widget.hide()
        self.grid_2 = QGridLayout(self.grid_2_widget)
        self.gb_2.setLayout(self.gb_2_v_box)
        self.grid_2.setSpacing(20)
        self.grid_2_scroll.setWidgetResizable(True)
        self.grid_2_scroll.setWidget(self.grid_2_widget)
        self.gb_2_v_box.addWidget(self.grid_2_scroll)
        self.gb_2_v_box.setContentsMargins(0, 0, 0, 0)

        self.gb_3 = QGroupBox('Progress')
        self.gb_3.setFixedHeight(self.height_p * 35)
        self.gb_3.setFixedWidth(self.width_p * 49)
        self.grid_3 = QGridLayout()
        self.gb_3.setLayout(self.grid_3)

        self.progress_pte = QPlainTextEdit()
        self.progress_pte.setFont(QFont('JetBrains Mono', 9))
        self.progress_pte.setStyleSheet('background-color: transparent;')
        self.progress_pte.setFrameShape(QFrame.NoFrame)
        self.progress_pte.setReadOnly(True)
        self.progress_pte.setWordWrapMode(QTextOption.WordWrap)
        self.grid_3.addWidget(self.progress_pte, 0, 0)

        self.gb_4 = QGroupBox('Visualization')
        self.gb_4.setFixedHeight(self.height_p * 99)
        self.gb_4.setFixedWidth(self.width_p * 50)
        self.grid_4_scroll = QScrollArea()
        self.grid_4_scroll.setFrameShape(QFrame.NoFrame)
        self.gb_4_v_box = QVBoxLayout()
        self.grid_4_widget = QWidget()
        self.grid_4 = QGridLayout(self.grid_4_widget)
        self.grid_4.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.grid_4_scroll.setWidgetResizable(True)
        self.grid_4_scroll.setWidget(self.grid_4_widget)
        self.gb_4_v_box.addWidget(self.grid_4_scroll)
        self.gb_4_v_box.setContentsMargins(0, 0, 0, 0)
        self.gb_4.setLayout(self.gb_4_v_box)

        self.main_layout.addLayout(self.left_layout, 0, 0)
        self.main_layout.addLayout(self.right_layout, 0, 1)

        self.left_layout.addWidget(self.gb_1)
        self.left_layout.addWidget(self.gb_2)
        self.left_layout.addWidget(self.gb_3)
        self.right_layout.addWidget(self.gb_4)

        self.thread_pool = QThreadPool()

        sys.stdout = Stream(fn=self.update_progress)

        self.original_df = []
        self.mg_df = pd.DataFrame()
        self.pp_df = pd.DataFrame()
        self.fs_df = pd.DataFrame()
        self.thread_pool = QThreadPool()
        self.index = 0

        self.reset()
        self.setLayout(self.main_layout)
        self.showMaximized()

    def update_progress(self, text):
        cursor = self.progress_pte.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.progress_pte.setTextCursor(cursor)
        self.progress_pte.ensureCursorVisible()

    def add_table(self, df):
        table_view = QTableView(self)
        model = PandasDfToPyqtTable(df)
        table_view.setFixedWidth((self.gb_2.width() // 100) * 95)
        table_view.setFixedHeight((self.gb_2.height() // 100) * 95)
        table_view.setModel(model)
        table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_view.verticalHeader().hide()
        table_view.resizeColumnsToContents()
        self.grid_2.addWidget(
            table_view, self.grid_2.count() + 1, 0, Qt.AlignHCenter)

    def add_plot(self, fig, title):
        canvas = FigureCanvasQTAgg(figure=fig)
        canvas.setFixedSize((self.gb_4.width() // 100) * 95, (self.gb_4.width() // 100) * 70)
        self.grid_4.addWidget(QLabel(title), self.index, 0)
        self.grid_4.addWidget(canvas, self.index + 1, 0)
        self.index += 2

    def load_thread(self):
        self.reset()
        worker = Worker(self.load_runner)
        worker.signals.finished.connect(self.load_finisher)
        self.thread_pool.start(worker)
        self.load_btn.setEnabled(False)

    def load_runner(self):
        self.original_df = load_data()

    def load_finisher(self):
        for df in self.original_df:
            self.add_table(df.head(100))
        self.mg_btn.setEnabled(True)

    def mg_thread(self):
        worker = Worker(self.mg_runner)
        worker.signals.finished.connect(self.mg_finisher)
        self.thread_pool.start(worker)
        self.mg_btn.setEnabled(False)

    def mg_runner(self):
        self.mg_df = merge_data(*[c.copy() for c in self.original_df])

    def mg_finisher(self):
        self.add_table(self.mg_df.head(100))
        self.pp_btn.setEnabled(True)

    def pp_thread(self):
        worker = Worker(self.pp_runner)
        worker.signals.finished.connect(self.pp_finisher)
        self.thread_pool.start(worker)
        self.pp_btn.setEnabled(False)

    def pp_runner(self):
        self.pp_df = preprocess_data(self.mg_df.copy())

    def pp_finisher(self):
        self.add_table(self.pp_df.head(100))
        self.train_btn.setEnabled(True)

    def train_thread(self):
        worker = Worker(train.train)
        worker.signals.finished.connect(self.train_finisher)
        self.thread_pool.start(worker)
        self.train_btn.setEnabled(False)
        self.add_plot(train.ACC_PLOT, 'Accuracy Plot')
        self.add_plot(train.LOSS_PLOT, 'Loss Plot')

    def train_finisher(self):
        for v in train.RESULTS_PLOT:
            for v1 in train.RESULTS_PLOT[v]:
                self.add_plot(train.RESULTS_PLOT[v][v1], '{1} {0}'.format(v, v1))

    @staticmethod
    def clear_layout(layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            if not item:
                continue
            w = item.widget()
            if w:
                w.deleteLater()

    @staticmethod
    def show_message_box(title, msg, icon=QMessageBox.Critical):
        msg_box = QMessageBox()
        msg_box.setFont(QFont('Fira Code', 10, 1))
        msg_box.setWindowTitle(title)
        msg_box.setText(msg)
        msg_box.setIcon(icon)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.exec_()

    def disable(self):
        self.load_btn.setEnabled(True)
        self.mg_btn.setEnabled(False)
        self.pp_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.original_df = []
        self.mg_df = pd.DataFrame()
        self.pp_df = pd.DataFrame()
        self.fs_df = pd.DataFrame()
        self.progress_pte.clear()
        self.index = 0

    def reset(self):
        self.disable()
        self.clear_layout(self.grid_2)
        self.clear_layout(self.grid_4)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainGUI()
    sys.exit(app.exec_())
