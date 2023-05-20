from functools import wraps
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, \
    QMessageBox
import sys
from PyQt5.QtGui import QPixmap, QIcon, QFont
import os


def check_file_loaded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.fname:
            self.show_popup_window('Сначала загрузите файлы!')
        else:
            return func(self, *args, **kwargs)
    return wrapper


class ImageGallery(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Приложение для автоматической разметки лебедей"
        self.current = 0
        self.fname = ''
        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QIcon("icon.ico"))
        self.setWindowTitle('icon')
        self.showMaximized()

        vbox = QVBoxLayout()

        topButtons = QHBoxLayout()
        btnOpenImages = NoFocusButton("Загрузить изображения")
        btnUseModel = NoFocusButton("Применить модель")

        mainWindow = QHBoxLayout()
        self.label = QLabel("Разметка лебедей будет тут!")
        self.label.setFont(QFont('Times font', 20))
        self.label.setAlignment(Qt.AlignCenter)

        arrows = QHBoxLayout()
        btnPrevImage = NoFocusButton("Назад")
        btnNextImage = NoFocusButton("Дальше")

        topButtons.addWidget(btnOpenImages)
        topButtons.addWidget(btnUseModel)
        vbox.addLayout(topButtons)

        mainWindow.addWidget(self.label)
        vbox.addLayout(mainWindow)

        arrows.addWidget(btnPrevImage)
        arrows.addWidget(btnNextImage)
        vbox.addLayout(arrows)
        self.setLayout(vbox)

        btnOpenImages.clicked.connect(self.getImage)
        btnUseModel.clicked.connect(self.useModel)

        btnNextImage.clicked.connect(lambda: self.nextImage())
        btnPrevImage.clicked.connect(lambda: self.prevImage())

        self.label.setFocus()

        self.show()

    def show_popup_window(self, error):
        msg = QMessageBox()
        msg.setWindowTitle("Внимание!")
        msg.setText(error)
        msg.exec_()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left: self.prevImage()
        elif event.key() == Qt.Key_Right: self.nextImage()

    def showingImage(self):
        imagePath = self.fname[0][self.current]
        pixmap = QPixmap(imagePath)
        scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(scaled_pixmap)
        print(self.fname[0][self.current])

    def getImage(self):
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file', os.getcwd(), "Image files (*.jpg *.gif *.jpeg)")
        try:
            imagePath = self.fname[0][0]
            # print("first image Path  = {}".format(imagePath))
            # print("list of image fname = {}".format(self.fname))
            # print("type imagePath{}".format(type(imagePath)))
            # print("type fname {}".format(type(self.fname)))
            self.showingImage()

        except IndexError as e:
            print(e)

        self.label.setFocus()

    @check_file_loaded
    def nextImage(self):
        try:
            if self.current >= len(self.fname[0]) - 1:
                self.show_popup_window('Это последнее изображение!')
            else:
                self.current += 1
                self.showingImage()

        except IndexError as e:  # сначала загрузите датасет функция
            print(e)

    @check_file_loaded
    def prevImage(self):
        if self.current > 0:
            self.current -= 1
            self.showingImage()
        else:
            self.show_popup_window('Это первое изображение!')



    @check_file_loaded
    def useModel(self):
        pass


class NoFocusButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.NoFocus)


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = ImageGallery()
    sys.exit(App.exec())