from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, \
    QMessageBox
import sys
from PyQt5.QtGui import QPixmap, QIcon, QFont


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
        btnOpenImages = QPushButton("Загрузить изображения")
        btnUseModel = QPushButton("Применить модель")
        btnSaveDetectedImages = QPushButton("Сохранить как")

        mainWindow = QHBoxLayout()
        self.label = QLabel("Разметка лебедей будет тут!")
        self.label.setFont(QFont('Times font', 20))
        self.label.setAlignment(Qt.AlignCenter)

        arrows = QHBoxLayout()
        btnPrevImage = QPushButton("Назад")
        btnNextImage = QPushButton("Дальше")


        topButtons.addWidget(btnOpenImages)
        topButtons.addWidget(btnUseModel)
        topButtons.addWidget(btnSaveDetectedImages)
        vbox.addLayout(topButtons)

        mainWindow.addWidget(self.label)
        vbox.addLayout(mainWindow)

        arrows.addWidget(btnPrevImage)
        arrows.addWidget(btnNextImage)
        vbox.addLayout(arrows)
        self.setLayout(vbox)


        btnOpenImages.clicked.connect(self.getImage)
        btnUseModel.clicked.connect(self.useModel)
        btnSaveDetectedImages.clicked.connect(self.saveAs)

        btnNextImage.clicked.connect(self.nextImage)
        btnPrevImage.clicked.connect(self.prevImage)

        self.show()


    def getImage(self):
        import os
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file', os.getcwd(), "Image files (*.jpg *.gif *.jpeg)")
        try:
            imagePath = self.fname[0][0]
            print("first image Path  = {}".format(imagePath))
            print("list of image fname = {}".format(self.fname))
            print("type imagePath{}".format(type(imagePath)))
            print("type fname {}".format(type(self.fname)))
            pixmap = QPixmap(imagePath)
            label_size = self.label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio)
            self.label.setPixmap(scaled_pixmap)

        except IndexError as e:
            print(e)

    def nextImage(self):
        try:
            # print(self.fname)
            if self.current >= len(self.fname[0]) - 1:
                self.show_popup_window('Это последнее изображение!')
            else:
                self.current += 1
                imagePath = self.fname[0][self.current]
                pixmap = QPixmap(imagePath)
                label_size = self.label.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio)
                self.label.setPixmap(scaled_pixmap)
                print(self.fname[0][self.current])
        except IndexError as e:
            print(e)

    def prevImage(self):
        if self.current > 0:
            self.current -= 1
            imagePath = self.fname[0][self.current]
            pixmap = QPixmap(imagePath)
            label_size = self.label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio)
            self.label.setPixmap(scaled_pixmap)
            print(self.fname[0][self.current])
        else:
            self.show_popup_window('Это первое изображение!')

    def show_popup_window(self, error):
        msg = QMessageBox()
        msg.setWindowTitle("Внимание!")
        msg.setText(error)
        msg.exec_()


    def saveAs(self):
        pass


    def useModel(self):
        pass


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = ImageGallery()
    sys.exit(App.exec())
