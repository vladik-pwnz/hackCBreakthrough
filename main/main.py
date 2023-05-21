from functools import wraps

import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, \
    QMessageBox
import sys
from PyQt5.QtGui import QPixmap, QIcon, QFont, QImage
import os
# from PyQt5 import QtGui
from detector.detector import load_model_detector
import cv2
from draw.drawer import Drawer
import numpy as np
# from classifier.classificatorResnet18 import ResNet
#
# from torchvision.models import resnet18

def check_file_loaded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.fname:
            self.show_popup_window('Сначала загрузите файлы!')
        else:
            return func(self, *args, **kwargs)
    return wrapper


class ImageGallery(QWidget):
    def __init__(self, detector_modelm, classifier_model=None):
        super().__init__()
        self.title = "Приложение для автоматической разметки лебедей"
        self.current = 0
        self.fname = ''
        self.InitWindow()
        self.detector_model = detector_model
        self.classifier_model = classifier_model
        self.drawer = Drawer()

    def InitWindow(self):
        self.setWindowIcon(QIcon("icon.ico"))
        self.setWindowTitle('icon')

        vbox = QVBoxLayout()

        topButtons = QHBoxLayout()
        btnOpenImages = NoFocusButton("Загрузить изображения")
        btnUseModel = NoFocusButton("Применить модель")

        mainWindow = QHBoxLayout()
        self.label = QLabel("Добро пожаловать!")
        self.label.setFont(QFont('Times font', 20))
        self.label.setAlignment(Qt.AlignCenter)

        arrows = QHBoxLayout()
        btnPrevImage = NoFocusButton("Назад")
        btnNextImage = NoFocusButton("Вперед")

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
        ####
        img = cv2.imread(imagePath)
        input_size = (640,640)
        ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR)

        self.drawer.set_image(img[:])
        img = img[:, :, ::-1]
        name_conf_bb_list = self.detector_model(img)  # BGR to RGB

        if self.classifier_model:
            conf_class = self.classifier_model(img)
            res = torch.argmax(conf_class)
            print(res)

        max_conf = 0
        pred_name = 0
        for cls_name, conf, *bbox in name_conf_bb_list:
            bbox = np.array(bbox).reshape(-1, 2)
            self.drawer.draw_bbox(bbox, cls_name, conf)
            if conf > max_conf:
                max_conf = conf
                pred_name = cls_name


        #TODO: add catboost
        self.label.setText(pred_name)
        # self.drawer.draw_text()
        ####
        # pixmap = QPixmap(imagePath)
        self.drawer.show()

        # cvImg = self.drawer.image[:, :, ::-1]
        # height, width, channel = cvImg.shape
        # bytesPerLine = 3 * width
        # image = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)


        # image = self.drawer.image[:,:,::-1]
        # image = QtGui.QImage(image, image.shape[1],
        #                      image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        # pixmap = QtGui.QPixmap(image)
        # pixmap = QPixmap(self.drawer.image)
        # scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio)
        # self.label.setPixmap(scaled_pixmap)
        # print(self.fname[0][self.current])

    def getImage(self):
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file', os.getcwd(), "Image files (*.jpg *.gif *.jpeg)")
        try:
            imagePaths = self.fname[0]
            # Iterate over selected file paths
            for imagePath in imagePaths:
                # Check if the file has an image extension
                if not any(imagePath.lower().endswith(ext) for ext in ['.jpg', '.gif', '.jpeg']):
                    self.show_popup_window('Добавлено не фото! можно добавлять только фото.')
                    return

            # All files are images
            self.showingImage()
        except IndexError as e:
            # Handle index error if no files were selected
            pass

        self.label.setFocus()

    @check_file_loaded
    def nextImage(self):
        """След кадр"""
        try:
            if self.current >= len(self.fname[0]) - 1:
                self.show_popup_window('Это последнее изображение!')
            else:
                self.current += 1
                self.showingImage()
                # self.label=

        except IndexError as e:  # сначала загрузите датасет функция
            print(e)

    @check_file_loaded
    def prevImage(self):
        """Пред кадр"""
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
    path_detector_onnx = r"./model_data/best.onnx"
    path_classifier_pkl = r"C:\workspace\hakaton\hackCBreakthrough\main\model_data\logs_model_ensemble_cpu.pkl"

    class_names = ['кликун', 'малый', 'щипун']
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    providers = ['CPUExecutionProvider']
    detector_model = load_model_detector(
        model_path=path_detector_onnx,
        class_names=class_names,
        providers=providers,
        input_shape=(640, 640),
        score_thresh=[0.2, 0.2, 0.2]
    )
    classifier_model = None
    # classifier_model = ResNet()
    # classifier_model.load_state_dict(torch.load('C:\workspace\hakaton\hackCBreakthrough\main\model_data\logs_model_ensemble_cpu.pth', map_location='cpu'))
    App = QApplication(sys.argv)
    window = ImageGallery(detector_model, classifier_model)
    sys.exit(App.exec())