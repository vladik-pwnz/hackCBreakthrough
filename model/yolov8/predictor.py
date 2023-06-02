import os
from ultralytics import YOLO
from numba import njit
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathbook.pathbook import *


@njit
def soft_crop(image, box): # TODO optimese batch
    '''
    Crops the largest rectangle from the image overlapped by the square 
    with the box in the center and 
    sidelenght of 1.4 times maximum of height and width od the box

    Args:
        image (np.ndarray): np.ndarray HWC,BGR like cv2.imread('im.jpg')
        box (np.ndarray): xywh boundary box in absolute values

    Returns:
        np.ndarray: cropped image
    '''
    h, w, c = image.shape
    x, y = box[0], box[1]
    size = max(box[2],box[3])

    X_min = int(x - 1.4*size/2)
    X_max = int(x + 1.4*size/2)
    Y_min = int(y - 1.4*size/2)
    Y_max = int(y + 1.4*size/2)

    if X_min < 0: 
        X_max = min(X_max - X_min, w)
        X_min = 0
    if X_max > w:
        X_min = max(X_min - X_max + w, 0)
        X_max = w
    if Y_min < 0:
        Y_max = min(Y_max - Y_min, h)
        Y_min = 0
    if Y_max > h:
        Y_min = max(Y_min - Y_max + h, 0)
        Y_max = h

    return image[Y_min : Y_max, X_min : X_max]

class ComplexSegOnlyYOLOPredictor:
    def __init__(self, device='cpu') -> None:
        self.model_seg = YOLO(path_segmentor_only)
        self.model_cls = YOLO(path_classifier)
        self.device = device
        self.input = 320
        self.output = ['path','seg conf','width','height','klikun p','maliy p','shipun p']

    def _predict(self, imgs):
        rows=[]

        batch = self.model_seg.predict(imgs, device=self.device, imgsz = self.input, stream=True, verbose=False)

        for res_seg in tqdm(batch, leave=False):
            for box, conf in zip(res_seg.boxes.xywh.numpy(), res_seg.boxes.conf.numpy()): # TODO: use batch instead of loop

                crop = soft_crop(res_seg.orig_img, box)
                res_cls = self.model_cls.predict(crop, device=self.device, imgsz=320, verbose=False)

                rows.append([res_seg.path, conf, box[2], box[3]]+res_cls[0].probs.tolist())
        return rows

    def predict(self, imgs):
        '''
        Makes a prediction for every image in imgs
        If imgs is a directory path, the function walks through each subdirectory also

        Args:
            imgs (path or YOLO compatable type)
        Returns:
            pd.DataFrame: results frame with a row for each detection, 
                          the columns are equal for self.output
        '''
        rows=[]

        try:
            isdir = os.path.isdir(imgs)
        except:
            isdir = False

        if isdir:
            for leaf in tqdm(os.walk(imgs)):
                if len(leaf[2])>0: # TODO: check if contatins images
                    rows.extend(self._predict(leaf[0]))
        else:
            rows.extend(self._predict(imgs))

        return pd.DataFrame(data=rows,columns=self.output)
    
class ComplexYOLOPredictor:
    def __init__(self, device='cpu') -> None:
        self.model_seg = YOLO(path_segmentor)
        self.model_cls = YOLO(path_classifier)
        self.device = device
        self.input = 640
        self.output = ['path','seg conf','seg pred','width','height','klikun p','maliy p','shipun p']

    def _predict(self, imgs):
        rows=[]

        batch = self.model_seg.predict(imgs, device=self.device, imgsz = self.input, stream=True, verbose=False)

        for res_seg in tqdm(batch, leave=False):
            for box, conf in zip(res_seg.boxes.xywh.numpy(), res_seg.boxes.conf.numpy()): # TODO: use batch instead of loop

                crop = soft_crop(res_seg.orig_img, box)
                res_cls = self.model_cls.predict(crop, device=self.device, imgsz=320, verbose=False)

                rows.append([res_seg.path, conf, box[2], box[3]]+res_cls[0].probs.tolist())
        return rows

    def predict(self, imgs):
        '''
        Makes a prediction for every image in imgs
        If imgs is a directory path, the function walks through each subdirectory also

        Args:
            imgs (path or YOLO compatable type)
        Returns:
            pd.DataFrame: results frame with a row for each detection, 
                          the columns are equal for self.output
        '''
        rows=[]

        try:
            isdir = os.path.isdir(imgs)
        except:
            isdir = False

        if isdir:
            for leaf in tqdm(os.walk(imgs)):
                if len(leaf[2])>0: # TODO: check if contatins images
                    rows.extend(self._predict(leaf[0]))
        else:
            rows.extend(self._predict(imgs))

        return pd.DataFrame(data=rows,columns=self.output)
    