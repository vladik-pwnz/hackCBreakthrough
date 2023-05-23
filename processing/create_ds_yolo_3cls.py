import os

import cv2
import pandas as pd
from tqdm import tqdm

from pathbook.pathbook import *

# Путь к папке с тренировочными данными
root_yolo = r"C:\workspace\hakaton\dataset_yolo"
data_folder_train = os.path.join(root_yolo, 'train')
data_folder_val = os.path.join(root_yolo, 'val')
root_dataset = path_initial_train_dataset

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('test.csv')
for df, data_folder in [(train_df, data_folder_train), (val_df, data_folder_val)]:

    # Создание папок для изображений и разметки, если они не существуют
    image_folder = os.path.join(data_folder, 'images')
    label_folder = os.path.join(data_folder, 'labels')
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    # Проходимся по каждой строке в DataFrame train_df
    for index, row in tqdm(df.iterrows()):
        # Получаем путь к изображению и разметке
        image_path = row['path_img']
        # image_path = image_path.replace('=', '-')
        # label_path = row['path_mask']
        # image_path = os.path.join(root_dataset, row['type_name'], image_path)
        image_path = os.path.join(root_dataset, image_path)
        image_path = image_path.replace('=', '-')
        # print(image_path)
        if not os.path.exists(image_path):
            print(image_path)
            continue
        # Загружаем изображение
        image = cv2.imread(image_path)

        # Получаем размеры изображения
        height, width, _ = image.shape

        # Сохраняем изображение в папку "train/images"
        image_filename = os.path.basename(image_path)
        image_save_path = os.path.join(image_folder, image_filename)
        cv2.imwrite(image_save_path, image)

        # Сохраняем разметку в папку "train/labels" в формате YOLO
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_save_path = os.path.join(label_folder, label_filename)

        # Получаем координаты бокса в нормализованных координатах YOLO
        x1 = row['x1'] / width
        y1 = row['y1'] / height
        x2 = row['x2'] / width
        y2 = row['y2'] / height

        # Расчет центра и размера бокса
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        box_width = x2 - x1
        box_height = y2 - y1

        # Создаем строку разметки в формате YOLO
        label = f"{row['type_id']} {x_center} {y_center} {box_width} {box_height}\n"

        # Записываем строку разметки в файл
        with open(label_save_path.replace('=', '-'), 'a') as f:
            f.write(label)

    print("Сохранение завершено.")
