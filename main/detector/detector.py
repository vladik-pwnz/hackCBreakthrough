import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

def load_model_detector(model_path, input_shape=(1024, 1280), class_names=['klikun', 'maliy', 'shipun'], score_thresh=0.2,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
    if model_path.endswith(".onnx"):
        model_detect = ModelDetectorYoloONNX(model_path, providers,
                                                      input_shape=input_shape,
                                                      score_thresh=score_thresh,
                                                      class_names=class_names,
                                                      )
    else:
        raise ValueError(f"{model_path} модель должна быть в onnx формате.")
    test_img = np.random.random((*input_shape, 3)).astype(np.float32)
    # print(test_img.shape)
    model_detect(test_img)  # 1 проход для проверки
    return model_detect

class ModelDetectorYoloONNX:
    def __init__(self, model_path, providers, input_shape,
                 score_thresh: list, class_names: list):
        self.class_names = class_names
        self.class_names += [f'c{i}' for i in range(len(class_names), 100)]
        if type(score_thresh) is float:
            score_thresh = [score_thresh]
        if len(score_thresh) < len(class_names):
            score_thresh += [score_thresh[-1] for _ in range(len(score_thresh), len(class_names))]
        self.score_thresh = score_thresh
        self.input_shape = input_shape
        self.input_h = input_shape[0]
        self.input_w = input_shape[1]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.inname = [i.name for i in self.session.get_inputs()]
        self.outname = [i.name for i in self.session.get_outputs()]
        # self.session.get_providers()
        self._ratio = None

    def __call__(self, image):
        '''

        :param image:
        :return: list [class_name, score, x0, y0, x1, y1], [..], ..
        '''
        image = self.preproc_img(image)
        cls_score_bbox = []

        im = image
        inp = {self.inname[0]: im[None, :]}
        outputs = self.session.run(self.outname, inp)[0]

        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            cls_id = int(cls_id)
            if score > self.score_thresh[cls_id]:
                x0 = max(0, int(x0)) if self._ratio is None else max(0, int(x0 / self._ratio))
                y0 = max(0, int(y0)) if self._ratio is None else max(0, int(y0 / self._ratio))
                x1 = min(int(x1), self.input_w) if self._ratio is None else min(int(x1 / self._ratio), self.input_w)
                y1 = min(int(y1), self.input_h) if self._ratio is None else min(int(y1 / self._ratio), self.input_h)
                cls_score_bbox.append([self.class_names[cls_id], score, x0, y0, x1, y1])

        return cls_score_bbox

    def preproc_img(self, image: np.ndarray, mean=None, std=None) -> np.ndarray:
        """
        Входящее изображение будет отресайзено с паддингами,
        если размер не соответсвует размеру детектора.
        Переведено в диапазон 0..1
        Перемещение размерности каналов в начала.

        :param image: Изображение
        :return: img_out, scale
        """
        img = np.array(image)
        # print(img.shape)
        # BGR to RGB ##############################!!!!!!!!!!!!!!!!!
        # img = img[:, :, ::-1]

        self._ratio = None

        # ресайз с паддингом если размер изображения не соответствует входу детектора
        if self.input_shape[0] != image.shape[0] \
                or self.input_shape[1] != image.shape[1] \
                or image.shape[2] != 3:
            input_size = self.input_shape
            if len(image.shape) == 3:
                padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
            else:
                padded_img = np.ones(input_size) * 114.0
            ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img
            img = padded_img
            self._ratio = ratio

        # нормализация
        img = img.astype(np.float32)
        img /= 255.0
        if mean is not None:
            img -= mean
        if std is not None:
            img /= std
        # каналы перемещаем в первую размерность
        img = img.transpose((2, 0, 1))
        # img = np.ascontiguousarray(img, dtype=np.float32)
        return img  # img (3, h, w)


if __name__ == "__main__":
    path_to_onnx = "yolo.onnx"
    path_img = "img.jpg"
    img = cv2.imread(path_img)
    detector_model = load_model_detector(model_path=path_to_onnx)
    result = detector_model(img)
    print(result)
