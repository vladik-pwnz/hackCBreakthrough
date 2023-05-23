import pickle
import PIL
from torchvision import transforms
import pickle
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import cv2
# import detectron2
# import PIL
import torch
import numpy as np

# скачать нужные библиотеки:
'''
!pip install torchensemble
!python -m pip install pyyaml==5.1
import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities.
    # See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))
'''

# код модели:

from torchvision.models import resnet18


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.requires_grad_(False)
        self.resnet.layer4[1].requires_grad_(True)
        self.classificator = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=False),
            torch.nn.Linear(1000, 3, bias=True)
        )

    def forward(self, x):
        return self.classificator(self.resnet(x))


# код обучения ансабля резнетов:
from torchensemble import VotingClassifier
from tqdm import tqdm

model = ResNet()
learning_rate = 5e-4
weight_decay = 1e-4
epochs = 7

ensemble = VotingClassifier(
    estimator=model,               # estimator is your pytorch model
    n_estimators=3,
    cuda=True                 # number of base estimators
)

# Set the optimizer
ensemble.set_optimizer(
    "Adam",                                 # type of parameter optimizer
    lr=learning_rate,                       # learning rate of parameter optimizer
    weight_decay=weight_decay,              # weight decay of parameter optimizer
)

# Set the learning rate scheduler
ensemble.set_scheduler(
    "CosineAnnealingLR",                    # type of learning rate scheduler
    T_max=epochs,                           # additional arguments on the scheduler
)

# Train the ensemble
# tqdm(ensemble.fit(
#     dataloader['train'],
#     epochs=epochs                # number of training epochs
# ))


class HandleDetectionHead:
    def __init__(self, pickle_path, device):
        with open(pickle_path, "rb") as file:
            self.model = pickle.load(file)
            self.model = self.model.to(device)


def predict_boxes(self, image_path):
    image = cv2.imread(image_path)
    pred_dict = self.model(image)
    return pred_dict['instances'].pred_boxes.tensor


def draw_boxes(self, image_path):
    image = read_image(image_path)
    pred = self.predict_boxes(image_path)
    image_with_boxes = draw_bounding_boxes(image, pred, width=3, colors=(255, 255, 0))
    img = transforms.ToPILImage()(image_with_boxes)


class ClassificationModel:
    def write(self, file_path, obj):
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
            return obj

    def __init__(self, model_path, detection_model_path=None, device="cuda"):
        self.model = self.load(model_path)
        if detection_model_path is not None:
            self.detection_heads = HandleDetectionHead(detection_model_path, device)
        self.device = device

    def predict(self, image, batched=False):
        # image = PIL.Image.open(image_path)
        # image = transforms.ToTensor()(image)
        image = image.astype(np.float32)
        image = image / 255.0
        image = image.transpose((2, 0, 1))

        if not batched:
            image = image[None,...]
        image = torch.Tensor(image)
        self.model = self.model.to(self.device)
        image = image.to(self.device)

        class_pred = self.model.predict(image)
        # head_bboxes = self.detection_heads.predict_boxes(image_path)
        # print(head_bboxes.shape)
        # head_preds = class_pred
        # for box in head_bboxes:
        #     top = int(box[1])
        #     left = int(box[0])
        #     height = int(abs(box[1] - box[3]))
        #     width = int(abs(box[0] - box[2]))
        #     crop_image = transforms.functional.crop(image.squeeze(0), top, left, height, width)
        #     head_preds += self.model.predict(crop_image.unsqueeze(0))
        #     crop_image = transforms.ToPILImage()(crop_image)
        #     crop_image.show()
        # head_preds /= len(head_bboxes) + 1
        return class_pred


if __name__ == "__main__":

    # detection_model_pkl_path = r"C:\workspace\hakaton\hackCBreakthrough\main\model_data\model_detection_heads_cpu.pkl"
    image_path = r"C:\workspace\hakaton\dataset\klikun\images\1.jpg"


    model_pkl_path = r"C:\workspace\hakaton\hackCBreakthrough\main\model_data\logs_model_ensemble_cpu.pkl"
    cls_model = ClassificationModel(model_pkl_path)

    img = cv2.imread(image_path)[:,:,::-1]
    print(cls_model.predict(img))
    torch.save(model.cpu().state_dict(), '../model_data/logs_model_ensemble_cpu.pth')

    # C:\workspace\hakaton\hackCBreakthrough\main\model_data\logs_model_ensemble.pkl
    # C:\workspace\hakaton\hackCBreakthrough\main\model_data\model_detection_heads.pkl