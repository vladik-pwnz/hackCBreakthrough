from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torchvision import transforms
from matplotlib import colors, pyplot as plt
import seaborn as sns
import torch.nn as nn

%matplotlib inline

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from func_classes import *

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

DATA_MODES = ['train', 'val', 'test'] #засунуть в конфиги поменяв,установив значения
RESCALE_SIZE = 224 #засунуть в конфиги поменяв,установив значения
TEST_PIC = Path('path/to/one/pic') #засунуть в конфиги поменяв,установив значения
PATH_TO_WEIGHTS = Path('/content/drive/MyDrive/model_weights_3.pth') #засунуть в конфиги поменяв,установив значения

DEVICE = torch.device("cuda") 

test_files = [TEST_PIC]

simple_cnn = AlexNetV1(3).to(DEVICE)
simple_cnn.load_state_dict(torch.load(PATH_TO_WEIGHTS))

test_dataset = Picture_Dataset(test_files, mode="test")
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)
probs = predict(simple_cnn, test_loader)

RESULT = probs