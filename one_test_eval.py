{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from torch.utils.data import Dataset, DataLoader\n",
    "import torch\n",
    "import os\n",
    "from PIL import Image\n",
    "from pathlib import Path\n",
    "import pickle\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import numpy as np\n",
    "from torchvision import transforms\n",
    "from matplotlib import colors, pyplot as plt\n",
    "import seaborn as sns\n",
    "import torch.nn as nn\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(action='ignore', category=DeprecationWarning)\n",
    "\n",
    "from func_classes import *\n",
    "\n",
    "np.random.seed(42)\n",
    "torch.manual_seed(42)\n",
    "torch.cuda.manual_seed(42)\n",
    "torch.backends.cudnn.deterministic = True\n",
    "\n",
    "DATA_MODES = ['train', 'val', 'test'] #засунуть в конфиги поменяв,установив значения\n",
    "RESCALE_SIZE = 224 #засунуть в конфиги поменяв,установив значения\n",
    "TEST_PIC = Path('path/to/one/pic') #засунуть в конфиги поменяв,установив значения\n",
    "PATH_TO_WEIGHTS = Path('/content/drive/MyDrive/model_weights_3.pth') #засунуть в конфиги поменяв,установив значения\n",
    "\n",
    "DEVICE = torch.device(\"cuda\") \n",
    "\n",
    "test_files = [TEST_PIC]\n",
    "\n",
    "simple_cnn = AlexNetV1(3).to(DEVICE)\n",
    "simple_cnn.load_state_dict(torch.load(PATH_TO_WEIGHTS))\n",
    "\n",
    "test_dataset = Picture_Dataset(test_files, mode=\"test\")\n",
    "test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)\n",
    "probs = predict(simple_cnn, test_loader)\n",
    "\n",
    "RESULT = probs"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
